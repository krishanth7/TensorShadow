"""Train the TensorShadow DNN and export it for production serving.

Produces two artefacts from the *same* trained weights:

    tensorflow_model/serving/tensorshadow/1/        TF SavedModel  -> TensorFlow Serving
    tensorflow_model/onnx/tensorshadow/1/model.onnx ONNX graph     -> ONNX Runtime
                                                    (tensorflow_model/onnx_server.py,
                                                     Triton onnxruntime backend, KServe)

Both use the input tensor ``features`` (float32, [batch, 10]) and the output
``probabilities`` (float32, [batch, 2]). The exporter checks that ONNX Runtime
reproduces the TensorFlow outputs before it finishes.

The repository's sample CSV has only five rows and two features, so training
uses a deterministic synthetic 10-feature, two-class dataset (a noisy linear
decision boundary). Swap ``make_dataset`` for real data in production.

Usage:
    pip install "tensorflow-cpu>=2.15" onnx onnxruntime
    python tensorflow_model/export_models.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_FEATURES = 10
INPUT_NAME = "features"
OUTPUT_NAME = "probabilities"
SEED = 7


def make_dataset(n: int = 6000, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = np.linspace(-1.0, 1.2, N_FEATURES)
    x = rng.normal(0.0, 1.0, size=(n, N_FEATURES)).astype(np.float32)
    y = ((x @ w + rng.normal(0.0, 0.5, size=n)) > 0).astype(np.int64)
    return x, y


def dense_stack_to_onnx(model, path: str) -> None:
    """Convert a Sequential stack of Dense layers to ONNX with the ONNX helper API.

    The model is plain Dense(relu) → Dense(relu) → Dense(softmax), so an exact
    graph can be written directly from the weights without a converter that
    must track the TensorFlow/Keras release.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits = [], []
    prev = INPUT_NAME
    dense_layers = [layer for layer in model.layers if layer.get_weights()]
    for i, layer in enumerate(dense_layers):
        kernel, bias = layer.get_weights()
        inits += [numpy_helper.from_array(kernel.astype(np.float32), f"W{i}"),
                  numpy_helper.from_array(bias.astype(np.float32), f"B{i}")]
        nodes += [helper.make_node("MatMul", [prev, f"W{i}"], [f"mm{i}"]),
                  helper.make_node("Add", [f"mm{i}", f"B{i}"], [f"z{i}"])]
        activation = layer.get_config().get("activation")
        out = OUTPUT_NAME if i == len(dense_layers) - 1 else f"a{i}"
        if activation == "relu":
            nodes.append(helper.make_node("Relu", [f"z{i}"], [out]))
        elif activation == "softmax":
            nodes.append(helper.make_node("Softmax", [f"z{i}"], [out], axis=-1))
        elif activation in (None, "linear"):
            nodes.append(helper.make_node("Identity", [f"z{i}"], [out]))
        else:
            raise ValueError(f"unsupported activation {activation!r}")
        prev = out

    n_out = dense_layers[-1].get_weights()[1].shape[0]
    graph = helper.make_graph(
        nodes, "tensorshadow",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, ["batch", N_FEATURES])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, ["batch", n_out])],
        initializer=inits,
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                                   producer_name="tensorshadow-export")
    onnx_model.ir_version = 8
    onnx.checker.check_model(onnx_model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(onnx_model, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out", default=HERE, help="directory that receives serving/ and onnx/")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import onnxruntime as ort
    from model import build_model

    tf.keras.utils.set_random_seed(SEED)
    x, y = make_dataset()
    split = int(0.8 * len(x))
    model = build_model((N_FEATURES,), 2)
    model.fit(x[:split], y[:split], epochs=args.epochs, batch_size=64, verbose=0)
    _, acc = model.evaluate(x[split:], y[split:], verbose=0)
    print(f"held-out accuracy: {acc:.4f} ({len(x) - split} samples)")

    # --- TensorFlow Serving: SavedModel with a named serving signature -------
    saved_dir = os.path.join(args.out, "serving", "tensorshadow", "1")
    shutil.rmtree(saved_dir, ignore_errors=True)

    def serve(features):
        return {OUTPUT_NAME: model(features, training=False)}

    # ExportArchive tracks the Keras 3 variables so TF Serving can restore them.
    archive = tf.keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint(name="serving_default", fn=serve,
                         input_signature=[tf.TensorSpec([None, N_FEATURES], tf.float32, name=INPUT_NAME)])
    archive.write_out(saved_dir, verbose=False)
    print(f"SavedModel  -> {os.path.relpath(saved_dir)}")

    # --- ONNX Runtime ----------------------------------------------------------
    onnx_path = os.path.join(args.out, "onnx", "tensorshadow", "1", "model.onnx")
    dense_stack_to_onnx(model, onnx_path)
    print(f"ONNX        -> {os.path.relpath(onnx_path)}")

    # --- Parity: the reloaded SavedModel and ONNX Runtime must agree -----------
    probe = x[split:split + 256]
    reloaded = tf.saved_model.load(saved_dir).signatures["serving_default"]
    tf_out = reloaded(**{INPUT_NAME: tf.constant(probe)})[OUTPUT_NAME].numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run([OUTPUT_NAME], {INPUT_NAME: probe})[0]
    max_diff = float(np.max(np.abs(tf_out - ort_out)))
    print(f"TF vs ONNX Runtime max |diff| over {len(probe)} samples: {max_diff:.2e}")
    if max_diff > 1e-5:
        raise SystemExit("ONNX export does not match TensorFlow")

    reference = {
        "input": [float(v) for v in probe[0]],
        "tensorflow": [float(v) for v in tf_out[0]],
        "onnxruntime": [float(v) for v in ort_out[0]],
        "held_out_accuracy": float(acc),
    }
    with open(os.path.join(args.out, "export_reference.json"), "w") as fh:
        json.dump(reference, fh, indent=2)
    print("reference sample -> export_reference.json")


if __name__ == "__main__":
    main()
