"""Minimal ONNX Runtime model server speaking the Open Inference Protocol (v2).

This is the same wire protocol as NVIDIA Triton and KServe, so the Go backend's
``onnxruntime`` backend works unchanged against this server, Triton's
onnxruntime backend, or a KServe InferenceService.

Model repository layout (Triton compatible)::

    <repository>/<model_name>/<version>/model.onnx

Endpoints::

    GET  /v2/health/live                    GET  /v2/health/ready
    GET  /v2/models/{name}[/versions/{v}]   GET  /v2/models/{name}[/versions/{v}]/ready
    POST /v2/models/{name}[/versions/{v}]/infer

Usage::

    pip install onnxruntime numpy
    python tensorflow_model/onnx_server.py --repository tensorflow_model/onnx --port 8001
"""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort

V2_DTYPES = {
    "tensor(float)": ("FP32", np.float32), "tensor(double)": ("FP64", np.float64),
    "tensor(int64)": ("INT64", np.int64), "tensor(int32)": ("INT32", np.int32),
}
NUMPY_BY_V2 = {"FP32": np.float32, "FP64": np.float64, "INT64": np.int64, "INT32": np.int32}
MAX_BODY = 32 << 20
ROUTE = re.compile(r"^/v2/models/(?P<name>[^/]+)(?:/versions/(?P<version>[^/]+))?(?P<action>/ready|/infer)?$")


class Model:
    def __init__(self, name: str, version: str, path: str, threads: int) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.name, self.version = name, version
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])

    def tensors(self, items) -> list:
        return [{"name": t.name, "datatype": V2_DTYPES.get(t.type, ("BYTES", None))[0],
                 "shape": [d if isinstance(d, int) else -1 for d in t.shape]} for t in items]

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "versions": [self.version], "platform": "onnxruntime_onnx",
                "inputs": self.tensors(self.session.get_inputs()),
                "outputs": self.tensors(self.session.get_outputs())}


class Repository:
    def __init__(self, root: str, threads: int) -> None:
        self.models: Dict[str, Dict[str, Model]] = {}
        for name in sorted(os.listdir(root)):
            mdir = os.path.join(root, name)
            if not os.path.isdir(mdir):
                continue
            for version in sorted(os.listdir(mdir)):
                path = os.path.join(mdir, version, "model.onnx")
                if os.path.isfile(path):
                    self.models.setdefault(name, {})[version] = Model(name, version, path, threads)
        if not self.models:
            raise SystemExit(f"no <name>/<version>/model.onnx found under {root}")

    def get(self, name: str, version: Optional[str]) -> Optional[Model]:
        versions = self.models.get(name)
        if not versions:
            return None
        if version is None:  # latest numeric version
            version = max(versions, key=lambda v: (v.isdigit(), int(v) if v.isdigit() else 0, v))
        return versions.get(version)


class Handler(BaseHTTPRequestHandler):
    repo: Repository
    server_version = "TensorShadowONNX/1.0"
    protocol_version = "HTTP/1.1"
    # Headers and body go out as separate writes; without TCP_NODELAY, Nagle's
    # algorithm plus the client's delayed ACK adds ~40 ms to every response.
    disable_nagle_algorithm = True

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        if os.environ.get("ONNX_SERVER_ACCESS_LOG"):
            super().log_message(fmt, *args)

    def reply(self, status: int, body: Optional[Dict[str, Any]] = None) -> None:
        raw = json.dumps(body if body is not None else {}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def route(self) -> Tuple[Optional[Model], Optional[str]]:
        m = ROUTE.match(self.path.split("?", 1)[0])
        if not m:
            self.reply(404, {"error": f"unknown endpoint {self.path}"})
            return None, None
        model = self.repo.get(m["name"], m["version"])
        if model is None:
            self.reply(404, {"error": f"model {m['name']!r} version {m['version'] or 'latest'} not found"})
            return None, None
        return model, m["action"]

    def do_GET(self) -> None:
        if self.path in ("/v2/health/live", "/v2/health/ready"):
            return self.reply(200, {"live": True, "ready": True})
        if self.path == "/v2":
            return self.reply(200, {"name": "tensorshadow-onnx", "version": ort.__version__, "extensions": []})
        model, action = self.route()
        if model is None:
            return
        if action == "/ready":
            return self.reply(200, {"name": model.name, "ready": True})
        if action is None:
            return self.reply(200, model.metadata())
        self.reply(405, {"error": "use POST for /infer"})

    def do_POST(self) -> None:
        model, action = self.route()
        if model is None:
            return
        if action != "/infer":
            return self.reply(405, {"error": "only /infer accepts POST"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return self.reply(400, {"error": f"body must be 1..{MAX_BODY} bytes"})
        try:
            req = json.loads(self.rfile.read(length))
            feeds = {}
            for t in req["inputs"]:
                dtype = NUMPY_BY_V2.get(t.get("datatype", "FP32"))
                if dtype is None:
                    raise ValueError(f"unsupported datatype {t.get('datatype')!r}")
                feeds[t["name"]] = np.asarray(t["data"], dtype=dtype).reshape(t["shape"])
            wanted = [o["name"] for o in req.get("outputs", [])] or None
            results = model.session.run(wanted, feeds)
            names = wanted or [o.name for o in model.session.get_outputs()]
        except (KeyError, ValueError, TypeError) as err:
            return self.reply(400, {"error": f"invalid inference request: {err}"})
        except Exception as err:  # onnxruntime errors (bad input names, shapes)
            return self.reply(400, {"error": str(err).splitlines()[0]})
        outputs = []
        for name, arr in zip(names, results):
            v2type = {np.dtype(np.float32): "FP32", np.dtype(np.float64): "FP64",
                      np.dtype(np.int64): "INT64", np.dtype(np.int32): "INT32"}.get(arr.dtype, "FP32")
            outputs.append({"name": name, "datatype": v2type, "shape": list(arr.shape),
                            "data": arr.reshape(-1).tolist()})
        self.reply(200, {"model_name": model.name, "model_version": model.version,
                         "id": req.get("id", ""), "outputs": outputs})


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="ONNX Runtime Open Inference Protocol server")
    parser.add_argument("--repository", default=os.path.join(here, "onnx"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--threads", type=int, default=1, help="intra-op threads per session")
    args = parser.parse_args()

    Handler.repo = Repository(args.repository, args.threads)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    for name, versions in Handler.repo.models.items():
        print(f"loaded {name} versions {sorted(versions)} (onnxruntime {ort.__version__})", flush=True)
    print(f"Open Inference Protocol v2 server on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
