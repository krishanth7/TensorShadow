import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAS_TF = True
except ImportError:
    HAS_TF = False
    from sklearn.neural_network import MLPClassifier

def build_model(input_shape: tuple, num_classes: int):
    """
    Builds a deep learning model. Falls back to Scikit-Learn if TensorFlow is unavailable.
    """
    if HAS_TF:
        print("[Engine] Using TensorFlow backend")
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    else:
        print("[Engine] TensorFlow not found. Falling back to Scikit-Learn MLP backend.")
        # Simulating a similar architecture with MLP
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=10)
        return model

if __name__ == "__main__":
    if HAS_TF:
        model = build_model((10,), 2)
        model.summary()
    else:
        print("Model architecture defined (Ready for Scikit-Learn fallback)")
