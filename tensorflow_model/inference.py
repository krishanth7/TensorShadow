import tensorflow as tf
import numpy as np
import json
import sys

class OmniInference:
    def __init__(self, model_path: str):
        print(f"Loading TensorShadow Model from {model_path}...")
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, input_data: list):
        if not self.model:
            return {"error": "Model not loaded"}
        
        # Prepare data
        arr = np.array(input_data).reshape(1, -1)
        
        # Run inference
        predictions = self.model.predict(arr)
        result = float(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))
        
        return {
            "prediction": result,
            "confidence": confidence,
            "status": "success"
        }

if __name__ == "__main__":
    # Example usage for CLI testing
    if len(sys.argv) > 1:
        data = json.loads(sys.argv[1])
        engine = OmniInference("tensorflow_model/saved_model")
        print(json.dumps(engine.predict(data)))
    else:
        print("Usage: python inference.py '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]'")


