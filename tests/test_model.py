import unittest
import numpy as np
from tensorflow_model.model import build_model

class TestOmniModel(unittest.TestCase):
    def test_model_build(self):
        model = build_model((10,), 2)
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.input_shape, (None, 10))

if __name__ == '__main__':
    unittest.main()
