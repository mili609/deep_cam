"""Swapper failures must propagate instead of returning camera pixels."""
import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from modules.processors.frame import face_swapper as swapper

class SwapFailureTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.source = SimpleNamespace(normed_embedding=np.ones(512))
        self.target = SimpleNamespace(kps=np.ones((5, 2)))

    def test_missing_model_raises(self):
        with patch.object(swapper, "get_face_swapper", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "not loaded"):
                swapper.swap_face(self.source, self.target, self.frame)

    def test_missing_embedding_raises(self):
        with patch.object(swapper, "get_face_swapper", return_value=object()):
            with self.assertRaisesRegex(ValueError, "recognition embedding"):
                swapper.swap_face(SimpleNamespace(), self.target, self.frame)

    def test_empty_model_output_raises(self):
        model = SimpleNamespace(get=lambda *a, **k: (None, np.eye(2, 3)))
        with patch.object(swapper, "get_face_swapper", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "returned no image"):
                swapper.swap_face(self.source, self.target, self.frame)

if __name__ == "__main__":
    unittest.main()
