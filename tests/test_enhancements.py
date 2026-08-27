import unittest

import numpy as np

from src.config import METHODS
from src.enhancements import apply_enhancement


class EnhancementTests(unittest.TestCase):
    def test_output_contract(self):
        image = np.random.default_rng(42).integers(0, 256, (48, 48, 3), dtype=np.uint8)
        for degradation, methods in METHODS.items():
            for method in methods:
                output = apply_enhancement(image, degradation, "medium", method)
                self.assertEqual(output.shape, image.shape)
                self.assertEqual(output.dtype, np.uint8)
                self.assertTrue(np.isfinite(output).all())

    def test_invalid_mapping_rejected(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_enhancement(image, "gaussian_blur", "light", "median_filter")


if __name__ == "__main__": unittest.main()
