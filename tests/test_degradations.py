import unittest

import numpy as np

from src.config import LEVELS
from src.degradations import apply_degradation, stable_seed


class DegradationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((64, 64, 3), 128, dtype=np.uint8)

    def test_shape_dtype_range(self):
        for kind in ("low_light", "gaussian_noise", "salt_pepper", "gaussian_blur", "color_cast"):
            for level in LEVELS:
                output = apply_degradation(self.image, kind, level, stable_seed("sample.jpg", kind, level))
                self.assertEqual(output.shape, self.image.shape)
                self.assertEqual(output.dtype, np.uint8)
                self.assertGreaterEqual(int(output.min()), 0)
                self.assertLessEqual(int(output.max()), 255)

    def test_noise_reproducibility(self):
        seed = stable_seed("sample.jpg", "gaussian_noise", "medium")
        first = apply_degradation(self.image, "gaussian_noise", "medium", seed)
        second = apply_degradation(self.image, "gaussian_noise", "medium", seed)
        self.assertTrue(np.array_equal(first, second))

    def test_strong_low_light_is_darker(self):
        light = apply_degradation(self.image, "low_light", "light", 42)
        strong = apply_degradation(self.image, "low_light", "strong", 42)
        self.assertLess(float(strong.mean()), float(light.mean()))


if __name__ == "__main__": unittest.main()
