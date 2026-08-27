import unittest

import numpy as np

from src.preprocessing import ensure_rgb_uint8, prepare_batch, resize_with_letterbox


class PreprocessingTests(unittest.TestCase):
    def test_grayscale_and_rgba_are_normalized(self):
        gray = np.full((8, 12), 127, dtype=np.uint8)
        rgba = np.zeros((8, 12, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        self.assertEqual(ensure_rgb_uint8(gray).shape, (8, 12, 3))
        self.assertEqual(ensure_rgb_uint8(rgba).shape, (8, 12, 3))

    def test_letterbox_is_square_and_deterministic(self):
        image = np.random.default_rng(42).integers(0, 256, (40, 80, 3), dtype=np.uint8)
        first = resize_with_letterbox(image, 64)
        second = resize_with_letterbox(image, 64)
        self.assertEqual(first.shape, (64, 64, 3))
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first[:16] == 0))

    def test_batch_contract(self):
        image = np.zeros((16, 20, 3), dtype=np.uint8)
        batch = prepare_batch([image, image], 32)
        self.assertEqual(batch.shape, (2, 32, 32, 3))
        self.assertEqual(batch.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
