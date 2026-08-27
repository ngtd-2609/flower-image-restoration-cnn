import unittest

import numpy as np

from src.classification_metrics import holm_bonferroni, mcnemar_exact, paired_bootstrap_difference
from src.image_metrics import delta_e_2000, full_reference_metrics, psnr, ssim


class MetricTests(unittest.TestCase):
    def test_identity_metrics(self):
        image = np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        self.assertGreaterEqual(psnr(image, image), 99)
        self.assertAlmostEqual(ssim(image, image), 1.0, places=6)
        self.assertAlmostEqual(delta_e_2000(image, image), 0.0, places=6)

    def test_holm_bonferroni_is_monotone_and_bounded(self):
        result = holm_bonferroni([0.01, 0.04, 0.03])
        np.testing.assert_allclose(result["adjusted_p_values"], [0.03, 0.06, 0.06])
        np.testing.assert_array_equal(result["reject"], [True, False, False])

    def test_image_metric_sampling_protocol_is_explicit(self):
        image = np.full((224, 224, 3), 128, dtype=np.uint8)
        result = full_reference_metrics(image, image, include_delta_e=True)
        self.assertEqual(result["image_metric_pixel_stride"], 4.0)
        self.assertEqual(result["delta_e_pixel_stride"], 4.0)
        self.assertAlmostEqual(result["ssim"], 1.0, places=6)
        self.assertAlmostEqual(result["delta_e_2000"], 0.0, places=6)

    def test_paired_statistics_contract(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        baseline = np.array([0, 1, 1, 0, 2, 0])
        candidate = np.array([0, 0, 1, 1, 2, 0])
        interval = paired_bootstrap_difference(
            truth, baseline, candidate, samples=100, seed=7
        )
        self.assertGreater(interval["difference"], 0)
        test = mcnemar_exact(truth, baseline, candidate)
        self.assertEqual(test["a_correct_b_wrong"], 0)
        self.assertEqual(test["a_wrong_b_correct"], 2)
        self.assertGreaterEqual(test["p_value"], 0)


if __name__ == "__main__": unittest.main()
