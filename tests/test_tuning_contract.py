from __future__ import annotations

import inspect
import unittest

import pandas as pd

from src import tuning


class TuningContractTests(unittest.TestCase):
    def test_three_level_tie_break(self) -> None:
        frame = pd.DataFrame([
            {"degradation": "low_light", "level": "light", "method": "clahe", "params": "a", "macro_f1": 0.8, "ssim": 0.90, "latency_ms_per_image": 3.0},
            {"degradation": "low_light", "level": "light", "method": "clahe", "params": "b", "macro_f1": 0.8, "ssim": 0.91, "latency_ms_per_image": 9.0},
            {"degradation": "low_light", "level": "light", "method": "clahe", "params": "c", "macro_f1": 0.8, "ssim": 0.91, "latency_ms_per_image": 2.0},
        ])
        selected = tuning.rank_candidates(frame).iloc[0]
        self.assertEqual(selected["params"], "c")

    def test_validation_tuning_has_no_test_split_dependency(self) -> None:
        source = inspect.getsource(tuning).lower()
        self.assertNotIn("test.csv", source)
        self.assertNotIn("splits/test", source)


if __name__ == "__main__":
    unittest.main()
