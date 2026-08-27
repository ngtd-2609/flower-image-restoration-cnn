import unittest
from pathlib import Path

import pandas as pd


class SplitTests(unittest.TestCase):
    def test_split_contract(self):
        root = Path(__file__).resolve().parents[1]
        frames = {name: pd.read_csv(root / "splits" / f"{name}.csv") for name in ("train", "validation", "test")}
        for frame in frames.values():
            self.assertEqual(set(frame["label"]), {"daisy", "dandelion", "roses", "sunflowers", "tulips"})
            self.assertIn("sha256", frame.columns)
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(set(frames[left]["relative_path"]) & set(frames[right]["relative_path"]))
            self.assertFalse(set(frames[left]["sha256"]) & set(frames[right]["sha256"]))


if __name__ == "__main__": unittest.main()
