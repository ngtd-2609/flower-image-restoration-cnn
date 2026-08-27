from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import SEED


def grouped_stratified_split(inventory: pd.DataFrame, n_splits: int = 20) -> dict[str, pd.DataFrame]:
    """Create 14/3/3 grouped folds = approximately 70/15/15."""
    required = {"relative_path", "label", "sha256"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"Inventory missing columns: {sorted(missing)}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = [test_idx for _, test_idx in splitter.split(inventory, inventory["label"], inventory["sha256"])]
    result = {
        "train": inventory.iloc[np.concatenate(folds[:14])].copy(),
        "validation": inventory.iloc[np.concatenate(folds[14:17])].copy(),
        "test": inventory.iloc[np.concatenate(folds[17:])].copy(),
    }
    assert_no_leakage(result)
    return result


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            paths = set(splits[left]["relative_path"]) & set(splits[right]["relative_path"])
            hashes = set(splits[left]["sha256"]) & set(splits[right]["sha256"])
            if paths or hashes:
                raise AssertionError(f"Leakage {left}/{right}: paths={len(paths)}, hashes={len(hashes)}")


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame[["relative_path", "label", "sha256"]].sort_values(
            ["label", "relative_path"]
        ).to_csv(output_dir / f"{name}.csv", index=False)
