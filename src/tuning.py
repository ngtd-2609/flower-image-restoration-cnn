from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .classification_metrics import summary_metrics
from .config import CLASS_NAMES, LEVELS, METHODS
from .degradations import apply_degradation, stable_seed
from .enhancements import apply_enhancement
from .image_metrics import full_reference_metrics

PARAMETER_GRID = {
    "gamma_correction": [{}],
    "clahe": [{"clip_limit": value} for value in (1.5, 2.0, 2.5)],
    "gaussian_filter": [{"sigma": value} for value in (0.7, 1.0, 1.3)],
    "bilateral_filter": [{"range_sigma": value} for value in (25, 40, 60)],
    "median_filter": [{"kernel": value} for value in (3, 5)],
    "unsharp_mask": [{"radius": radius, "amount": amount} for radius, amount in ((1.0, 0.7), (1.5, 1.0), (2.0, 1.2))],
    "sharpening": [{}],
    "rgb_balance": [{}],
    "hsv_correction": [{}],
    "lab_correction": [{}],
}


def rank_candidates(all_results: pd.DataFrame) -> pd.DataFrame:
    """Select by Macro F1, then SSIM, then lower enhancement latency."""
    required = {"degradation", "level", "method", "macro_f1", "ssim", "latency_ms_per_image"}
    missing = required.difference(all_results.columns)
    if missing:
        raise ValueError(f"Tuning results missing columns: {sorted(missing)}")
    ranked = all_results.sort_values(
        ["degradation", "level", "method", "macro_f1", "ssim", "latency_ms_per_image"],
        ascending=[True, True, True, False, False, True],
    )
    return ranked.groupby(["degradation", "level", "method"], as_index=False).first()


def tune_on_validation(model_service, clean_images, relative_paths, labels):
    y_true = np.asarray([CLASS_NAMES.index(label) for label in labels])
    records = []
    for degradation, methods in METHODS.items():
        for level in LEVELS:
            degraded = [apply_degradation(image, degradation, level, stable_seed(path, degradation, level)) for image, path in zip(clean_images, relative_paths)]
            for method in methods:
                for params in PARAMETER_GRID[method]:
                    enhancement_started = time.perf_counter()
                    enhanced = [apply_enhancement(image, degradation, level, method, params) for image in degraded]
                    enhancement_latency = (time.perf_counter() - enhancement_started) * 1000 / len(enhanced)
                    outputs = model_service.predict_batch(enhanced)
                    predictions = [item["label"] for item in outputs]
                    y_pred = np.asarray([CLASS_NAMES.index(label) for label in predictions])
                    # Delta E is not a selection criterion. It is computed for
                    # final Test reporting only, avoiding expensive irrelevant
                    # color-distance work during Validation grid search.
                    quality = pd.DataFrame([full_reference_metrics(a, b, False) for a, b in zip(clean_images, enhanced)]).mean(numeric_only=True).to_dict()
                    records.append({"degradation": degradation, "level": level, "method": method,
                                    "params": json.dumps(params, sort_keys=True),
                                    "latency_ms_per_image": float(enhancement_latency),
                                    "latency_scope": "enhancement_only",
                                    **quality, **summary_metrics(y_true, y_pred)})
    all_results = pd.DataFrame(records)
    best = rank_candidates(all_results)
    all_results["selected"] = False
    selected_keys = {
        (row.degradation, row.level, row.method, row.params) for row in best.itertuples()
    }
    all_results["selected"] = [
        (row.degradation, row.level, row.method, row.params) in selected_keys
        for row in all_results.itertuples()
    ]
    locked = {f"{row.degradation}|{row.level}|{row.method}": json.loads(row.params) for row in best.itertuples()}
    return all_results, best, locked


def save_tuning(all_results, best, locked, results_dir: Path, config_path: Path, metadata: dict | None = None):
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(results_dir / "validation_tuning_all.csv", index=False)
    best.to_csv(results_dir / "validation_tuning_best.csv", index=False)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_metadata": metadata or {}, "parameters": locked}
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
