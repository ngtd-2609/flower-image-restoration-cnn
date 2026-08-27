from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEVELS, METHODS
from .degradations import apply_degradation, stable_seed
from .enhancements import apply_enhancement


def build_error_analysis_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Derive paired transition errors from canonical stored predictions.

    This avoids a second model pass and guarantees error cases have the same
    condition IDs, hashes, labels and confidences as results/final/predictions.csv.
    """
    required = {
        "condition_id", "image_type", "degradation", "level", "enhancement_method",
        "relative_path", "sha256", "true_label", "predicted_label", "confidence",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    clean = predictions[predictions["condition_id"] == "clean"].set_index("relative_path")
    rows: list[dict] = []
    for condition_id, enhanced in predictions[predictions["image_type"] == "enhanced"].groupby("condition_id"):
        sample = enhanced.iloc[0]
        baseline_id = f'{sample["degradation"]}__{sample["level"]}__degraded'
        degraded = predictions[predictions["condition_id"] == baseline_id].set_index("relative_path")
        for item in enhanced.itertuples(index=False):
            c = clean.loc[item.relative_path]
            d = degraded.loc[item.relative_path]
            clean_ok = c.predicted_label == item.true_label
            degraded_ok = d.predicted_label == item.true_label
            enhanced_ok = item.predicted_label == item.true_label
            if clean_ok and not degraded_ok and enhanced_ok:
                group = "recovered_by_enhancement"
            elif degraded_ok and not enhanced_ok:
                group = "harmed_by_enhancement"
            elif clean_ok and not degraded_ok:
                group = "clean_correct_degraded_wrong"
            elif not enhanced_ok and item.confidence > d.confidence:
                group = "confidence_increased_still_wrong"
            elif not clean_ok and not degraded_ok and not enhanced_ok:
                group = "always_wrong"
            elif not enhanced_ok:
                group = "enhanced_still_wrong"
            else:
                continue
            rows.append({
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "condition_id": condition_id,
                "true_label": item.true_label,
                "degradation": item.degradation,
                "level": item.level,
                "enhancement_method": item.enhancement_method,
                "error_group": group,
                "clean_pred": c.predicted_label,
                "clean_confidence": c.confidence,
                "degraded_pred": d.predicted_label,
                "degraded_confidence": d.confidence,
                "enhanced_pred": item.predicted_label,
                "enhanced_confidence": item.confidence,
                "confidence_delta": item.confidence - d.confidence,
            })
    return pd.DataFrame(rows)


def top_confusion_pairs(predictions: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    wrong = predictions[predictions["true_label"] != predictions["predicted_label"]]
    if wrong.empty:
        return pd.DataFrame(columns=["condition_id", "degradation", "level", "enhancement_method", "true_label", "predicted_label", "count"])
    counts = (
        wrong.groupby([
            "condition_id", "degradation", "level", "enhancement_method",
            "true_label", "predicted_label",
        ])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["condition_id", "count"], ascending=[True, False])
    )
    return counts.groupby("condition_id", group_keys=False).head(top_n).reset_index(drop=True)


def categorize_errors(paths, y_true, clean_pred, degraded_pred, enhanced_pred) -> pd.DataFrame:
    rows = []
    for path, truth, clean, degraded, enhanced in zip(paths, y_true, clean_pred, degraded_pred, enhanced_pred):
        if clean == truth and degraded != truth:
            category = "clean_correct_degraded_wrong"
        elif degraded != truth and enhanced == truth:
            category = "recovered_by_enhancement"
        elif degraded == truth and enhanced != truth:
            category = "harmed_by_enhancement"
        elif enhanced != truth:
            category = "enhanced_still_wrong"
        else:
            continue
        rows.append({"relative_path": path, "true_label": truth, "clean_pred": clean,
                     "degraded_pred": degraded, "enhanced_pred": enhanced, "category": category})
    return pd.DataFrame(rows)


def build_error_analysis(
    model_service,
    clean_images,
    relative_paths,
    labels,
    hashes: list[str] | None = None,
    locked_params: dict | None = None,
    max_examples_per_group: int = 20,
) -> pd.DataFrame:
    """Trace prediction transitions for every strong corruption and mapped enhancement.

    The returned rows keep prediction confidence so the report can distinguish a genuine
    recovery from a low-confidence label flip.  Images are never written by this function.
    """
    locked_params = locked_params or {}
    hashes = hashes or [""] * len(labels)
    clean_outputs = model_service.predict_batch(clean_images)
    _clean_pred = [item["label"] for item in clean_outputs]
    rows: list[dict] = []
    for degradation, methods in METHODS.items():
        for level in LEVELS:
            degraded_images = [
                apply_degradation(image, degradation, level, stable_seed(path, degradation, level))
                for image, path in zip(clean_images, relative_paths)
            ]
            degraded_outputs = model_service.predict_batch(degraded_images)
            for method in methods:
                params = locked_params.get(f"{degradation}|{level}|{method}", {})
                enhanced_images = [
                    apply_enhancement(image, degradation, level, method, params)
                    for image in degraded_images
                ]
                enhanced_outputs = model_service.predict_batch(enhanced_images)
                condition_id = f"enhanced__{degradation}__{level}__{method}"
                for path, digest, truth, clean_out, degraded_out, enhanced_out in zip(
                    relative_paths, hashes, labels, clean_outputs, degraded_outputs, enhanced_outputs
                ):
                    clean_ok = clean_out["label"] == truth
                    degraded_ok = degraded_out["label"] == truth
                    enhanced_ok = enhanced_out["label"] == truth
                    if clean_ok and not degraded_ok and enhanced_ok:
                        group = "recovered_by_enhancement"
                    elif degraded_ok and not enhanced_ok:
                        group = "harmed_by_enhancement"
                    elif clean_ok and not degraded_ok:
                        group = "clean_correct_degraded_wrong"
                    elif not enhanced_ok:
                        if not clean_ok and not degraded_ok:
                            group = "always_wrong"
                        elif enhanced_out["confidence"] > degraded_out["confidence"]:
                            group = "confidence_increased_still_wrong"
                        else:
                            group = "enhanced_still_wrong"
                    else:
                        continue
                    rows.append({
                        "relative_path": path,
                        "sha256": digest,
                        "condition_id": condition_id,
                        "true_label": truth,
                        "degradation": degradation,
                        "level": level,
                        "enhancement_method": method,
                        "error_group": group,
                        "clean_pred": clean_out["label"],
                        "clean_confidence": clean_out["confidence"],
                        "degraded_pred": degraded_out["label"],
                        "degraded_confidence": degraded_out["confidence"],
                        "enhanced_pred": enhanced_out["label"],
                        "enhanced_confidence": enhanced_out["confidence"],
                        "confidence_delta": enhanced_out["confidence"] - degraded_out["confidence"],
                    })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.sort_values("confidence_delta", key=lambda values: np.abs(values), ascending=False)
        .groupby(["degradation", "level", "enhancement_method", "error_group"], group_keys=False)
        .head(max_examples_per_group)
        .reset_index(drop=True)
    )
