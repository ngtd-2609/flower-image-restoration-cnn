from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .classification_metrics import (
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_difference,
    per_class_frame,
    summary_metrics,
)
from .config import CLASS_NAMES, DEGRADATION_PARAMS
from .degradations import apply_degradation, stable_seed
from .enhancements import apply_enhancement
from .experiment_matrix import ExperimentCondition, build_experiment_matrix
from .image_metrics import full_reference_metrics
from .locked_params import params_for


@dataclass
class EvaluationArtifacts:
    condition_metrics: pd.DataFrame
    predictions: pd.DataFrame
    per_class_metrics: pd.DataFrame
    statistical_tests: pd.DataFrame
    confusion_matrices: dict[str, np.ndarray]


def expected_condition_count() -> int:
    return len(build_experiment_matrix())


def _condition_images(
    condition: ExperimentCondition,
    clean_images: list[np.ndarray],
    relative_paths: list[str],
    locked_params: dict,
    degraded_cache: dict[tuple[str, str], list[np.ndarray]],
) -> list[np.ndarray]:
    if condition.image_type == "clean":
        return clean_images
    key = (condition.degradation, condition.level)
    if key not in degraded_cache:
        degraded_cache[key] = [
            apply_degradation(
                image,
                condition.degradation,
                condition.level,
                stable_seed(path, condition.degradation, condition.level),
            )
            for image, path in zip(clean_images, relative_paths)
        ]
    degraded = degraded_cache[key]
    if condition.image_type == "degraded":
        return degraded
    parameters = params_for(
        locked_params,
        condition.degradation,
        condition.level,
        condition.enhancement_method,
    )
    return [
        apply_enhancement(
            image,
            condition.degradation,
            condition.level,
            condition.enhancement_method,
            parameters,
        )
        for image in degraded
    ]


def evaluate_full_experiment(
    model_service,
    clean_images: list[np.ndarray],
    relative_paths: list[str],
    labels: list[str],
    hashes: list[str] | None = None,
    locked_params: dict | None = None,
    batch_size: int = 32,
    bootstrap_samples: int = 2000,
    latency_runs: int = 3,
    run_id: str | None = None,
) -> EvaluationArtifacts:
    """Evaluate one immutable model over the canonical 49-condition matrix."""
    if not (len(clean_images) == len(relative_paths) == len(labels)) or not clean_images:
        raise ValueError("images, paths and labels must have the same non-zero length")
    if any(label not in CLASS_NAMES for label in labels):
        raise ValueError("Unknown class label")
    if latency_runs < 1:
        raise ValueError("latency_runs must be at least 1")
    hashes = hashes or [""] * len(labels)
    if len(hashes) != len(labels):
        raise ValueError("hashes must have the same length as labels")

    run_id = run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    locked_params = locked_params or {}
    y_true = np.asarray([CLASS_NAMES.index(label) for label in labels])
    degraded_cache: dict[tuple[str, str], list[np.ndarray]] = {}
    condition_rows: list[dict] = []
    prediction_rows: list[dict] = []
    per_class_rows: list[pd.DataFrame] = []
    prediction_indices: dict[str, np.ndarray] = {}
    confusion_matrices: dict[str, np.ndarray] = {}

    for condition in build_experiment_matrix():
        images = _condition_images(condition, clean_images, relative_paths, locked_params, degraded_cache)
        # Produce the scientific predictions once. Benchmark a representative
        # batch separately so repeated latency measurements do not multiply the
        # full 49 x N inference cost or change stored predictions.
        outputs = model_service.predict_batch(images, batch_size=batch_size)
        benchmark_images = images[: min(batch_size, len(images))]
        model_service.predict_batch(benchmark_images, batch_size=batch_size)
        latency_samples = []
        for _ in range(latency_runs):
            started = time.perf_counter()
            model_service.predict_batch(benchmark_images, batch_size=batch_size)
            latency_samples.append((time.perf_counter() - started) * 1000 / len(benchmark_images))
        if outputs is None or len(outputs) != len(images):
            raise RuntimeError("Model output count does not match the input batch")
        y_pred = np.asarray([CLASS_NAMES.index(item["label"]) for item in outputs])
        prediction_indices[condition.condition_id] = y_pred
        confusion_matrices[condition.condition_id] = confusion_matrix(
            y_true, y_pred, labels=range(len(CLASS_NAMES))
        )

        if condition.image_type == "clean":
            quality = {
                "psnr": 100.0,
                "ssim": 1.0,
                "delta_e_2000": 0.0,
                "image_metric_pixel_stride": 4.0,
                "delta_e_pixel_stride": 4.0,
                "mean_brightness": float(np.mean([image.mean() for image in clean_images])),
                "rms_contrast": float(np.mean([image.std() for image in clean_images])),
                "edge_preservation_ratio": 1.0,
                "histogram_distance": 0.0,
            }
        else:
            quality = pd.DataFrame(
                [
                    full_reference_metrics(clean, candidate, condition.degradation == "color_cast")
                    for clean, candidate in zip(clean_images, images)
                ]
            ).mean(numeric_only=True).to_dict()

        enhancement_params = (
            params_for(
                locked_params,
                condition.degradation,
                condition.level,
                condition.enhancement_method,
            )
            if condition.image_type == "enhanced"
            else {}
        )
        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "image_type": condition.image_type,
                "degradation": condition.degradation,
                "level": condition.level,
                "degradation_params": json.dumps(
                    DEGRADATION_PARAMS.get(condition.degradation, {}).get(condition.level, {}),
                    sort_keys=True,
                ),
                "enhancement_method": condition.enhancement_method,
                "enhancement_params": json.dumps(enhancement_params, sort_keys=True),
                "sample_count": len(labels),
                "inference_time_ms_per_image_mean": float(np.mean(latency_samples)),
                "inference_time_ms_per_image_median": float(np.median(latency_samples)),
                "inference_time_ms_per_image_p95": float(np.percentile(latency_samples, 95)),
                "latency_runs": latency_runs,
                "latency_batch_size": len(benchmark_images),
                "latency_warmup_runs": 1,
                **quality,
                **summary_metrics(y_true, y_pred),
            }
        )

        class_frame = per_class_frame(y_true, y_pred, CLASS_NAMES)
        class_frame.insert(0, "condition_id", condition.condition_id)
        class_frame.insert(1, "image_type", condition.image_type)
        class_frame.insert(2, "degradation", condition.degradation)
        class_frame.insert(3, "level", condition.level)
        class_frame.insert(4, "enhancement_method", condition.enhancement_method)
        per_class_rows.append(class_frame)

        for path, digest, truth, output in zip(relative_paths, hashes, labels, outputs):
            prediction_rows.append(
                {
                    "run_id": run_id,
                    "condition_id": condition.condition_id,
                    "image_type": condition.image_type,
                    "degradation": condition.degradation,
                    "level": condition.level,
                    "enhancement_method": condition.enhancement_method,
                    "relative_path": path,
                    "sha256": digest,
                    "true_label": truth,
                    "predicted_label": output["label"],
                    "confidence": output["confidence"],
                    "correct": output["label"] == truth,
                    "probabilities_json": json.dumps(output["probabilities"], sort_keys=True),
                }
            )

    statistical_rows: list[dict] = []
    for condition in build_experiment_matrix():
        if condition.image_type != "enhanced":
            continue
        degraded_id = f"{condition.degradation}__{condition.level}__degraded"
        degraded_pred = prediction_indices[degraded_id]
        enhanced_pred = prediction_indices[condition.condition_id]
        mcnemar = mcnemar_exact(y_true, degraded_pred, enhanced_pred)
        for metric in ("accuracy", "macro_f1"):
            bootstrap = paired_bootstrap_difference(
                y_true,
                degraded_pred,
                enhanced_pred,
                metric=metric,
                samples=bootstrap_samples,
            )
            statistical_rows.append(
                {
                    "condition_id": condition.condition_id,
                    "baseline_condition_id": degraded_id,
                    **bootstrap,
                    "mcnemar_a_correct_b_wrong": mcnemar["a_correct_b_wrong"],
                    "mcnemar_a_wrong_b_correct": mcnemar["a_wrong_b_correct"],
                    "mcnemar_p_value_raw": mcnemar["p_value"],
                }
            )

    if statistical_rows:
        unique_mcnemar = {}
        for row in statistical_rows:
            unique_mcnemar[row["condition_id"]] = row["mcnemar_p_value_raw"]
        condition_ids = list(unique_mcnemar)
        correction = holm_bonferroni([unique_mcnemar[item] for item in condition_ids])
        adjusted_by_condition = dict(zip(condition_ids, correction["adjusted_p_values"]))
        rejected_by_condition = dict(zip(condition_ids, correction["reject"]))
        for row in statistical_rows:
            condition_id = row["condition_id"]
            row["mcnemar_p_value_holm"] = float(adjusted_by_condition[condition_id])
            row["mcnemar_reject_h0_0_05"] = bool(rejected_by_condition[condition_id])
            row["multiple_testing_family"] = "33 enhanced-vs-degraded McNemar comparisons"

    condition_metrics = pd.DataFrame(condition_rows)
    if len(condition_metrics) != expected_condition_count() or condition_metrics["condition_id"].duplicated().any():
        raise AssertionError("49-condition evaluation contract failed")
    return EvaluationArtifacts(
        condition_metrics=condition_metrics,
        predictions=pd.DataFrame(prediction_rows),
        per_class_metrics=pd.concat(per_class_rows, ignore_index=True),
        statistical_tests=pd.DataFrame(statistical_rows),
        confusion_matrices=confusion_matrices,
    )


def evaluate_49_conditions(
    model_service,
    clean_images,
    relative_paths,
    labels,
    locked_params: dict | None = None,
) -> pd.DataFrame:
    """Backward-compatible metric table used by the notebook."""
    return evaluate_full_experiment(
        model_service,
        list(clean_images),
        list(relative_paths),
        list(labels),
        locked_params=locked_params,
    ).condition_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_evaluation_artifacts(
    artifacts: EvaluationArtifacts,
    output_dir: Path,
    *,
    model_path: Path,
    split_path: Path,
    metadata: dict | None = None,
) -> dict:
    """Persist all final tables, confusion matrices and a checksum manifest."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "condition_metrics": output_dir / "condition_metrics.csv",
        "predictions": output_dir / "predictions.csv",
        "per_class_metrics": output_dir / "per_class_metrics.csv",
        "statistical_tests": output_dir / "statistical_tests.csv",
    }
    artifacts.condition_metrics.to_csv(files["condition_metrics"], index=False)
    artifacts.predictions.to_csv(files["predictions"], index=False)
    artifacts.per_class_metrics.to_csv(files["per_class_metrics"], index=False)
    artifacts.statistical_tests.to_csv(files["statistical_tests"], index=False)

    for condition_id, matrix in artifacts.confusion_matrices.items():
        csv_path = confusion_dir / f"{condition_id}.csv"
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(csv_path)
        if condition_id == "clean" or "__strong__" in condition_id:
            figure, axis = plt.subplots(figsize=(6.4, 5.4))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=axis,
            )
            axis.set(xlabel="Dự đoán", ylabel="Nhãn thật", title=condition_id)
            figure.tight_layout()
            figure.savefig(confusion_dir / f"{condition_id}.png", dpi=180)
            plt.close(figure)

    confusion_files = sorted(confusion_dir.glob("*"))
    tracked_files = [*files.values(), *confusion_files, model_path, split_path]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "model_file": model_path.name,
        "model_sha256": _sha256(model_path),
        "split_file": split_path.name,
        "split_sha256": _sha256(split_path),
        "condition_count": len(artifacts.condition_metrics),
        "prediction_rows": len(artifacts.predictions),
        "per_class_rows": len(artifacts.per_class_metrics),
        "statistical_test_rows": len(artifacts.statistical_tests),
        "metadata": metadata or {},
        "files": {path.relative_to(output_dir.parent.parent).as_posix(): _sha256(path) for path in tracked_files},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
