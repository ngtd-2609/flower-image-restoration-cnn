from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, SEED
from src.data_loader import build_tf_dataset, load_rgb
from src.data_split import grouped_stratified_split, save_splits
from src.data_validation import save_validation, scan_dataset
from src.error_analysis import build_error_analysis_from_predictions, top_confusion_pairs
from src.evaluate import evaluate_full_experiment, save_evaluation_artifacts
from src.inference import ModelService
from src.train import train_two_stage
from src.tuning import PARAMETER_GRID, save_tuning, tune_on_validation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    import tensorflow as tf

    tf.random.set_seed(SEED)


def save_environment(output: Path, command: str, duration_seconds: float | None = None) -> None:
    import cv2
    import sklearn
    import tensorflow as tf

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cpu": platform.processor() or platform.machine(),
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "cuda_build": tf.sysconfig.get_build_info().get("cuda_version"),
        "cudnn_build": tf.sysconfig.get_build_info().get("cudnn_version"),
        "seed": SEED,
        "command": command,
        "duration_seconds": duration_seconds,
        "git_commit": "not_available_no_git_repository",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_history(history: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, train_name, val_name, title in (
        (axes[0], "loss", "val_loss", "Loss"),
        (axes[1], "accuracy", "val_accuracy", "Accuracy"),
    ):
        axis.plot(history["epoch"], history[train_name], label="Train")
        axis.plot(history["epoch"], history[val_name], label="Validation")
        axis.axvline((history["stage"] == "head").sum() + 0.5, color="#F79646", linestyle="--", label="Fine-tune")
        axis.set(title=title, xlabel="Epoch")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def load_split_images(frame: pd.DataFrame) -> list[np.ndarray]:
    # All degradations, enhancements, image metrics and inference use the same
    # canonical 224 x 224 letterboxed representation.
    return [load_rgb(ROOT / relative_path, size=224) for relative_path in frame["relative_path"]]


def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Audit, train, tune and evaluate the final scientific pipeline")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "flower_photos")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="Train and persist evidence, then stop before tuning/evaluation")
    parser.add_argument("--regenerate-splits", action="store_true")
    parser.add_argument("--quick-run", action="store_true", help="Use at most 10 samples per split; never for report metrics")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise SystemExit(f"Dataset missing: {data_dir}. Expected five class directories.")

    inventory, audit = scan_dataset(data_dir)
    save_validation(inventory, audit, ROOT / "results")
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    inventory.to_csv(ROOT / "data" / "inventory.csv", index=False)
    if args.regenerate_splits:
        save_splits(grouped_stratified_split(inventory), ROOT / "splits")
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    seed_everything()

    frames = {
        name: pd.read_csv(ROOT / "splits" / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    if args.quick_run:
        frames = {name: frame.groupby("label", group_keys=False).head(2).reset_index(drop=True) for name, frame in frames.items()}

    model_path = ROOT / "models" / "mobilenetv2_flowers.keras"
    history_path = artifacts_dir / "training" / "history.csv"
    training_started = time.perf_counter()
    trained_this_run = False
    if not args.skip_train and (args.retrain or not model_path.exists()):
        train_ds = build_tf_dataset(frames["train"], ROOT, CLASS_NAMES, training=True)
        validation_ds = build_tf_dataset(frames["validation"], ROOT, CLASS_NAMES, training=False)
        _, history = train_two_stage(
            train_ds,
            validation_ds,
            model_path,
            head_epochs=2 if args.quick_run else 15,
            fine_tune_epochs=1 if args.quick_run else 10,
            history_path=history_path,
        )
        plot_history(history, artifacts_dir / "training" / "learning_curves.png")
        trained_this_run = True
    if not model_path.exists():
        raise SystemExit("Real MobileNetV2 checkpoint is missing; training cannot be skipped.")

    if args.train_only:
        import tensorflow as tf

        metadata_path = ROOT / "models" / "model_metadata.json"
        metadata = {
            "architecture": "MobileNetV2",
            "weights": "ImageNet",
            "status": "FULL_RUN_TRAINED — EVALUATION_PENDING",
            "model_file": model_path.relative_to(ROOT).as_posix(),
            "model_sha256": sha256_file(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "tensorflow_version": tf.__version__,
            "cuda": tf.sysconfig.get_build_info().get("cuda_version"),
            "cudnn": tf.sysconfig.get_build_info().get("cudnn_version"),
            "input_shape": [None, 224, 224, 3],
            "output_shape": [None, len(CLASS_NAMES)],
            "class_names": CLASS_NAMES,
            "train_split_sha256": sha256_file(ROOT / "splits" / "train.csv"),
            "validation_split_sha256": sha256_file(ROOT / "splits" / "validation.csv"),
            "seed": SEED,
            "git_commit": "not_available_no_git_repository",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "training_duration_seconds": time.perf_counter() - training_started,
            "preprocessing": "EXIF transpose + RGB + letterbox LANCZOS 224 + in-graph MobileNetV2 preprocess_input",
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        save_environment(
            artifacts_dir / "environment.json",
            command=" ".join(sys.argv),
            duration_seconds=time.perf_counter() - run_started,
        )
        print("Training completed; evaluation intentionally pending.")
        return

    service = ModelService(model_path)
    validation_images = load_split_images(frames["validation"])
    tuning_all, tuning_best, locked = tune_on_validation(
        service,
        validation_images,
        frames["validation"]["relative_path"].tolist(),
        frames["validation"]["label"].tolist(),
    )
    validation_split_path = ROOT / "splits" / "validation.csv"
    save_tuning(
        tuning_all,
        tuning_best,
        locked,
        ROOT / "results",
        ROOT / "configs" / "locked_enhancement_params.json",
        metadata={
            "selection_split": "validation",
            "metric": "macro_f1",
            "tie_break_1": "ssim",
            "tie_break_2": "latency_ms_per_image",
            "seed": SEED,
            "candidate_grid": PARAMETER_GRID,
            "image_metric_protocol": "all images; deterministic pixel stride 4 (56x56 lattice at 224x224)",
            "delta_e_protocol": "Test only; same deterministic 56x56 lattice; not used for selection",
            "validation_split_sha256": sha256_file(validation_split_path),
            "model_sha256": sha256_file(model_path),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": "not_available_no_git_repository",
            "quick_run": args.quick_run,
        },
    )

    test_images = load_split_images(frames["test"])
    evaluation = evaluate_full_experiment(
        service,
        test_images,
        frames["test"]["relative_path"].tolist(),
        frames["test"]["label"].tolist(),
        hashes=frames["test"]["sha256"].tolist(),
        locked_params=locked,
        bootstrap_samples=200 if args.quick_run else 2000,
        latency_runs=1 if args.quick_run else 5,
    )
    manifest = save_evaluation_artifacts(
        evaluation,
        ROOT / "results" / "final",
        model_path=model_path,
        split_path=ROOT / "splits" / "test.csv",
        metadata={
            "seed": SEED,
            "quick_run": args.quick_run,
            "selection_split": "validation",
            "image_metric_protocol": "all images; deterministic pixel stride 4 (56x56 lattice at 224x224)",
            "delta_e_protocol": "color-cast and clean; same deterministic 56x56 lattice",
        },
    )
    error_frame = build_error_analysis_from_predictions(evaluation.predictions)
    error_path = ROOT / "results" / "final" / "error_analysis.csv"
    error_frame.to_csv(error_path, index=False)
    confusion_pair_path = ROOT / "results" / "final" / "top_confusion_pairs.csv"
    top_confusion_pairs(evaluation.predictions).to_csv(confusion_pair_path, index=False)
    for path in (error_path, confusion_pair_path):
        manifest["files"][path.relative_to(ROOT).as_posix()] = sha256_file(path)
    manifest["error_analysis_rows"] = len(error_frame)
    (ROOT / "results" / "final" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata_path = ROOT / "models" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    final_metadata = {
            "status": "FULL_RUN_COMPLETE" if not args.quick_run else "QUICK_RUN_ONLY_NOT_FOR_REPORT",
            "model_file": model_path.relative_to(ROOT).as_posix(),
            "model_sha256": sha256_file(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "tensorflow_version": __import__("tensorflow").__version__,
            "cuda": __import__("tensorflow").sysconfig.get_build_info().get("cuda_version"),
            "cudnn": __import__("tensorflow").sysconfig.get_build_info().get("cudnn_version"),
            "input_shape": [None, 224, 224, 3],
            "output_shape": [None, len(CLASS_NAMES)],
            "class_names": CLASS_NAMES,
            "train_split_sha256": sha256_file(ROOT / "splits" / "train.csv"),
            "validation_split_sha256": sha256_file(validation_split_path),
            "seed": SEED,
            "git_commit": "not_available_no_git_repository",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "preprocessing": "EXIF transpose + RGB + letterbox LANCZOS 224 + in-graph MobileNetV2 preprocess_input",
        }
    if trained_this_run:
        final_metadata["training_duration_seconds"] = time.perf_counter() - training_started
    metadata.update(final_metadata)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    full_run_metadata = {
        "status": final_metadata["status"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_sha256": final_metadata["model_sha256"],
        "test_split_sha256": sha256_file(ROOT / "splits" / "test.csv"),
        "experiment_rows": int(manifest["condition_count"]),
        "prediction_rows": int(manifest["prediction_rows"]),
        "per_class_rows": int(manifest["per_class_rows"]),
        "statistical_test_rows": int(manifest["statistical_test_rows"]),
        "quick_run": bool(args.quick_run),
        "selection_split": "validation",
        "deployment_status": "DEPLOY_READY_BUT_NOT_DEPLOYED",
        "image_metric_protocol": "all images; deterministic pixel stride 4 (56x56 lattice at 224x224)",
    }
    (artifacts_dir / "full_run_metadata.json").write_text(
        json.dumps(full_run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_environment(
        artifacts_dir / "environment.json",
        command=" ".join(sys.argv),
        duration_seconds=time.perf_counter() - run_started,
    )
    print("Full pipeline completed. Run scripts/validate_project.py --require-final.")


if __name__ == "__main__":
    main()
