from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config import CLASS_NAMES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_artifact_readiness(
    model_path: Path,
    metadata_path: Path,
    class_names_path: Path,
    locked_path: Path,
) -> dict:
    """Return an honest readiness verdict without loading TensorFlow."""
    errors: list[str] = []
    metadata: dict = {}
    locked: dict = {}

    if not model_path.exists():
        errors.append("Chưa có checkpoint MobileNetV2 thật")
    if not metadata_path.exists():
        errors.append("Thiếu model_metadata.json")
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("model_metadata.json không hợp lệ")
    if metadata and metadata.get("status") != "FULL_RUN_COMPLETE":
        errors.append("Model metadata chưa chứng minh FULL_RUN_COMPLETE")

    if not class_names_path.exists():
        errors.append("Thiếu class_names.json")
    else:
        try:
            class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
            if class_names != CLASS_NAMES or metadata.get("class_names", CLASS_NAMES) != CLASS_NAMES:
                errors.append("Thứ tự lớp không khớp contract")
        except (OSError, json.JSONDecodeError):
            errors.append("class_names.json không hợp lệ")

    if model_path.exists() and metadata:
        expected_hash = metadata.get("model_sha256")
        if not expected_hash or sha256_file(model_path) != expected_hash:
            errors.append("Checksum checkpoint không khớp model metadata")

    if not locked_path.exists():
        errors.append("Chưa có locked_enhancement_params.json từ Validation tuning")
    else:
        try:
            locked = json.loads(locked_path.read_text(encoding="utf-8"))
            lock_metadata = locked.get("_metadata", {})
            parameters = locked.get("parameters", {})
            if lock_metadata.get("selection_split") != "validation":
                errors.append("Locked params không chứng minh nguồn Validation")
            if lock_metadata.get("quick_run") is True or lock_metadata.get("run_mode") == "QUICK_RUN":
                errors.append("Locked params chỉ đến từ QUICK_RUN")
            if len(parameters) != 33:
                errors.append("Locked params không đủ 33 tổ hợp")
        except (OSError, json.JSONDecodeError):
            errors.append("locked_enhancement_params.json không hợp lệ")

    return {
        "ready": not errors,
        "errors": errors,
        "metadata": metadata,
        "locked": locked.get("parameters", locked) if locked else {},
        "model_sha256": sha256_file(model_path) if model_path.exists() else None,
    }
