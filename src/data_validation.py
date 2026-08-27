from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from .config import CLASS_NAMES


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Validate images and return an auditable inventory plus duplicate report."""
    rows: list[dict] = []
    bad: list[dict] = []
    for label in CLASS_NAMES:
        class_dir = data_dir / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            try:
                # Do not let junction resolution, drive casing, or platform separators
                # leak into the scientific identity used by splits and manifests.
                relative_path = (Path("data") / "flower_photos" / label / path.name).as_posix()
                with Image.open(path) as image:
                    original_format = image.format or "unknown"
                    original_mode = image.mode
                    exif_orientation = image.getexif().get(274)
                    image.verify()
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.load()
                    width, height = image.size
                    mode = image.mode
                digest = sha256_file(path)
                rows.append({
                    "path": relative_path,
                    "relative_path": relative_path,
                    "label": label,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / max(height, 1),
                    "mode": mode,
                    "original_format": original_format,
                    "original_mode": original_mode,
                    "exif_orientation": exif_orientation,
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                    "decode_status": "ok",
                })
            except Exception as exc:  # noqa: BLE001
                try:
                    display_path = str(path.relative_to(data_dir.parent.parent))
                except ValueError:
                    display_path = path.name
                bad.append({"path": display_path, "error": repr(exc)})

    inventory = pd.DataFrame(rows)
    duplicate_hashes = {
        digest: f"dup_{index:03d}"
        for index, (digest, group) in enumerate(inventory.groupby("sha256"), start=1)
        if len(group) > 1
    }
    if not inventory.empty:
        inventory["duplicate_group"] = inventory["sha256"].map(duplicate_hashes).fillna("")
    groups = [group.to_dict("records") for _, group in inventory.groupby("sha256") if len(group) > 1]
    cross_label = [group for group in groups if len({row["label"] for row in group}) > 1]
    report = {
        "valid_images": len(inventory),
        "source_file_count": int(len(inventory) + len(bad)),
        "bad_images": bad,
        "duplicate_group_count": len(groups),
        "cross_label_duplicate_group_count": len(cross_label),
        "duplicate_groups": groups,
        "class_counts": inventory["label"].value_counts().sort_index().to_dict(),
        "inventory_columns": list(inventory.columns),
        "audit_contract": "PIL verify + full decode + EXIF transpose + RGB + SHA256 + duplicate/cross-label grouping",
    }
    return inventory, report


def save_validation(inventory: pd.DataFrame, report: dict, results_dir: Path) -> None:
    """Save audit diagnostics; the canonical inventory lives only in data/inventory.csv."""
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report.get("bad_images", [])).to_csv(results_dir / "bad_images.csv", index=False)
    (results_dir / "data_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
