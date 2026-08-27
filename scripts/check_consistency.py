"""Cross-check canonical facts across source tables and final visible artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/cross_artifact_consistency.json"
CLASS_COUNTS = {"daisy": 633, "dandelion": 898, "roses": 641, "sunflowers": 699, "tulips": 799}
MEMBERS = {"24100358": "Nguyễn Tùng Dương", "24100065": "Trịnh Ngọc Nga", "24106898": "Trương Việt Thành"}


def office_text(path: Path) -> str:
    with ZipFile(path) as archive:
        raw = " ".join(archive.read(n).decode("utf-8", errors="ignore") for n in archive.namelist() if n.endswith(".xml"))
    return re.sub(r"<[^>]+>", " ", raw)


def main() -> None:
    inventory = pd.read_csv(ROOT / "data/inventory.csv")
    splits = {n: pd.read_csv(ROOT / f"splits/{n}.csv") for n in ("train", "validation", "test")}
    metadata = json.loads((ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
    final_manifest_path = ROOT / "results/final/manifest.json"
    full_complete = final_manifest_path.exists() and metadata.get("status") == "FULL_RUN_COMPLETE"
    paths = [ROOT / "README.md", ROOT / "docs/DATA_CARD.md", ROOT / "docs/MODEL_CARD.md",
             ROOT / "docs/REPORT_FINAL.docx", ROOT / "docs/SLIDES_FINAL.pptx", ROOT / "docs/ASSIGNMENT_FINAL.xlsx"]
    texts = [p.read_text(encoding="utf-8") if p.suffix == ".md" else office_text(p) for p in paths if p.exists()]
    joined = "\n".join(texts); compact = re.sub(r"\s+", "", joined)
    checks = {
        "inventory_3670": len(inventory) == 3670,
        "class_counts": inventory.label.value_counts().to_dict() == CLASS_COUNTS,
        "split_counts": {k: len(v) for k, v in splits.items()} == {"train": 2571, "validation": 549, "test": 550},
        "evaluation_state_truthful": metadata.get("status") in {
            "FULL_RUN_TRAINED — EVALUATION_PENDING", "FULL_RUN_COMPLETE"
        },
        "model_checkpoint_hashed": len(metadata.get("model_sha256", "")) == 64,
        "canonical_members": all(mid in joined and re.sub(r"\s+", "", name) in compact for mid, name in MEMBERS.items()),
        "office_files_exist": all(p.exists() for p in paths[3:]),
        "deployment_truthful": "DEPLOY_PENDING" in joined or "DEPLOY_READY_BUT_NOT_DEPLOYED" in joined,
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        checks[f"{left}_{right}_path_disjoint"] = set(splits[left].relative_path).isdisjoint(set(splits[right].relative_path))
        checks[f"{left}_{right}_hash_disjoint"] = set(splits[left].sha256).isdisjoint(set(splits[right].sha256))
    if full_complete:
        metrics = pd.read_csv(ROOT / "results/final/condition_metrics.csv")
        predictions = pd.read_csv(ROOT / "results/final/predictions.csv")
        manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        checks.update({
            "condition_rows": len(metrics) == 49 and metrics.condition_id.nunique() == 49,
            "prediction_rows": len(predictions) == 26950,
            "manifest_counts": manifest.get("condition_count") == 49 and manifest.get("prediction_rows") == 26950,
        })
    payload = {
        "checks": checks,
        "pass": all(checks.values()),
        "evaluation": "FULL_RUN_COMPLETE" if full_complete else "FULL_49_PENDING_USER_GPU",
        "deployment": "DEPLOY_PENDING_USER_ACTION",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
