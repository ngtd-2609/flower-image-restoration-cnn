from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app_components.pipeline import run_pipeline
from app_components.readiness import inspect_artifact_readiness
from src.config import DEGRADATION_PARAMS
from src.data_loader import load_rgb
from src.inference import ModelService

ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "models/mobilenetv2_flowers.keras"
readiness = inspect_artifact_readiness(
    model_path, ROOT / "models/model_metadata.json", ROOT / "models/class_names.json",
    ROOT / "configs/locked_enhancement_params.json",
)
if not readiness["ready"]:
    raise SystemExit(f"App not ready: {readiness['errors']}")

test_rows = __import__("pandas").read_csv(ROOT / "splits/test.csv").head(3)
service = ModelService(model_path)
records = []
for row in test_rows.itertuples():
    image = load_rgb(ROOT / row.relative_path, size=224)
    output = run_pipeline(
        image, identifier=row.relative_path, degradation="low_light", level="medium",
        method="gamma_correction", model_service=service, locked_params=readiness["locked"],
        degradation_params=DEGRADATION_PARAMS["low_light"]["medium"],
    )
    records.append({
        "relative_path": row.relative_path,
        "clean_label": output.predictions["clean"]["label"],
        "degraded_label": output.predictions["degraded"]["label"],
        "enhanced_label": output.predictions["enhanced"]["label"],
        "processing_time_ms": output.processing_time_ms,
    })

payload = {
    "status": "PASS",
    "verified_at_utc": datetime.now(UTC).isoformat(),
    "model_sha256": readiness["model_sha256"],
    "readiness": True,
    "single_and_batch_core_count": len(records),
    "condition": "low_light|medium|gamma_correction",
    "records": records,
    "scope": "local core smoke; public deployment not verified",
}
path = ROOT / "artifacts/app_smoke_test.json"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
