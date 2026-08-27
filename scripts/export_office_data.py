from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "results/final/condition_metrics.csv")
p = pd.read_csv(ROOT / "results/final/per_class_metrics.csv")
s = pd.read_csv(ROOT / "results/final/statistical_tests.csv")
facts = json.loads((ROOT / "artifacts/canonical_facts.json").read_text(encoding="utf-8"))
clean = m[m.image_type.eq("clean")].iloc[0]
deg = m[m.image_type.eq("degraded")]
enh = m[m.image_type.eq("enhanced")]
best = enh.loc[enh.macro_f1.idxmax()]
worst = deg.loc[deg.macro_f1.idxmin()]
clean_pc = p[p.condition_id.eq("clean")].sort_values("f1", ascending=False)
payload = {
    "facts": facts,
    "clean": {"accuracy": float(clean.accuracy), "macro_f1": float(clean.macro_f1), "latency": float(clean.inference_time_ms_per_image_mean)},
    "best_enhanced": {"condition_id": best.condition_id, "macro_f1": float(best.macro_f1), "accuracy": float(best.accuracy)},
    "worst_degraded": {"condition_id": worst.condition_id, "macro_f1": float(worst.macro_f1), "accuracy": float(worst.accuracy)},
    "top_conditions": m.nlargest(10, "macro_f1")[["condition_id", "image_type", "accuracy", "macro_f1", "ssim", "inference_time_ms_per_image_mean"]].to_dict("records"),
    "degradation_summary": deg.groupby("degradation", as_index=False).agg(macro_f1=("macro_f1", "mean"), ssim=("ssim", "mean")).to_dict("records"),
    "type_summary": m.groupby("image_type", as_index=False).agg(macro_f1=("macro_f1", "mean"), accuracy=("accuracy", "mean"), ssim=("ssim", "mean")).to_dict("records"),
    "clean_per_class": clean_pc[["class", "precision", "recall", "f1", "support"]].to_dict("records"),
    "significant_rows": int(s.mcnemar_reject_h0_0_05.astype(str).str.lower().eq("true").sum()),
    "history_epochs": len(pd.read_csv(ROOT / "artifacts/training/history.csv")),
}
(ROOT / "artifacts/office_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(ROOT / "artifacts/office_data.json")
