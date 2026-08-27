from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
metrics = pd.read_csv(ROOT / "results/final/condition_metrics.csv")
out = ROOT / "figures/classification"; out.mkdir(parents=True, exist_ok=True)

image_columns = [
    "condition_id", "image_type", "degradation", "level", "enhancement_method", "sample_count",
    "psnr", "ssim", "delta_e_2000", "mean_brightness", "rms_contrast",
    "edge_preservation_ratio", "histogram_distance",
]
metrics[image_columns].to_csv(ROOT / "results/image_quality_results_49_conditions.csv", index=False)


def plot_metric(column: str, filename: str, title: str, ylabel: str) -> None:
    frame = metrics.sort_values(column, ascending=False).head(16).sort_values(column)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = frame.image_type.map({"clean": "#17365D", "degraded": "#E76F51", "enhanced": "#2A9D8F"})
    ax.barh(frame.condition_id.str.replace("__", " / ", regex=False), frame[column], color=colors)
    ax.set(title=title, xlabel=ylabel); ax.grid(axis="x", alpha=.2); fig.tight_layout()
    fig.savefig(out / filename, dpi=190); plt.close(fig)


plot_metric("psnr", "psnr_comparison.png", "Top 16 điều kiện theo PSNR", "PSNR (dB)")
plot_metric("ssim", "ssim_comparison.png", "Top 16 điều kiện theo SSIM", "SSIM")
plot_metric("edge_preservation_ratio", "edge_preservation.png", "Top 16 điều kiện theo bảo toàn biên", "Edge preservation ratio")
delta = metrics.dropna(subset=["delta_e_2000"]).sort_values("delta_e_2000").head(16)
fig, ax = plt.subplots(figsize=(10, 6)); ax.barh(delta.condition_id.str.replace("__", " / ", regex=False), delta.delta_e_2000, color="#8B5CF6")
ax.set(title="16 điều kiện có sai khác màu thấp nhất", xlabel="Delta E 2000 (thấp hơn tốt hơn)"); ax.grid(axis="x", alpha=.2); fig.tight_layout()
fig.savefig(out / "delta_e_comparison.png", dpi=190); plt.close(fig)

error_out = ROOT / "figures/error_analysis"; error_out.mkdir(parents=True, exist_ok=True)
errors = pd.read_csv(ROOT / "results/final/error_analysis.csv")
pairs = pd.read_csv(ROOT / "results/final/top_confusion_pairs.csv")
counts = errors.error_group.value_counts().sort_values()
fig, ax = plt.subplots(figsize=(8, 4.8)); ax.barh(counts.index, counts.values, color="#E76F51")
ax.set(title="Phân bố nhóm lỗi trên 33 cặp enhanced/degraded", xlabel="Số bản ghi"); ax.grid(axis="x", alpha=.2); fig.tight_layout()
fig.savefig(error_out / "error_groups.png", dpi=190); plt.close(fig)
if not pairs.empty:
    pair_label = pairs["true_label"].astype(str) + " → " + pairs["predicted_label"].astype(str)
    values = pd.to_numeric(pairs["count"], errors="coerce").fillna(0)
    subset = pd.DataFrame({"pair": pair_label, "count": values}).nlargest(12, "count").sort_values("count")
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(subset.pair, subset["count"], color="#1D4ED8")
    ax.set(title="Top confusion pairs", xlabel="Số lần"); ax.grid(axis="x", alpha=.2); fig.tight_layout()
    fig.savefig(error_out / "top_confusion_pairs.png", dpi=190); plt.close(fig)

(ROOT / "results/README.md").write_text(
    "# Final results\n\n`results/final/` là nguồn metric canonical duy nhất.\n\n"
    "- `condition_metrics.csv`: 49 điều kiện;\n"
    "- `predictions.csv`: 26.950 dự đoán;\n"
    "- `per_class_metrics.csv`: 245 hàng;\n"
    "- `statistical_tests.csv`: 66 hàng so sánh cặp;\n"
    "- `confusion_matrices/`: 49 CSV;\n"
    "- `error_analysis.csv` và `top_confusion_pairs.csv`: phân tích lỗi truy vết.\n\n"
    "Các file ngoài `results/final/` là view dẫn xuất và không được dùng làm nguồn số liệu.\n",
    encoding="utf-8",
)
print("Built final-result figures and derived image-quality table")
