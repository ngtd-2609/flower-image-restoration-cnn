from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEGRADATION_PARAMS, LEVELS, METHODS
from src.data_loader import load_rgb
from src.degradations import apply_degradation, stable_seed
from src.enhancements import apply_enhancement
from src.image_metrics import full_reference_metrics


def mean_metrics(references, candidates, color=False):
    return pd.DataFrame([full_reference_metrics(a, b, color) for a, b in zip(references, candidates)]).mean(numeric_only=True).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-project-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    root = args.project_root.resolve()
    test = pd.read_csv(root / "splits/test.csv")
    clean = [load_rgb(args.data_project_root / path, args.size) for path in test.relative_path]
    rows = [{"image_type": "clean", "degradation": "clean", "level": "none", "enhancement_method": "none",
             "degradation_params": "none", "enhancement_params": "none", **mean_metrics(clean, clean, True)}]
    for degradation in METHODS:
        for level in LEVELS:
            degraded = [apply_degradation(image, degradation, level, stable_seed(path, degradation, level)) for image, path in zip(clean, test.relative_path)]
            rows.append({"image_type": "degraded", "degradation": degradation, "level": level, "enhancement_method": "none",
                         "degradation_params": json.dumps(DEGRADATION_PARAMS[degradation][level]), "enhancement_params": "none",
                         **mean_metrics(clean, degraded, degradation == "color_cast")})
            for method in METHODS[degradation]:
                enhanced = [apply_enhancement(image, degradation, level, method) for image in degraded]
                rows.append({"image_type": "enhanced", "degradation": degradation, "level": level, "enhancement_method": method,
                             "degradation_params": json.dumps(DEGRADATION_PARAMS[degradation][level]), "enhancement_params": "quality_pipeline_defaults",
                             **mean_metrics(clean, enhanced, degradation == "color_cast")})
    result = pd.DataFrame(rows)
    assert len(result) == 49
    result.to_csv(root / "results/image_quality_results_49_conditions.csv", index=False)
    degraded = result[result.image_type == "degraded"]
    enhanced = result[result.image_type == "enhanced"]
    for metric, title, file in [
        ("psnr", "PSNR theo dạng và mức suy giảm", "psnr_comparison.png"),
        ("ssim", "SSIM theo dạng và mức suy giảm", "ssim_comparison.png"),
        ("edge_preservation_ratio", "Tỷ lệ bảo toàn biên", "edge_preservation.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot = result[result.image_type != "clean"].copy()
        plot["condition"] = plot.degradation + "\n" + plot.level + "\n" + plot.enhancement_method
        sns.barplot(plot, x="condition", y=metric, hue="image_type", ax=ax, palette=["#F79646", "#4F81BD"])
        ax.tick_params(axis="x", labelrotation=90, labelsize=7); ax.set_title(title, color="#17365D", weight="bold")
        fig.tight_layout(); fig.savefig(root / "figures/classification" / file, dpi=200); plt.close(fig)
    color = result[result.degradation == "color_cast"]
    fig, ax = plt.subplots(figsize=(10, 5)); sns.barplot(color, x="level", y="delta_e_2000", hue="enhancement_method", ax=ax)
    ax.set_title("Delta E 2000 cho color cast", color="#17365D", weight="bold"); fig.tight_layout()
    fig.savefig(root / "figures/classification/delta_e_comparison.png", dpi=200); plt.close(fig)
    print(result.groupby(["degradation", "image_type"])[["psnr", "ssim"]].mean())


if __name__ == "__main__": main()
