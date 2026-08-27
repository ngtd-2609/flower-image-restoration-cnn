from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

NAVY = "#17365D"
BLUE = "#4F81BD"
ORANGE = "#F79646"


def apply_academic_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlecolor": NAVY})


def save_condition_metric(results: pd.DataFrame, metric: str, output: Path, title: str):
    apply_academic_style()
    plot = results.copy()
    plot["condition"] = plot["degradation"] + "\n" + plot["level"] + "\n" + plot["enhancement_method"]
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(data=plot, x="condition", y=metric, hue="image_type", ax=ax, palette=["#7F8C8D", ORANGE, BLUE])
    ax.set_title(title); ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout(); fig.savefig(output, dpi=200); plt.close(fig)
