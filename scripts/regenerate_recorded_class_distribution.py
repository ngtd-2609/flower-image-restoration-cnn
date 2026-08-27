"""Regenerate the class-count figure from the recorded valid inventory."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def main() -> None:
    inventory = pd.read_csv(ROOT / "data" / "inventory.csv")
    valid = inventory[inventory["decode_status"].astype(str).str.startswith("valid")]
    counts = valid["label"].value_counts().reindex(ORDER).fillna(0).astype(int)
    figure, axis = plt.subplots(figsize=(10, 5.2))
    bars = axis.bar(counts.index, counts.values, color="#4F81BD")
    axis.set_title(f"Phân bố {int(counts.sum()):,} ảnh hợp lệ theo lớp".replace(",", "."), color="#17365D", weight="bold")
    axis.set_ylabel("Số ảnh")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, counts.values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 10, str(value), ha="center", va="bottom")
    figure.tight_layout()
    output = ROOT / "figures" / "eda" / "class_distribution.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
