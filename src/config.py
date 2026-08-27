from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED = 42
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
LEVELS = ["light", "medium", "strong"]
LEVEL_LABELS_VI = {"light": "Nhẹ", "medium": "Vừa", "strong": "Mạnh"}

DEGRADATION_PARAMS = {
    "low_light": {"light": {"gamma": 1.5}, "medium": {"gamma": 2.5}, "strong": {"gamma": 4.0}},
    "gaussian_noise": {"light": {"sigma": 10.0}, "medium": {"sigma": 25.0}, "strong": {"sigma": 50.0}},
    "salt_pepper": {"light": {"amount": 0.01}, "medium": {"amount": 0.03}, "strong": {"amount": 0.07}},
    "gaussian_blur": {"light": {"kernel": 3}, "medium": {"kernel": 7}, "strong": {"kernel": 15}},
    "color_cast": {
        "light": {"gains": [1.05, 1.00, 0.95]},
        "medium": {"gains": [1.15, 1.00, 0.85]},
        "strong": {"gains": [1.30, 1.00, 0.70]},
    },
}

METHODS = {
    "low_light": ["gamma_correction", "clahe"],
    "gaussian_noise": ["gaussian_filter", "bilateral_filter"],
    "salt_pepper": ["median_filter", "gaussian_filter"],
    "gaussian_blur": ["unsharp_mask", "sharpening"],
    "color_cast": ["rgb_balance", "hsv_correction", "lab_correction"],
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path: return self.root / "data" / "flower_photos"
    @property
    def splits(self) -> Path: return self.root / "splits"
    @property
    def models(self) -> Path: return self.root / "models"
    @property
    def results(self) -> Path: return self.root / "results"
    @property
    def figures(self) -> Path: return self.root / "figures"

    def ensure_outputs(self) -> None:
        for path in (self.splits, self.models, self.results, self.figures):
            path.mkdir(parents=True, exist_ok=True)
