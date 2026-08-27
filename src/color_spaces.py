from __future__ import annotations

import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float64) / 255.0
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = (x @ matrix.T) / np.array([0.95047, 1.0, 1.08883])
    epsilon, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    epsilon, kappa = 216 / 24389, 24389 / 27
    xyz = np.where(f ** 3 > epsilon, f ** 3, (116 * f - 16) / kappa) * np.array([0.95047, 1.0, 1.08883])
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    rgb = xyz @ np.linalg.inv(matrix.T)
    rgb = np.where(rgb > 0.0031308, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055, 12.92 * rgb)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)
