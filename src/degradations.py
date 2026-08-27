from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import DEGRADATION_PARAMS, LEVELS


def stable_seed(relative_path: str, degradation: str, level: str) -> int:
    token = f"{relative_path}|{degradation}|{level}".encode()
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def apply_degradation(image: np.ndarray, kind: str, level: str, seed: int | None = None) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB uint8 image")
    if kind not in DEGRADATION_PARAMS or level not in LEVELS:
        raise ValueError(f"Unsupported degradation: {kind}/{level}")
    params = DEGRADATION_PARAMS[kind][level]
    x = image.astype(np.float32)
    if kind == "low_light":
        return _uint8(255 * (x / 255) ** params["gamma"])
    if kind == "gaussian_noise":
        rng = np.random.default_rng(seed)
        return _uint8(x + rng.normal(0, params["sigma"], x.shape))
    if kind == "salt_pepper":
        rng = np.random.default_rng(seed)
        mask = rng.random(image.shape[:2])
        amount = params["amount"]
        output = image.copy()
        output[mask < amount / 2] = 0
        output[(mask >= amount / 2) & (mask < amount)] = 255
        return output
    if kind == "gaussian_blur":
        kernel = params["kernel"]
        sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
        return _uint8(gaussian_filter(x, sigma=(sigma, sigma, 0), mode="reflect"))
    if kind == "color_cast":
        return _uint8(x * np.asarray(params["gains"])[None, None, :])
    raise AssertionError("Unreachable")


def assert_degradation_output(source: np.ndarray, output: np.ndarray) -> None:
    assert output.shape == source.shape
    assert output.dtype == np.uint8
    assert int(output.min()) >= 0 and int(output.max()) <= 255
