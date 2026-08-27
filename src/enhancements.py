from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, median_filter

from .color_spaces import lab_to_rgb, rgb_to_lab
from .config import DEGRADATION_PARAMS, METHODS


def _uint8(x): return np.clip(x, 0, 255).astype(np.uint8)


def _equalize_channel(channel: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    hist = np.bincount(channel.ravel(), minlength=256).astype(float)
    threshold = max(1.0, clip_limit * channel.size / 256)
    excess = np.maximum(hist - threshold, 0).sum()
    hist = np.minimum(hist, threshold) + excess / 256
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-12)
    return _uint8(cdf[channel] * 255)


def gamma_correction(image: np.ndarray, degradation_level: str) -> np.ndarray:
    gamma = DEGRADATION_PARAMS["low_light"][degradation_level]["gamma"]
    return _uint8(255 * (image.astype(float) / 255) ** (1 / gamma))


def clahe_hsv_value(image: np.ndarray, clip_limit=2.0, tiles=8) -> np.ndarray:
    """CLAHE on HSV Value with clipped tile histograms and bilinear LUT interpolation."""
    hsv = np.asarray(Image.fromarray(image).convert("HSV"), dtype=np.uint8).copy()
    channel = hsv[..., 2]
    height, width = channel.shape
    tiles_y, tiles_x = min(tiles, height), min(tiles, width)
    y_edges = np.linspace(0, height, tiles_y + 1, dtype=int)
    x_edges = np.linspace(0, width, tiles_x + 1, dtype=int)
    luts = np.zeros((tiles_y, tiles_x, 256), dtype=np.float32)
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            tile = channel[y_edges[tile_y]:y_edges[tile_y + 1], x_edges[tile_x]:x_edges[tile_x + 1]]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
            threshold = max(1, round(clip_limit * tile.size / 256))
            excess = int(np.maximum(hist - threshold, 0).sum())
            hist = np.minimum(hist, threshold)
            hist += excess // 256
            hist[: excess % 256] += 1
            cdf = hist.cumsum()
            nonzero = cdf[cdf > 0]
            cdf_min = nonzero[0] if len(nonzero) else 0
            luts[tile_y, tile_x] = np.clip(
                (cdf - cdf_min) * 255 / max(tile.size - cdf_min, 1), 0, 255
            )

    y_centers = (y_edges[:-1] + y_edges[1:] - 1) / 2
    x_centers = (x_edges[:-1] + x_edges[1:] - 1) / 2
    y_hi = np.clip(np.searchsorted(y_centers, np.arange(height), side="right"), 0, tiles_y - 1)
    x_hi = np.clip(np.searchsorted(x_centers, np.arange(width), side="right"), 0, tiles_x - 1)
    y_lo = np.clip(y_hi - 1, 0, tiles_y - 1)
    x_lo = np.clip(x_hi - 1, 0, tiles_x - 1)
    y_den = np.maximum(y_centers[y_hi] - y_centers[y_lo], 1)
    x_den = np.maximum(x_centers[x_hi] - x_centers[x_lo], 1)
    wy = np.where(y_hi == y_lo, 0, (np.arange(height) - y_centers[y_lo]) / y_den)
    wx = np.where(x_hi == x_lo, 0, (np.arange(width) - x_centers[x_lo]) / x_den)
    wy, wx = np.clip(wy, 0, 1), np.clip(wx, 0, 1)

    output = np.empty_like(channel, dtype=np.float32)
    for y in range(height):
        values = channel[y]
        top = (1 - wx) * luts[y_lo[y], x_lo, values] + wx * luts[y_lo[y], x_hi, values]
        bottom = (1 - wx) * luts[y_hi[y], x_lo, values] + wx * luts[y_hi[y], x_hi, values]
        output[y] = (1 - wy[y]) * top + wy[y] * bottom
    hsv[..., 2] = _uint8(output)
    return np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"))


def bilateral_filter(image: np.ndarray, spatial_sigma=1.6, range_sigma=40.0, radius=2) -> np.ndarray:
    x = image.astype(np.float32)
    padded = np.pad(x, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    accum = np.zeros_like(x)
    normalizer = np.zeros(x.shape[:2], dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[radius + dy:radius + dy + x.shape[0], radius + dx:radius + dx + x.shape[1]]
            spatial = math.exp(-(dx * dx + dy * dy) / (2 * spatial_sigma ** 2))
            color_distance = np.sum((neighbor - x) ** 2, axis=2)
            weight = spatial * np.exp(-color_distance / (2 * range_sigma ** 2))
            accum += neighbor * weight[..., None]
            normalizer += weight
    return _uint8(accum / (normalizer[..., None] + 1e-12))


def rgb_balance(image: np.ndarray) -> np.ndarray:
    x = image.astype(float)
    means = x.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    return _uint8(x * (target / (means + 1e-12))[None, None, :])


def hsv_correction(image: np.ndarray) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(image).convert("HSV"), dtype=np.uint8).copy()
    hsv[..., 1] = _uint8(hsv[..., 1].astype(float) * 0.92)
    hsv[..., 2] = _equalize_channel(hsv[..., 2], clip_limit=1.5)
    return np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"))


def lab_correction(image: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(image)
    lab[..., 1] -= np.mean(lab[..., 1])
    lab[..., 2] -= np.mean(lab[..., 2])
    low, high = np.percentile(lab[..., 0], [1, 99])
    lab[..., 0] = np.clip((lab[..., 0] - low) * 100 / (high - low + 1e-12), 0, 100)
    return lab_to_rgb(lab)


def apply_enhancement(image: np.ndarray, degradation: str, level: str, method: str, params: dict | None = None) -> np.ndarray:
    if method not in METHODS.get(degradation, []):
        raise ValueError(f"Method {method} is not mapped to {degradation}")
    params = params or {}
    x = image.astype(float)
    if method == "gamma_correction": return gamma_correction(image, level)
    if method == "clahe": return clahe_hsv_value(image, clip_limit=params.get("clip_limit", 2.0))
    if method == "gaussian_filter": return _uint8(gaussian_filter(x, sigma=(params.get("sigma", 1.0), params.get("sigma", 1.0), 0)))
    if method == "median_filter":
        kernel = int(params.get("kernel", 3)); return median_filter(image, size=(kernel, kernel, 1), mode="reflect").astype(np.uint8)
    if method == "bilateral_filter": return bilateral_filter(image, range_sigma=float(params.get("range_sigma", 40)))
    if method == "unsharp_mask":
        radius, amount = float(params.get("radius", 1.5)), float(params.get("amount", 1.0))
        smooth = gaussian_filter(x, sigma=(radius, radius, 0)); return _uint8(x + amount * (x - smooth))
    if method == "sharpening": return np.asarray(Image.fromarray(image).filter(ImageFilter.SHARPEN))
    if method == "rgb_balance": return rgb_balance(image)
    if method == "hsv_correction": return hsv_correction(image)
    if method == "lab_correction": return lab_correction(image)
    raise AssertionError("Unreachable")
