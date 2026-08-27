from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import sobel, uniform_filter

from .color_spaces import rgb_to_lab

IMAGE_METRIC_PIXEL_STRIDE = 4
DELTA_E_PIXEL_STRIDE = IMAGE_METRIC_PIXEL_STRIDE


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = np.mean((reference.astype(float) - candidate.astype(float)) ** 2)
    return 100.0 if mse < 1e-12 else 20 * math.log10(255 / math.sqrt(mse))


def ssim(reference: np.ndarray, candidate: np.ndarray, window=7) -> float:
    a, b = reference.astype(float), candidate.astype(float)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    channels = []
    for index in range(3):
        x, y = a[..., index], b[..., index]
        ux, uy = uniform_filter(x, window), uniform_filter(y, window)
        vx = uniform_filter(x * x, window) - ux * ux
        vy = uniform_filter(y * y, window) - uy * uy
        covariance = uniform_filter(x * y, window) - ux * uy
        score = ((2 * ux * uy + c1) * (2 * covariance + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-12)
        channels.append(np.mean(score))
    return float(np.mean(channels))


def delta_e_2000(reference: np.ndarray, candidate: np.ndarray, pixel_stride: int = DELTA_E_PIXEL_STRIDE) -> float:
    """Vectorized CIEDE2000 on a deterministic regular pixel lattice.

    Every image is included. A stride of four evaluates a 56 x 56 lattice for
    canonical 224 x 224 inputs, reducing redundant spatial computation while
    keeping the protocol deterministic and traceable.
    """
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be at least 1")
    reference = reference[::pixel_stride, ::pixel_stride]
    candidate = candidate[::pixel_stride, ::pixel_stride]
    l1, a1, b1 = np.moveaxis(rgb_to_lab(reference), -1, 0)
    l2, a2, b2 = np.moveaxis(rgb_to_lab(candidate), -1, 0)
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25 ** 7 + 1e-12)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = (np.degrees(np.arctan2(b1, ap1)) + 360) % 360
    hp2 = (np.degrees(np.arctan2(b2, ap2)) + 360) % 360
    dl, dc = l2 - l1, cp2 - cp1
    dh = hp2 - hp1
    dh = np.where(cp1 * cp2 == 0, 0, dh)
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    d_h = 2 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh / 2))
    l_bar, cp_bar = (l1 + l2) / 2, (cp1 + cp2) / 2
    hp_bar = np.where(
        cp1 * cp2 == 0,
        hp1 + hp2,
        np.where(np.abs(hp1 - hp2) <= 180, (hp1 + hp2) / 2,
                 np.where(hp1 + hp2 < 360, (hp1 + hp2 + 360) / 2, (hp1 + hp2 - 360) / 2)),
    )
    t = 1 - 0.17 * np.cos(np.radians(hp_bar - 30)) + 0.24 * np.cos(np.radians(2 * hp_bar)) + 0.32 * np.cos(np.radians(3 * hp_bar + 6)) - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * cp_bar
    s_h = 1 + 0.015 * cp_bar * t
    r_t = -2 * np.sqrt(cp_bar ** 7 / (cp_bar ** 7 + 25 ** 7 + 1e-12)) * np.sin(np.radians(60 * np.exp(-((hp_bar - 275) / 25) ** 2)))
    distance = np.sqrt((dl / s_l) ** 2 + (dc / s_c) ** 2 + (d_h / s_h) ** 2 + r_t * (dc / s_c) * (d_h / s_h))
    return float(np.mean(distance))


def mean_brightness(image: np.ndarray) -> float:
    return float(np.mean(0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]))


def rms_contrast(image: np.ndarray) -> float:
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return float(np.std(gray))


def edge_energy(image: np.ndarray) -> float:
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return float(np.mean(np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))))


def edge_preservation_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    return edge_energy(candidate) / (edge_energy(reference) + 1e-12)


def histogram_distance(reference: np.ndarray, candidate: np.ndarray, bins=32) -> float:
    distances = []
    for channel in range(3):
        h1, _ = np.histogram(reference[..., channel], bins=bins, range=(0, 256), density=True)
        h2, _ = np.histogram(candidate[..., channel], bins=bins, range=(0, 256), density=True)
        distances.append(np.sqrt(0.5 * np.sum((np.sqrt(h1) - np.sqrt(h2)) ** 2)))
    return float(np.mean(distances))


def full_reference_metrics(reference: np.ndarray, candidate: np.ndarray, include_delta_e=False) -> dict[str, float]:
    reference_sample = reference[::IMAGE_METRIC_PIXEL_STRIDE, ::IMAGE_METRIC_PIXEL_STRIDE]
    candidate_sample = candidate[::IMAGE_METRIC_PIXEL_STRIDE, ::IMAGE_METRIC_PIXEL_STRIDE]
    return {
        "psnr": psnr(reference_sample, candidate_sample),
        "ssim": ssim(reference_sample, candidate_sample),
        "delta_e_2000": delta_e_2000(reference_sample, candidate_sample, pixel_stride=1) if include_delta_e else float("nan"),
        "image_metric_pixel_stride": float(IMAGE_METRIC_PIXEL_STRIDE),
        "delta_e_pixel_stride": float(DELTA_E_PIXEL_STRIDE),
        "mean_brightness": mean_brightness(candidate_sample),
        "rms_contrast": rms_contrast(candidate_sample),
        "edge_preservation_ratio": edge_preservation_ratio(reference_sample, candidate_sample),
        "histogram_distance": histogram_distance(reference_sample, candidate_sample),
    }
