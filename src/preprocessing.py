from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Validate and normalize an in-memory image to RGB uint8."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/HxWx4 image, received shape={array.shape}")
    if array.shape[2] == 4:
        array = np.asarray(Image.fromarray(array.astype(np.uint8), mode="RGBA").convert("RGB"))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def resize_with_letterbox(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize without aspect-ratio distortion, using one canonical implementation."""
    if size <= 0:
        raise ValueError("size must be positive")
    rgb = ensure_rgb_uint8(image)
    pil = Image.fromarray(rgb, mode="RGB")
    padded = ImageOps.pad(
        pil,
        (size, size),
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )
    return np.asarray(padded, dtype=np.uint8)


def load_image(path: str | Path) -> np.ndarray:
    """Decode a file completely, apply EXIF orientation and return RGB uint8."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
        return np.asarray(image, dtype=np.uint8)


def prepare_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Return the raw 0..255 float tensor expected by the model graph."""
    return resize_with_letterbox(image, size=size).astype(np.float32)


def prepare_batch(images: Iterable[np.ndarray], size: int = 224) -> np.ndarray:
    prepared = [prepare_image(image, size=size) for image in images]
    if not prepared:
        return np.empty((0, size, size, 3), dtype=np.float32)
    return np.stack(prepared, axis=0)
