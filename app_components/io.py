from __future__ import annotations

import hashlib
import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def decode_uploaded_image(content: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> np.ndarray:
    """Validate image bytes by content, decode fully, transpose EXIF and return RGB."""
    if not content:
        raise ValueError("Tệp tải lên đang trống.")
    if len(content) > max_bytes:
        raise ValueError(f"Ảnh vượt giới hạn {max_bytes / 1024 / 1024:.0f} MB.")
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as probe:
                image_format = (probe.format or "").upper()
                if image_format not in ALLOWED_FORMATS:
                    raise ValueError("Chỉ chấp nhận nội dung ảnh JPEG hoặc PNG hợp lệ.")
                probe.verify()
            with Image.open(io.BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
                return np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Không thể đọc ảnh hoặc kích thước giải nén không an toàn.") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def content_identifier(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
