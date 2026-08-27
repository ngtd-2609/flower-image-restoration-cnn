from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocessing import load_image, prepare_image


def load_rgb(path: str | Path, size: int | None = None) -> np.ndarray:
    image = load_image(path)
    return prepare_image(image, size).astype(np.uint8) if size is not None else image


def build_tf_dataset(frame, project_root: Path, class_names: list[str], image_size=224, batch_size=32, training=False):
    """Lazy TensorFlow loader so non-TensorFlow utilities remain importable."""
    import tensorflow as tf

    paths = [str(project_root / path) for path in frame["relative_path"]]
    labels = [class_names.index(label) for label in frame["label"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _decode_numpy(path):
        # tf.numpy_function passes a NumPy scalar/bytes value, not an EagerTensor.
        raw = path.item() if hasattr(path, "item") else path
        value = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return prepare_image(load_image(value), image_size)

    def decode(path, label):
        image = tf.numpy_function(_decode_numpy, [path], tf.float32)
        image.set_shape((image_size, image_size, 3))
        return image, label

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(len(frame), seed=42, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
