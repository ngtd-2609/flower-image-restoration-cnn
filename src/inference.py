from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np

from .config import CLASS_NAMES
from .preprocessing import prepare_batch


class ModelService:
    """Own one immutable Keras checkpoint and serialize predict calls safely."""

    def __init__(self, model_path: Path, image_size: int = 224):
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run the Colab notebook first.")
        import tensorflow as tf
        self.model_path = model_path
        self.image_size = image_size
        self.model = tf.keras.models.load_model(model_path)
        self.load_count = 1
        self._predict_lock = Lock()

    def predict_batch(self, images, batch_size: int = 32) -> list[dict]:
        tensor = prepare_batch(images, size=self.image_size)
        if len(tensor) == 0:
            return []
        with self._predict_lock:
            probabilities = self.model.predict(tensor, batch_size=batch_size, verbose=0)
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(tensor), len(CLASS_NAMES)):
            raise RuntimeError(
                f"Unexpected model output shape {probabilities.shape}; expected "
                f"({len(tensor)}, {len(CLASS_NAMES)})"
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model returned non-finite probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
            raise RuntimeError("Model outputs do not sum to one")
        outputs = []
        for values in probabilities:
            index = int(np.argmax(values))
            outputs.append({
                "label": CLASS_NAMES[index],
                "confidence": float(values[index]),
                "probabilities": {name: float(value) for name, value in zip(CLASS_NAMES, values)},
            })
        return outputs

    def predict(self, image: np.ndarray) -> dict:
        return self.predict_batch([image], batch_size=1)[0]
