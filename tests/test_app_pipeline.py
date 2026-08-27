import io
import unittest

import numpy as np
from PIL import Image

from app_components.io import content_identifier, decode_uploaded_image
from app_components.pipeline import run_pipeline
from src.config import DEGRADATION_PARAMS


class StubModelService:
    def __init__(self):
        self.calls = 0

    def predict_batch(self, images, batch_size=32):
        self.calls += 1
        return [
            {
                "label": "daisy",
                "confidence": 0.6,
                "probabilities": {
                    "daisy": 0.6,
                    "dandelion": 0.1,
                    "roses": 0.1,
                    "sunflowers": 0.1,
                    "tulips": 0.1,
                },
            }
            for _ in images
        ]


class StandaloneAppTests(unittest.TestCase):
    def test_decode_uses_real_content(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (20, 10), (10, 20, 30, 255)).save(buffer, format="PNG")
        image = decode_uploaded_image(buffer.getvalue())
        self.assertEqual(image.shape, (10, 20, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_invalid_content_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_uploaded_image(b"not-an-image")

    def test_pipeline_batches_three_states_once(self):
        service = StubModelService()
        image = np.full((32, 48, 3), 128, dtype=np.uint8)
        locked = {"low_light|medium|gamma_correction": {}}
        result = run_pipeline(
            image,
            identifier=content_identifier(image.tobytes()),
            degradation="low_light",
            level="medium",
            method="gamma_correction",
            model_service=service,
            locked_params=locked,
            degradation_params=DEGRADATION_PARAMS["low_light"]["medium"],
        )
        self.assertEqual(service.calls, 1)
        self.assertEqual(set(result.predictions), {"clean", "degraded", "enhanced"})
        self.assertEqual(result.enhanced.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
