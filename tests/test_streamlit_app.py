from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(
    STREAMLIT_AVAILABLE,
    "Streamlit is not installed in this lightweight verification environment.",
)
class StreamlitAppTest(unittest.TestCase):
    """Smoke-test both honest blocked and model-ready startup states."""

    def test_missing_checkpoint_is_reported_without_uncaught_exception(self) -> None:
        sys.path.insert(0, str(ROOT))
        try:
            from streamlit.testing.v1 import AppTest

            app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=20).run()
            self.assertEqual(list(app.exception), [])
            messages = [element.value for element in app.error]
            from app_components.readiness import inspect_artifact_readiness

            readiness = inspect_artifact_readiness(
                ROOT / "models" / "mobilenetv2_flowers.keras",
                ROOT / "models" / "model_metadata.json",
                ROOT / "models" / "class_names.json",
                ROOT / "configs" / "locked_enhancement_params.json",
            )
            if readiness["ready"]:
                self.assertEqual(messages, [])
                self.assertTrue(list(app.file_uploader))
            else:
                self.assertTrue(messages)
        finally:
            if sys.path and sys.path[0] == str(ROOT):
                sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
