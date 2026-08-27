import json
import unittest
from pathlib import Path

from src.locked_params import load_locked_params, params_for

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts" / "test_runtime"


class LockedParameterTests(unittest.TestCase):
    def test_metadata_wrapper_is_supported(self):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        path = RUNTIME / "locked-params.json"
        try:
            path.write_text(
                json.dumps({"_metadata": {"selection_split": "validation"}, "parameters": {"low_light|light|clahe": {"clip_limit": 2.0}}}),
                encoding="utf-8",
            )
            locked = load_locked_params(path, required=True)
            self.assertEqual(params_for(locked, "low_light", "light", "clahe"), {"clip_limit": 2.0})
        finally:
            path.unlink(missing_ok=True)

    def test_required_file_must_exist(self):
        with self.assertRaises(FileNotFoundError):
            load_locked_params("does-not-exist.json", required=True)


if __name__ == "__main__":
    unittest.main()
