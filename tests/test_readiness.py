import hashlib
import json
import unittest
from pathlib import Path

from app_components.readiness import inspect_artifact_readiness
from src.config import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts" / "test_runtime"


class ReadinessTests(unittest.TestCase):
    def test_missing_artifacts_are_blocked(self):
        result = inspect_artifact_readiness(
            RUNTIME / "missing-model.keras",
            RUNTIME / "missing-metadata.json",
            RUNTIME / "missing-classes.json",
            RUNTIME / "missing-locked.json",
        )
        self.assertFalse(result["ready"])
        self.assertTrue(any("checkpoint" in item for item in result["errors"]))
        self.assertTrue(any("locked_enhancement" in item for item in result["errors"]))

    def test_complete_metadata_contract_is_accepted(self):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        root = RUNTIME
        model = root / "model.keras"
        try:
            model.write_bytes(b"contract-test-bytes")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "FULL_RUN_COMPLETE",
                        "model_sha256": digest,
                        "class_names": CLASS_NAMES,
                    }
                ),
                encoding="utf-8",
            )
            (root / "classes.json").write_text(json.dumps(CLASS_NAMES), encoding="utf-8")
            parameters = {f"key-{index}": {} for index in range(33)}
            (root / "locked.json").write_text(
                json.dumps(
                    {
                        "_metadata": {"selection_split": "validation", "quick_run": False},
                        "parameters": parameters,
                    }
                ),
                encoding="utf-8",
            )
            result = inspect_artifact_readiness(
                model,
                root / "metadata.json",
                root / "classes.json",
                root / "locked.json",
            )
        finally:
            for name in ("model.keras", "metadata.json", "classes.json", "locked.json"):
                (root / name).unlink(missing_ok=True)
        self.assertTrue(result["ready"], result["errors"])


if __name__ == "__main__":
    unittest.main()
