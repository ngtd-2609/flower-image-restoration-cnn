from __future__ import annotations

import json
from pathlib import Path


def load_locked_params(path: str | Path, *, required: bool = False) -> dict:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Locked Validation parameters not found: {path}. Run Validation tuning first."
            )
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload.get("parameters", payload)
    if not isinstance(parameters, dict):
        raise ValueError("locked_enhancement_params.json must contain an object")  # noqa: TRY004
    return parameters


def params_for(locked: dict, degradation: str, level: str, method: str) -> dict:
    value = locked.get(f"{degradation}|{level}|{method}", {})
    if not isinstance(value, dict):
        raise ValueError(f"Locked parameters for {degradation}/{level}/{method} must be an object")  # noqa: TRY004
    return value
