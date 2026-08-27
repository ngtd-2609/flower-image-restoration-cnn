from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.degradations import apply_degradation, stable_seed
from src.enhancements import apply_enhancement
from src.locked_params import params_for


@dataclass
class PipelineOutput:
    clean: np.ndarray
    degraded: np.ndarray
    enhanced: np.ndarray
    predictions: dict[str, dict]
    degradation_params: dict
    enhancement_params: dict
    processing_time_ms: float


def run_pipeline(
    image: np.ndarray,
    *,
    identifier: str,
    degradation: str,
    level: str,
    method: str,
    model_service,
    locked_params: dict,
    degradation_params: dict,
) -> PipelineOutput:
    """Run the standalone app pipeline with exactly one batched model call."""
    started = time.perf_counter()
    degraded = apply_degradation(
        image,
        degradation,
        level,
        stable_seed(identifier, degradation, level),
    )
    enhancement_params = params_for(locked_params, degradation, level, method)
    enhanced = apply_enhancement(degraded, degradation, level, method, enhancement_params)
    outputs = model_service.predict_batch([image, degraded, enhanced], batch_size=3)
    if len(outputs) != 3:
        raise RuntimeError("Model inference did not return three predictions")
    return PipelineOutput(
        clean=image,
        degraded=degraded,
        enhanced=enhanced,
        predictions=dict(zip(("clean", "degraded", "enhanced"), outputs)),
        degradation_params=degradation_params,
        enhancement_params=enhancement_params,
        processing_time_ms=(time.perf_counter() - started) * 1000,
    )
