from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import DEGRADATION_PARAMS, LEVELS, METHODS


@dataclass(frozen=True)
class ExperimentCondition:
    condition_id: str
    image_type: str
    degradation: str
    level: str
    enhancement_method: str


def build_experiment_matrix() -> list[ExperimentCondition]:
    conditions = [ExperimentCondition("clean", "clean", "clean", "none", "none")]
    for degradation, methods in METHODS.items():
        for level in LEVELS:
            conditions.append(
                ExperimentCondition(
                    f"{degradation}__{level}__degraded",
                    "degraded",
                    degradation,
                    level,
                    "none",
                )
            )
            conditions.extend(
                ExperimentCondition(
                    f"{degradation}__{level}__{method}",
                    "enhanced",
                    degradation,
                    level,
                    method,
                )
                for method in methods
            )
    if len(conditions) != 49:
        raise AssertionError(f"Expected 49 conditions, generated {len(conditions)}")
    ids = [condition.condition_id for condition in conditions]
    if len(ids) != len(set(ids)):
        raise AssertionError("Condition IDs must be unique")
    return conditions


def matrix_as_records() -> list[dict]:
    records = []
    for condition in build_experiment_matrix():
        record = asdict(condition)
        record["degradation_params"] = (
            DEGRADATION_PARAMS[condition.degradation][condition.level]
            if condition.image_type != "clean"
            else {}
        )
        records.append(record)
    return records
