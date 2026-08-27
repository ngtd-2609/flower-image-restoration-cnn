from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support


def summary_metrics(y_true, y_pred) -> dict[str, float]:
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
    }


def per_class_frame(y_true, y_pred, class_names):
    import pandas as pd
    report = classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)
    return pd.DataFrame([
        {"class": name, "precision": report[name]["precision"], "recall": report[name]["recall"], "f1": report[name]["f1-score"], "support": report[name]["support"]}
        for name in class_names
    ])


def bootstrap_accuracy_interval(y_true, y_pred, seed=42, samples=2000, confidence=0.95):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(np.mean(y_true[idx] == y_pred[idx]))
    alpha = (1 - confidence) / 2
    return tuple(float(v) for v in np.quantile(scores, [alpha, 1 - alpha]))


def paired_bootstrap_difference(y_true, pred_a, pred_b, metric="accuracy", seed=42, samples=2000, confidence=0.95):
    """Paired bootstrap CI for metric(pred_b) - metric(pred_a)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    if not (len(y_true) == len(pred_a) == len(pred_b)) or len(y_true) == 0:
        raise ValueError("Paired predictions must have the same non-zero length")
    if metric == "accuracy":
        score = lambda truth, pred: float(np.mean(truth == pred))
    elif metric == "macro_f1":
        score = lambda truth, pred: float(f1_score(truth, pred, average="macro", zero_division=0))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), len(y_true))
        differences.append(score(y_true[indices], pred_b[indices]) - score(y_true[indices], pred_a[indices]))
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(differences, [alpha, 1 - alpha])
    return {
        "metric": metric,
        "difference": score(y_true, pred_b) - score(y_true, pred_a),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
        "bootstrap_samples": samples,
    }


def mcnemar_exact(y_true, pred_a, pred_b):
    from scipy.stats import binomtest
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    p_value = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c else 1.0
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "p_value": p_value}


def holm_bonferroni(p_values, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Adjust a family of p-values with Holm's step-down procedure.

    Returned adjusted values are monotone in the sorted order and mapped back
    to the input order.  The implementation deliberately exposes both the
    adjusted p-value and the family-wise rejection decision so downstream
    reports cannot confuse an uncorrected McNemar result with final evidence.
    """
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p_values must be finite values in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    order = np.argsort(values, kind="stable")
    ranked = values[order]
    factors = values.size - np.arange(values.size)
    adjusted_ranked = np.maximum.accumulate(ranked * factors)
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return {
        "adjusted_p_values": adjusted,
        "reject": adjusted <= alpha,
    }
