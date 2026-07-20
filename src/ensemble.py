"""Confidence-weighted soft-voting ensemble.

Each model contributes its softmax probabilities weighted by its **validation
macro-F1** (normalised to sum to 1), so more reliable models count for more. The
ensemble prediction is the argmax of the weighted-average probability.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.evaluate import compute_metrics


def normalise_weights(val_f1: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in val_f1.values())
    if total <= 0:
        n = len(val_f1)
        return {k: 1.0 / n for k in val_f1}
    return {k: max(v, 0.0) / total for k, v in val_f1.items()}


def ensemble_probs(
    model_probs: Dict[str, np.ndarray], val_f1: Dict[str, float]
) -> np.ndarray:
    """Weighted average of per-model probability matrices.

    ``model_probs[key]`` is ``[N, C]``; ``val_f1[key]`` is that model's validation
    macro-F1 (the vote weight).
    """
    keys = list(model_probs.keys())
    weights = normalise_weights({k: val_f1[k] for k in keys})
    stacked = np.stack([model_probs[k] * weights[k] for k in keys], axis=0)
    return stacked.sum(axis=0)


def build_ensemble(
    model_probs: Dict[str, np.ndarray], val_f1: Dict[str, float],
    targets: np.ndarray, class_names: List[str],
) -> Dict:
    """Build the ensemble and return its probs, preds and full metrics."""
    weights = normalise_weights({k: val_f1[k] for k in model_probs})
    probs = ensemble_probs(model_probs, val_f1)
    preds = probs.argmax(1)
    metrics = compute_metrics(targets, preds, class_names)
    metrics.update({"probs": probs, "preds": preds, "targets": np.asarray(targets),
                    "weights": weights})
    return metrics
