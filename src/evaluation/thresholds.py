from __future__ import annotations

import numpy as np


def threshold_for_acceptance_rate(y_proba, target_rate: float) -> float:
    """
    Find a threshold such that the fraction of predictions >= threshold
    approximates target_rate.
    """

    probs = np.asarray(y_proba, dtype=float).ravel()
    if probs.size == 0:
        raise ValueError("y_proba must contain at least one element.")
    if not (0.0 < target_rate < 1.0):
        raise ValueError("target_rate must be in (0, 1).")

    quantile = max(0.0, min(1.0, 1.0 - target_rate))
    threshold = float(np.quantile(probs, quantile, method="higher"))
    return threshold
