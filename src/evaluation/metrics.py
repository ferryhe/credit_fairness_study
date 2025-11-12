from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def compute_accuracy_metrics(y_true, y_proba) -> dict:
    """
    Compute standard accuracy metrics for probabilistic binary predictions.
    """

    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_proba_arr = np.asarray(y_proba, dtype=float).ravel()

    if y_true_arr.shape[0] != y_proba_arr.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")

    y_proba_clipped = np.clip(y_proba_arr, 1e-7, 1 - 1e-7)

    unique_classes = np.unique(y_true_arr)
    has_both_classes = unique_classes.size == 2

    roc_auc = (
        roc_auc_score(y_true_arr, y_proba_arr) if has_both_classes else np.nan
    )
    pr_auc = (
        average_precision_score(y_true_arr, y_proba_arr)
        if has_both_classes
        else np.nan
    )

    brier = brier_score_loss(y_true_arr, y_proba_clipped)
    ll = log_loss(y_true_arr, y_proba_clipped)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "log_loss": ll,
    }
