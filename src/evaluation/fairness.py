from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _group_confusion_counts(y_true_g: np.ndarray, y_pred_g: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true_g == 1) & (y_pred_g == 1)))
    fp = int(np.sum((y_true_g == 0) & (y_pred_g == 1)))
    tn = int(np.sum((y_true_g == 0) & (y_pred_g == 0)))
    fn = int(np.sum((y_true_g == 1) & (y_pred_g == 0)))
    return tp, fp, tn, fn


def fairness_metrics(y_true, y_proba, A, threshold: float = 0.5) -> dict:
    """
    Compute common fairness metrics for binary classification.
    """

    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_proba_arr = np.asarray(y_proba, dtype=float).ravel()
    A_arr = np.asarray(A).astype(int).ravel()

    if not (y_true_arr.shape == y_proba_arr.shape == A_arr.shape):
        raise ValueError("y_true, y_proba, and A must all have the same shape.")

    y_pred = (y_proba_arr >= threshold).astype(int)

    metrics = {
        "tpr_0": np.nan,
        "tpr_1": np.nan,
        "fpr_0": np.nan,
        "fpr_1": np.nan,
        "selection_rate_0": np.nan,
        "selection_rate_1": np.nan,
        "auc_roc_group0": np.nan,
        "auc_roc_group1": np.nan,
    }

    for g in (0, 1):
        mask = A_arr == g
        count = np.sum(mask)
        if count == 0:
            continue

        y_true_g = y_true_arr[mask]
        y_pred_g = y_pred[mask]
        y_proba_g = y_proba_arr[mask]

        tp, fp, tn, fn = _group_confusion_counts(y_true_g, y_pred_g)

        tpr_den = tp + fn
        fpr_den = fp + tn

        metrics[f"tpr_{g}"] = tp / tpr_den if tpr_den > 0 else np.nan
        metrics[f"fpr_{g}"] = fp / fpr_den if fpr_den > 0 else np.nan
        metrics[f"selection_rate_{g}"] = (
            y_pred_g.mean() if count > 0 else np.nan
        )

        unique_classes = np.unique(y_true_g)
        if unique_classes.size == 2:
            metrics[f"auc_roc_group{g}"] = roc_auc_score(y_true_g, y_proba_g)

    tpr_0, tpr_1 = metrics["tpr_0"], metrics["tpr_1"]
    fpr_0, fpr_1 = metrics["fpr_0"], metrics["fpr_1"]
    sel_0, sel_1 = metrics["selection_rate_0"], metrics["selection_rate_1"]

    metrics["eo_gap_tpr"] = (
        abs(tpr_1 - tpr_0) if not np.isnan(tpr_0) and not np.isnan(tpr_1) else np.nan
    )
    metrics["eo_gap_fpr"] = (
        abs(fpr_1 - fpr_0) if not np.isnan(fpr_0) and not np.isnan(fpr_1) else np.nan
    )

    if not np.isnan(sel_0) and not np.isnan(sel_1):
        metrics["dp_diff"] = sel_1 - sel_0
        metrics["dp_ratio"] = sel_1 / sel_0 if sel_0 != 0 else np.nan
    else:
        metrics["dp_diff"] = np.nan
        metrics["dp_ratio"] = np.nan

    return metrics


def find_threshold_for_target_rate(y_proba, target_rate: float) -> float:
    """
    Determine a probability threshold that yields approximately target_rate positives.
    """

    probs = np.asarray(y_proba, dtype=float).ravel()
    if probs.size == 0:
        raise ValueError("y_proba must contain at least one element.")
    if not (0.0 < target_rate < 1.0):
        raise ValueError("target_rate must be in (0, 1).")

    sorted_probs = np.sort(probs)[::-1]
    n = sorted_probs.size
    k = int(target_rate * n)

    if k <= 0:
        threshold = float(sorted_probs[0] + 1e-8)
    elif k >= n:
        threshold = float(sorted_probs[-1] - 1e-8)
    else:
        threshold = float(sorted_probs[k - 1])
    return threshold


def fairness_at_target_rate(y_true, y_proba, A, target_rate: float) -> dict:
    """
    Compute fairness metrics at a threshold chosen to hit a desired selection rate.
    """

    threshold = find_threshold_for_target_rate(y_proba, target_rate)
    fairness = fairness_metrics(y_true, y_proba, A, threshold=threshold)
    actual_rate = float(
        np.mean(np.asarray(y_proba, dtype=float).ravel() >= threshold)
    )
    extended = dict(fairness)
    extended.update(
        {
            "threshold": threshold,
            "target_rate": target_rate,
            "actual_rate": actual_rate,
        }
    )
    return extended
