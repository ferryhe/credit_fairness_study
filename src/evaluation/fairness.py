from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_curve

from src.evaluation.thresholds import threshold_for_acceptance_rate


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _group_confusion_counts(y_true_g: np.ndarray, y_pred_g: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true_g == 1) & (y_pred_g == 1)))
    fp = int(np.sum((y_true_g == 0) & (y_pred_g == 1)))
    tn = int(np.sum((y_true_g == 0) & (y_pred_g == 0)))
    fn = int(np.sum((y_true_g == 1) & (y_pred_g == 0)))
    return tp, fp, tn, fn


def _calc_group_stats(
    y_true_arr: np.ndarray,
    y_score_arr: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    stats: dict[str, float] = {
        "tpr": np.nan,
        "fpr": np.nan,
        "selection_rate": np.nan,
        "auc": np.nan,
    }
    if mask.sum() == 0:
        return stats
    y_true_g = y_true_arr[mask]
    y_score_g = y_score_arr[mask]
    y_pred_g = (y_score_g >= threshold).astype(int)

    tp, fp, tn, fn = _group_confusion_counts(y_true_g, y_pred_g)
    tpr_den = tp + fn
    fpr_den = fp + tn
    stats["tpr"] = tp / tpr_den if tpr_den > 0 else np.nan
    stats["fpr"] = fp / fpr_den if fpr_den > 0 else np.nan
    stats["selection_rate"] = float(y_pred_g.mean())
    if np.unique(y_true_g).size == 2:
        fpr_vals, tpr_vals, _ = roc_curve(y_true_g, y_score_g)
        stats["auc"] = auc(fpr_vals, tpr_vals)
    return stats


def compute_fairness_metrics(
    y_true,
    y_score,
    A,
    threshold: float = 0.5,
    target_rate: float | None = 0.02,
) -> dict[str, float]:
    """
    Returns fairness metrics computed at a fixed threshold and an optional target rate.
    """

    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_score_arr = np.asarray(y_score, dtype=float).ravel()
    A_arr = np.asarray(A).astype(int).ravel()

    if not (y_true_arr.shape == y_score_arr.shape == A_arr.shape):
        raise ValueError("y_true, y_score, and A must all have the same shape.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1.")

    metrics: dict[str, float] = {
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
        stats = _calc_group_stats(y_true_arr, y_score_arr, mask, threshold)
        metrics[f"tpr_{g}"] = stats["tpr"]
        metrics[f"fpr_{g}"] = stats["fpr"]
        metrics[f"selection_rate_{g}"] = stats["selection_rate"]
        metrics[f"auc_roc_group{g}"] = stats["auc"]
        if mask.sum() == 0:
            _warn(f"No samples found for group {g} when computing fairness at threshold={threshold}.")

    tpr_0, tpr_1 = metrics["tpr_0"], metrics["tpr_1"]
    fpr_0, fpr_1 = metrics["fpr_0"], metrics["fpr_1"]
    sel_0, sel_1 = metrics["selection_rate_0"], metrics["selection_rate_1"]

    metrics["eo_gap_tpr"] = (
        abs(tpr_1 - tpr_0) if not np.isnan(tpr_0) and not np.isnan(tpr_1) else np.nan
    )
    metrics["eo_gap_fpr"] = (
        abs(fpr_1 - fpr_0) if not np.isnan(fpr_0) and not np.isnan(fpr_1) else np.nan
    )

    if sel_0 == 0:
        metrics["dp_ratio"] = np.nan
        _warn("DP ratio at threshold is undefined because group 0 selection rate is 0.")
    else:
        metrics["dp_ratio"] = sel_1 / sel_0
    if not np.isnan(sel_0) and not np.isnan(sel_1):
        metrics["dp_diff"] = sel_1 - sel_0
    else:
        metrics["dp_diff"] = np.nan

    if target_rate is not None:
        if not (0.0 < target_rate < 1.0):
            _warn(f"Skipping target rate {target_rate} because it is not within (0, 1).")
        else:
            thr_fixed = threshold_for_acceptance_rate(y_score_arr, target_rate)
            fixed_stats = {
                g: _calc_group_stats(
                    y_true_arr,
                    y_score_arr,
                    A_arr == g,
                    thr_fixed,
                )
                for g in (0, 1)
            }

            metrics["threshold_fixed_r"] = thr_fixed
            metrics["selection_rate_0_fixed_r"] = fixed_stats[0]["selection_rate"]
            metrics["selection_rate_1_fixed_r"] = fixed_stats[1]["selection_rate"]

            sel0_fixed = metrics["selection_rate_0_fixed_r"]
            sel1_fixed = metrics["selection_rate_1_fixed_r"]

            if sel0_fixed == 0:
                metrics["dp_ratio_fixed_r"] = np.nan
                _warn(
                    "DP ratio at fixed rate is undefined because group 0 selection rate is 0."
                )
            else:
                metrics["dp_ratio_fixed_r"] = sel1_fixed / sel0_fixed
            if not np.isnan(sel0_fixed) and not np.isnan(sel1_fixed):
                metrics["dp_diff_fixed_r"] = sel1_fixed - sel0_fixed
            else:
                metrics["dp_diff_fixed_r"] = np.nan
    return metrics


def fairness_metrics(y_true, y_proba, A, threshold: float = 0.5) -> dict:
    """
    Wrapper for backwards compatibility that omits target-rate keys.
    """

    metrics = compute_fairness_metrics(
        y_true,
        y_proba,
        A,
        threshold=threshold,
        target_rate=None,
    )
    return metrics


def fairness_at_target_rate(y_true, y_proba, A, target_rate: float) -> dict:
    """
    Compute fairness metrics at a threshold chosen to hit a desired selection rate.
    """

    metrics = compute_fairness_metrics(
        y_true,
        y_proba,
        A,
        threshold=0.5,
        target_rate=target_rate,
    )
    thr = metrics.get("threshold_fixed_r", 0.5)
    actual_rate = float(
        np.mean(np.asarray(y_proba, dtype=float).ravel() >= thr)
    )
    extended = dict(metrics)
    extended.update(
        {
            "threshold": thr,
            "target_rate": target_rate,
            "actual_rate": actual_rate,
        }
    )
    return extended
