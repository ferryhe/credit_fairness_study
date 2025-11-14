from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve

from src.auto import generate_auto_insurance_data
from src.config import EvalConfig, get_default_configs, get_default_auto_simulation_config
from src.credit import train_test_split_df
from src.evaluation.fairness import fairness_metrics
from src.experiments.auto_baseline_utils import AUTO_FEATURE_SPEC
from src.training.train_nn import (
    PlainNN,
    _build_dataloaders,
    _prepare_features,
    train_plain_nn,
    predict_proba_plain_nn,
)


RUN_DIR = Path("results") / "auto" / "20251113_191359"
PLOTS_DIR = RUN_DIR / "diagnostics" / "nn_bias_strength_2_diagnostics"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_score_distributions(y_proba: np.ndarray, A: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {0: "tab:blue", 1: "tab:orange"}
    labels = {0: "majority (A=0)", 1: "minority (A=1)"}
    bins = np.linspace(0.0, 1.0, 50)
    for group in (0, 1):
        mask = A == group
        if not mask.any():
            continue
        ax.hist(
            y_proba[mask],
            bins=bins,
            alpha=0.5,
            density=True,
            color=colors[group],
            label=labels[group],
        )
    ax.set_title("NN score distributions by group")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, A: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {0: "tab:blue", 1: "tab:orange"}
    labels = {0: "majority (A=0)", 1: "minority (A=1)"}
    for group in (0, 1):
        mask = (A == group)
        if mask.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[mask], y_proba[mask])
        ax.plot(fpr, tpr, label=f"{labels[group]} (AUC={np.trapezoid(tpr, fpr):.2f})", color=colors[group])
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC curves by protected group")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_tpr_fpr_vs_threshold(y_true: np.ndarray, y_proba: np.ndarray, A: np.ndarray, out_path: Path) -> None:
    thresholds = np.linspace(0.0, 1.0, 201)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    colors = {0: "tab:blue", 1: "tab:orange"}
    labels = {0: "majority (A=0)", 1: "minority (A=1)"}

    for group in (0, 1):
        mask = A == group
        if mask.sum() == 0:
            continue
        tprs = []
        fprs = []
        y_true_g = y_true[mask]
        y_proba_g = y_proba[mask]
        for thr in thresholds:
            y_pred_g = (y_proba_g >= thr).astype(int)
            tp = ((y_true_g == 1) & (y_pred_g == 1)).sum()
            fn = ((y_true_g == 1) & (y_pred_g == 0)).sum()
            fp = ((y_true_g == 0) & (y_pred_g == 1)).sum()
            tn = ((y_true_g == 0) & (y_pred_g == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tprs.append(tpr)
            fprs.append(fpr)
        axes[0].plot(thresholds, tprs, label=labels[group], color=colors[group])
        axes[1].plot(thresholds, fprs, label=labels[group], color=colors[group])

    axes[0].set_ylabel("TPR")
    axes[1].set_ylabel("FPR")
    axes[1].set_xlabel("Threshold")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend()
    fig.suptitle("TPR / FPR vs threshold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_confusion_matrices(y_true: np.ndarray, y_proba: np.ndarray, A: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    default_threshold = 0.5
    colors = {0: "tab:blue", 1: "tab:orange"}
    labels = {0: "majority", 1: "minority"}
    for idx, group in enumerate((0, 1)):
        mask = A == group
        ax = axes[idx]
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center")
            ax.set_axis_off()
            continue
        cm = confusion_matrix(y_true[mask], (y_proba[mask] >= default_threshold).astype(int))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{labels[group]} (A={group})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{val}", ha="center", va="center", color="white" if val > cm.max() / 2 else "black")
    fig.suptitle("Confusion matrices by protected group (threshold=0.5)")
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_output_histogram(y_proba: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_proba, bins=40, color="tab:green", alpha=0.75)
    ax.set_title("NN output distribution (all samples)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    sim_cfg = get_default_auto_simulation_config()
    sim_cfg.bias_strength = 2.0
    df = generate_auto_insurance_data(sim_cfg)
    df_train, df_test = train_test_split_df(
        df, test_size=0.2, seed=sim_cfg.seed, target_col="claim_indicator"
    )

    _, train_cfg, eval_cfg = get_default_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, _, scaler = _prepare_features(df_train, AUTO_FEATURE_SPEC)
    X_test, y_test, A_test, _ = _prepare_features(df_test, AUTO_FEATURE_SPEC, scaler=scaler)
    train_loader, val_loader = _build_dataloaders(
        X_train, y_train, batch_size=train_cfg.batch_size, seed=sim_cfg.seed
    )

    input_dim = X_train.shape[1]
    model = PlainNN(input_dim=input_dim).to(device)
    train_plain_nn(model, train_loader, val_loader, train_cfg, device)
    y_proba = predict_proba_plain_nn(model, X_test, device=device)

    _ensure_dir(PLOTS_DIR)
    _plot_score_distributions(y_proba, A_test, PLOTS_DIR / "score_distribution.png")
    _plot_roc_curves(y_test, y_proba, A_test, PLOTS_DIR / "roc_curves.png")
    _plot_tpr_fpr_vs_threshold(y_test, y_proba, A_test, PLOTS_DIR / "tpr_fpr_vs_threshold.png")
    _plot_confusion_matrices(y_test, y_proba, A_test, PLOTS_DIR / "confusion_matrices.png")
    _plot_output_histogram(y_proba, PLOTS_DIR / "nn_output_histogram.png")

    fairness = fairness_metrics(y_test, y_proba, A_test, threshold=eval_cfg.threshold)
    score_means = {
        group: float(np.mean(y_proba[A_test == group])) if (A_test == group).sum() > 0 else float("nan")
        for group in (0, 1)
    }
    score_std = np.nanstd(y_proba)

    mean_diff = abs(score_means[0] - score_means[1])
    distributions_identical = mean_diff < 1e-3
    collapsed = score_std < 1e-3
    eo_gap_zero = max(fairness["eo_gap_tpr"], fairness["eo_gap_fpr"]) < 1e-3

    print(f"Bias-strength diagnostics (run={RUN_DIR.name})")
    print(f" - Score means: majority={score_means[0]:.4f}, minority={score_means[1]:.4f}")
    print(f" - Mean difference {mean_diff:.4f} -> distributions identical? {distributions_identical}")
    print(f" - NN output std={score_std:.4f} -> collapse to constant? {collapsed}")
    print(f" - EO gaps: TPR={fairness['eo_gap_tpr']:.4f}, FPR={fairness['eo_gap_fpr']:.4f} -> EO gap zero at threshold=0.5? {eo_gap_zero}")


if __name__ == "__main__":
    main()
