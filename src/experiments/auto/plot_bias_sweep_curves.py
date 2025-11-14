from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 200
MODELS = ("GLM", "NN", "ADV_NN")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot(series_df, x_col, y_col, ylabel, filename, plot_dir, horizontal=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS:
        subset = series_df[series_df["model_name"] == model].sort_values(x_col)
        subset = subset.dropna(subset=[y_col])
        if subset.empty:
            logging.warning("%s missing for %s; skipping curve", y_col, model)
            continue
        ax.plot(
            subset[x_col],
            subset[y_col],
            marker="o",
            label=model,
            linewidth=2,
        )
    if horizontal is not None:
        ax.axhline(horizontal, color="gray", linestyle="--", linewidth=1.0, label="perfect")
    ax.set_xlabel("bias_strength")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend()
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    save_path = plot_dir / filename
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logging.info("Saved %s", save_path)


def main() -> None:
    metrics_path = Path("results") / "auto" / "auto_bias_sweep_metrics.csv"
    df = pd.read_csv(metrics_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = Path("results") / "auto" / "bias_sweep_plots" / timestamp
    _ensure_dir(plot_dir)

    _plot(
        df,
        "bias_strength",
        "roc_auc",
        "ROC AUC",
        "roc_auc_vs_bias.png",
        plot_dir,
    )
    _plot(
        df,
        "bias_strength",
        "eo_gap_tpr",
        "Equalized Odds TPR gap",
        "eo_tpr_gap_vs_bias.png",
        plot_dir,
    )
    _plot(
        df,
        "bias_strength",
        "eo_gap_fpr",
        "Equalized Odds FPR gap",
        "eo_fpr_gap_vs_bias.png",
        plot_dir,
    )
    _plot(
        df,
        "bias_strength",
        "dp_ratio_fixed_2pct",
        "DP ratio (fixed 2%)",
        "dp_ratio_vs_bias.png",
        plot_dir,
        horizontal=1.0,
    )


if __name__ == "__main__":
    main()
