from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from src.config import get_default_configs
from src.data_generation import generate_credit_insurance_data, train_test_split_df
from src.evaluation.reporting import save_metrics
from src.training.train_adv_nn import train_and_eval_adv_nn


def main() -> None:
    sim_cfg, train_cfg, eval_cfg = get_default_configs()

    df = generate_credit_insurance_data(sim_cfg)
    df_train, df_test = train_test_split_df(df, test_size=0.2, seed=sim_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg.target_acceptance_rate = 0.02

    lambdas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
    rows: list[dict[str, float]] = []

    for lambda_val in lambdas:
        cfg = replace(train_cfg, lambda_adv=lambda_val)
        metrics = train_and_eval_adv_nn(
            df_train, df_test, sim_cfg, cfg, eval_cfg, device
        )
        row = {
            "lambda_adv": lambda_val,
            "roc_auc": metrics.get("roc_auc", np.nan),
            "eo_gap_tpr": metrics.get("eo_gap_tpr", np.nan),
            "eo_gap_fpr": metrics.get("eo_gap_fpr", np.nan),
            "tpr_0": metrics.get("tpr_0", np.nan),
            "tpr_1": metrics.get("tpr_1", np.nan),
            "fpr_0": metrics.get("fpr_0", np.nan),
            "fpr_1": metrics.get("fpr_1", np.nan),
            "selection_rate_0": metrics.get("selection_rate_0", np.nan),
            "selection_rate_1": metrics.get("selection_rate_1", np.nan),
            "dp_diff": metrics.get("dp_diff", np.nan),
            "dp_ratio": metrics.get("dp_ratio", np.nan),
        }
        rows.append(row)

    sweep_df = pd.DataFrame(rows)
    print(sweep_df)

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "lambda_sweep.csv")
    save_metrics(sweep_df, csv_path)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            sweep_df["lambda_adv"],
            sweep_df["roc_auc"],
            marker="o",
            label="ROC AUC",
        )
        ax.plot(
            sweep_df["lambda_adv"],
            sweep_df["eo_gap_tpr"],
            marker="s",
            label="EO gap (TPR)",
        )
        ax.plot(
            sweep_df["lambda_adv"],
            sweep_df["eo_gap_fpr"],
            marker="^",
            label="EO gap (FPR)",
        )
        ax.set_xlabel("lambda_adv")
        ax.set_ylabel("Metric value")
        ax.set_title("Lambda Sweep: Accuracy vs Fairness")
        ax.legend()
        plot_path = os.path.join("results", "lambda_sweep_plot.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
