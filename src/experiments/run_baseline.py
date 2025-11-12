from __future__ import annotations

import os

import torch

from src.config import get_default_configs
from src.data_generation import generate_credit_insurance_data, train_test_split_df
from src.evaluation.reporting import format_metrics_table, save_metrics
from src.training.train_adv_nn import train_and_eval_adv_nn
from src.training.train_glm import train_and_eval_glm
from src.training.train_nn import train_and_eval_plain_nn


def main() -> None:
    sim_cfg, train_cfg, eval_cfg = get_default_configs()

    df = generate_credit_insurance_data(sim_cfg)
    df_train, df_test = train_test_split_df(df, test_size=0.2, seed=sim_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    results.append(train_and_eval_glm(df_train, df_test, eval_cfg))
    results.append(
        train_and_eval_plain_nn(df_train, df_test, sim_cfg, train_cfg, eval_cfg, device)
    )
    results.append(
        train_and_eval_adv_nn(df_train, df_test, sim_cfg, train_cfg, eval_cfg, device)
    )

    metrics_table = format_metrics_table(results)
    print(metrics_table)

    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "baseline_results.csv")
    save_metrics(metrics_table, save_path)


if __name__ == "__main__":
    main()
