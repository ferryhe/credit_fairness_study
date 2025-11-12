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
from src.training.train_glm import train_and_eval_glm

FAIRNESS_KEYS = ["eo_gap_tpr", "eo_gap_fpr", "dp_diff", "dp_ratio"]


def _collect_fairness(metrics: dict) -> dict:
    return {key: metrics.get(key, np.nan) for key in FAIRNESS_KEYS}


def _maybe_remove_proxy(df, use_proxy: bool):
    if use_proxy:
        return df
    df_copy = df.copy()
    df_copy["Z"] = 0.0
    return df_copy


def run_measurement_bias_experiment(sim_cfg, device, train_cfg, eval_cfg):
    scenarios = [
        ("biased", sim_cfg.bias_b),
        ("unbiased", 0.0),
    ]

    rows = []
    for label, bias in scenarios:
        sim_case = replace(sim_cfg, bias_b=bias)
        df = generate_credit_insurance_data(sim_case)
        df_train, df_test = train_test_split_df(df, test_size=0.2, seed=sim_case.seed)

        glm_metrics = train_and_eval_glm(df_train, df_test, eval_cfg)
        adv_metrics = train_and_eval_adv_nn(df_train, df_test, sim_case, train_cfg, eval_cfg, device)

        for metrics in (glm_metrics, adv_metrics):
            record = {
                "experiment": "measurement_bias",
                "scenario": label,
                "bias_b": bias,
                "model_name": metrics["model_name"],
            }
            record.update(_collect_fairness(metrics))
            rows.append(record)

    df_bias = pd.DataFrame(rows)
    print("Measurement bias comparison:")
    print(df_bias)

    os.makedirs("results", exist_ok=True)
    save_metrics(df_bias, os.path.join("results", "sanity_bias_vs_nobias.csv"))


def run_proxy_experiment(sim_cfg, device, train_cfg, eval_cfg):
    df = generate_credit_insurance_data(sim_cfg)
    df_train, df_test = train_test_split_df(df, test_size=0.2, seed=sim_cfg.seed)

    scenarios = [
        ("with_proxy", True),
        ("without_proxy", False),
    ]

    rows = []
    for label, use_proxy in scenarios:
        train_df_case = _maybe_remove_proxy(df_train, use_proxy)
        test_df_case = _maybe_remove_proxy(df_test, use_proxy)

        glm_metrics = train_and_eval_glm(train_df_case, test_df_case, eval_cfg)
        adv_metrics = train_and_eval_adv_nn(
            train_df_case, test_df_case, sim_cfg, train_cfg, eval_cfg, device
        )

        for metrics in (glm_metrics, adv_metrics):
            record = {
                "experiment": "proxy_feature",
                "scenario": label,
                "use_proxy": use_proxy,
                "model_name": metrics["model_name"],
            }
            record.update(_collect_fairness(metrics))
            rows.append(record)

    df_proxy = pd.DataFrame(rows)
    print("Proxy feature comparison:")
    print(df_proxy)

    os.makedirs("results", exist_ok=True)
    save_metrics(df_proxy, os.path.join("results", "sanity_with_vs_without_proxy.csv"))


def main() -> None:
    sim_cfg, train_cfg, eval_cfg = get_default_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_measurement_bias_experiment(sim_cfg, device, train_cfg, eval_cfg)
    run_proxy_experiment(sim_cfg, device, train_cfg, eval_cfg)


if __name__ == "__main__":
    main()
