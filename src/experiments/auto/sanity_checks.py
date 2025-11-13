from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.auto import generate_auto_insurance_data
from src.common.feature_spec import FeatureSpec
from src.config import get_default_auto_simulation_config, get_default_configs
from src.credit import train_test_split_df
from src.evaluation.reporting import save_metrics
from src.training.train_adv_nn import train_and_eval_adv_nn
from src.training.train_glm import train_and_eval_glm


AUTO_FEATURE_SPEC = FeatureSpec(
    numeric_features=(
        "territory",
        "age",
        "years_licensed",
        "annual_mileage",
        "vehicle_use",
        "vehicle_age",
        "vehicle_value",
        "safety_score",
        "past_claims_obs",
        "violations_obs",
        "credit_score",
        "income",
    ),
    protected_feature="A",
    proxy_feature="territory",
    target_feature="claim_indicator",
)


def _create_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path("results") / "auto" / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _collect_fairness(metrics: dict) -> dict:
    keys = ["eo_gap_tpr", "eo_gap_fpr", "dp_diff", "dp_ratio"]
    return {key: metrics.get(key, np.nan) for key in keys}


def _maybe_remove_proxy(df: pd.DataFrame, use_proxy: bool) -> pd.DataFrame:
    df_copy = df.copy()
    if use_proxy:
        return df_copy
    for col in ("territory", "credit_score", "income"):
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].mean()
    return df_copy


def main() -> None:
    sim_cfg = get_default_auto_simulation_config()
    _, train_cfg, eval_cfg = get_default_configs()
    run_root = _create_run_dir()

    eval_cfg.target_acceptance_rate = 0.02
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows_bias: list[dict[str, float | str | bool]] = []
    rows_proxy: list[dict[str, float | str | bool]] = []

    for label, bias in (("biased", sim_cfg.bias_strength), ("unbiased", 0.0)):
        sim_case = replace(sim_cfg, bias_strength=bias)
        df = generate_auto_insurance_data(sim_case)
        df_train, df_test = train_test_split_df(
            df, test_size=0.2, seed=sim_case.seed, target_col="claim_indicator"
        )

        glm_metrics = train_and_eval_glm(
            df_train,
            df_test,
            eval_cfg,
            feature_spec=AUTO_FEATURE_SPEC,
        )
        adv_metrics = train_and_eval_adv_nn(
            df_train,
            df_test,
            sim_case,
            train_cfg,
            eval_cfg,
            device,
            feature_spec=AUTO_FEATURE_SPEC,
        )

        for metrics in (glm_metrics, adv_metrics):
            record = {
                "experiment": "measurement_bias",
                "scenario": label,
                "bias_strength": bias,
                "model_name": metrics["model_name"],
            }
            record.update(_collect_fairness(metrics))
            rows_bias.append(record)

    df_base = generate_auto_insurance_data(sim_cfg)
    df_train, df_test = train_test_split_df(
        df_base, test_size=0.2, seed=sim_cfg.seed, target_col="claim_indicator"
    )

    scenarios = [("with_proxy", True), ("without_proxy", False)]
    for label, use_proxy in scenarios:
        train_df_case = _maybe_remove_proxy(df_train, use_proxy)
        test_df_case = _maybe_remove_proxy(df_test, use_proxy)

        glm_metrics = train_and_eval_glm(
            train_df_case,
            test_df_case,
            eval_cfg,
            feature_spec=AUTO_FEATURE_SPEC,
        )
        adv_metrics = train_and_eval_adv_nn(
            train_df_case,
            test_df_case,
            sim_cfg,
            train_cfg,
            eval_cfg,
            device,
            feature_spec=AUTO_FEATURE_SPEC,
        )

        for metrics in (glm_metrics, adv_metrics):
            record = {
                "experiment": "proxy_ablation",
                "scenario": label,
                "use_proxy": use_proxy,
                "model_name": metrics["model_name"],
            }
            record.update(_collect_fairness(metrics))
            rows_proxy.append(record)

    bias_df = pd.DataFrame(rows_bias)
    proxy_df = pd.DataFrame(rows_proxy)

    save_metrics(bias_df, run_root / "measurement_bias.csv")
    save_metrics(proxy_df, run_root / "proxy_ablation.csv")

    readme_path = run_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Auto sanity checks",
                "",
                "1. Measurement-bias comparison by toggling `bias_strength`.",
                "2. Proxy ablation by zeroing `territory`, `credit_score`, and `income`.",
                "- Metrics in `measurement_bias.csv` and `proxy_ablation.csv`.",
                "- Device used: cuda if available.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved sanity metrics to {run_root}")


if __name__ == "__main__":
    main()
