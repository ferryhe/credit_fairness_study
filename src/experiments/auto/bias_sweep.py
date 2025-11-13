from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from src.auto import generate_auto_insurance_data
from src.config import get_default_auto_simulation_config, get_default_configs
from src.credit import train_test_split_df
from src.experiments.auto_baseline_utils import AUTO_FEATURE_SPEC, run_auto_models


BIAS_STRENGTHS = [0.0, 0.25, 0.5, 1.0, 2.0]


def main() -> None:
    base_sim_cfg = get_default_auto_simulation_config()
    _, train_cfg, eval_cfg = get_default_configs()
    eval_cfg.target_acceptance_rate = 0.02

    rows: list[dict[str, float | str]] = []

    for bias_strength in BIAS_STRENGTHS:
        sim_cfg = replace(base_sim_cfg, bias_strength=bias_strength)
        df = generate_auto_insurance_data(
            sim_cfg,
        )
        df_train, df_test = train_test_split_df(
            df, test_size=0.2, seed=sim_cfg.seed, target_col="claim_indicator"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = run_auto_models(
            df_train,
            df_test,
            sim_cfg,
            train_cfg,
            eval_cfg,
            device,
            feature_spec=AUTO_FEATURE_SPEC,
        )
        for metrics in results:
            rows.append(
                {
                    "bias_strength": bias_strength,
                    "model_name": metrics["model_name"],
                    "roc_auc": metrics.get("roc_auc", np.nan),
                    "eo_gap_tpr": metrics.get("eo_gap_tpr", np.nan),
                    "eo_gap_fpr": metrics.get("eo_gap_fpr", np.nan),
                    "dp_ratio_fixed_2pct": metrics.get("dp_ratio_fixed_r", np.nan),
                    "dp_diff": metrics.get("dp_diff", np.nan),
                    "dp_ratio": metrics.get("dp_ratio", np.nan),
                }
            )

    output_path = Path("results") / "auto" / "auto_bias_sweep_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved bias sweep metrics to {output_path}")


if __name__ == "__main__":
    main()
