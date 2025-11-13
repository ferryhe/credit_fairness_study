from __future__ import annotations

import torch

from src.common.feature_spec import FeatureSpec
from src.training.train_adv_nn import train_and_eval_adv_nn
from src.training.train_glm import train_and_eval_glm
from src.training.train_nn import train_and_eval_plain_nn


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


def run_auto_models(
    df_train,
    df_test,
    sim_cfg,
    train_cfg,
    eval_cfg,
    device: torch.device,
    feature_spec: FeatureSpec | None = None,
) -> list[dict]:
    feature_spec = feature_spec or AUTO_FEATURE_SPEC
    results = [
        train_and_eval_glm(
            df_train,
            df_test,
            eval_cfg,
            feature_spec=feature_spec,
        ),
        train_and_eval_plain_nn(
            df_train,
            df_test,
            sim_cfg,
            train_cfg,
            eval_cfg,
            device,
            feature_spec=feature_spec,
        ),
        train_and_eval_adv_nn(
            df_train,
            df_test,
            sim_cfg,
            train_cfg,
            eval_cfg,
            device,
            feature_spec=feature_spec,
        ),
    ]
    return results
