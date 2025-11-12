from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..evaluation.metrics import compute_accuracy_metrics
from ..evaluation.fairness import fairness_metrics
from ..evaluation.thresholds import threshold_for_acceptance_rate
from ..models.glm_model import GLMClassifier


NUMERIC_FEATURES = ["S", "D", "L"]
PROTECTED_FEATURE = "A"
PROXY_FEATURE = "Z"
TARGET = "Y"


def _prepare_features(df, scaler: StandardScaler | None = None):
    numeric = df[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    proxy = df[[PROXY_FEATURE]].to_numpy(dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(numeric)
    else:
        scaled_numeric = scaler.transform(numeric)

    X = np.concatenate([scaled_numeric, proxy], axis=1)
    y = df[TARGET].to_numpy(dtype=np.int64)
    A = df[PROTECTED_FEATURE].to_numpy(dtype=np.int64)
    return X, y, A, scaler


def train_and_eval_glm(df_train, df_test, eval_cfg) -> dict:
    """
    Train a GLMClassifier on df_train and evaluate on df_test.
    """

    X_train, y_train, A_train, scaler = _prepare_features(df_train)
    X_test, y_test, A_test, _ = _prepare_features(df_test, scaler=scaler)

    model = GLMClassifier()
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)

    accuracy = compute_accuracy_metrics(y_test, y_proba)
    fairness = fairness_metrics(y_test, y_proba, A_test, threshold=eval_cfg.threshold)
    metrics = {
        "model_name": "GLM",
        **accuracy,
        **fairness,
    }

    target_rate = getattr(eval_cfg, "target_acceptance_rate", None)
    if target_rate is not None:
        thr = threshold_for_acceptance_rate(y_proba, target_rate)
        fairness_fixed = fairness_metrics(y_test, y_proba, A_test, threshold=thr)
        metrics["target_acceptance_rate"] = target_rate
        metrics["threshold_fixed_r"] = thr
        metrics.update({f"{k}_fixed_r": v for k, v in fairness_fixed.items()})

    return metrics
