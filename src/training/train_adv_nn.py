from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.metrics import compute_accuracy_metrics
from ..evaluation.fairness import fairness_metrics
from ..evaluation.thresholds import threshold_for_acceptance_rate
from ..models.adv_nn_model import (
    AdvPredictor,
    predict_proba_adv_nn,
    train_adv_nn,
)

NUMERIC_FEATURES = ["S", "D", "L"]
PROXY_FEATURE = "Z"
PROTECTED_FEATURE = "A"
TARGET = "Y"


def _prepare_features(df, scaler: StandardScaler | None = None):
    numeric = df[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    proxy = df[[PROXY_FEATURE]].to_numpy(dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(numeric)
    else:
        scaled_numeric = scaler.transform(numeric)

    X = np.concatenate([scaled_numeric, proxy], axis=1).astype(np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    A = df[PROTECTED_FEATURE].to_numpy(dtype=np.int64)
    return X, y, A, scaler


def _build_loader(
    X: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
        torch.from_numpy(A).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader


def train_and_eval_adv_nn(
    df_train,
    df_test,
    sim_cfg: Any,
    train_cfg,
    eval_cfg,
    device: torch.device,
) -> dict:
    """
    Train and evaluate an adversarial neural network with EO regularization.
    """

    X_train, y_train, A_train, scaler = _prepare_features(df_train)
    X_test, y_test, A_test, _ = _prepare_features(df_test, scaler=scaler)

    train_loader = _build_loader(
        X_train, y_train, A_train, batch_size=train_cfg.batch_size
    )

    input_dim = X_train.shape[1]
    model = AdvPredictor(input_dim=input_dim).to(device)
    train_adv_nn(model, train_loader, train_cfg, device=device)

    y_proba = predict_proba_adv_nn(model, X_test, device=device)

    accuracy = compute_accuracy_metrics(y_test, y_proba)
    fairness = fairness_metrics(y_test, y_proba, A_test, threshold=eval_cfg.threshold)
    metrics = {
        "model_name": "ADV_NN",
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
