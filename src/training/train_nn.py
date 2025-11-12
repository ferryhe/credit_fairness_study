from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.metrics import compute_accuracy_metrics
from ..evaluation.fairness import fairness_metrics
from ..evaluation.thresholds import threshold_for_acceptance_rate
from ..models.nn_model import (
    PlainNN,
    predict_proba_plain_nn,
    train_plain_nn,
)

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

    X = np.concatenate([scaled_numeric, proxy], axis=1).astype(np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    A = df[PROTECTED_FEATURE].to_numpy(dtype=np.int64)
    return X, y, A, scaler


def _build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader


def train_and_eval_plain_nn(
    df_train,
    df_test,
    sim_cfg: Any,
    train_cfg,
    eval_cfg,
    device: torch.device,
) -> dict:
    """
    Train and evaluate a plain neural network classifier.
    """

    X_train, y_train, _, scaler = _prepare_features(df_train)
    X_test, y_test, A_test, _ = _prepare_features(df_test, scaler=scaler)

    train_loader, val_loader = _build_dataloaders(
        X_train, y_train, batch_size=train_cfg.batch_size, seed=sim_cfg.seed
    )

    input_dim = X_train.shape[1]
    model = PlainNN(input_dim=input_dim).to(device)

    train_plain_nn(model, train_loader, val_loader, train_cfg, device)

    y_proba = predict_proba_plain_nn(model, X_test, device=device)

    accuracy = compute_accuracy_metrics(y_test, y_proba)
    fairness = fairness_metrics(y_test, y_proba, A_test, threshold=eval_cfg.threshold)
    metrics = {
        "model_name": "NN",
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
