from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.common.feature_spec import FeatureSpec, CREDIT_FEATURE_SPEC

from ..evaluation.metrics import compute_accuracy_metrics
from ..evaluation.fairness import compute_fairness_metrics
from ..models.nn_model import (
    PlainNN,
    predict_proba_plain_nn,
    train_plain_nn,
)


def _prepare_features(
    df,
    feature_spec: FeatureSpec,
    scaler: StandardScaler | None = None,
):
    numeric = df[list(feature_spec.numeric_features)].to_numpy(dtype=np.float32)
    proxy = df[[feature_spec.proxy_feature]].to_numpy(dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(numeric)
    else:
        scaled_numeric = scaler.transform(numeric)

    X = np.concatenate([scaled_numeric, proxy], axis=1).astype(np.float32)
    y = df[feature_spec.target_feature].to_numpy(dtype=np.float32)
    A = df[feature_spec.protected_feature].to_numpy(dtype=np.int64)
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
    feature_spec: FeatureSpec | None = None,
) -> dict:
    """
    Train and evaluate a plain neural network classifier.
    """

    feature_spec = feature_spec or CREDIT_FEATURE_SPEC
    X_train, y_train, _, scaler = _prepare_features(df_train, feature_spec)
    X_test, y_test, A_test, _ = _prepare_features(
        df_test, feature_spec, scaler=scaler
    )

    train_loader, val_loader = _build_dataloaders(
        X_train, y_train, batch_size=train_cfg.batch_size, seed=sim_cfg.seed
    )

    input_dim = X_train.shape[1]
    model = PlainNN(input_dim=input_dim).to(device)

    train_plain_nn(model, train_loader, val_loader, train_cfg, device)

    y_proba = predict_proba_plain_nn(model, X_test, device=device)

    accuracy = compute_accuracy_metrics(y_test, y_proba)
    fairness = compute_fairness_metrics(
        y_test,
        y_proba,
        A_test,
        threshold=eval_cfg.threshold,
        target_rate=getattr(eval_cfg, "target_acceptance_rate", None),
    )
    metrics = {
        "model_name": "NN",
        **accuracy,
        **fairness,
    }

    return metrics
