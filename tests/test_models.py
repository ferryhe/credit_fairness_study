from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import SimulationConfig, TrainingConfig
from src.credit import (
    generate_credit_underwriting_data,
    train_test_split_df,
)
from src.models.adv_nn_model import AdvPredictor, predict_proba_adv_nn, train_adv_nn
from src.models.glm_model import GLMClassifier
from src.models.nn_model import PlainNN, predict_proba_plain_nn, train_plain_nn

NUMERIC_FEATURES = ["S", "D", "L"]


def _prepare_features(df, scaler: StandardScaler | None = None):
    numeric = df[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    proxy = df[["Z"]].to_numpy(dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric)
    else:
        scaled = scaler.transform(numeric)

    X = np.concatenate([scaled, proxy], axis=1).astype(np.float32)
    y = df["Y"].to_numpy(dtype=np.float32)
    A = df["A"].to_numpy(dtype=np.int64)
    return X, y, A, scaler


def test_glm_classifier_probabilities():
    sim_cfg = SimulationConfig(n_samples=400, seed=101)
    df = generate_credit_underwriting_data(sim_cfg)
    df_train, df_test = train_test_split_df(df, test_size=0.3, seed=sim_cfg.seed)

    X_train, y_train, _, scaler = _prepare_features(df_train)
    X_test, _, _, _ = _prepare_features(df_test, scaler=scaler)

    glm = GLMClassifier().fit(X_train, y_train)
    probs = glm.predict_proba(X_test)

    assert probs.shape[0] == X_test.shape[0]
    assert np.all((probs > 0.0) & (probs < 1.0))


def _build_plain_nn_loaders(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def test_plain_nn_training_and_prediction():
    sim_cfg = SimulationConfig(n_samples=500, seed=202)
    df = generate_credit_underwriting_data(sim_cfg)
    df_train, df_test = train_test_split_df(df, test_size=0.3, seed=sim_cfg.seed)

    X_train, y_train, _, scaler = _prepare_features(df_train)
    X_test, _, _, _ = _prepare_features(df_test, scaler=scaler)

    train_cfg = replace(
        TrainingConfig(),
        n_epochs_nn=2,
        batch_size=64,
    )

    train_loader, val_loader = _build_plain_nn_loaders(
        X_train, y_train, train_cfg.batch_size, sim_cfg.seed
    )

    device = torch.device("cpu")
    model = PlainNN(input_dim=X_train.shape[1]).to(device)
    train_plain_nn(model, train_loader, val_loader, train_cfg, device)

    probs = predict_proba_plain_nn(model, X_test, device=device)
    assert probs.shape[0] == X_test.shape[0]
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert not np.isnan(probs).any()


def test_adv_nn_training_and_prediction():
    sim_cfg = SimulationConfig(n_samples=500, seed=303)
    df = generate_credit_underwriting_data(sim_cfg)
    df_train, df_test = train_test_split_df(df, test_size=0.3, seed=sim_cfg.seed)

    X_train, y_train, A_train, scaler = _prepare_features(df_train)
    X_test, _, _, _ = _prepare_features(df_test, scaler=scaler)

    train_cfg = replace(
        TrainingConfig(),
        n_epochs_adv=2,
        batch_size=64,
        lambda_adv=0.1,
    )

    dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(A_train).long(),
    )
    train_loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    device = torch.device("cpu")
    model = AdvPredictor(input_dim=X_train.shape[1]).to(device)
    train_adv_nn(model, train_loader, train_cfg, device=device)

    probs = predict_proba_adv_nn(model, X_test, device=device)
    assert probs.shape[0] == X_test.shape[0]
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert not np.isnan(probs).any()
