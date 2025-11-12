from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


class PlainNN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_plain_nn(
    model: PlainNN,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: Any,
    device: torch.device,
) -> PlainNN:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    model.to(device)
    for _ in range(config.n_epochs_nn):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            y_true_list: list[np.ndarray] = []
            y_score_list: list[np.ndarray] = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device).float()
                    y_val = y_val.to(device).float()
                    logits_val = model(X_val).squeeze(-1)
                    y_true_list.append(y_val.cpu().numpy())
                    y_score_list.append(torch.sigmoid(logits_val).cpu().numpy())

            y_true_concat = np.concatenate(y_true_list) if y_true_list else None
            y_score_concat = np.concatenate(y_score_list) if y_score_list else None
            if (
                y_true_concat is not None
                and y_score_concat is not None
                and np.unique(y_true_concat).size == 2
            ):
                _ = roc_auc_score(y_true_concat, y_score_concat)

    return model


def predict_proba_plain_nn(
    model: PlainNN, X: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor_X = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(tensor_X).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs
