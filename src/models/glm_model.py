from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class GLMClassifier:
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        penalty: str = "l2",
        solver: str = "lbfgs",
    ) -> None:
        self.model = LogisticRegression(
            C=C, max_iter=max_iter, penalty=penalty, solver=solver
        )

    def fit(self, X, y) -> "GLMClassifier":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1]
