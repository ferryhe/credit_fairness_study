from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SimulationConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std == 0.0:
        return np.zeros_like(values)
    mean = values.mean()
    return (values - mean) / std


def _solve_intercept(
    sim_cfg: SimulationConfig,
    s_tilde: np.ndarray,
    d_values: np.ndarray,
    l_values: np.ndarray,
) -> tuple[float, np.ndarray]:
    target = sim_cfg.target_default_rate
    lower, upper = -6.0, 6.0
    intercept = 0.0
    p = np.zeros_like(s_tilde)

    for _ in range(100):
        intercept = 0.5 * (lower + upper)
        logits = (
            intercept
            + sim_cfg.alpha_s * s_tilde
            + sim_cfg.alpha_d * d_values
            + sim_cfg.alpha_l * l_values
        )
        p = _sigmoid(logits)
        mean_p = p.mean()
        if abs(mean_p - target) < 1e-4:
            break
        if mean_p > target:
            upper = intercept
        else:
            lower = intercept

    return intercept, p


def generate_credit_insurance_data(sim_cfg: SimulationConfig) -> pd.DataFrame:
    """
    Simulate a biased credit-insurance dataset.
    """

    rng = np.random.default_rng(sim_cfg.seed)
    n = sim_cfg.n_samples

    A = rng.binomial(1, sim_cfg.p_raceA, size=n)
    p_z = np.where(A == 1, sim_cfg.p_z_given_raceA, sim_cfg.p_z_given_raceB)
    Z = rng.binomial(1, p_z)

    S_star = rng.normal(sim_cfg.mu_s, sim_cfg.sigma_s, size=n)
    S = S_star - sim_cfg.bias_b * A + rng.normal(0.0, sim_cfg.tau_s, size=n)

    log_D = rng.normal(sim_cfg.mu_d, sim_cfg.sigma_d, size=n)
    D = np.clip(np.exp(log_D), 0.05, 1.50)

    L = rng.poisson(sim_cfg.lambda_l, size=n)

    S_tilde = _standardize(S_star)
    _, p_true = _solve_intercept(sim_cfg, S_tilde, D, L)

    Y = rng.binomial(1, p_true)

    data = {
        "Y": Y,
        "S": S,
        "D": D,
        "L": L,
        "Z": Z,
        "A": A,
        "S_star": S_star,
        "p_true": p_true,
    }
    return pd.DataFrame(data)


def train_test_split_df(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split on Y. Return (df_train, df_test).
    """

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["Y"]
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)
