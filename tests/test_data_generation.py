from __future__ import annotations

import numpy as np

from src.config import SimulationConfig
from src.credit import generate_credit_underwriting_data


def test_generate_credit_underwriting_data_basic():
    sim_cfg = SimulationConfig(n_samples=2000, seed=123)
    df = generate_credit_underwriting_data(sim_cfg)

    required_cols = ["Y", "S", "D", "L", "Z", "A", "S_star", "p_true"]
    assert all(col in df.columns for col in required_cols)

    assert not df.isna().any().any()

    y_mean = df["Y"].mean()
    assert 0 < y_mean < 1
    assert abs(y_mean - sim_cfg.target_default_rate) < 0.03
