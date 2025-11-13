from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # plotting is optional
    plt = None

from src.config import AutoSimulationConfig


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_histograms(df: pd.DataFrame, out_dir: Path, prefix: str = "") -> None:
    if plt is None:
        print("matplotlib is not available; skipping debug plots.")
        return

    _ensure_dir(out_dir)

    vars_to_plot = [
        "A",
        "territory",
        "age",
        "years_licensed",
        "annual_mileage",
        "vehicle_age",
        "vehicle_value",
        "safety_score",
        "past_claims_true",
        "violations_true",
        "past_claims_obs",
        "violations_obs",
        "income",
        "credit_score",
        "claim_count",
        "pure_premium_true",
        "premium_charged",
    ]

    for col in vars_to_plot:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        plt.figure(figsize=(6, 4))
        plt.hist(series, bins=30)
        plt.title(col)
        plt.tight_layout()
        fname = f"{prefix}{col}.png" if prefix else f"{col}.png"
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()


def generate_auto_insurance_data(
    sim_cfg: AutoSimulationConfig,
    debug_plots: bool = False,
    debug_plots_dir: str | Path = "debug_auto_distributions",
) -> pd.DataFrame:
    rng = np.random.default_rng(sim_cfg.seed)
    n_samples = sim_cfg.n_samples
    bias_strength = sim_cfg.bias_strength

    A = rng.binomial(1, sim_cfg.p_protected, size=n_samples)
    mask_prot = A == 1
    mask_nonprot = ~mask_prot

    territory = np.empty(n_samples, dtype=int)
    territory[mask_prot] = rng.choice([1, 2], size=mask_prot.sum(), p=[0.3, 0.7])
    territory[mask_nonprot] = rng.choice(
        [0, 1, 2], size=mask_nonprot.sum(), p=[0.6, 0.35, 0.05]
    )

    age = 18.0 + rng.beta(a=2.2, b=1.8, size=n_samples) * (80.0 - 18.0)

    max_years = np.maximum(age - 16.0, 0.0)
    u = rng.beta(a=2.0, b=1.0, size=n_samples)
    years_licensed = u * max_years

    log_mileage = rng.normal(loc=10.0, scale=0.5, size=n_samples)
    annual_mileage = np.exp(log_mileage)
    annual_mileage = np.clip(annual_mileage, 3_000.0, 60_000.0)

    vehicle_use = rng.choice([0, 1, 2], size=n_samples, p=[0.5, 0.4, 0.1])

    vehicle_age = rng.beta(a=2.0, b=4.0, size=n_samples) * 25.0

    base_log_value = 10.0 - 0.08 * vehicle_age
    log_value = base_log_value + rng.normal(loc=0.0, scale=0.3, size=n_samples)
    vehicle_value = np.exp(log_value)
    vehicle_value = np.clip(vehicle_value, 3_000.0, 80_000.0)

    safety_score = rng.beta(a=2.0, b=3.0, size=n_samples)

    past_claims_true = rng.poisson(lam=0.2, size=n_samples)
    violations_true = rng.poisson(lam=0.4, size=n_samples)

    log_income = rng.normal(
        loc=11.0 - 0.5 * A,
        scale=0.7,
        size=n_samples,
    )
    income = np.exp(log_income)
    income = np.clip(income, 10_000.0, 500_000.0)

    credit_score = rng.normal(
        loc=700.0 - 80.0 * A,
        scale=60.0,
        size=n_samples,
    )
    credit_score = np.clip(credit_score, 300.0, 900.0)

    age_c = (age - 40.0) / 10.0
    is_non_pleasure = (vehicle_use != 0).astype(float)

    lin_no_intercept = (
        sim_cfg.beta_age2 * (age_c ** 2)
        - sim_cfg.beta_exp * np.log(years_licensed + 1.0)
        + sim_cfg.beta_mileage * np.log(annual_mileage)
        + sim_cfg.beta_territory * territory
        + sim_cfg.beta_use * is_non_pleasure
        + sim_cfg.beta_past_claims * past_claims_true
        + sim_cfg.beta_violations * violations_true
    )

    lambda_temp = np.exp(lin_no_intercept)
    prob_temp = 1.0 - np.exp(-lambda_temp)
    current_freq = prob_temp.mean()
    if current_freq <= 0.0:
        raise RuntimeError("Current claim frequency is zero; check parameters.")

    scale = sim_cfg.target_claim_freq / current_freq
    lambda_ = lambda_temp * scale

    claim_count = rng.poisson(lam=lambda_)
    claim_indicator = (claim_count > 0).astype(int)

    log_severity_mean = (
        sim_cfg.alpha_sev0
        + sim_cfg.alpha_log_value * np.log(vehicle_value)
        + sim_cfg.alpha_safety * safety_score
    )
    expected_severity = np.exp(log_severity_mean + 0.5 * (sim_cfg.sigma_sev ** 2))
    pure_premium_true = lambda_ * expected_severity

    past_claims_obs = past_claims_true.copy()
    violations_obs = violations_true.copy()

    extra_viol = rng.binomial(1, sim_cfg.p_extra_violation * bias_strength, size=n_samples)
    extra_claim = rng.binomial(1, sim_cfg.p_extra_claim * bias_strength, size=n_samples)

    past_claims_obs[mask_prot] += extra_claim[mask_prot]
    violations_obs[mask_prot] += extra_viol[mask_prot]

    credit_penalty = (800.0 - credit_score) / 100.0
    log_income_centered = np.log(income) - 11.0

    lin_premium = (
        sim_cfg.premium_base_log
        + sim_cfg.gamma_age2 * (age_c ** 2)
        - sim_cfg.gamma_exp * np.log(years_licensed + 1.0)
        + sim_cfg.gamma_mileage * np.log(annual_mileage)
        + sim_cfg.gamma_territory * territory
        + sim_cfg.gamma_use * is_non_pleasure
        + sim_cfg.gamma_past_claims * past_claims_obs
        + sim_cfg.gamma_violations * violations_obs
        + sim_cfg.gamma_credit * credit_penalty
        + sim_cfg.gamma_income * log_income_centered
    )

    premium_charged = np.exp(lin_premium)

    df = pd.DataFrame(
        {
            "A": A,
            "territory": territory,
            "age": age,
            "years_licensed": years_licensed,
            "annual_mileage": annual_mileage,
            "vehicle_use": vehicle_use,
            "vehicle_age": vehicle_age,
            "vehicle_value": vehicle_value,
            "safety_score": safety_score,
            "past_claims_true": past_claims_true,
            "violations_true": violations_true,
            "past_claims_obs": past_claims_obs,
            "violations_obs": violations_obs,
            "income": income,
            "credit_score": credit_score,
            "lambda_true": lambda_,
            "claim_count": claim_count,
            "claim_indicator": claim_indicator,
            "pure_premium_true": pure_premium_true,
            "premium_charged": premium_charged,
        }
    )

    if debug_plots:
        out_dir = Path(debug_plots_dir)
        _plot_histograms(df, out_dir=out_dir, prefix="auto_")

    return df


if __name__ == "__main__":
    default_cfg = AutoSimulationConfig()
    demo_df = generate_auto_insurance_data(
        default_cfg,
        debug_plots=True,
        debug_plots_dir="debug_auto_distributions",
    )
    print("Demo dataset shape:", demo_df.shape)
    print("Overall claim frequency:", demo_df["claim_indicator"].mean())
    print("Mean pure premium:", demo_df["pure_premium_true"].mean())
    print("Mean charged premium:", demo_df["premium_charged"].mean())
    print("Mean premium by A:", demo_df.groupby("A")["premium_charged"].mean().to_dict())
