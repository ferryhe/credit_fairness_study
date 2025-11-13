from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimulationConfig:
    n_samples: int = 60_000
    seed: int = 42
    p_raceA: float = 0.35
    p_z_given_raceA: float = 0.65
    p_z_given_raceB: float = 0.25
    mu_s: float = 700.0
    sigma_s: float = 60.0
    bias_b: float = 35.0
    tau_s: float = 25.0
    mu_d: float = -0.9
    sigma_d: float = 0.45
    lambda_l: float = 0.30
    alpha_s: float = -1.20
    alpha_d: float = 1.25
    alpha_l: float = 0.60
    target_default_rate: float = 0.12


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 1_024
    n_epochs_nn: int = 10
    n_epochs_adv: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_adv: float = 0.1
    warmup_epochs_adv: int = 5


@dataclass(slots=True)
class EvalConfig:
    threshold: float = 0.5
    target_acceptance_rate: float | None = None


def get_default_configs() -> tuple[SimulationConfig, TrainingConfig, EvalConfig]:
    """
    Returns the default (simulation, training, evaluation) configurations.
    """

    return SimulationConfig(), TrainingConfig(), EvalConfig()


@dataclass(slots=True)
class AutoSimulationConfig:
    n_samples: int = 60_000
    seed: int = 202
    p_protected: float = 0.3
    target_claim_freq: float = 0.10
    p_extra_violation: float = 0.3
    p_extra_claim: float = 0.2
    beta_age2: float = 0.3
    beta_exp: float = 0.4
    beta_mileage: float = 0.25
    beta_territory: float = 0.3
    beta_use: float = 0.25
    beta_past_claims: float = 0.5
    beta_violations: float = 0.3
    alpha_sev0: float = 7.5
    alpha_log_value: float = 0.4
    alpha_safety: float = -0.5
    sigma_sev: float = 0.5
    gamma_age2: float = 0.25
    gamma_exp: float = 0.3
    gamma_mileage: float = 0.2
    gamma_territory: float = 0.25
    gamma_use: float = 0.2
    gamma_past_claims: float = 0.6
    gamma_violations: float = 0.4
    gamma_credit: float = 0.08
    gamma_income: float = -0.10
    premium_base_log: float = 5.5
    bias_strength: float = 1.0


def get_default_auto_simulation_config() -> AutoSimulationConfig:
    return AutoSimulationConfig()
