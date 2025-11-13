from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.auto import generate_auto_insurance_data
from src.config import (
    SimulationConfig,
    get_default_auto_simulation_config,
    get_default_configs,
)
from src.credit import generate_credit_underwriting_data
from src.health import simulate_health_insurance_data
from src.life import simulate_life_insurance_data
from src.mortgage import simulate_mortgage_insurance_data


@dataclass(frozen=True)
class ProductSpec:
    name: str
    generator: Callable[..., Any]
    config_getter: Callable[[], Any | None]


def _credit_config() -> SimulationConfig:
    sim_cfg, *_ = get_default_configs()
    return sim_cfg


PRODUCT_REGISTRY: list[ProductSpec] = [
    ProductSpec(
        name="credit",
        generator=generate_credit_underwriting_data,
        config_getter=_credit_config,
    ),
    ProductSpec(
        name="auto",
        generator=generate_auto_insurance_data,
        config_getter=get_default_auto_simulation_config,
    ),
    ProductSpec(
        name="life",
        generator=simulate_life_insurance_data,
        config_getter=lambda: None,
    ),
    ProductSpec(
        name="mortgage",
        generator=simulate_mortgage_insurance_data,
        config_getter=lambda: None,
    ),
    ProductSpec(
        name="health",
        generator=simulate_health_insurance_data,
        config_getter=lambda: None,
    ),
]
