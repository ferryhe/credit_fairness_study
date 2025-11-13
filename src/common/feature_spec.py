from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FeatureSpec:
    numeric_features: tuple[str, ...]
    protected_feature: str = "A"
    proxy_feature: str = "Z"
    target_feature: str = "Y"


def _ensure_tuple(features: Iterable[str]) -> tuple[str, ...]:
    return tuple(features)


CREDIT_FEATURE_SPEC = FeatureSpec(
    numeric_features=_ensure_tuple(["S", "D", "L"]),
    protected_feature="A",
    proxy_feature="Z",
    target_feature="Y",
)
