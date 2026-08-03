"""Deterministic category-stratified fixed-suite bootstrap."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_VERSION = "bootstrap-v1"
EVIDENCE_SCOPE = "fixed-suite sensitivity interval"


@dataclass(frozen=True)
class BootstrapInterval:
    effect: float
    lower_2_5: float
    upper_97_5: float
    draws: int
    seed: int
    evidence_scope: str


def derive_bootstrap_seed(
    manifest_sha256: str,
    model_or_reference: str,
    contrast_id: str,
    outcome: str,
    version: str = BOOTSTRAP_VERSION,
) -> int:
    parts = (manifest_sha256, model_or_reference, contrast_id, outcome, version)
    if any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("Bootstrap seed material must be non-empty text.")
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stratified_bootstrap(
    category_values: Mapping[str, Sequence[float]],
    *,
    samples_per_category: int,
    draws: int,
    seed: int,
) -> BootstrapInterval:
    values = _validated_categories(category_values, samples_per_category)
    if draws != BOOTSTRAP_DRAWS:
        raise ValueError("Registered bootstrap requires exactly 20,000 draws.")
    rng = random.Random(seed)
    effects = [
        fmean(_draw_stratified(values, rng, samples_per_category=samples_per_category))
        for _ in range(draws)
    ]
    effects.sort()
    original = [value for category in sorted(values) for value in values[category]]
    return BootstrapInterval(
        fmean(original),
        _percentile(effects, 0.025),
        _percentile(effects, 0.975),
        draws,
        seed,
        EVIDENCE_SCOPE,
    )


def _draw_stratified(
    category_values: Mapping[str, Sequence[float]],
    rng: Any,
    *,
    samples_per_category: int,
) -> tuple[float, ...]:
    return tuple(
        value
        for category in sorted(category_values)
        for value in rng.choices(
            category_values[category],
            k=samples_per_category,
        )
    )


def _validated_categories(
    category_values: Mapping[str, Sequence[float]],
    samples_per_category: int,
) -> dict[str, tuple[float, ...]]:
    if samples_per_category not in {3, 12} or len(category_values) != 10:
        raise ValueError("Bootstrap requires 10 categories with 3 or 12 cases each.")
    values = {
        str(category): tuple(float(value) for value in category_values[category])
        for category in category_values
    }
    if any(len(items) != samples_per_category for items in values.values()):
        raise ValueError("Bootstrap category denominator drifted.")
    if not all(math.isfinite(value) for items in values.values() for value in items):
        raise ValueError("Bootstrap outcomes must be finite.")
    return values


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_VERSION",
    "BootstrapInterval",
    "derive_bootstrap_seed",
    "stratified_bootstrap",
]
