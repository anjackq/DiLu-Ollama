"""Fail-closed outcome schema checks for registered analysis rows."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

_EPISODE_NUMBERS = (
    "driving_score_balanced_v1",
    "decision_latency_ms_avg",
)
_EPISODE_BOOLEANS = ("task_completed", "crashed")
_EPISODE_COUNTS = (
    "decisions_made",
    "decision_calls_total",
    "responses_strict_format",
    "fallback_action_count",
    "decision_timeout_count",
    "analysis_any_shield_intervention_count",
    "analysis_lane_change_shield_count",
    "analysis_longitudinal_safety_shield_count",
    "analysis_low_speed_recovery_shield_count",
    "analysis_proposal_action_change_count",
)
_BASELINE_NUMBERS = ("driving_score_balanced_v1",)
_BASELINE_BOOLEANS = ("task_completed", "crashed")


def validate_analysis_metrics(
    episodes: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    errors: set[str] = set()
    for row in episodes:
        _require_numbers(row, _EPISODE_NUMBERS, "episode", errors)
        _require_booleans(row, _EPISODE_BOOLEANS, "episode", errors)
        _require_counts(row, _EPISODE_COUNTS, "episode", errors)
        decisions = _positive_count(row, "decisions_made", "episode", errors)
        calls = _positive_count(row, "decision_calls_total", "episode", errors)
        _bounded_count(row, "responses_strict_format", decisions, errors)
        for field in _EPISODE_COUNTS[3:]:
            _bounded_count(row, field, calls, errors)
        latency = row.get("decision_latency_ms_avg")
        if _finite_number(latency) and float(latency) < 0:
            errors.add("episode decision_latency_ms_avg must be nonnegative")
        _unit_interval(row, "driving_score_balanced_v1", "episode", errors)
    for row in baseline_rows:
        _require_numbers(row, _BASELINE_NUMBERS, "baseline", errors)
        _require_booleans(row, _BASELINE_BOOLEANS, "baseline", errors)
        _unit_interval(row, "driving_score_balanced_v1", "baseline", errors)
    return tuple(sorted(errors))


def _require_numbers(
    row: Mapping[str, Any],
    fields: Sequence[str],
    scope: str,
    errors: set[str],
) -> None:
    for field in fields:
        if not _finite_number(row.get(field)):
            errors.add(f"{scope} {field} must be a finite number")


def _require_booleans(
    row: Mapping[str, Any],
    fields: Sequence[str],
    scope: str,
    errors: set[str],
) -> None:
    for field in fields:
        if not isinstance(row.get(field), bool):
            errors.add(f"{scope} {field} must be a boolean")


def _require_counts(
    row: Mapping[str, Any],
    fields: Sequence[str],
    scope: str,
    errors: set[str],
) -> None:
    for field in fields:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.add(f"{scope} {field} must be a nonnegative integer")


def _positive_count(
    row: Mapping[str, Any],
    field: str,
    scope: str,
    errors: set[str],
) -> int | None:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.add(f"{scope} {field} must be a positive integer")
        return None
    return value


def _bounded_count(
    row: Mapping[str, Any],
    field: str,
    denominator: int | None,
    errors: set[str],
) -> None:
    value = row.get(field)
    if (
        denominator is not None
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value > denominator
    ):
        errors.add(f"episode {field} exceeds its registered denominator")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit_interval(
    row: Mapping[str, Any],
    field: str,
    scope: str,
    errors: set[str],
) -> None:
    value = row.get(field)
    if _finite_number(value) and not 0.0 <= float(value) <= 1.0:
        errors.add(f"{scope} {field} must be within [0, 1]")


__all__ = ["validate_analysis_metrics"]
