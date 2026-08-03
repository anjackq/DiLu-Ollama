"""Outcome readers and descriptive summaries for registered analysis."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

OutcomeReader = Callable[[Mapping[str, Any]], float]


def outcomes() -> dict[str, tuple[OutcomeReader, str]]:
    return {
        "driving_score_balanced_v1": (_number("driving_score_balanced_v1"), "higher"),
        "task_completion": (_boolean("task_completed"), "higher"),
        "crash": (_boolean("crashed"), "lower"),
        "strict_format_rate": (
            _rate("responses_strict_format", "decisions_made"),
            "higher",
        ),
        "fallback_rate": (
            _rate("fallback_action_count", "decision_calls_total"),
            "lower",
        ),
        "timeout_rate": (
            _rate("decision_timeout_count", "decision_calls_total"),
            "lower",
        ),
        "any_shield_intervention": (
            _rate("analysis_any_shield_intervention_count", "decision_calls_total"),
            "diagnostic",
        ),
        "lane_change_shield": (
            _rate("analysis_lane_change_shield_count", "decision_calls_total"),
            "diagnostic",
        ),
        "longitudinal_safety_shield": (
            _rate(
                "analysis_longitudinal_safety_shield_count",
                "decision_calls_total",
            ),
            "diagnostic",
        ),
        "low_speed_recovery_shield": (
            _rate(
                "analysis_low_speed_recovery_shield_count",
                "decision_calls_total",
            ),
            "diagnostic",
        ),
        "proposal_action_change": (
            _rate("analysis_proposal_action_change_count", "decision_calls_total"),
            "diagnostic",
        ),
        "decision_latency_ms_avg": (_number("decision_latency_ms_avg"), "lower"),
    }


def condition_summaries(
    episodes: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    readers: Mapping[str, tuple[OutcomeReader, str]],
    provenance: Mapping[str, str],
) -> list[dict[str, Any]]:
    output = []
    for model in models:
        grouped = _group(
            [row for row in episodes if row.get("model_slot") == model],
            "condition_id",
        )
        for condition, rows in sorted(grouped.items()):
            for outcome, (reader, direction) in readers.items():
                output.append(
                    {
                        **_summary_row(
                            provenance,
                            model,
                            condition,
                            outcome,
                            direction,
                            [reader(row) for row in rows],
                        ),
                        **_diagnostics(rows),
                    }
                )
    return output


def category_summaries(
    episodes: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    readers: Mapping[str, tuple[OutcomeReader, str]],
    provenance: Mapping[str, str],
) -> list[dict[str, Any]]:
    output = []
    for model in models:
        rows = [row for row in episodes if row.get("model_slot") == model]
        for (condition, category), subset in sorted(
            _group_pairs(rows, "condition_id", "category").items()
        ):
            for outcome, (reader, direction) in readers.items():
                output.append(
                    {
                        **_summary_row(
                            provenance,
                            model,
                            condition,
                            outcome,
                            direction,
                            [reader(row) for row in subset],
                        ),
                        **_diagnostics(subset),
                        "category": category,
                        "evidence_scope": "fixed-suite category description",
                    }
                )
    return output


def _summary_row(
    provenance: Mapping[str, str],
    model: str,
    condition: str,
    outcome: str,
    direction: str,
    values: Sequence[float],
) -> dict[str, Any]:
    return {
        **provenance,
        "model_or_reference": model,
        "condition_id": condition,
        "outcome": outcome,
        "outcome_direction": direction,
        "numerator": sum(values),
        "denominator": len(values),
        "effect": fmean(values),
        "lower_2_5": "",
        "upper_97_5": "",
        "evidence_scope": "fixed-suite descriptive summary",
    }


def _diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    fallback_dominated = sum(
        _count(row, "fallback_action_count")
        > _positive_count(row, "decision_calls_total") / 2
        for row in rows
    )
    return {
        "missing_count": 0,
        "blocked_count": 0,
        "trace_invalid_count": 0,
        "fallback_dominated_count": fallback_dominated,
    }


def _group(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_text(row, key)].append(row)
    return dict(grouped)


def _group_pairs(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(_text(row, left), _text(row, right))].append(row)
    return dict(grouped)


def _number(name: str) -> OutcomeReader:
    def read(row: Mapping[str, Any]) -> float:
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be a finite number.")
        return float(value)

    return read


def _boolean(name: str) -> OutcomeReader:
    def read(row: Mapping[str, Any]) -> float:
        value = row.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")
        return float(value)

    return read


def _rate(numerator: str, denominator: str) -> OutcomeReader:
    def read(row: Mapping[str, Any]) -> float:
        count = _count(row, numerator)
        total = _positive_count(row, denominator)
        if count > total:
            raise ValueError(f"{numerator} exceeds {denominator}.")
        return count / total

    return read


def _count(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _positive_count(row: Mapping[str, Any], name: str) -> int:
    value = _count(row, name)
    if value == 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text.")
    return value


__all__ = ["OutcomeReader", "category_summaries", "condition_summaries", "outcomes"]
