"""Registered category and stage-denominator checks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

ENDPOINTS = frozenset({"c000", "c111"})
STAGES = frozenset({"stage1", "stage2_additional"})


def category_errors(episodes: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    categories = {str(row.get("category") or "") for row in episodes}
    errors: list[str] = []
    if "" in categories or len(categories) != 10:
        return ("analysis denominator must contain exactly 10 categories",)
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in episodes:
        stage = row.get("stage")
        if stage not in STAGES:
            errors.append("episode contains an unregistered stage label")
            continue
        stage_scope = "stage1" if stage == "stage1" else "endpoint"
        counts[
            (
                str(row.get("model_slot")),
                str(row.get("condition_id")),
                stage_scope,
                str(row.get("category")),
            )
        ] += 1
    for (_model, condition, scope, _category), count in counts.items():
        expected = 3 if scope == "stage1" else 9
        if condition not in ENDPOINTS and scope != "stage1":
            errors.append("non-endpoint category contains Stage 2 rows")
        elif count != expected:
            errors.append("category denominator drifted from registered 3/12 design")
    return tuple(errors)


__all__ = ["ENDPOINTS", "STAGES", "category_errors"]
