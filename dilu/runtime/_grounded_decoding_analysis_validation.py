"""Fail-closed validation gate for the registered V8 grounded-decoding analysis.

Mirrors the blocked-output contract already established by
:mod:`dilu.runtime._minimal_factorial_analysis_validation` (collect every
violated gate, sorted and deduplicated, and report ``status="blocked"``
with ``contrast_artifacts_written=False``) rather than raising on the first
error. V8 adds exactly one new gate beyond the five inherited ones (missing
row, duplicate id, digest mismatch, fingerprint drift, scoring-version
drift): Family M, the manipulation check that grounded decoding actually
restricted the schema (zero ``action_unavailable`` violations per model).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._grounded_decoding_analysis_pairing import (
    build_frozen_comparator_contract,
    resolve_v8_pair,
)
from ._grounded_decoding_comparator_support import ComparatorContractError

EXPECTED_STAGE1_CASES = 30
EXPECTED_STAGE2_CASES = 90
EXPECTED_STAGE1_ROWS = 300
EXPECTED_STAGE2_ROWS = 180
EXPECTED_TOTAL_ROWS = 480
STAGE1_CONDITIONS = frozenset({"c120", "c121"})
STAGE2_CONDITION = "c121"
# The two small models Family D's endpoint contrast covers; the only models
# V8's schedule extends into Stage 2 (see grounded_decoding_schedule.py).
SMALL_MODEL_SLOTS = frozenset({"qwen_06b", "llama_1b"})
_ACTION_UNAVAILABLE_FIELD = "analysis_action_unavailable_count"


@dataclass(frozen=True)
class GroundedAnalysisValidation:
    status: str
    errors: tuple[str, ...]
    contrast_artifacts_written: bool


def validate_v8_rows(
    v8_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
) -> GroundedAnalysisValidation:
    """Validate the 480 claim-bearing V8 rows against the registered design."""
    errors: set[str] = set()
    errors.update(_identity_errors(v8_rows))
    errors.update(_shape_errors(v8_rows))
    errors.update(_pairing_errors(v8_rows, frozen_rows))
    errors.update(_family_m_errors(v8_rows))
    if errors:
        return GroundedAnalysisValidation("blocked", tuple(sorted(errors)), False)
    return GroundedAnalysisValidation("complete", (), True)


def _identity_errors(v8_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    errors: set[str] = set()
    seen_ids: set[str] = set()
    seen_natural: set[tuple[str, str, str, int]] = set()
    for row in v8_rows:
        attempt_id = row.get("episode_attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            errors.add("V8 row is missing a valid episode_attempt_id")
        elif attempt_id in seen_ids:
            errors.add("V8 rows contain a duplicate episode_attempt_id")
        else:
            seen_ids.add(attempt_id)
        try:
            key = (
                str(row["model_slot"]),
                str(row["condition_id"]),
                str(row["case_id"]),
                int(row["simulator_seed"]),
            )
        except (KeyError, TypeError, ValueError):
            errors.add("V8 row is missing a required identity field")
            continue
        if key in seen_natural:
            errors.add(
                "V8 rows contain a duplicate (model_slot, condition_id, case_id, "
                "simulator_seed) identity"
            )
        seen_natural.add(key)
    return errors


def _shape_errors(v8_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    errors: set[str] = set()
    if len(v8_rows) != EXPECTED_TOTAL_ROWS:
        errors.add(
            f"V8 schedule must contain exactly {EXPECTED_TOTAL_ROWS} rows, "
            f"observed {len(v8_rows)}"
        )
    stage1 = [row for row in v8_rows if row.get("stage") == "stage1"]
    stage2 = [row for row in v8_rows if row.get("stage") == "stage2_additional"]
    if len(stage1) + len(stage2) != len(v8_rows):
        errors.add("V8 rows contain an unregistered stage label")
    errors.update(_stage1_shape_errors(stage1))
    errors.update(_stage2_shape_errors(stage2))
    stage1_cases = {str(row.get("case_id")) for row in stage1}
    stage2_cases = {str(row.get("case_id")) for row in stage2}
    if stage1_cases & stage2_cases:
        errors.add("Stage 1 and Stage 2 case identities are not disjoint")
    return errors


def _stage1_shape_errors(stage1: Sequence[Mapping[str, Any]]) -> set[str]:
    errors: set[str] = set()
    if len(stage1) != EXPECTED_STAGE1_ROWS:
        errors.add(
            f"V8 Stage 1 must contain exactly {EXPECTED_STAGE1_ROWS} rows, "
            f"observed {len(stage1)}"
        )
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in stage1:
        cells[(str(row.get("model_slot")), str(row.get("condition_id")))].append(row)
    if any(condition not in STAGE1_CONDITIONS for _model, condition in cells):
        errors.add("V8 Stage 1 contains an unregistered condition_id")
    case_sets = []
    for (_model, _condition), rows in cells.items():
        if len(rows) != EXPECTED_STAGE1_CASES:
            errors.add(
                f"V8 Stage 1 cell does not contain exactly {EXPECTED_STAGE1_CASES} rows"
            )
        case_sets.append(frozenset(str(row.get("case_id")) for row in rows))
    if case_sets and any(cases != case_sets[0] for cases in case_sets[1:]):
        errors.add("V8 Stage 1 case identities are not reused across cells")
    models = {model for model, _condition in cells}
    conditions_by_model: dict[str, set[str]] = defaultdict(set)
    for model, condition in cells:
        conditions_by_model[model].add(condition)
    if any(conditions != STAGE1_CONDITIONS for conditions in conditions_by_model.values()):
        errors.add("V8 Stage 1 does not contain both c120 and c121 for every model")
    if models and len(models) < 2:
        errors.add("V8 Stage 1 does not span multiple models")
    return errors


def _stage2_shape_errors(stage2: Sequence[Mapping[str, Any]]) -> set[str]:
    errors: set[str] = set()
    if len(stage2) != EXPECTED_STAGE2_ROWS:
        errors.add(
            f"V8 Stage 2 must contain exactly {EXPECTED_STAGE2_ROWS} rows, "
            f"observed {len(stage2)}"
        )
    if any(row.get("condition_id") != STAGE2_CONDITION for row in stage2):
        errors.add("V8 Stage 2 contains a condition_id other than c121")
    models = {str(row.get("model_slot")) for row in stage2}
    if stage2 and models != SMALL_MODEL_SLOTS:
        errors.add("V8 Stage 2 model slots do not match the registered small-model set")
    by_model: dict[str, set[str]] = defaultdict(set)
    for row in stage2:
        by_model[str(row.get("model_slot"))].add(str(row.get("case_id")))
    for model, cases in by_model.items():
        if len(cases) != EXPECTED_STAGE2_CASES:
            errors.add(
                f"V8 Stage 2 model {model} does not contain exactly "
                f"{EXPECTED_STAGE2_CASES} case identities"
            )
    case_sets = list(by_model.values())
    if case_sets and any(cases != case_sets[0] for cases in case_sets[1:]):
        errors.add("V8 Stage 2 case identities are not reused across models")
    return errors


def _pairing_errors(
    v8_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    errors: set[str] = set()
    try:
        contract = build_frozen_comparator_contract(frozen_rows)
    except ComparatorContractError as exc:
        return {str(exc)}
    for row in v8_rows:
        try:
            resolve_v8_pair(contract, row)
        except ComparatorContractError as exc:
            errors.add(str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            errors.add(f"V8 row could not be paired to a frozen comparator: {exc}")
    return errors


def _family_m_errors(v8_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Family M gate: zero ``action_unavailable`` violations under O2, per model.

    Filtered to ``condition_id in {"c120", "c121"}`` (O2) rather than
    relying on "every V8 row is O2": the registered digest-drift
    contingency (``rerun_comparators_for``) can append within-V8 O1
    comparator rows (``c110``/``c111``) to the schedule, and this gate must
    stay scoped to its literal "under O2" spec even then.
    """
    errors: set[str] = set()
    totals: dict[str, int] = defaultdict(int)
    for row in v8_rows:
        if row.get("condition_id") not in STAGE1_CONDITIONS:
            continue
        model = str(row.get("model_slot"))
        count = row.get(_ACTION_UNAVAILABLE_FIELD)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            errors.add(
                f"V8 row is missing a valid {_ACTION_UNAVAILABLE_FIELD} (Family M gate)"
            )
            continue
        totals[model] += count
    for model, total in sorted(totals.items()):
        if total != 0:
            errors.add(
                f"Family M gate failed: model {model} recorded {total} "
                "action_unavailable violation(s) under grounded decoding (O2); "
                "the transport or schema failed to restrict the enum."
            )
    return errors


__all__ = [
    "EXPECTED_STAGE1_CASES",
    "EXPECTED_STAGE1_ROWS",
    "EXPECTED_STAGE2_CASES",
    "EXPECTED_STAGE2_ROWS",
    "EXPECTED_TOTAL_ROWS",
    "SMALL_MODEL_SLOTS",
    "STAGE1_CONDITIONS",
    "STAGE2_CONDITION",
    "GroundedAnalysisValidation",
    "validate_v8_rows",
]
