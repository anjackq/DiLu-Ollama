"""Shared V8-to-frozen-O1 row pairing for the grounded-decoding analysis.

Both the validation gate and the contrast computation need to resolve a V8
grounded-decoding (O2) row to its frozen O1 comparator row. The actual
resolution -- and every fail-closed gate (missing row, digest mismatch) --
is delegated to :class:`~dilu.runtime._grounded_decoding_comparator_support.
ComparatorContract` and :func:`pair_v8_row`, exactly as Task 4 built them;
this module does not re-derive that logic.

Two gates that ``pair_v8_row``/``ComparatorContract`` do not themselves
perform are reproduced here, narrowly: the frozen row's case-set fingerprint
and scoring-policy-version equality checks. Those live in
``_load_comparator_rows`` in ``_grounded_decoding_comparator_support.py``,
which is bound to a file path (it reads ``episodes.jsonl`` off disk) and
therefore cannot run over in-memory synthetic rows in a unit test. The
checks themselves are copied verbatim (same registered constants, same
typed exceptions) rather than approximated.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from ._grounded_decoding_comparator_support import (
    CaseSetFingerprintDriftError,
    ComparatorContract,
    ComparatorContractError,
    ComparatorRow,
    ScoringVersionDriftError,
    pair_v8_row,
)
from ._harness_config_support import ConditionSpec, OutputEnforcement
from ._minimal_factorial_manifest import CASE_FINGERPRINT
from .dilu_scoring import (
    BALANCED_DRIVING_SCORE_POLICY_VERSION,
    SPLIT_SCORING_POLICY_VERSION,
)

_REQUIRED_COMPARATOR_FIELDS = (
    "campaign_id",
    "model_tag",
    "model_digest",
    "condition_id",
    "case_id",
    "simulator_seed",
    "benchmark_fingerprint",
    "split_scoring_policy_version",
    "balanced_driving_score_policy_version",
)


def paired_o1_condition_id(condition_id: str) -> str:
    """The frozen O1 condition id a V8 O2 ``condition_id`` pairs against.

    Reuses :meth:`ConditionSpec.from_condition_id`/``condition_id`` (the
    registered condition-id codec) rather than a hardcoded ``{"c120":
    "c110", ...}`` table, so it stays correct if the digit grammar ever
    grows another output-enforcement level.
    """
    spec = ConditionSpec.from_condition_id(condition_id)
    if spec.output_enforcement is not OutputEnforcement.BACKEND_SCHEMA_GROUNDED:
        raise ValueError(
            f"paired_o1_condition_id expects a grounded-decoding (O2) condition_id, "
            f"got {condition_id!r}."
        )
    return replace(spec, output_enforcement=OutputEnforcement.BACKEND_SCHEMA).condition_id()


def natural_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    """The ``(model_tag, condition_id, case_id, simulator_seed)`` identity key."""
    return (
        str(row["model_tag"]),
        str(row["condition_id"]),
        str(row["case_id"]),
        int(row["simulator_seed"]),
    )


def build_frozen_comparator_contract(
    frozen_rows: Sequence[Mapping[str, Any]],
) -> ComparatorContract:
    """Build the frozen O1 :class:`ComparatorContract` from full frozen rows.

    Validates the fingerprint and scoring-version gates that
    ``_load_comparator_rows`` would otherwise perform at file-read time
    (see module docstring), then delegates identity indexing to
    ``ComparatorContract`` unmodified.
    """
    rows: list[ComparatorRow] = []
    for row in frozen_rows:
        for field in _REQUIRED_COMPARATOR_FIELDS:
            if field not in row:
                raise ComparatorContractError(
                    f"frozen comparator row is missing required field {field!r}."
                )
        if row["benchmark_fingerprint"] != CASE_FINGERPRINT:
            raise CaseSetFingerprintDriftError(
                "frozen comparator row benchmark_fingerprint drifted from the "
                "registered V8 case set."
            )
        if (
            row["split_scoring_policy_version"] != SPLIT_SCORING_POLICY_VERSION
            or row["balanced_driving_score_policy_version"]
            != BALANCED_DRIVING_SCORE_POLICY_VERSION
        ):
            raise ScoringVersionDriftError(
                "frozen comparator row scoring policy version drifted from the "
                "registered version."
            )
        rows.append(
            ComparatorRow(
                campaign_id=str(row["campaign_id"]),
                model_tag=str(row["model_tag"]),
                model_digest=str(row["model_digest"]),
                condition_id=str(row["condition_id"]),
                case_id=str(row["case_id"]),
                simulator_seed=int(row["simulator_seed"]),
                benchmark_fingerprint=str(row["benchmark_fingerprint"]),
                split_scoring_policy_version=str(row["split_scoring_policy_version"]),
                balanced_driving_score_policy_version=str(
                    row["balanced_driving_score_policy_version"]
                ),
            )
        )
    return ComparatorContract(rows)


def resolve_v8_pair(contract: ComparatorContract, row: Mapping[str, Any]) -> ComparatorRow:
    """Resolve one V8 (O2) row to its frozen O1 :class:`ComparatorRow`.

    Adapts the flat analysis-time row mapping to the
    ``row.condition.condition.output_enforcement`` shape that
    :func:`pair_v8_row` expects from a live ``ScheduledEpisode``, so the
    *exact* Task-4 pairing function runs unmodified instead of being
    re-derived for the analysis's row shape.
    """
    spec = ConditionSpec.from_condition_id(str(row["condition_id"]))
    shim = SimpleNamespace(
        condition=SimpleNamespace(condition=spec),
        model_tag=str(row["model_tag"]),
        model_digest=str(row["model_digest"]),
        case_id=str(row["case_id"]),
        simulator_seed=int(row["simulator_seed"]),
    )
    return pair_v8_row(contract, shim)


__all__ = [
    "build_frozen_comparator_contract",
    "natural_key",
    "paired_o1_condition_id",
    "resolve_v8_pair",
]
