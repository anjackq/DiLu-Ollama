"""Public deterministic schedule API for the ICLR 2027 grounded-decoding V8 campaign.

V8 adds exactly one new output-enforcement level (O2, "backend_schema_grounded")
on top of the already-frozen P1 policy content, at both execution modes
(E0, E1): condition ids ``c120``/``c121``. Everything else -- manifest
parsing conventions, Stage-1 case selection, the deterministic SHA-256
identity recipe, and ``HarnessConfig`` construction -- is imported and
reused from :mod:`dilu.runtime.minimal_factorial_schedule` rather than
reimplemented. The only genuinely new logic here is: (a) the two-cell O2
condition grid, (b) the Stage-2 model/mode restriction to the two small
models on ``c121`` only, and (c) the cross-campaign comparator contract
(delegated to :mod:`dilu.runtime._grounded_decoding_comparator_support`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from ._grounded_decoding_comparator_support import (
    CaseSetFingerprintDriftError,
    ComparatorContract,
    ComparatorContractError,
    ComparatorDigestMismatchError,
    ComparatorRow,
    MissingComparatorRowError,
    ScoringVersionDriftError,
    build_comparator_contract,
    pair_v8_row,
)
from ._grounded_decoding_manifest_support import (
    GROUNDED_DECODING_MANIFEST_SHA256,
    ComparatorPaths,
    GroundedConditionSpec,
    GroundedDecodingManifest,
    GroundedSelectionSpec,
    load_grounded_decoding_manifest,
)
from ._harness_config_support import (
    ConditionSpec,
    ExecutionMode,
    OutputEnforcement,
    PolicyContent,
)
from .minimal_factorial_schedule import (
    RuntimeSnapshot,
    ScheduledEpisode,
    _binding,
    _episodes,
    _trusted_model_digests,
    build_runtime_snapshot,
    select_stage1_cases,
)
from .ollama_transport import OllamaModelIdentity

_REQUIRED_EXECUTION_MODES = frozenset(
    {ExecutionMode.UNSHIELDED_OPERATIONAL, ExecutionMode.SHIELDED}
)


@dataclass(frozen=True)
class V8Schedule:
    """The deterministic V8 schedule.

    ``stage1`` (300 rows: 30 cases x 5 models x {c120, c121}) and
    ``stage2_additional`` (180 rows: 90 remaining cases x 2 small models x
    c121) together are the 480 claim-bearing V8 episodes. ``rerun_rows``
    holds the optional registered-contingency P1 O1 comparator cells
    (added only when a model's live digest no longer matches the frozen
    comparator, so it must be re-paired within V8 instead of across
    campaigns); it is empty unless ``rerun_comparators_for`` was supplied.
    ``all_claim_bearing`` is the union of all three, with uniqueness
    already checked.
    """

    stage1: tuple[ScheduledEpisode, ...]
    stage2_additional: tuple[ScheduledEpisode, ...]
    rerun_rows: tuple[ScheduledEpisode, ...]
    all_claim_bearing: tuple[ScheduledEpisode, ...]


def build_v8_schedule(
    manifest: GroundedDecodingManifest,
    case_set: Mapping[str, Any],
    model_bindings: Mapping[str, OllamaModelIdentity],
    *,
    runtime_snapshot: RuntimeSnapshot,
    rerun_comparators_for: frozenset[str] = frozenset(),
) -> V8Schedule:
    digests = _trusted_model_digests(manifest, model_bindings)
    fingerprint, revision = _binding(runtime_snapshot, case_set)

    policy = manifest.conditions.policy()
    grounded_output = manifest.conditions.output()
    if grounded_output is not OutputEnforcement.BACKEND_SCHEMA_GROUNDED:
        raise ValueError("V8 conditions.output_enforcement must be backend_schema_grounded.")
    grounded_executions = manifest.conditions.executions()
    if set(grounded_executions) != _REQUIRED_EXECUTION_MODES:
        raise ValueError("V8 conditions.execution_modes must be exactly E0 and E1.")
    grounded_specs = tuple(
        ConditionSpec(policy, grounded_output, mode) for mode in grounded_executions
    )

    stage2_mode = manifest.selection.stage2_mode()
    if stage2_mode is not ExecutionMode.SHIELDED:
        raise ValueError("V8 selection.stage2_execution_mode must be shielded (E1).")
    stage2_spec = (ConditionSpec(policy, grounded_output, stage2_mode),)

    known_slots = manifest.model_slots()
    unknown_rerun_slots = frozenset(rerun_comparators_for) - known_slots
    if unknown_rerun_slots:
        raise ValueError(
            f"rerun_comparators_for has unknown model slots: {sorted(unknown_rerun_slots)}."
        )
    stage2_slots = frozenset(manifest.selection.stage2_models)
    if stage2_slots - known_slots:
        raise ValueError("selection.stage2_models has unknown model slots.")

    stage1_cases = select_stage1_cases(case_set, manifest.selection.stage1_hash_prefix)
    stage1_case_ids = {case["case_id"] for case in stage1_cases}
    remaining_cases = tuple(
        case for case in case_set["cases"] if case["case_id"] not in stage1_case_ids
    )

    stage1_rows = _episodes(
        "stage1",
        manifest.campaign_id,
        manifest,
        stage1_cases,
        grounded_specs,
        digests,
        revision,
        fingerprint,
    )
    stage2_rows = _episodes(
        "stage2_additional",
        manifest.campaign_id,
        _restrict_models(manifest, stage2_slots),
        remaining_cases,
        stage2_spec,
        digests,
        revision,
        fingerprint,
    )
    rerun_rows = _build_rerun_rows(
        manifest,
        stage1_cases,
        remaining_cases,
        digests,
        revision,
        fingerprint,
        policy=policy,
        rerun_comparators_for=rerun_comparators_for,
        stage2_slots=stage2_slots,
    )

    all_rows = stage1_rows + stage2_rows + rerun_rows
    if len({row.episode_attempt_id for row in all_rows}) != len(all_rows):
        raise ValueError("V8 schedule contains duplicate episode ids.")
    return V8Schedule(stage1_rows, stage2_rows, rerun_rows, all_rows)


def validate_comparator_pairing(
    schedule: V8Schedule,
    contract: ComparatorContract,
    *,
    rerun_comparators_for: frozenset[str] = frozenset(),
) -> None:
    """Enforce the comparator contract's equality gates over a built schedule.

    Every Stage-1/Stage-2 row for a model *not* under
    ``rerun_comparators_for`` must resolve to exactly one frozen V5/V7 row
    (same model, case, seed, O1 in place of O2); ``pair_v8_row`` raises the
    appropriate typed error otherwise. Models under ``rerun_comparators_for``
    are paired within V8 itself (see ``V8Schedule.rerun_rows``) and are
    skipped here by design.
    """
    for row in (*schedule.stage1, *schedule.stage2_additional):
        if row.model_slot in rerun_comparators_for:
            continue
        pair_v8_row(contract, row)


def _build_rerun_rows(
    manifest: GroundedDecodingManifest,
    stage1_cases: tuple[Mapping[str, Any], ...],
    remaining_cases: tuple[Mapping[str, Any], ...],
    digests: Mapping[str, str],
    revision: str,
    fingerprint: str,
    *,
    policy: PolicyContent,
    rerun_comparators_for: frozenset[str],
    stage2_slots: frozenset[str],
) -> tuple[ScheduledEpisode, ...]:
    if not rerun_comparators_for:
        return ()
    o1_specs = (
        ConditionSpec(policy, OutputEnforcement.BACKEND_SCHEMA, ExecutionMode.UNSHIELDED_OPERATIONAL),
        ConditionSpec(policy, OutputEnforcement.BACKEND_SCHEMA, ExecutionMode.SHIELDED),
    )
    o1_endpoint_spec = (
        ConditionSpec(policy, OutputEnforcement.BACKEND_SCHEMA, ExecutionMode.SHIELDED),
    )
    rows: list[ScheduledEpisode] = []
    for slot in sorted(rerun_comparators_for):
        slot_manifest = _restrict_models(manifest, frozenset({slot}))
        rows.extend(
            _episodes(
                "comparator_rerun_stage1",
                manifest.campaign_id,
                slot_manifest,
                stage1_cases,
                o1_specs,
                digests,
                revision,
                fingerprint,
            )
        )
        if slot in stage2_slots:
            rows.extend(
                _episodes(
                    "comparator_rerun_stage2",
                    manifest.campaign_id,
                    slot_manifest,
                    remaining_cases,
                    o1_endpoint_spec,
                    digests,
                    revision,
                    fingerprint,
                )
            )
    return tuple(rows)


def _restrict_models(
    manifest: GroundedDecodingManifest, slots: frozenset[str]
) -> GroundedDecodingManifest:
    restricted = tuple(model for model in manifest.models if model.slot in slots)
    if len(restricted) != len(slots):
        raise ValueError("Cannot restrict the V8 manifest to unknown model slots.")
    return replace(manifest, models=restricted)


__all__ = [
    "GROUNDED_DECODING_MANIFEST_SHA256",
    "CaseSetFingerprintDriftError",
    "ComparatorContract",
    "ComparatorContractError",
    "ComparatorDigestMismatchError",
    "ComparatorPaths",
    "ComparatorRow",
    "GroundedConditionSpec",
    "GroundedDecodingManifest",
    "GroundedSelectionSpec",
    "MissingComparatorRowError",
    "ScoringVersionDriftError",
    "V8Schedule",
    "build_comparator_contract",
    "build_runtime_snapshot",
    "build_v8_schedule",
    "load_grounded_decoding_manifest",
    "pair_v8_row",
    "validate_comparator_pairing",
]
