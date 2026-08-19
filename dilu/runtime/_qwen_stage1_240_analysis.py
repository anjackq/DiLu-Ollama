"""Fixed, non-promotable analysis of completed Qwen Stage-1 evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._campaign_attempt_state import read_validated_attempt_snapshot
from ._minimal_factorial_analysis_io import _campaign_sha256
from ._minimal_factorial_analysis_locks import _validate_binding
from ._minimal_factorial_analysis_summaries import condition_summaries, outcomes
from ._minimal_factorial_analysis_tables import compute_stage1_factor_rows
from ._minimal_factorial_manifest import _manifest_payload, load_experiment_manifest
from ._minimal_factorial_runner_summaries import (
    load_summary_records,
    summary_root_sha256,
)
from ._minimal_factorial_schedule_support import canonical_sha256
from ._qwen_stage1_240_artifacts import (
    QwenStage1Tables,
    publish_qwen_stage1_bundle,
)
from ._scientific_runtime_binding import load_verified_runtime_lock_binding
from ._scientific_trace_store import read_validated_trace_snapshot
from .campaign_attempts import AttemptStatus

SCOPE = "qwen_stage1_240"
CAMPAIGN_ID = "iclr2027-minimal-factorial-v5"
MODEL_SLOT = "qwen_06b"
STAGE = "stage1"
CONDITIONS = tuple(f"c{index:03b}" for index in range(8))
CASES_PER_CELL = 30
EXPECTED_ROWS = 240
CLAIM_ELIGIBLE = False
_FAST_OUTCOMES = (
    "driving_score_balanced_v1",
    "task_completion",
    "crash",
    "strict_format_rate",
    "fallback_rate",
    "timeout_rate",
    "decision_latency_ms_avg",
)


def build_qwen_stage1_tables(
    episodes: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, str],
    manifest_sha256: str,
) -> QwenStage1Tables:
    _validate_fixed_rows(episodes)
    ordered = tuple(
        sorted(
            episodes,
            key=lambda row: (_text(row, "condition_id"), _text(row, "case_id")),
        )
    )
    readers = {name: outcomes()[name] for name in _FAST_OUTCOMES}
    models = (MODEL_SLOT,)
    condition_rows = condition_summaries(ordered, models, readers, provenance)
    factor_rows = compute_stage1_factor_rows(
        ordered,
        models,
        readers,
        provenance,
        manifest_sha256,
    )
    return QwenStage1Tables(tuple(condition_rows), tuple(factor_rows))


def run_qwen_stage1_analysis(repo_root: Path, *, output_root: Path) -> Path:
    observed_rows = 0
    try:
        root = Path(repo_root).resolve(strict=True)
        manifest = load_experiment_manifest(
            root / "configs" / "iclr2027" / "minimal_factorial.yaml"
        )
        if manifest.campaign_id != CAMPAIGN_ID:
            raise ValueError("Registered campaign ID drifted.")
        campaign_root = root / manifest.outputs.root / manifest.outputs.llm_campaign
        claim_path = campaign_root / "campaign_manifest.json"
        claim_bytes = claim_path.read_bytes()
        claim = _object(json.loads(claim_bytes), "campaign manifest")
        if claim.get("manifest") != _manifest_payload(manifest):
            raise ValueError("Frozen campaign manifest drifted from registration.")
        schedule = _rows(claim.get("schedule"), "campaign schedule")
        if len(schedule) != 840:
            raise ValueError("Frozen V5 campaign schedule must contain 840 rows.")
        schedule_by_id = _unique_index(schedule, "episode_attempt_id", "schedule")
        selected_schedule = tuple(
            row
            for row in schedule
            if row.get("model_slot") == MODEL_SLOT and row.get("stage") == STAGE
        )
        _validate_fixed_rows(selected_schedule)
        selected_ids = {_text(row, "episode_attempt_id") for row in selected_schedule}

        ledger = read_validated_attempt_snapshot(
            campaign_root / "campaign_attempts.jsonl",
            campaign_id=CAMPAIGN_ID,
        )
        summaries = load_summary_records(
            campaign_root / "episodes.jsonl",
            expected_campaign_provenance_sha256=_campaign_sha256(claim),
        )
        traces = read_validated_trace_snapshot(
            campaign_root / "traces" / "decision_traces.jsonl",
            artifact_root=campaign_root,
        )
        summary_by_id = _unique_index(summaries, "episode_attempt_id", "summaries")
        terminal_by_id = _unique_terminal_index(ledger.terminal_records)
        trace_references = {
            key: tuple(reference.to_dict() for reference in references)
            for key, references in traces.references_by_attempt.items()
        }
        _validate_global_evidence(
            schedule_by_id,
            ledger.statuses,
            summary_by_id,
            terminal_by_id,
            trace_references,
        )
        if any(
            ledger.statuses.get(attempt_id) is not AttemptStatus.COMPLETED
            for attempt_id in selected_ids
        ):
            raise ValueError("Qwen Stage-1 scope is not exactly 240 completed attempts.")
        selected = tuple(summary_by_id[attempt_id] for attempt_id in sorted(selected_ids))
        observed_rows = len(selected)
        _validate_fixed_rows(selected)
        _validate_locks(root, manifest, claim, selected_schedule, selected)

        manifest_sha256 = "sha256:" + hashlib.sha256(claim_bytes).hexdigest()
        selected_trace_sha256 = "sha256:" + canonical_sha256(
            {
                attempt_id: trace_references[(CAMPAIGN_ID, attempt_id)]
                for attempt_id in sorted(selected_ids)
            }
        )
        validation = {
            "status": "complete",
            "errors": [],
            "scope": SCOPE,
            "claim_eligible": CLAIM_ELIGIBLE,
            "campaign_id": CAMPAIGN_ID,
            "model_slot": MODEL_SLOT,
            "stage": STAGE,
            "conditions": list(CONDITIONS),
            "rows_per_condition": CASES_PER_CELL,
            "expected_rows": EXPECTED_ROWS,
            "observed_rows": observed_rows,
            "run_source_revision": _text(
                _object(claim.get("runtime_snapshot"), "runtime snapshot"),
                "code_revision",
            ),
            "manifest_sha256": manifest_sha256,
            "summary_root_sha256": summary_root_sha256(summaries),
            "selected_trace_references_sha256": selected_trace_sha256,
        }
        provenance = {
            key: str(validation[key]).lower()
            for key in (
                "scope",
                "claim_eligible",
                "campaign_id",
                "model_slot",
                "run_source_revision",
                "manifest_sha256",
                "summary_root_sha256",
                "selected_trace_references_sha256",
            )
        }
        tables = build_qwen_stage1_tables(
            selected,
            provenance=provenance,
            manifest_sha256=manifest_sha256,
        )
        return publish_qwen_stage1_bundle(output_root, validation, tables)
    except Exception as exc:  # noqa: BLE001 - fail closed into validation artifact
        validation = {
            "status": "blocked",
            "errors": [f"Qwen Stage-1 diagnostic validation failed: {exc}"],
            "scope": SCOPE,
            "claim_eligible": CLAIM_ELIGIBLE,
            "campaign_id": CAMPAIGN_ID,
            "model_slot": MODEL_SLOT,
            "stage": STAGE,
            "expected_rows": EXPECTED_ROWS,
            "observed_rows": observed_rows,
        }
        return publish_qwen_stage1_bundle(output_root, validation)


def _validate_fixed_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("Qwen Stage-1 diagnostic requires exactly 240 rows.")
    cells: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if (
            row.get("campaign_id") != CAMPAIGN_ID
            or row.get("model_slot") != MODEL_SLOT
            or row.get("stage") != STAGE
        ):
            raise ValueError("Qwen Stage-1 identity drifted.")
        condition = _text(row, "condition_id")
        cells.setdefault(condition, []).append(row)
    if set(cells) != set(CONDITIONS) or any(
        len(cell) != CASES_PER_CELL for cell in cells.values()
    ):
        raise ValueError("Qwen Stage-1 cells are not the fixed 8 x 30 design.")
    identities = [
        {(_text(row, "case_id"), row.get("simulator_seed")) for row in cells[name]}
        for name in CONDITIONS
    ]
    if any(values != identities[0] for values in identities[1:]):
        raise ValueError("Qwen Stage-1 case/seed identities are not reused.")
    if all("category" in row for row in rows):
        category_maps = [
            {_text(row, "case_id"): _text(row, "category") for row in cells[name]}
            for name in CONDITIONS
        ]
        if any(values != category_maps[0] for values in category_maps[1:]):
            raise ValueError("Qwen Stage-1 category identities drifted.")
        counts: dict[str, int] = {}
        for category in category_maps[0].values():
            counts[category] = counts.get(category, 0) + 1
        if len(counts) != 10 or set(counts.values()) != {3}:
            raise ValueError("Qwen Stage-1 requires 10 categories x 3 cases.")


def _validate_global_evidence(
    schedule: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, AttemptStatus],
    summaries: Mapping[str, Mapping[str, Any]],
    terminals: Mapping[str, Any],
    traces: Mapping[tuple[str, str], tuple[Mapping[str, object], ...]],
) -> None:
    if set(statuses) - set(schedule) or set(summaries) - set(schedule):
        raise ValueError("Observed evidence contains IDs outside the frozen schedule.")
    completed = {
        attempt_id
        for attempt_id, status in statuses.items()
        if status is AttemptStatus.COMPLETED
    }
    if completed != set(summaries) or completed != set(terminals):
        raise ValueError("Completed ledger, summary, and terminal IDs do not match.")
    if set(traces) != {(CAMPAIGN_ID, attempt_id) for attempt_id in completed}:
        raise ValueError("Trace attempt identities do not match completed evidence.")
    for attempt_id, summary in summaries.items():
        scheduled = schedule[attempt_id]
        if any(summary.get(key) != value for key, value in scheduled.items()):
            raise ValueError(f"Summary identity drifted for {attempt_id}.")
        expected = traces[(CAMPAIGN_ID, attempt_id)]
        terminal = terminals[attempt_id]
        terminal_refs = tuple(reference.to_dict() for reference in terminal.trace_references)
        if not expected or tuple(summary.get("scientific_trace_references", ())) != expected:
            raise ValueError(f"Summary trace references drifted for {attempt_id}.")
        if terminal_refs != expected:
            raise ValueError(f"Terminal trace references drifted for {attempt_id}.")


def _validate_locks(
    repo_root: Path,
    manifest: Any,
    claim: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
) -> None:
    snapshot = _object(claim.get("runtime_snapshot"), "runtime snapshot")
    transport = _object(claim["manifest"].get("transport"), "registered transport")
    errors: set[str] = set()
    lock_root = repo_root / manifest.outputs.root / manifest.outputs.s1 / "locks"
    for condition in CONDITIONS:
        scheduled = [row for row in schedule if row.get("condition_id") == condition]
        observed = [row for row in episodes if row.get("condition_id") == condition]
        lock_dir = lock_root / MODEL_SLOT / condition
        binding = load_verified_runtime_lock_binding(
            runtime_lock_path=lock_dir / "RUNTIME_PROTOCOL_LOCK.json",
            authorization_path=lock_dir / "PROTOCOL_FROZEN.json",
        )
        _validate_binding(
            binding,
            scheduled[0],
            observed,
            transport,
            snapshot,
            errors,
        )
    if errors:
        raise ValueError("; ".join(sorted(errors)))


def _unique_index(
    rows: Sequence[Mapping[str, Any]], key: str, name: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = _text(row, key)
        if value in output:
            raise ValueError(f"{name} contains duplicate {key}.")
        output[value] = row
    return output


def _unique_terminal_index(rows: Sequence[Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for row in rows:
        attempt_id = str(row.episode_attempt_id)
        if attempt_id in output:
            raise ValueError("Attempt ledger contains duplicate terminal IDs.")
        output[attempt_id] = row
    return output


def _rows(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"{name} must be a list of objects.")
    return tuple(value)


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object.")
    return value


def _text(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{name} must be non-empty text.")
    return item


__all__ = [
    "CAMPAIGN_ID",
    "CLAIM_ELIGIBLE",
    "EXPECTED_ROWS",
    "SCOPE",
    "QwenStage1Tables",
    "build_qwen_stage1_tables",
    "publish_qwen_stage1_bundle",
    "run_qwen_stage1_analysis",
]
