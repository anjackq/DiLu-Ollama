"""Read-only status reporting for frozen minimal-factorial campaigns."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

from ._campaign_attempt_state import read_validated_attempt_snapshot
from ._minimal_factorial_runner_summaries import (
    campaign_provenance_sha256,
    load_summary_records,
)
from ._scientific_trace_store import (
    ScientificTraceWriteError,
    read_validated_trace_snapshot,
)
from .campaign_attempts import (
    AttemptStatus,
    ScientificAttemptRecord,
    ScientificAttemptWriteError,
)

_COUNT_NAMES = (
    "scheduled",
    "completed",
    "blocked",
    "failed",
    "ambiguous",
    "resumable",
    "pending",
)


def summarize_status(
    rows: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, AttemptStatus],
    *,
    resumable_attempt_ids: AbstractSet[str] | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {name: 0 for name in _COUNT_NAMES}
    )
    for row in rows:
        key = (
            _text(row, "stage"),
            _text(row, "model_slot"),
            _text(row, "condition_id"),
        )
        counts = grouped[key]
        counts["scheduled"] += 1
        attempt_id = _text(row, "episode_attempt_id")
        status = statuses.get(attempt_id)
        if status is AttemptStatus.COMPLETED:
            counts["completed"] += 1
        elif status is AttemptStatus.BLOCKED:
            counts["blocked"] += 1
        elif status is AttemptStatus.FAILED:
            counts["failed"] += 1
        elif status is AttemptStatus.WRITE_AMBIGUOUS:
            counts["ambiguous"] += 1
        elif status is AttemptStatus.STARTED:
            if resumable_attempt_ids is None or attempt_id in resumable_attempt_ids:
                counts["resumable"] += 1
            else:
                counts["ambiguous"] += 1
        else:
            counts["pending"] += 1
    groups = [
        {
            "stage": key[0],
            "model_slot": key[1],
            "condition_id": key[2],
            **grouped[key],
        }
        for key in sorted(grouped)
    ]
    totals = {name: sum(int(group[name]) for group in groups) for name in _COUNT_NAMES}
    return {"groups": groups, "totals": totals}


def campaign_status(prepared_campaigns: Sequence[Any]) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    statuses: dict[str, AttemptStatus] = {}
    resumable: set[str] = set()
    artifact_errors: list[str] = []
    claim_promotion_allowed = False
    for prepared in prepared_campaigns:
        scheduled = tuple(prepared.schedule)
        campaign_ids = {row.campaign_id for row in scheduled}
        if len(campaign_ids) != 1:
            raise ValueError("Frozen status schedule must have one campaign ID.")
        campaign_id = campaign_ids.pop()
        rows.extend(row.to_payload() for row in scheduled)
        try:
            snapshot = _read_attempt_snapshot(
                prepared.output_root / "campaign_attempts.jsonl",
                campaign_id=campaign_id,
            )
            campaign_statuses = dict(snapshot.statuses)
            campaign_resumable = set(snapshot.resumable_attempt_ids)
            terminal_records = tuple(snapshot.terminal_records)
        except (OSError, ValueError) as exc:
            campaign_statuses = {}
            campaign_resumable = set()
            terminal_records = ()
            artifact_errors.append(
                f"{campaign_id}: attempt ledger status invalid: {exc}"
            )
        overlap = set(statuses).intersection(campaign_statuses)
        if overlap:
            artifact_errors.append(
                f"{campaign_id}: campaign status contains duplicate attempt IDs."
            )
        statuses.update(campaign_statuses)
        resumable.update(campaign_resumable)
        campaign_errors, promotion_allowed = _validate_campaign_artifacts(
            prepared,
            scheduled,
            campaign_id,
            campaign_statuses,
            terminal_records,
        )
        artifact_errors.extend(campaign_errors)
        claim_promotion_allowed = claim_promotion_allowed or promotion_allowed
    report = summarize_status(
        rows,
        statuses,
        resumable_attempt_ids=resumable,
    )
    report["artifact_validation"] = {
        "valid": not artifact_errors,
        "errors": tuple(artifact_errors),
        "claim_promotion_allowed": (
            not artifact_errors and claim_promotion_allowed
        ),
    }
    return report


def _validate_campaign_artifacts(
    prepared: Any,
    scheduled: Sequence[Any],
    campaign_id: str,
    statuses: Mapping[str, AttemptStatus],
    terminal_records: Sequence[ScientificAttemptRecord],
) -> tuple[tuple[str, ...], bool]:
    errors: list[str] = []
    summaries_path = prepared.output_root / "episodes.jsonl"
    expected_provenance = None
    snapshot = getattr(prepared, "snapshot", None)
    snapshot_sha256 = getattr(snapshot, "sha256", None)
    if isinstance(snapshot_sha256, str):
        expected_provenance = campaign_provenance_sha256(
            getattr(prepared, "schedule", scheduled),
            snapshot_sha256,
        )
    try:
        summaries = load_summary_records(
            summaries_path,
            expected_campaign_provenance_sha256=expected_provenance,
        )
    except (OSError, ValueError) as exc:
        summaries = ()
        errors.append(f"{campaign_id}: episode summaries invalid: {exc}")
    try:
        trace_references = _read_validated_trace_references(
            prepared.output_root / "traces" / "decision_traces.jsonl",
            artifact_root=prepared.output_root,
        )
    except (OSError, ValueError, ScientificTraceWriteError) as exc:
        trace_references = {}
        errors.append(f"{campaign_id}: scientific trace evidence invalid: {exc}")
    try:
        _validate_ledger_trace_evidence(
            terminal_records,
            trace_references,
        )
    except (
        OSError,
        ValueError,
        ScientificAttemptWriteError,
        ScientificTraceWriteError,
    ) as exc:
        errors.append(f"{campaign_id}: ledger trace evidence invalid: {exc}")

    scheduled_ids = [row.episode_attempt_id for row in scheduled]
    scheduled_set = set(scheduled_ids)
    unexpected_attempt_ids = set(statuses) - scheduled_set
    if unexpected_attempt_ids:
        errors.append(
            f"{campaign_id}: unexpected attempt ledger IDs are outside the "
            "frozen schedule."
        )
    orphan_trace_keys = tuple(
        key
        for key in trace_references
        if (
            key[0] != campaign_id
            or key[1] not in scheduled_set
            or key[1] not in statuses
        )
    )
    if orphan_trace_keys:
        errors.append(
            f"{campaign_id}: orphan trace episode identities are not present in "
            "both the frozen schedule and attempt ledger."
        )

    summary_ids = [summary.get("episode_attempt_id") for summary in summaries]
    if any(not isinstance(attempt_id, str) or not attempt_id for attempt_id in summary_ids):
        errors.append(f"{campaign_id}: episode summary IDs must be non-empty text.")
    if len(summary_ids) != len(set(summary_ids)):
        errors.append(f"{campaign_id}: duplicate episode summary IDs.")
    completed_ids = {
        attempt_id
        for attempt_id, status in statuses.items()
        if status is AttemptStatus.COMPLETED
    }
    if completed_ids != set(summary_ids):
        errors.append(
            f"{campaign_id}: completed attempt IDs do not exactly match summary IDs."
        )
    for summary in summaries:
        attempt_id = summary.get("episode_attempt_id")
        if not isinstance(attempt_id, str):
            continue
        expected = trace_references.get((campaign_id, attempt_id), ())
        if not expected or summary.get("scientific_trace_references") != list(expected):
            errors.append(
                f"{campaign_id}: summary ordered trace references do not match "
                f"validated trace evidence for {attempt_id}."
            )

    promotion_allowed = (
        _is_claim_schedule(scheduled)
        and len(scheduled_ids) == 840
        and len(scheduled_set) == 840
        and len(summaries) == 840
        and len(set(summary_ids)) == 840
        and completed_ids == scheduled_set == set(summary_ids)
        and set(statuses) == scheduled_set
        and all(status is AttemptStatus.COMPLETED for status in statuses.values())
        and all(
            statuses.get(attempt_id) is AttemptStatus.COMPLETED
            for attempt_id in scheduled_ids
        )
        and not errors
    )
    return tuple(errors), promotion_allowed


def _validate_ledger_trace_evidence(
    terminal_records: Sequence[ScientificAttemptRecord],
    trace_references: Mapping[
        tuple[str, str],
        tuple[Mapping[str, object], ...],
    ],
) -> None:
    for record in terminal_records:
        episode = (record.campaign_id, record.episode_attempt_id)
        expected = tuple(reference.to_dict() for reference in record.trace_references)
        if expected != trace_references.get(episode, ()):
            raise ScientificAttemptWriteError(
                "Attempt ledger does not match ordered trace evidence."
            )


def _read_validated_trace_references(
    path: Path,
    *,
    artifact_root: Path,
) -> Mapping[tuple[str, str], tuple[Mapping[str, object], ...]]:
    snapshot = read_validated_trace_snapshot(
        path,
        artifact_root=artifact_root,
    )
    return {
        key: tuple(reference.to_dict() for reference in references)
        for key, references in snapshot.references_by_attempt.items()
    }


def _is_claim_schedule(rows: Sequence[Any]) -> bool:
    return bool(rows) and all(
        row.stage in {"stage1", "stage2_additional"} for row in rows
    )


def _read_attempt_snapshot(path: Any, *, campaign_id: str) -> Any:
    try:
        return read_validated_attempt_snapshot(path, campaign_id=campaign_id)
    except ScientificAttemptWriteError as exc:
        raise ValueError("Attempt ledger failed status validation.") from exc


def _read_attempt_statuses(path: Any, *, campaign_id: str) -> dict[str, AttemptStatus]:
    return dict(_read_attempt_snapshot(path, campaign_id=campaign_id).statuses)


def _text(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{name} must be non-empty text.")
    return item


__all__ = ["campaign_status", "summarize_status"]
