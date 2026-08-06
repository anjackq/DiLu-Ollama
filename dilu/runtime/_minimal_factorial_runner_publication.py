"""Summary-first completion publication and resume reconciliation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .campaign_attempts import AttemptStatus
from .scientific_trace import TraceReference


def build_completion_publisher(
    *,
    row: Any,
    summaries_path: Path,
    runtime_snapshot_sha256: str,
    campaign_provenance_sha256: str,
    ledger: Any,
    summary_appender: Callable[[Path, Mapping[str, Any], Any], None],
) -> Callable[[Mapping[str, Any], tuple[TraceReference, ...]], None]:
    def publish(
        result: Mapping[str, Any],
        references: tuple[TraceReference, ...],
    ) -> None:
        if not references:
            raise RuntimeError("Completed episode returned without trace evidence.")
        summary = {
            **dict(result),
            **row.to_payload(),
            "runtime_snapshot_sha256": "sha256:" + runtime_snapshot_sha256,
            "campaign_provenance_sha256": campaign_provenance_sha256,
            "scientific_trace_references": [
                reference.to_dict() for reference in references
            ],
        }
        summary_appender(summaries_path, summary, ledger)
        ledger.append_terminal(
            row.episode_attempt_id,
            status=AttemptStatus.COMPLETED,
            decision_count=len(references),
            trace_references=references,
        )

    return publish


def reconcile_published_summaries(
    *,
    rows: Sequence[Any],
    summaries: Sequence[Mapping[str, Any]],
    runtime_snapshot_sha256: str,
    campaign_provenance_sha256: str,
    statuses: Mapping[str, AttemptStatus],
    ledger: Any,
    trace_writer: Any,
) -> None:
    row_by_id = {row.episode_attempt_id: row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("Campaign contains duplicate scheduled episode IDs.")
    summary_by_id: dict[str, Mapping[str, Any]] = {}
    pending_completions: list[tuple[str, tuple[TraceReference, ...]]] = []
    for summary in summaries:
        attempt_id = summary.get("episode_attempt_id")
        if not isinstance(attempt_id, str) or attempt_id not in row_by_id:
            raise ValueError("Episode summary does not match the campaign schedule.")
        if attempt_id in summary_by_id:
            raise ValueError("Episode summary evidence contains duplicate attempts.")
        references = _validate_summary_binding(
            summary,
            row_by_id[attempt_id],
            runtime_snapshot_sha256,
            campaign_provenance_sha256,
            trace_writer,
        )
        status = statuses.get(attempt_id)
        if status is AttemptStatus.STARTED:
            pending_completions.append((attempt_id, references))
        elif status is not AttemptStatus.COMPLETED:
            value = status.value if isinstance(status, AttemptStatus) else "unknown"
            raise ValueError(
                f"Episode summary is paired with invalid attempt status: {value}."
            )
        summary_by_id[attempt_id] = summary
    for attempt_id, status in statuses.items():
        if status is AttemptStatus.COMPLETED and attempt_id not in summary_by_id:
            raise ValueError("Completed attempt has no durable episode summary.")
    for attempt_id, references in pending_completions:
        ledger.append_terminal(
            attempt_id,
            status=AttemptStatus.COMPLETED,
            decision_count=len(references),
            trace_references=references,
        )


def _validate_summary_binding(
    summary: Mapping[str, Any],
    row: Any,
    runtime_snapshot_sha256: str,
    campaign_provenance_sha256: str,
    trace_writer: Any,
) -> tuple[TraceReference, ...]:
    for key, expected in row.to_payload().items():
        if summary.get(key) != expected:
            raise ValueError("Episode summary does not match the campaign schedule.")
    if summary.get("campaign_provenance_sha256") != campaign_provenance_sha256:
        raise ValueError("Episode summary campaign provenance drifted.")
    if summary.get("runtime_snapshot_sha256") != "sha256:" + runtime_snapshot_sha256:
        raise ValueError("Episode summary runtime snapshot drifted.")
    references = trace_writer.references_for_attempt(
        row.campaign_id,
        row.episode_attempt_id,
    )
    expected_references = [reference.to_dict() for reference in references]
    if (
        not references
        or summary.get("scientific_trace_references") != expected_references
    ):
        raise ValueError(
            "Episode summary trace references do not match ordered evidence."
        )
    return references


__all__ = ["build_completion_publisher", "reconcile_published_summaries"]
