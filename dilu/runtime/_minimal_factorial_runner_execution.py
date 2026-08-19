"""Campaign-scoped execution and completion gates."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .campaign_attempts import AttemptStatus
from .harness_config import RetryPolicy
from ._minimal_factorial_runner_summaries import (
    campaign_provenance_sha256 as _campaign_provenance_sha256,
    load_summary_records,
    summary_root_sha256,
)
from ._minimal_factorial_runner_publication import (
    build_completion_publisher,
    reconcile_published_summaries,
)
from .ollama_scientific_client import OllamaScientificClient
from .ollama_transport import inspect_ollama_model_identity
from .scientific_transport_types import ScientificTransportCapabilities


@dataclass(frozen=True)
class RunSummary:
    stage: str
    output_root: Path
    scheduled: int
    completed: int
    blocked: int
    failed: int
    ambiguous: int
    resumable: int
    pending: int
    promotion_allowed: bool = False
    errors: tuple[str, ...] = ()
    campaign_provenance_sha256: str | None = None
    summary_root_sha256: str | None = None
    selected_this_invocation: int = 0


def build_model_clients(
    capabilities: Mapping[str, ScientificTransportCapabilities],
    retry_policy: RetryPolicy,
) -> dict[str, OllamaScientificClient]:
    return {
        slot: OllamaScientificClient(
            capabilities=capability,
            retry_policy=retry_policy,
            identity_inspector=inspect_ollama_model_identity,
        )
        for slot, capability in capabilities.items()
    }


def execute_campaign(
    prepared: Any,
    *,
    scheduled_rows: Sequence[Any],
    denominator_rows: Sequence[Any],
    resume: bool,
    stage: str,
    ledger_type: Any,
    trace_type: Any,
    client_builder: Callable[..., Mapping[str, Any]],
    episode_runner: Callable[..., Mapping[str, Any]],
    pending_selector: Callable[..., tuple[Any, ...]],
    summary_appender: Callable[..., None],
    failure_recorder: Callable[..., None],
    completion_checker: Callable[..., tuple[str, ...]],
    max_episodes: int | None = None,
) -> RunSummary:
    _validate_max_episodes(max_episodes)
    output_root = prepared.output_root
    ledger_path = output_root / "campaign_attempts.jsonl"
    trace_path = output_root / "traces" / "decision_traces.jsonl"
    summaries_path = output_root / "episodes.jsonl"
    campaign_provenance = _campaign_provenance_sha256(
        getattr(prepared, "schedule", denominator_rows),
        prepared.snapshot.sha256,
    )
    summaries = _load_summaries(
        summaries_path,
        expected_campaign_provenance_sha256=campaign_provenance,
    )
    artifact_resume = resume or ledger_path.exists()
    ledger = ledger_type(
        ledger_path,
        campaign_id=_campaign_id(denominator_rows),
        resume=artifact_resume,
    )
    trace_writer = trace_type(
        trace_path,
        artifact_root=output_root,
        resume=resume or trace_path.exists(),
    )
    statuses = ledger.attempt_statuses()
    reconcile_published_summaries(
        rows=denominator_rows,
        summaries=summaries,
        runtime_snapshot_sha256=prepared.snapshot.sha256,
        campaign_provenance_sha256=campaign_provenance,
        statuses=statuses,
        ledger=ledger,
        trace_writer=trace_writer,
    )
    ledger.validate_trace_evidence(trace_writer)
    statuses = ledger.attempt_statuses()
    pending = pending_selector(scheduled_rows, statuses, resume=artifact_resume)
    pending = _ledger_approved_rows(pending, statuses, ledger)
    selected = pending if max_episodes is None else pending[:max_episodes]
    clients: Mapping[str, Any] = {}
    if selected:
        retry_policy = selected[0].condition.retry_policy
        clients = client_builder(prepared.capabilities, retry_policy)

    for row in selected:
        try:
            client = clients[row.model_slot]
            capability = prepared.capabilities[row.model_slot]
            if (
                capability.model_tag != row.model_tag
                or capability.model_digest != row.model_digest
            ):
                raise ValueError("Scheduled model binding drifted.")
            with tempfile.TemporaryDirectory(
                prefix=f"dilu-{row.episode_attempt_id[:16]}-"
            ) as temporary:
                episode_runner(
                    prepared,
                    row,
                    ledger=ledger,
                    trace_writer=trace_writer,
                    client=client,
                    episode_temp_dir=Path(temporary),
                    completion_publisher=build_completion_publisher(
                        row=row,
                        summaries_path=summaries_path,
                        runtime_snapshot_sha256=prepared.snapshot.sha256,
                        campaign_provenance_sha256=campaign_provenance,
                        ledger=ledger,
                        summary_appender=summary_appender,
                    ),
                )
            terminal = ledger.attempt_status(row.episode_attempt_id)
            if terminal is not AttemptStatus.COMPLETED:
                raise RuntimeError("Episode returned without completed evidence.")
        except Exception as exc:
            failure_recorder(ledger, row, exc)

    ledger.validate_trace_evidence(trace_writer)
    statuses = ledger.attempt_statuses()
    summary_errors: tuple[str, ...] = ()
    try:
        summaries = _load_summaries(
            summaries_path,
            expected_campaign_provenance_sha256=campaign_provenance,
        )
    except ValueError as exc:
        summaries = ()
        summary_errors = (f"episode summary evidence invalid: {exc}",)
    errors = summary_errors + completion_checker(
        denominator_rows,
        summaries,
        statuses,
    )
    counts = _counts(denominator_rows, statuses)
    resumable, interrupted = _started_resume_counts(
        denominator_rows,
        statuses,
        ledger,
    )
    pending_count = sum(
        statuses.get(row.episode_attempt_id) is None for row in scheduled_rows
    )
    return RunSummary(
        stage=stage,
        output_root=output_root,
        scheduled=len(scheduled_rows),
        completed=counts[AttemptStatus.COMPLETED],
        blocked=counts[AttemptStatus.BLOCKED],
        failed=counts[AttemptStatus.FAILED],
        ambiguous=counts[AttemptStatus.WRITE_AMBIGUOUS] + interrupted,
        resumable=resumable,
        pending=pending_count,
        promotion_allowed=not errors,
        errors=errors,
        campaign_provenance_sha256=campaign_provenance,
        summary_root_sha256=_summary_root_sha256(summaries),
        selected_this_invocation=len(selected),
    )


def _validate_max_episodes(max_episodes: int | None) -> None:
    if max_episodes is None:
        return
    if isinstance(max_episodes, bool) or not isinstance(max_episodes, int):
        raise ValueError("max_episodes must be a positive integer or None.")
    if max_episodes < 1:
        raise ValueError("max_episodes must be a positive integer or None.")


def _campaign_id(rows: Sequence[Any]) -> str:
    campaigns = {row.campaign_id for row in rows}
    if len(campaigns) != 1:
        raise ValueError("Campaign denominator must have one campaign ID.")
    return campaigns.pop()


def _ledger_approved_rows(
    rows: Sequence[Any],
    statuses: Mapping[str, AttemptStatus],
    ledger: Any,
) -> tuple[Any, ...]:
    return tuple(
        row
        for row in rows
        if statuses.get(row.episode_attempt_id) is not AttemptStatus.STARTED
        or ledger.can_resume(row.episode_attempt_id)
    )


def _started_resume_counts(
    rows: Sequence[Any],
    statuses: Mapping[str, AttemptStatus],
    ledger: Any,
) -> tuple[int, int]:
    started = tuple(
        row
        for row in rows
        if statuses.get(row.episode_attempt_id) is AttemptStatus.STARTED
    )
    resumable = sum(ledger.can_resume(row.episode_attempt_id) for row in started)
    return resumable, len(started) - resumable


def _counts(
    rows: Sequence[Any],
    statuses: Mapping[str, AttemptStatus],
) -> dict[AttemptStatus, int]:
    return {
        status: sum(statuses.get(row.episode_attempt_id) is status for row in rows)
        for status in (
            AttemptStatus.COMPLETED,
            AttemptStatus.BLOCKED,
            AttemptStatus.FAILED,
            AttemptStatus.WRITE_AMBIGUOUS,
        )
    }


def _load_summaries(
    path: Path,
    *,
    expected_campaign_provenance_sha256: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    return load_summary_records(
        path,
        expected_campaign_provenance_sha256=(expected_campaign_provenance_sha256),
    )


def _summary_root_sha256(
    summaries: Sequence[Mapping[str, Any]],
) -> str | None:
    return summary_root_sha256(summaries)


__all__ = ["RunSummary", "build_model_clients", "execute_campaign"]
