"""Completion evidence gate for matched non-LLM calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._campaign_attempt_serialization import hash_payload
from ._campaign_attempt_state import read_validated_attempt_snapshot
from ._campaign_attempt_store import ScientificAttemptLedger
from ._minimal_factorial_runner_summaries import load_summary_records
from ._scientific_trace_store import ScientificTraceWriter
from .campaign_attempts import AttemptStatus, ScientificAttemptWriteError


def require_complete_claim_campaign(
    claim_path: Path,
    claim: Mapping[str, Any],
) -> None:
    schedule = _rows(claim.get("schedule"))
    attempt_ids = tuple(_text(row, "episode_attempt_id") for row in schedule)
    if len(schedule) != 840 or len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("Calibration requires a unique 840-row claim denominator.")
    campaign_ids = {_text(row, "campaign_id") for row in schedule}
    if len(campaign_ids) != 1:
        raise ValueError("Claim denominator must contain one campaign ID.")
    campaign_id = campaign_ids.pop()
    snapshot_sha256 = _digest(claim.get("runtime_snapshot_sha256"))
    campaign_sha256 = hash_payload(
        {
            "schema_version": "iclr2027.campaign_provenance.v1",
            "runtime_snapshot_sha256": "sha256:" + snapshot_sha256,
            "scheduled_denominator": list(schedule),
        }
    )
    try:
        ledger = read_validated_attempt_snapshot(
            claim_path.parent / "campaign_attempts.jsonl",
            campaign_id=campaign_id,
        )
    except ScientificAttemptWriteError as exc:
        raise ValueError("Claim attempt ledger failed completion validation.") from exc
    expected = set(attempt_ids)
    if set(ledger.statuses) != expected or any(
        status is not AttemptStatus.COMPLETED for status in ledger.statuses.values()
    ):
        raise ValueError("Claim attempt ledger is not exactly 840/840 completed.")
    try:
        summaries = load_summary_records(
            claim_path.parent / "episode_summaries.jsonl",
            expected_campaign_provenance_sha256=campaign_sha256,
        )
    except ValueError as exc:
        raise ValueError(
            "Claim episode summaries failed completion validation."
        ) from exc
    summary_ids = tuple(_text(row, "episode_attempt_id") for row in summaries)
    if len(summary_ids) != len(expected) or set(summary_ids) != expected:
        raise ValueError("Claim episode summaries are not exactly 840/840 complete.")
    try:
        durable_ledger = ScientificAttemptLedger(
            claim_path.parent / "campaign_attempts.jsonl",
            campaign_id=campaign_id,
            resume=True,
        )
        trace_writer = ScientificTraceWriter(
            claim_path.parent / "traces" / "decision_traces.jsonl",
            artifact_root=claim_path.parent,
            resume=True,
        )
        durable_ledger.validate_trace_evidence(trace_writer)
    except Exception as exc:
        raise ValueError("Claim ledger-to-trace evidence validation failed.") from exc


def _rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError("Claim schedule must be a list of objects.")
    return tuple(value)


def _text(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item or item != item.strip():
        raise ValueError(f"{name} must be non-empty canonical text.")
    return item


def _digest(value: Any) -> str:
    text = str(value or "")
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError("Runtime snapshot digest is invalid.")
    return text


__all__ = ["require_complete_claim_campaign"]
