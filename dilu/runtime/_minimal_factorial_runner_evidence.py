"""Append-only evidence for post-terminal campaign publication failures."""

from __future__ import annotations

from typing import Any

from ._campaign_attempt_serialization import require_keys
from .campaign_attempts import AttemptReference, AttemptStatus, _require_text


SUMMARY_FAILURE_FIELDS = {
    "schema_version",
    "sequence",
    "previous_record_sha256",
    "record_sha256",
    "event_type",
    "campaign_id",
    "episode_attempt_id",
    "prior_status",
    "failure_class",
    "failure_message",
}


def append_summary_failure(
    ledger: Any,
    episode_attempt_id: str,
    *,
    failure_class: str,
    failure_message: str,
) -> AttemptReference:
    _require_text("episode_attempt_id", episode_attempt_id)
    _require_text("failure_class", failure_class)
    _require_text("failure_message", failure_message)
    with ledger._append_guard():
        if (
            ledger._attempt_status.get(episode_attempt_id)
            is not AttemptStatus.COMPLETED
        ):
            raise ValueError(
                "Summary publication failure requires completed terminal evidence."
            )
        reference = ledger._append_event(
            {
                "event_type": "summary_publication_failed",
                "campaign_id": ledger.campaign_id,
                "episode_attempt_id": episode_attempt_id,
                "prior_status": AttemptStatus.COMPLETED.value,
                "failure_class": failure_class,
                "failure_message": failure_message,
            },
            episode_attempt_id,
        )
        ledger._attempt_status[episode_attempt_id] = AttemptStatus.WRITE_AMBIGUOUS
        return reference


def replay_summary_failure(ledger: Any, payload: dict[str, Any]) -> None:
    require_keys(payload, SUMMARY_FAILURE_FIELDS)
    campaign_id = _require_text("campaign_id", payload.get("campaign_id"))
    attempt_id = _require_text(
        "episode_attempt_id",
        payload.get("episode_attempt_id"),
    )
    _require_text("failure_class", payload.get("failure_class"))
    _require_text("failure_message", payload.get("failure_message"))
    if campaign_id != ledger.campaign_id:
        raise ValueError("Summary failure belongs to another campaign.")
    if payload.get("prior_status") != AttemptStatus.COMPLETED.value:
        raise ValueError("Summary failure prior status is invalid.")
    if ledger._attempt_status.get(attempt_id) is not AttemptStatus.COMPLETED:
        raise ValueError("Summary failure has no completed terminal evidence.")
    ledger._attempt_status[attempt_id] = AttemptStatus.WRITE_AMBIGUOUS


__all__ = ["append_summary_failure", "replay_summary_failure"]
