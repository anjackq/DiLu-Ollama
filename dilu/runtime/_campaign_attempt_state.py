"""Shared in-memory state and read-only validation for attempt ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._append_intent_io import append_intent_path_for
from ._campaign_attempt_io import lock_path_for, poison_path_for
from .campaign_attempts import (
    AttemptStatus,
    ScientificAttemptWriteError,
    _require_text,
)


@dataclass(frozen=True)
class AttemptLedgerSnapshot:
    statuses: Mapping[str, AttemptStatus]
    resumable_attempt_ids: frozenset[str]


def initialize_attempt_ledger_state(
    ledger: Any,
    path: Path,
    campaign_id: str,
) -> None:
    _require_text("campaign_id", campaign_id)
    ledger.path = Path(path).resolve()
    ledger._lock_path = lock_path_for(ledger.path)
    ledger._poison_path = poison_path_for(ledger.path)
    ledger._pending_path = append_intent_path_for(ledger.path)
    ledger.campaign_id = campaign_id
    ledger._attempt_status = {}
    ledger._request_owner = {}
    ledger._line_count = 0
    ledger._byte_offset = 0
    ledger._last_hash = None
    ledger._poisoned = False
    ledger._poisoned_attempts = set()
    ledger._terminal_records = []


def attempt_is_resumable(ledger: Any, episode_attempt_id: str) -> bool:
    return (
        not ledger._poisoned
        and episode_attempt_id not in ledger._poisoned_attempts
        and ledger._attempt_status.get(episode_attempt_id) is AttemptStatus.STARTED
        and episode_attempt_id not in ledger._request_owner.values()
    )


def read_validated_attempt_snapshot(
    path: Path,
    *,
    campaign_id: str,
) -> AttemptLedgerSnapshot:
    from ._campaign_attempt_store import ScientificAttemptLedger

    ledger = object.__new__(ScientificAttemptLedger)
    initialize_attempt_ledger_state(ledger, path, campaign_id)
    _require_quiescent(ledger)
    before = _file_state(ledger.path)
    if ledger.path.exists() and ledger.path.stat().st_size:
        ledger._scan_existing()
    after = _file_state(ledger.path)
    _require_quiescent(ledger)
    final = _file_state(ledger.path)
    if before != after or after != final:
        raise ScientificAttemptWriteError(
            "Attempt ledger changed during read-only status validation."
        )
    _require_quiescent(ledger)
    statuses = dict(ledger._attempt_status)
    resumable = frozenset(
        attempt_id
        for attempt_id in statuses
        if attempt_is_resumable(ledger, attempt_id)
    )
    return AttemptLedgerSnapshot(statuses, resumable)


def _require_quiescent(ledger: Any) -> None:
    if any(
        marker.exists()
        for marker in (
            ledger._lock_path,
            ledger._pending_path,
            ledger._poison_path,
        )
    ):
        raise ScientificAttemptWriteError(
            "Attempt ledger is busy or has ambiguous durable-append evidence."
        )


def _file_state(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


__all__ = [
    "AttemptLedgerSnapshot",
    "attempt_is_resumable",
    "initialize_attempt_ledger_state",
    "read_validated_attempt_snapshot",
]
