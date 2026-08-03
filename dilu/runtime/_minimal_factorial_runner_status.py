"""Read-only status reporting for frozen minimal-factorial campaigns."""

from __future__ import annotations

from collections import defaultdict
from typing import AbstractSet, Any, Mapping, Sequence

from ._campaign_attempt_state import read_validated_attempt_snapshot
from .campaign_attempts import AttemptStatus, ScientificAttemptWriteError

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
    for prepared in prepared_campaigns:
        scheduled = tuple(prepared.schedule)
        campaign_ids = {row.campaign_id for row in scheduled}
        if len(campaign_ids) != 1:
            raise ValueError("Frozen status schedule must have one campaign ID.")
        rows.extend(row.to_payload() for row in scheduled)
        snapshot = _read_attempt_snapshot(
            prepared.output_root / "campaign_attempts.jsonl",
            campaign_id=campaign_ids.pop(),
        )
        overlap = set(statuses).intersection(snapshot.statuses)
        if overlap:
            raise ValueError("Campaign status contains duplicate attempt IDs.")
        statuses.update(snapshot.statuses)
        resumable.update(snapshot.resumable_attempt_ids)
    return summarize_status(
        rows,
        statuses,
        resumable_attempt_ids=resumable,
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
