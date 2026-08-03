"""Canonical append-only episode summary evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._append_intent_io import (
    append_intent_path_for,
    durable_append_with_intent,
)
from ._campaign_attempt_serialization import (
    canonical_bytes,
    hash_payload,
    reject_json_constant,
)
from .campaign_attempts import _is_sha256, _require_text


SUMMARY_SCHEMA_VERSION = "iclr2027.episode_summary.v1"
_METADATA_FIELDS = {
    "schema_version",
    "sequence",
    "previous_record_sha256",
    "summary_provenance_sha256",
    "record_sha256",
}


def campaign_provenance_sha256(
    rows: Sequence[Any],
    runtime_snapshot_sha256: str,
) -> str:
    return hash_payload(
        {
            "schema_version": "iclr2027.campaign_provenance.v1",
            "runtime_snapshot_sha256": "sha256:" + runtime_snapshot_sha256,
            "scheduled_denominator": [row.to_payload() for row in rows],
        }
    )


def append_summary_record(path: Path, summary: Mapping[str, Any]) -> None:
    payload = dict(summary)
    if _METADATA_FIELDS.intersection(payload):
        raise ValueError("Episode summary contains reserved evidence fields.")
    attempt_id = _require_text(
        "episode_attempt_id",
        payload.get("episode_attempt_id"),
    )
    campaign_hash = payload.get("campaign_provenance_sha256")
    if not _is_sha256(campaign_hash):
        raise ValueError("campaign_provenance_sha256 must be a full sha256 digest.")
    existing = load_summary_records(
        path,
        expected_campaign_provenance_sha256=campaign_hash,
    )
    record: dict[str, Any] = {
        **payload,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "sequence": len(existing) + 1,
        "previous_record_sha256": summary_root_sha256(existing),
        "summary_provenance_sha256": hash_payload(payload),
    }
    record["record_sha256"] = hash_payload(record)
    line = canonical_bytes(record) + b"\n"
    offset = path.stat().st_size if path.exists() else 0
    durable_append_with_intent(
        path,
        line,
        artifact_kind="episode_summaries",
        episode_attempt_id=attempt_id,
        expected_offset=offset,
        record_sha256=record["record_sha256"],
    )


def load_summary_records(
    path: Path,
    *,
    expected_campaign_provenance_sha256: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    if append_intent_path_for(path).exists():
        raise ValueError("Episode summary append outcome is ambiguous.")
    if not path.exists():
        return ()
    records: list[Mapping[str, Any]] = []
    previous: str | None = None
    with path.open("rb") as handle:
        for sequence, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                raise ValueError("Episode summary has a truncated tail.")
            encoded = raw_line[:-1]
            payload = json.loads(
                encoded.decode("utf-8"),
                parse_constant=reject_json_constant,
            )
            if not isinstance(payload, dict):
                raise ValueError("Episode summary line must be an object.")
            if encoded != canonical_bytes(payload):
                raise ValueError("Episode summary line is not canonical JSON.")
            _validate_record(
                payload,
                sequence,
                previous,
                expected_campaign_provenance_sha256,
            )
            records.append(payload)
            previous = payload["record_sha256"]
    return tuple(records)


def summary_root_sha256(records: Sequence[Mapping[str, Any]]) -> str | None:
    if not records:
        return None
    root = records[-1].get("record_sha256")
    if not _is_sha256(root):
        raise ValueError("Episode summary campaign root hash is invalid.")
    return root


def _validate_record(
    payload: dict[str, Any],
    sequence: int,
    previous: str | None,
    expected_campaign_provenance_sha256: str | None,
) -> None:
    if payload.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("Episode summary schema version drifted.")
    if payload.get("sequence") != sequence:
        raise ValueError("Episode summary sequence is not contiguous.")
    if payload.get("previous_record_sha256") != previous:
        raise ValueError("Episode summary campaign root chain is broken.")
    campaign_hash = payload.get("campaign_provenance_sha256")
    if not _is_sha256(campaign_hash):
        raise ValueError("Episode summary campaign provenance hash is invalid.")
    if (
        expected_campaign_provenance_sha256 is not None
        and campaign_hash != expected_campaign_provenance_sha256
    ):
        raise ValueError("Episode summary campaign provenance drifted.")
    summary_payload = {
        key: value for key, value in payload.items() if key not in _METADATA_FIELDS
    }
    if payload.get("summary_provenance_sha256") != hash_payload(summary_payload):
        raise ValueError("Episode summary provenance integrity hash is invalid.")
    unhashed = dict(payload)
    claimed = unhashed.pop("record_sha256", None)
    if not _is_sha256(claimed) or claimed != hash_payload(unhashed):
        raise ValueError("Episode summary record integrity hash is invalid.")


__all__ = [
    "append_summary_record",
    "campaign_provenance_sha256",
    "load_summary_records",
    "summary_root_sha256",
]
