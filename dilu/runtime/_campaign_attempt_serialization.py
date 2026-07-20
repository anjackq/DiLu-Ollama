from __future__ import annotations

import hashlib
import json
from typing import Any

from ._scientific_trace_store import TraceReference
from .campaign_attempts import AttemptStatus, ScientificAttemptRecord


BASE_FIELDS = {
    "schema_version",
    "sequence",
    "previous_record_sha256",
    "record_sha256",
    "event_type",
}
ATTEMPT_EVENT_FIELDS = BASE_FIELDS | {"attempt"}
REQUEST_EVENT_FIELDS = BASE_FIELDS | {
    "campaign_id",
    "request_id",
    "episode_attempt_id",
}
ATTEMPT_FIELDS = {
    "campaign_id",
    "episode_attempt_id",
    "status",
    "decision_count",
    "trace_references",
    "failure_class",
    "failure_message",
    "trace_absence_reason",
}
TRACE_FIELDS = {
    "relative_path",
    "line_number",
    "record_sha256",
    "schema_version",
    "schema_sha256",
}


def record_from_dict(value: object) -> ScientificAttemptRecord:
    if not isinstance(value, dict):
        raise ValueError("Attempt lifecycle payload must be an object.")
    require_keys(value, ATTEMPT_FIELDS)
    references = value["trace_references"]
    if not isinstance(references, list):
        raise ValueError("Stored trace_references must be an array.")
    return ScientificAttemptRecord(
        value["campaign_id"],
        value["episode_attempt_id"],
        AttemptStatus(value["status"]),
        value["decision_count"],
        tuple(trace_from_dict(item) for item in references),
        value["failure_class"],
        value["failure_message"],
        value["trace_absence_reason"],
    )


def trace_from_dict(value: object) -> TraceReference:
    if not isinstance(value, dict):
        raise ValueError("Stored trace reference must be an object.")
    require_keys(value, TRACE_FIELDS)
    return TraceReference(**value)


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def hash_payload(payload: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def require_keys(payload: dict[str, Any], expected: set[str]) -> None:
    if set(payload) != expected:
        raise ValueError("Campaign attempt payload fields drifted.")


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is prohibited: {value}.")


__all__ = [
    "ATTEMPT_EVENT_FIELDS",
    "REQUEST_EVENT_FIELDS",
    "canonical_bytes",
    "hash_payload",
    "record_from_dict",
    "reject_json_constant",
    "require_keys",
]
