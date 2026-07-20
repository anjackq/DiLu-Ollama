from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ._scientific_trace_store import TraceReference

if TYPE_CHECKING:
    from ._campaign_attempt_store import ScientificAttemptLedger


ATTEMPT_SCHEMA_VERSION = "iclr2027.campaign_attempt.v1"
TRACE_COMMIT_AMBIGUOUS = "trace_commit_ambiguous"
_SHA256_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z")


class AttemptStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    WRITE_AMBIGUOUS = "write_ambiguous"


@dataclass(frozen=True)
class ScientificAttemptRecord:
    campaign_id: str
    episode_attempt_id: str
    status: AttemptStatus
    decision_count: int
    trace_references: tuple[TraceReference, ...] = ()
    failure_class: str | None = None
    failure_message: str | None = None
    trace_absence_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text("campaign_id", self.campaign_id)
        _require_text("episode_attempt_id", self.episode_attempt_id)
        if not isinstance(self.status, AttemptStatus):
            raise ValueError("status must be an AttemptStatus.")
        _require_nonnegative_int("decision_count", self.decision_count)
        if not isinstance(self.trace_references, tuple):
            raise ValueError("trace_references must be an immutable tuple.")
        for reference in self.trace_references:
            _validate_trace_reference(reference)
        for name in ("failure_class", "failure_message", "trace_absence_reason"):
            value = getattr(self, name)
            if value is not None:
                _require_text(name, value)
        self._validate_evidence()

    def _validate_evidence(self) -> None:
        reference_count = len(self.trace_references)
        failure_values = (
            self.failure_class,
            self.failure_message,
            self.trace_absence_reason,
        )
        if self.status is AttemptStatus.STARTED:
            if self.decision_count or reference_count or any(failure_values):
                raise ValueError("Started attempts cannot contain terminal evidence.")
            return
        if self.status is AttemptStatus.COMPLETED:
            if self.decision_count < 1 or reference_count != self.decision_count:
                raise ValueError("Completed attempts require all committed traces.")
            if any(failure_values):
                raise ValueError("Completed attempts cannot contain failure evidence.")
            return
        if self.status is AttemptStatus.WRITE_AMBIGUOUS:
            valid = (
                self.failure_class == TRACE_COMMIT_AMBIGUOUS
                and self.failure_message is not None
                and self.trace_absence_reason == TRACE_COMMIT_AMBIGUOUS
                and reference_count == self.decision_count
            )
            if not valid:
                raise ValueError(
                    "Write ambiguity requires trace_commit_ambiguous evidence."
                )
            return
        if self.failure_class is None or self.failure_message is None:
            raise ValueError("Blocked and failed attempts require failure evidence.")
        if self.decision_count == 0:
            if reference_count or self.trace_absence_reason is None:
                raise ValueError(
                    "Zero-decision failure requires a trace absence reason."
                )
        elif (
            reference_count != self.decision_count
            or self.trace_absence_reason is not None
        ):
            raise ValueError("Decided aborts require all committed traces.")

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "episode_attempt_id": self.episode_attempt_id,
            "status": self.status.value,
            "decision_count": self.decision_count,
            "trace_references": [item.to_dict() for item in self.trace_references],
            "failure_class": self.failure_class,
            "failure_message": self.failure_message,
            "trace_absence_reason": self.trace_absence_reason,
        }


@dataclass(frozen=True)
class AttemptReference:
    line_number: int
    record_sha256: str
    previous_record_sha256: str | None
    schema_version: str = ATTEMPT_SCHEMA_VERSION


class ScientificAttemptWriteError(RuntimeError):
    pass


def _validate_trace_reference(reference: object) -> None:
    if type(reference) is not TraceReference:
        raise ValueError("trace_references must contain TraceReference values.")
    _require_text("relative_path", reference.relative_path)
    relative = PurePosixPath(reference.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Trace reference path must be contained and relative.")
    _require_positive_int("line_number", reference.line_number)
    if not _is_sha256(reference.record_sha256):
        raise ValueError("record_sha256 must be a full sha256 digest.")
    if not _is_sha256(reference.schema_sha256):
        raise ValueError("schema_sha256 must be a full sha256 digest.")
    _require_text("schema_version", reference.schema_version)


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text.")
    return value


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.fullmatch(value))


def __getattr__(name: str) -> object:
    if name == "ScientificAttemptLedger":
        from ._campaign_attempt_store import ScientificAttemptLedger

        return ScientificAttemptLedger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "TRACE_COMMIT_AMBIGUOUS",
    "AttemptReference",
    "AttemptStatus",
    "ScientificAttemptLedger",
    "ScientificAttemptRecord",
    "ScientificAttemptWriteError",
]
