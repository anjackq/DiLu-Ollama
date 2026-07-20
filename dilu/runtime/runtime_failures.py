from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class RuntimeFailureClass(str, Enum):
    TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT = "transport_unavailable_before_accept"
    TRANSPORT_DRIFT = "transport_drift"
    GENERATION_TIMEOUT = "generation_timeout"
    MODEL_EMPTY_OUTPUT = "model_empty_output"
    SCHEMA_REJECTION = "schema_rejection"
    SYNTAX_INVALID = "syntax_invalid"
    ACTION_UNAVAILABLE = "action_unavailable"
    TRACE_WRITE_FAILURE = "trace_write_failure"
    SIMULATOR_FAILURE = "simulator_failure"


class ProtocolInvariantCode(str, Enum):
    ACTION_AVAILABILITY_UNRESOLVED = "action_availability_unresolved"
    ACTION_TOKEN_MAPPING_MISMATCH = "action_token_mapping_mismatch"
    FIXED_FALLBACK_UNAVAILABLE = "fixed_fallback_unavailable"
    RUNTIME_LOCK_MISMATCH = "runtime_lock_mismatch"
    TRACE_EVIDENCE_MISSING = "trace_evidence_missing"


@dataclass(frozen=True)
class RuntimeFailure:
    failure_class: RuntimeFailureClass
    message: str
    details: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        failure_class: RuntimeFailureClass,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> "RuntimeFailure":
        ordered_details = tuple(sorted((details or {}).items()))
        return cls(failure_class, str(message), ordered_details)


@dataclass(frozen=True)
class ProtocolInvariantViolation:
    invariant_code: ProtocolInvariantCode
    message: str
    details: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        invariant_code: ProtocolInvariantCode,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> "ProtocolInvariantViolation":
        ordered_details = tuple(sorted((details or {}).items()))
        return cls(invariant_code, str(message), ordered_details)


class RuntimeProtocolError(RuntimeError):
    def __init__(self, violation: ProtocolInvariantViolation) -> None:
        super().__init__(violation.message)
        self.violation = violation

    @property
    def invariant_code(self) -> ProtocolInvariantCode:
        return self.violation.invariant_code


__all__ = [
    "ProtocolInvariantCode",
    "ProtocolInvariantViolation",
    "RuntimeFailure",
    "RuntimeFailureClass",
    "RuntimeProtocolError",
]
