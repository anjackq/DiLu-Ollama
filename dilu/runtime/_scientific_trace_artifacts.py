from __future__ import annotations

from dataclasses import dataclass

from .runtime_failures import RuntimeFailureClass


class ScientificTraceWriteError(RuntimeError):
    failure_class = RuntimeFailureClass.TRACE_WRITE_FAILURE


class ScientificTraceValidationError(ScientificTraceWriteError):
    pass


class ScientificTraceCommitAmbiguousError(ScientificTraceWriteError):
    pass


@dataclass(frozen=True)
class TraceReference:
    relative_path: str
    line_number: int
    record_sha256: str
    schema_version: str
    schema_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "line_number": self.line_number,
            "record_sha256": self.record_sha256,
            "schema_version": self.schema_version,
            "schema_sha256": self.schema_sha256,
        }


class ScientificSimulatorAbort(RuntimeError):
    failure_class = RuntimeFailureClass.SIMULATOR_FAILURE

    def __init__(self, trace_reference: TraceReference, cause: Exception) -> None:
        if not isinstance(trace_reference, TraceReference):
            raise ValueError("trace_reference must be a TraceReference.")
        if not isinstance(cause, Exception):
            raise ValueError("cause must be an Exception.")
        super().__init__(f"Simulator step failed after trace commit: {cause}")
        self.trace_reference = trace_reference
        self.cause = cause

    def to_failure_record(self) -> dict[str, object]:
        return {
            "failure_class": self.failure_class.value,
            "message": str(self.cause),
            "error_type": type(self.cause).__name__,
            "trace_reference": self.trace_reference.to_dict(),
        }


__all__ = [
    "ScientificSimulatorAbort",
    "ScientificTraceCommitAmbiguousError",
    "ScientificTraceValidationError",
    "ScientificTraceWriteError",
    "TraceReference",
]
