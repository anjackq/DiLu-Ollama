from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

from ._scientific_trace_artifacts import ScientificSimulatorAbort, TraceReference
from ._scientific_trace_records import DecisionTraceRecord, TraceDisposition

if TYPE_CHECKING:
    from ._scientific_trace_store import ScientificTraceWriter


StepResult = TypeVar("StepResult")


def append_trace_before_step(
    writer: ScientificTraceWriter,
    record: DecisionTraceRecord,
    step_callable: Callable[[int], StepResult],
) -> tuple[TraceReference, StepResult]:
    if record.disposition is not TraceDisposition.READY_FOR_ENV_STEP:
        raise ValueError("Only ready traces may precede env.step.")
    if record.shield_stack is None:
        raise ValueError("Ready traces require an executed action.")
    reference = writer.append(record)
    try:
        step_result = step_callable(record.shield_stack.executed_action_id)
    except Exception as exc:
        raise ScientificSimulatorAbort(reference, exc) from exc
    return reference, step_result


__all__ = ["append_trace_before_step"]
