from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .runtime_failures import RuntimeFailureClass
from .scientific_transport_records import BackendTiming, GenerationAttempt
from .scientific_transport_types import GenerationRequest


def failed_attempt(
    request: GenerationRequest,
    attempt_id: str,
    attempt_index: int,
    *,
    accepted_by_server: bool | None,
    status_code: int | None,
    started: float,
    clock: Callable[[], float],
    failure: RuntimeFailureClass,
    error: Exception,
    response_body: str | None = None,
    transport_error_body: str | None = None,
) -> GenerationAttempt:
    return GenerationAttempt(
        request_id=request.request_id,
        attempt_id=attempt_id,
        attempt_index=attempt_index,
        accepted_by_server=accepted_by_server,
        http_status=status_code,
        response_body=response_body,
        raw_response=None,
        contract_text=None,
        transport_error_body=transport_error_body,
        thinking_response="",
        stop_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        backend_timing=None,
        latency_ms=elapsed_ms(started, clock),
        error_class=failure,
        error_message=f"{type(error).__name__}: {error}",
    )


def empty_output_attempt(
    request: GenerationRequest,
    attempt_id: str,
    attempt_index: int,
    status_code: int | None,
    response_body: str,
    raw_response: str,
    thinking_response: str,
    stop_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    backend_timing: BackendTiming,
    started: float,
    clock: Callable[[], float],
) -> GenerationAttempt:
    return GenerationAttempt(
        request_id=request.request_id,
        attempt_id=attempt_id,
        attempt_index=attempt_index,
        accepted_by_server=True,
        http_status=status_code,
        response_body=response_body,
        raw_response=raw_response,
        contract_text=raw_response,
        transport_error_body=None,
        thinking_response=thinking_response,
        stop_reason=stop_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        backend_timing=backend_timing,
        latency_ms=elapsed_ms(started, clock),
        error_class=RuntimeFailureClass.MODEL_EMPTY_OUTPUT,
        error_message="model returned empty output",
    )


def schema_failure_attempt(
    request: GenerationRequest,
    attempt_id: str,
    attempt_index: int,
    status_code: int | None,
    response_body: str,
    raw_response: str,
    thinking_response: str,
    stop_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    backend_timing: BackendTiming,
    started: float,
    clock: Callable[[], float],
    error: Exception,
) -> GenerationAttempt:
    return GenerationAttempt(
        request_id=request.request_id,
        attempt_id=attempt_id,
        attempt_index=attempt_index,
        accepted_by_server=True,
        http_status=status_code,
        response_body=response_body,
        raw_response=raw_response,
        contract_text=None,
        transport_error_body=None,
        thinking_response=thinking_response,
        stop_reason=stop_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        backend_timing=backend_timing,
        latency_ms=elapsed_ms(started, clock),
        error_class=RuntimeFailureClass.SCHEMA_REJECTION,
        error_message=f"{type(error).__name__}: {error}",
    )


def payload_drift_attempt(
    request: GenerationRequest,
    attempt_id: str,
    attempt_index: int,
    status_code: int | None,
    started: float,
    clock: Callable[[], float],
    reason: str,
    *,
    response_body: str | None = None,
    raw_response: str | None = None,
    thinking_response: str = "",
    transport_error_body: str | None = None,
) -> GenerationAttempt:
    return GenerationAttempt(
        request_id=request.request_id,
        attempt_id=attempt_id,
        attempt_index=attempt_index,
        accepted_by_server=True,
        http_status=status_code,
        response_body=response_body,
        raw_response=raw_response,
        contract_text=None,
        transport_error_body=transport_error_body,
        thinking_response=thinking_response,
        stop_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        backend_timing=None,
        latency_ms=elapsed_ms(started, clock),
        error_class=RuntimeFailureClass.TRANSPORT_DRIFT,
        error_message=reason,
    )


def backend_timing(data: dict[str, Any]) -> BackendTiming | None:
    values = tuple(
        data.get(field_name)
        for field_name in (
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        )
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        return None
    return BackendTiming(*values)


def message_evidence(data: Any) -> tuple[str | None, str]:
    if not isinstance(data, dict):
        return None, ""
    message = data.get("message")
    if not isinstance(message, dict):
        return None, ""
    raw_response = message.get("content")
    thinking_response = message.get("thinking", "")
    return (
        raw_response if isinstance(raw_response, str) else None,
        thinking_response if isinstance(thinking_response, str) else "",
    )


def strict_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def elapsed_ms(started: float, clock: Callable[[], float]) -> float:
    return max(0.0, (clock() - started) * 1000.0)


__all__ = ["elapsed_ms", "failed_attempt"]
