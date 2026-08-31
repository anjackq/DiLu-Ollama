from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ._scientific_transport_attempts import (
    backend_timing as _backend_timing,
    elapsed_ms,
    empty_output_attempt as _empty_output_attempt,
    failed_attempt,
    message_evidence as _message_evidence,
    payload_drift_attempt as _payload_drift,
    schema_failure_attempt as _schema_failure,
    strict_nonnegative_int as _strict_nonnegative_int,
    strict_positive_int as _strict_positive_int,
)
from .harness_config import OutputEnforcement, ThinkMode
from .runtime_failures import RuntimeFailureClass
from .scientific_transport_records import (
    GenerationAttempt,
    GenerationResult,
    ModelIdentityCheck,
)
from .scientific_transport_types import (
    CANONICAL_ACTION_TEXT_VALUES,
    GenerationRequest,
    ScientificTransportCapabilities,
)


def parse_native_response_attempt(
    request: GenerationRequest,
    attempt_id: str,
    attempt_index: int,
    status_code: int | None,
    data: Any,
    response_body: str | None,
    started: float,
    clock: Callable[[], float],
) -> GenerationAttempt:
    raw_response, early_thinking = _message_evidence(data)
    if not isinstance(data, dict):
        return _payload_drift(
            request,
            attempt_id,
            attempt_index,
            status_code,
            started,
            clock,
            "payload_not_object",
            response_body=response_body,
            transport_error_body=response_body,
        )
    if data.get("model") != request.model_tag:
        return _payload_drift(
            request,
            attempt_id,
            attempt_index,
            status_code,
            started,
            clock,
            "response_model_tag_drift",
            response_body=response_body,
            raw_response=raw_response,
            thinking_response=early_thinking,
        )
    if data.get("done") is not True:
        return _payload_drift(
            request,
            attempt_id,
            attempt_index,
            status_code,
            started,
            clock,
            "response_not_done",
            response_body=response_body,
            raw_response=raw_response,
            thinking_response=early_thinking,
        )
    message = data.get("message")
    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
    ):
        return _payload_drift(
            request,
            attempt_id,
            attempt_index,
            status_code,
            started,
            clock,
            "invalid_message_content",
            response_body=response_body,
            raw_response=raw_response,
            thinking_response=early_thinking,
            transport_error_body=response_body if raw_response is None else None,
        )
    raw_response = message["content"]
    thinking_response = message.get("thinking", "")
    if not isinstance(thinking_response, str):
        return _payload_drift(
            request,
            attempt_id,
            attempt_index,
            status_code,
            started,
            clock,
            "invalid_thinking_content",
            response_body=response_body,
            raw_response=raw_response,
        )
    if request.think_mode is ThinkMode.NO_THINK and thinking_response.strip():
        return _payload_drift(
            request,
            attempt_id,
            attempt_index,
            status_code,
            started,
            clock,
            "no_think_leakage",
            response_body=response_body,
            raw_response=raw_response,
            thinking_response=thinking_response,
        )
    prompt_tokens = _strict_positive_int(data.get("prompt_eval_count"))
    completion_tokens = _strict_nonnegative_int(data.get("eval_count"))
    stop_reason = data.get("done_reason")
    backend_timing = _backend_timing(data)
    if (
        prompt_tokens is None
        or completion_tokens is None
        or not isinstance(stop_reason, str)
        or not stop_reason
        or stop_reason != stop_reason.strip()
        or backend_timing is None
        or (bool(raw_response.strip()) and completion_tokens == 0)
    ):
        return _payload_drift(
            request,
            attempt_id,
            attempt_index,
            status_code,
            started,
            clock,
            "invalid_completion_metadata",
            response_body=response_body,
            raw_response=raw_response,
            thinking_response=thinking_response,
        )
    if not raw_response.strip():
        return _empty_output_attempt(
            request,
            attempt_id,
            attempt_index,
            status_code,
            response_body or "",
            raw_response,
            thinking_response,
            stop_reason,
            prompt_tokens,
            completion_tokens,
            backend_timing,
            started,
            clock,
        )

    contract_text = raw_response
    if request.output_enforcement in (
        OutputEnforcement.BACKEND_SCHEMA,
        OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
    ):
        try:
            decoded = json.loads(raw_response)
        except (TypeError, ValueError) as exc:
            return _schema_failure(
                request,
                attempt_id,
                attempt_index,
                status_code,
                response_body or "",
                raw_response,
                thinking_response,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                backend_timing,
                started,
                clock,
                exc,
            )
        allowed_values = (
            CANONICAL_ACTION_TEXT_VALUES
            if request.output_enforcement is OutputEnforcement.BACKEND_SCHEMA
            else tuple(
                f"Response to user:#### {action_id}"
                for action_id in request.available_action_ids or ()
            )
        )
        if not isinstance(decoded, str) or decoded not in allowed_values:
            return _schema_failure(
                request,
                attempt_id,
                attempt_index,
                status_code,
                response_body or "",
                raw_response,
                thinking_response,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                backend_timing,
                started,
                clock,
                ValueError("noncanonical_schema_contract"),
            )
        contract_text = decoded

    return GenerationAttempt(
        request_id=request.request_id,
        attempt_id=attempt_id,
        attempt_index=attempt_index,
        accepted_by_server=True,
        http_status=status_code,
        response_body=response_body,
        raw_response=raw_response,
        contract_text=contract_text,
        transport_error_body=None,
        thinking_response=thinking_response,
        stop_reason=stop_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        backend_timing=backend_timing,
        latency_ms=elapsed_ms(started, clock),
        error_class=None,
        error_message=None,
    )


def result_from_attempts(
    request: GenerationRequest,
    capabilities: ScientificTransportCapabilities,
    identity_checks: list[ModelIdentityCheck],
    attempts: list[GenerationAttempt],
    cooldown_observed_ms: float,
    cooldown_policy_ms: float,
) -> GenerationResult:
    final = attempts[-1]
    return GenerationResult(
        request=request,
        capabilities=capabilities,
        identity_checks=tuple(identity_checks),
        attempts=tuple(attempts),
        response_body=final.response_body,
        raw_response=final.raw_response,
        contract_text=final.contract_text,
        transport_error_body=final.transport_error_body,
        thinking_response=final.thinking_response,
        stop_reason=final.stop_reason,
        prompt_tokens=final.prompt_tokens,
        completion_tokens=final.completion_tokens,
        backend_timing=final.backend_timing,
        latency_ms=(
            sum(check.latency_ms for check in identity_checks)
            + sum(attempt.latency_ms for attempt in attempts)
            + cooldown_observed_ms
        ),
        identity_latency_ms=sum(check.latency_ms for check in identity_checks),
        generation_latency_ms=sum(attempt.latency_ms for attempt in attempts),
        retry_cooldown_ms=cooldown_observed_ms,
        retry_cooldown_policy_ms=cooldown_policy_ms,
        error_class=final.error_class,
        error_message=final.error_message,
    )


def blocked_result(
    request: GenerationRequest,
    capabilities: ScientificTransportCapabilities,
    reason: str,
    identity_checks: list[ModelIdentityCheck] | None = None,
) -> GenerationResult:
    checks = list(identity_checks or [])
    identity_latency_ms = sum(check.latency_ms for check in checks)
    return GenerationResult(
        request=request,
        capabilities=capabilities,
        identity_checks=tuple(checks),
        attempts=(),
        response_body=None,
        raw_response=None,
        contract_text=None,
        transport_error_body=None,
        thinking_response="",
        stop_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        backend_timing=None,
        latency_ms=identity_latency_ms,
        identity_latency_ms=identity_latency_ms,
        generation_latency_ms=0.0,
        retry_cooldown_ms=0.0,
        retry_cooldown_policy_ms=0.0,
        error_class=RuntimeFailureClass.TRANSPORT_DRIFT,
        error_message=reason,
    )


__all__ = [
    "blocked_result",
    "elapsed_ms",
    "failed_attempt",
    "parse_native_response_attempt",
    "result_from_attempts",
]
