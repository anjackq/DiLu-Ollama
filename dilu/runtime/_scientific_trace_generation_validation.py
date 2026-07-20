from __future__ import annotations

from typing import Any

from ._scientific_contract_validation import (
    validate_output_contract_semantics,
    validate_success_response_body,
)
from .runtime_failures import RuntimeFailureClass


_GENERATION_FAILURE_VALUES = frozenset(
    {
        RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT.value,
        RuntimeFailureClass.TRANSPORT_DRIFT.value,
        RuntimeFailureClass.GENERATION_TIMEOUT.value,
        RuntimeFailureClass.MODEL_EMPTY_OUTPUT.value,
        RuntimeFailureClass.SCHEMA_REJECTION.value,
    }
)
_OPERATIONAL_GENERATION_FAILURES = frozenset(
    {
        RuntimeFailureClass.GENERATION_TIMEOUT.value,
        RuntimeFailureClass.MODEL_EMPTY_OUTPUT.value,
    }
)
_RETRYABLE_FAILURE = RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT.value
_TRANSPORT_DRIFT = RuntimeFailureClass.TRANSPORT_DRIFT.value
_RETRY_POLICY_MS = 10000.0


def validate_serialized_generation_chain(
    generation: dict[str, Any],
    request: dict[str, Any],
) -> None:
    attempts = generation["attempts"]
    checks = generation["identity_checks"]
    _validate_failure_class(generation["error_class"], "generation")
    _validate_error_pair(
        generation["error_class"],
        generation["error_message"],
        "generation",
    )
    _validate_identity_checks(checks, request)
    expected_checks = _validate_attempts(attempts, request)
    observed_checks = [(item["attempt_index"], item["phase"]) for item in checks]
    if attempts and observed_checks != expected_checks:
        raise ValueError("Identity checks do not bind the serialized attempt chain.")
    if not attempts:
        _validate_preflight_block(generation, checks)
    else:
        _validate_retry_semantics(generation, attempts)
        _validate_final_attempt(generation, attempts[-1])
        _validate_identity_outcome(generation, checks)
    _validate_timing(generation, checks, attempts)


def _validate_identity_checks(
    checks: list[dict[str, Any]],
    request: dict[str, Any],
) -> None:
    for check in checks:
        if (
            check["requested_model_tag"] != request["model_tag"]
            or check["requested_model_digest"] != request["model_digest"]
            or check["succeeded"] != (check["error_message"] is None)
        ):
            raise ValueError("Serialized model identity evidence is inconsistent.")
        if check["succeeded"] and (
            check["observed_model_tag"] != request["model_tag"]
            or check["observed_model_digest"] != request["model_digest"]
        ):
            raise ValueError("Successful identity evidence does not match the request.")


def _validate_attempts(
    attempts: list[dict[str, Any]],
    request: dict[str, Any],
) -> list[tuple[int, str]]:
    expected_checks: list[tuple[int, str]] = []
    for expected_index, attempt in enumerate(attempts, start=1):
        _validate_failure_class(attempt["error_class"], "attempt")
        _validate_error_pair(
            attempt["error_class"],
            attempt["error_message"],
            "attempt",
        )
        if (
            attempt["request_id"] != request["request_id"]
            or attempt["attempt_index"] != expected_index
            or attempt["attempt_id"] != f'{request["request_id"]}:a{expected_index}'
        ):
            raise ValueError("Serialized generation attempts are not one chain.")
        expected_checks.append((expected_index, "pre"))
        if attempt["accepted_by_server"] is not False:
            expected_checks.append((expected_index, "post"))
        if (
            attempt["accepted_by_server"] is False
            and any(
                attempt[name] is not None
                for name in (
                    "response_body",
                    "raw_output",
                    "contract_text",
                    "transport_error_body",
                    "prompt_tokens",
                    "completion_tokens",
                    "backend_timing",
                )
            )
            or (attempt["accepted_by_server"] is False and attempt["thinking_output"])
        ):
            raise ValueError("Pre-accept attempt contains response evidence.")
        if attempt["error_class"] is None:
            success_metadata = (
                attempt["accepted_by_server"] is True
                and isinstance(attempt["http_status"], int)
                and 200 <= attempt["http_status"] < 300
                and attempt["response_body"] is not None
                and attempt["raw_output"] is not None
                and attempt["contract_text"] is not None
                and attempt["transport_error_body"] is None
                and attempt["prompt_tokens"] is not None
                and attempt["completion_tokens"] is not None
                and attempt["backend_timing"] is not None
                and isinstance(attempt["stop_reason"], str)
                and bool(attempt["stop_reason"])
                and attempt["stop_reason"] == attempt["stop_reason"].strip()
            )
            if not success_metadata:
                raise ValueError(
                    "Serialized successful attempt lacks accepted completion metadata."
                )
        validate_output_contract_semantics(
            output_enforcement=request["output_enforcement"],
            think_mode=request["think_mode"],
            error_class=attempt["error_class"],
            raw_output=attempt["raw_output"],
            contract_text=attempt["contract_text"],
            thinking_output=attempt["thinking_output"],
        )
        if attempt["error_class"] is None:
            timing = attempt["backend_timing"]
            validate_success_response_body(
                model_tag=request["model_tag"],
                response_body=attempt["response_body"],
                raw_output=attempt["raw_output"],
                thinking_output=attempt["thinking_output"],
                stop_reason=attempt["stop_reason"],
                prompt_tokens=attempt["prompt_tokens"],
                completion_tokens=attempt["completion_tokens"],
                backend_timing=(
                    timing["total_duration_ns"],
                    timing["load_duration_ns"],
                    timing["prompt_eval_duration_ns"],
                    timing["eval_duration_ns"],
                ),
            )
    return expected_checks


def _validate_preflight_block(
    generation: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    if (
        generation["error_class"] != _TRANSPORT_DRIFT
        or len(checks) > 1
        or (checks and (checks[0]["phase"] != "pre" or checks[0]["succeeded"]))
    ):
        raise ValueError("Blocked preflight evidence is invalid.")
    if (
        any(
            generation[name] is not None
            for name in (
                "response_body",
                "raw_output",
                "contract_text",
                "transport_error_body",
                "stop_reason",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "backend_timing",
            )
        )
        or generation["thinking_output"]
    ):
        raise ValueError("Blocked preflight contains generation evidence.")
    if (
        generation["generation_latency_ms"] != 0
        or generation["retry_cooldown_ms"] != 0
        or generation["retry_cooldown_policy_ms"] != 0
    ):
        raise ValueError("Blocked preflight contains generation timing.")


def _validate_retry_semantics(
    generation: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> None:
    if len(attempts) == 1:
        if attempts[0]["error_class"] == _RETRYABLE_FAILURE:
            raise ValueError("Pre-accept unavailability requires the fixed retry.")
        if generation["retry_cooldown_ms"] or generation["retry_cooldown_policy_ms"]:
            raise ValueError("Single-attempt result contains retry cooldown.")
        return
    if len(attempts) != 2:
        raise ValueError("Scientific generation permits at most two attempts.")
    first = attempts[0]
    if (
        first["error_class"] != _RETRYABLE_FAILURE
        or first["accepted_by_server"] is not False
    ):
        raise ValueError("Only proved pre-accept unavailability permits retry.")
    if float(generation["retry_cooldown_policy_ms"]) != _RETRY_POLICY_MS:
        raise ValueError("Scientific retry policy must be exactly 10 seconds.")
    if float(generation["retry_cooldown_ms"]) < _RETRY_POLICY_MS:
        raise ValueError("Scientific retry did not observe the fixed cooldown.")


def _validate_identity_outcome(
    generation: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    error_class = generation["error_class"]
    if error_class is None or error_class in _OPERATIONAL_GENERATION_FAILURES:
        if not all(check["succeeded"] for check in checks):
            raise ValueError("Accepted generation has failed identity evidence.")
    elif any(not check["succeeded"] for check in checks):
        if error_class != _TRANSPORT_DRIFT:
            raise ValueError("Identity-check failure must be transport drift.")


def _validate_final_attempt(
    generation: dict[str, Any],
    final: dict[str, Any],
) -> None:
    names = (
        "response_body",
        "raw_output",
        "contract_text",
        "transport_error_body",
        "thinking_output",
        "stop_reason",
        "prompt_tokens",
        "completion_tokens",
        "backend_timing",
        "error_class",
        "error_message",
    )
    if any(generation[name] != final[name] for name in names):
        raise ValueError("Serialized result does not preserve its final attempt.")
    expected_tokens = (
        None
        if generation["prompt_tokens"] is None
        or generation["completion_tokens"] is None
        else generation["prompt_tokens"] + generation["completion_tokens"]
    )
    if generation["total_tokens"] != expected_tokens:
        raise ValueError("Serialized total token count is inconsistent.")


def _validate_timing(
    generation: dict[str, Any],
    checks: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> None:
    if float(generation["identity_latency_ms"]) != sum(
        float(item["latency_ms"]) for item in checks
    ):
        raise ValueError("Serialized identity latency is inconsistent.")
    if float(generation["generation_latency_ms"]) != sum(
        float(item["latency_ms"]) for item in attempts
    ):
        raise ValueError("Serialized generation latency is inconsistent.")
    expected_total = (
        float(generation["identity_latency_ms"])
        + float(generation["generation_latency_ms"])
        + float(generation["retry_cooldown_ms"])
    )
    if float(generation["latency_ms"]) != expected_total:
        raise ValueError("Serialized total latency is inconsistent.")


def _validate_failure_class(value: str | None, layer: str) -> None:
    if value is not None and value not in _GENERATION_FAILURE_VALUES:
        raise ValueError(f"Serialized {layer} failure class is invalid for generation.")


def _validate_error_pair(
    failure_class: str | None,
    message: str | None,
    layer: str,
) -> None:
    if (failure_class is None) != (message is None):
        raise ValueError(f"Serialized {layer} error evidence is inconsistent.")
    if message is not None and (not message or message != message.strip()):
        raise ValueError(f"Serialized {layer} error message is not canonical.")


__all__ = ["validate_serialized_generation_chain"]
