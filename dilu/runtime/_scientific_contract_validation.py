from __future__ import annotations

import json

from .scientific_transport_types import CANONICAL_ACTION_TEXT_VALUES


_SCHEMA_REJECTION = "schema_rejection"
_TRANSPORT_DRIFT = "transport_drift"


def validate_output_contract_semantics(
    *,
    output_enforcement: str,
    think_mode: str,
    error_class: str | None,
    raw_output: str | None,
    contract_text: str | None,
    thinking_output: str,
) -> None:
    if think_mode == "no_think" and thinking_output.strip():
        if error_class != _TRANSPORT_DRIFT or contract_text is not None:
            raise ValueError(
                "No-think leakage must be preserved as transport drift evidence."
            )
    if output_enforcement == "prompt_only":
        if error_class == _SCHEMA_REJECTION:
            raise ValueError("Prompt-only generation cannot be schema rejection.")
        if error_class != _TRANSPORT_DRIFT and contract_text != raw_output:
            raise ValueError("Prompt-only contract text must preserve raw output.")
        return
    if output_enforcement != "backend_schema":
        raise ValueError("Unknown output-enforcement mode.")
    if error_class is not None:
        return
    try:
        decoded = json.loads(raw_output or "")
    except (TypeError, ValueError) as exc:
        raise ValueError("Backend-schema raw output is not valid JSON.") from exc
    if decoded != contract_text or decoded not in CANONICAL_ACTION_TEXT_VALUES:
        raise ValueError("Backend-schema raw output does not bind contract text.")


def validate_success_response_body(
    *,
    model_tag: str,
    response_body: str | None,
    raw_output: str | None,
    thinking_output: str,
    stop_reason: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    backend_timing: tuple[int, int, int, int] | None,
) -> None:
    try:
        payload = json.loads(response_body or "")
    except (TypeError, ValueError) as exc:
        raise ValueError("Successful response body is not valid JSON.") from exc
    message = payload.get("message") if isinstance(payload, dict) else None
    observed_timing = (
        (
            payload.get("total_duration"),
            payload.get("load_duration"),
            payload.get("prompt_eval_duration"),
            payload.get("eval_duration"),
        )
        if isinstance(payload, dict)
        else None
    )
    if (
        not isinstance(message, dict)
        or payload.get("model") != model_tag
        or payload.get("done") is not True
        or message.get("role") != "assistant"
        or message.get("content") != raw_output
        or message.get("thinking", "") != thinking_output
        or payload.get("done_reason") != stop_reason
        or payload.get("prompt_eval_count") != prompt_tokens
        or payload.get("eval_count") != completion_tokens
        or observed_timing != backend_timing
    ):
        raise ValueError("Successful response body does not bind derived evidence.")


__all__ = [
    "validate_output_contract_semantics",
    "validate_success_response_body",
]
