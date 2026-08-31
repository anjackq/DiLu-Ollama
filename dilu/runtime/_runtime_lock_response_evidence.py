"""Response-evidence derivation shared by runtime-lock authoring and reload."""

from __future__ import annotations

from dataclasses import asdict

from ._scientific_transport_response import parse_native_response_attempt
from .action_resolution import (
    CANONICAL_ACTION_IDS,
    ActionResolutionResult,
    ActionSyntaxStatus,
    resolve_action,
)
from .harness_config import (
    FallbackPolicy,
    OutputEnforcement,
    ParserMode,
    ResolverMode,
)
from .runtime_failures import RuntimeFailureClass
from .scientific_transport_types import GenerationRequest


def derive_response_evidence(
    request: GenerationRequest,
    status: int,
    response_payload: object,
    response_body: str,
) -> tuple[dict[str, object], ActionResolutionResult]:
    """Parse one response into the exact evidence fields persisted by authoring."""
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or not 200 <= status < 300
    ):
        raise ValueError("Native capability response requires a direct 2xx status.")
    if not isinstance(response_body, str):
        raise ValueError(  # noqa: TRY004 - preserve frozen API
            "Native capability response body is malformed."
        )
    attempt = parse_native_response_attempt(
        request,
        f"{request.request_id}:a1",
        1,
        status,
        response_payload,
        response_body,
        0.0,
        lambda: 0.0,
    )
    if attempt.error_class is not None:
        label = (
            "schema rejection"
            if attempt.error_class is RuntimeFailureClass.SCHEMA_REJECTION
            else "malformed native response"
        )
        raise ValueError(f"Native capability probe {label}: {attempt.error_message}.")
    if (
        attempt.contract_text is None
        or attempt.backend_timing is None
        or attempt.prompt_tokens is None
        or attempt.completion_tokens is None
    ):
        raise ValueError("Native capability probe omitted required evidence.")
    effective_action_ids = (
        request.available_action_ids
        if request.available_action_ids is not None
        else CANONICAL_ACTION_IDS
    )
    resolution = resolve_action(
        attempt.contract_text,
        available_action_ids=effective_action_ids,
        parser_mode=ParserMode.STRICT_ONLY,
        resolver_mode=ResolverMode.DISABLED,
        fallback_policy=FallbackPolicy.FIXED_IDLE,
    )
    if request.output_enforcement in (
        OutputEnforcement.BACKEND_SCHEMA,
        OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
    ) and (
        resolution.syntax_status is not ActionSyntaxStatus.STRICT_VALID
        or resolution.strict_action not in effective_action_ids
        or resolution.used_fallback
    ):
        raise ValueError(
            "Native capability probe backend schema did not return a strict canonical action."
        )
    evidence = {
        "http_status": status,
        "response_body": response_body,
        "raw_response": attempt.raw_response,
        "contract_text": attempt.contract_text,
        "action_resolution": serialize_action_resolution(resolution),
        "stop_reason": attempt.stop_reason,
        "prompt_tokens": attempt.prompt_tokens,
        "completion_tokens": attempt.completion_tokens,
        "total_tokens": attempt.prompt_tokens + attempt.completion_tokens,
        "backend_timing": asdict(attempt.backend_timing),
    }
    return evidence, resolution


def serialize_action_resolution(
    resolution: ActionResolutionResult,
) -> dict[str, object]:
    return {
        "raw_response": resolution.raw_response,
        "syntax_status": resolution.syntax_status.value,
        "strict_action": resolution.strict_action,
        "recovered_action": resolution.recovered_action,
        "recovery_stage": resolution.recovery_stage.value,
        "violation": None
        if resolution.violation is None
        else resolution.violation.value,
        "action_available": resolution.action_available.value,
        "fallback_action": resolution.fallback_action,
        "final_resolved_action": resolution.final_resolved_action,
        "used_fallback": resolution.used_fallback,
    }


__all__ = ["derive_response_evidence", "serialize_action_resolution"]
