from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._scientific_trace_payload_validation import (
    validate_serialized_trace_invariants,
)
from ._scientific_trace_hashing import capability_snapshot_sha256

if TYPE_CHECKING:
    from ._scientific_trace_records import DecisionTraceRecord


TRACE_SCHEMA_VERSION = "iclr2027.decision_trace.v1"


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is prohibited: {value}.")


def trace_schema_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "schemas"
        / "iclr2027"
        / "decision_trace.schema.json"
    )


@lru_cache(maxsize=1)
def trace_schema() -> dict[str, Any]:
    payload = json.loads(trace_schema_path().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Decision trace schema must be a JSON object.")
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "jsonschema is required for scientific trace validation."
        ) from exc
    jsonschema.Draft202012Validator.check_schema(payload)
    return payload


@lru_cache(maxsize=1)
def trace_schema_sha256() -> str:
    return "sha256:" + hashlib.sha256(trace_schema_path().read_bytes()).hexdigest()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def validate_trace_payload(payload: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "jsonschema is required for scientific trace validation."
        ) from exc
    jsonschema.Draft202012Validator(trace_schema()).validate(payload)
    if payload["schema_version"] != TRACE_SCHEMA_VERSION:
        raise ValueError("Decision trace schema version drifted.")
    if payload["schema_sha256"] != trace_schema_sha256():
        raise ValueError("Decision trace schema hash drifted.")
    validate_serialized_trace_invariants(payload)


def serialize_trace_record(record: "DecisionTraceRecord") -> dict[str, Any]:
    key = record.context.key
    prompt = record.prompt_artifact
    generation = record.generation
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "schema_sha256": trace_schema_sha256(),
        "trace_key": {
            "campaign_id": key.campaign_id,
            "episode_attempt_id": key.episode_attempt_id,
            "condition_id": key.condition_id,
            "case_id": key.case_id,
            "pair_id": key.pair_id,
            "template_id": key.template_id,
            "replicate_id": key.replicate_id,
            "decision_index": key.decision_index,
            "env_step_index": key.env_step_index,
        },
        "context": {
            "benchmark_fingerprint": record.context.benchmark_fingerprint,
            "code_revision": record.context.code_revision,
            "simulator_seed": record.context.simulator_seed,
            "generation_seed_master": record.context.generation_seed_master,
            "generation_seed_scope": record.context.generation_seed_scope.value,
            "decision_snapshot_id": record.context.decision_snapshot_id,
            "available_action_ids": list(record.context.available_action_ids),
            "event_phase": record.context.event_phase,
            "applied_event_ids": list(record.context.applied_event_ids),
        },
        "factors": record.harness_config.condition.to_canonical_dict(),
        "harness_config": record.harness_config.to_canonical_dict(),
        "config_sha256": "sha256:" + record.harness_config.config_hash(),
        "prompt": {
            "policy_content": prompt.policy_content.value,
            "output_enforcement": prompt.output_enforcement.value,
            "provenance_scope": prompt.provenance_scope,
            "few_shot_num": prompt.few_shot_num,
            "prompt_sha256": "sha256:" + prompt.prompt_hash(),
            "component_sha256": [
                {"name": name, "sha256": "sha256:" + digest}
                for name, digest in prompt.component_hashes()
            ],
        },
        "generation": _generation_payload(generation, record.harness_config),
        "action_resolution": _resolution_payload(record.resolution),
        "shield_stack": _shield_payload(record.shield_stack),
        "failure": _failure_payload(record),
        "disposition": record.disposition.value,
        "decision_latency_ms": float(record.decision_latency_ms),
    }
    return payload


def _generation_payload(result: Any, harness_config: Any) -> dict[str, Any]:
    request = result.request
    return {
        "transport_evidence": _transport_evidence_payload(result, harness_config),
        "request": {
            "request_id": request.request_id,
            "model_tag": request.model_tag,
            "model_digest": request.model_digest,
            "native_endpoint": request.native_endpoint,
            "messages": [
                {"role": role, "content": content} for role, content in request.messages
            ],
            "options": request.options.to_payload(),
            "output_enforcement": request.output_enforcement.value,
            "think_mode": request.think_mode.value,
            "timeout_sec": float(request.timeout_sec),
        },
        "identity_checks": [_identity_payload(item) for item in result.identity_checks],
        "attempts": [_attempt_payload(item) for item in result.attempts],
        "response_body": result.response_body,
        "raw_output": result.raw_response,
        "contract_text": result.contract_text,
        "transport_error_body": result.transport_error_body,
        "thinking_output": result.thinking_response,
        "stop_reason": result.stop_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "backend_timing": _backend_timing_payload(result.backend_timing),
        "latency_ms": float(result.latency_ms),
        "identity_latency_ms": float(result.identity_latency_ms),
        "generation_latency_ms": float(result.generation_latency_ms),
        "retry_cooldown_ms": float(result.retry_cooldown_ms),
        "retry_cooldown_policy_ms": float(result.retry_cooldown_policy_ms),
        "error_class": _enum_value(result.error_class),
        "error_message": result.error_message,
    }


def _transport_evidence_payload(result: Any, harness_config: Any) -> dict[str, Any]:
    capabilities = result.capabilities
    requested_profile = harness_config.transport.profile.value
    profile_effective = any(
        attempt.accepted_by_server is True for attempt in result.attempts
    )
    error_value = None if result.error_class is None else result.error_class.value
    think_mode_effective = profile_effective and error_value in {
        None,
        "model_empty_output",
    }
    evidence = {
        "requested_profile": requested_profile,
        "effective_profile": requested_profile if profile_effective else None,
        "requested_think_mode": harness_config.transport.think_mode.value,
        "effective_think_mode": (
            result.request.think_mode.value if think_mode_effective else None
        ),
        "capability_model_tag": capabilities.model_tag,
        "capability_model_digest": capabilities.model_digest,
        "capability_native_endpoint": capabilities.native_endpoint,
        "capability_supported_think_modes": sorted(
            mode.value for mode in capabilities.supported_think_modes
        ),
        "seed_verified": capabilities.seed_verified,
        "schema_verified": capabilities.schema_verified,
        "capability_probe_id": capabilities.capability_probe_id,
        "capability_artifact_sha256": capabilities.capability_artifact_hash,
        "schema_mechanism": capabilities.schema_mechanism,
    }
    evidence["capability_snapshot_sha256"] = capability_snapshot_sha256(evidence)
    return evidence


def _identity_payload(check: Any) -> dict[str, Any]:
    return {
        "attempt_index": check.attempt_index,
        "phase": check.phase,
        "requested_model_tag": check.requested_model_tag,
        "requested_model_digest": check.requested_model_digest,
        "observed_model_tag": check.observed_model_tag,
        "observed_model_digest": check.observed_model_digest,
        "latency_ms": float(check.latency_ms),
        "error_message": check.error_message,
        "succeeded": check.succeeded,
    }


def _attempt_payload(attempt: Any) -> dict[str, Any]:
    return {
        "request_id": attempt.request_id,
        "attempt_id": attempt.attempt_id,
        "attempt_index": attempt.attempt_index,
        "accepted_by_server": attempt.accepted_by_server,
        "http_status": attempt.http_status,
        "response_body": attempt.response_body,
        "raw_output": attempt.raw_response,
        "contract_text": attempt.contract_text,
        "transport_error_body": attempt.transport_error_body,
        "thinking_output": attempt.thinking_response,
        "stop_reason": attempt.stop_reason,
        "prompt_tokens": attempt.prompt_tokens,
        "completion_tokens": attempt.completion_tokens,
        "backend_timing": _backend_timing_payload(attempt.backend_timing),
        "latency_ms": float(attempt.latency_ms),
        "error_class": _enum_value(attempt.error_class),
        "error_message": attempt.error_message,
    }


def _backend_timing_payload(timing: Any) -> dict[str, int] | None:
    if timing is None:
        return None
    return {
        "total_duration_ns": timing.total_duration_ns,
        "load_duration_ns": timing.load_duration_ns,
        "prompt_eval_duration_ns": timing.prompt_eval_duration_ns,
        "eval_duration_ns": timing.eval_duration_ns,
    }


def _resolution_payload(resolution: Any) -> dict[str, Any] | None:
    if resolution is None:
        return None
    return {
        "parser_input": resolution.raw_response,
        "syntax_status": resolution.syntax_status.value,
        "strict_action": resolution.strict_action,
        "recovered_action": resolution.recovered_action,
        "recovery_stage": resolution.recovery_stage.value,
        "violation": _enum_value(resolution.violation),
        "action_available": resolution.action_available.value,
        "fallback_action": resolution.fallback_action,
        "used_fallback": resolution.used_fallback,
        "final_resolved_action": resolution.final_resolved_action,
    }


def _shield_payload(stack: Any) -> dict[str, Any] | None:
    if stack is None:
        return None
    return {
        "proposed_action_id": stack.proposed_action_id,
        "fallback_modified_action_id": stack.fallback_modified_action_id,
        "unshielded_action_id": stack.unshielded_action_id,
        "shielded_action_id": stack.shielded_action_id,
        "executed_action_id": stack.executed_action_id,
        "execution_mode": stack.execution_mode.value,
        "stages": [_shield_stage_payload(stage) for stage in stack.stages],
    }


def _shield_stage_payload(stage: Any) -> dict[str, Any]:
    primitive = stage.primitive_result
    return {
        "stage_name": stage.stage_name,
        "input_action_id": stage.input_action_id,
        "output_action_id": stage.output_action_id,
        "applied": stage.applied,
        "bypassed": stage.bypassed,
        "reason": stage.reason,
        "primitive": (
            None if primitive is None else _shield_primitive_payload(primitive)
        ),
    }


def _shield_primitive_payload(primitive: Any) -> dict[str, Any]:
    numeric_fields = (
        "front_gap_m",
        "rear_gap_m",
        "front_ttc_sec",
        "rear_ttc_sec",
        "required_front_gap_m",
        "required_rear_gap_m",
        "required_front_ttc_sec",
        "required_rear_ttc_sec",
        "projected_front_gap_m",
        "projected_front_ttc_sec",
        "projected_ego_speed_mps",
        "projection_horizon_sec",
    )
    nonfinite_values: dict[str, str] = {}
    payload = {
        "original_action_id": primitive.original_action_id,
        "action_id": primitive.action_id,
        "applied": primitive.applied,
        "reason": primitive.reason,
        "shield_type": primitive.shield_type,
        "target_lane_rank": primitive.target_lane_rank,
    }
    for field_name in numeric_fields:
        value = getattr(primitive, field_name)
        if value is None or math.isfinite(float(value)):
            payload[field_name] = value
            continue
        if math.isnan(float(value)):
            raise ValueError(f"Shield primitive {field_name} cannot be NaN.")
        payload[field_name] = None
        nonfinite_values[field_name] = (
            "positive_infinity" if float(value) > 0 else "negative_infinity"
        )
    payload["nonfinite_values"] = nonfinite_values
    return payload


def _failure_payload(record: Any) -> dict[str, str] | None:
    failure_class = record.generation.error_class
    message = record.generation.error_message
    if failure_class is None and record.resolution is not None:
        failure_class = record.resolution.violation
        message = None if failure_class is None else "action_resolution_violation"
    if failure_class is None:
        return None
    return {
        "failure_class": failure_class.value,
        "message": message or failure_class.value,
    }


def _enum_value(value: Any) -> str | None:
    return None if value is None else value.value


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "canonical_json_bytes",
    "serialize_trace_record",
    "trace_schema",
    "trace_schema_path",
    "trace_schema_sha256",
    "validate_trace_payload",
]
