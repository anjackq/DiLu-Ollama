from __future__ import annotations

import hashlib
import json
from typing import Any

from dilu.driver_agent.prompt_modules import build_prompt_artifact

from ._scientific_trace_generation_validation import (
    validate_serialized_generation_chain,
)
from ._scientific_trace_transport_validation import validate_transport_evidence
from .action_resolution import resolve_action
from .generation_seed import (
    post_divergence_generation_seed,
    primary_snapshot_generation_seed,
)
from .harness_config import (
    FallbackPolicy,
    OutputEnforcement,
    ParserMode,
    PolicyContent,
    ResolverMode,
)


_CONDITION_BITS = {
    "policy_content": {"historical_dilu_2024": "0", "modular_harness": "1"},
    "output_enforcement": {"prompt_only": "0", "backend_schema": "1"},
    "execution_mode": {"unshielded_operational": "0", "shielded": "1"},
}
_OPERATIONAL_GENERATION_FAILURES = {"generation_timeout", "model_empty_output"}
_PRIMITIVE_SHIELD_TYPE = {
    "lane_change": "lane_change",
    "longitudinal_safety": "longitudinal",
    "low_speed_recovery": "flow_recovery",
}


def validate_serialized_trace_invariants(payload: dict[str, Any]) -> None:
    key = payload["trace_key"]
    context = payload["context"]
    factors = payload["factors"]
    config = payload["harness_config"]
    prompt = payload["prompt"]
    generation = payload["generation"]
    request = generation["request"]

    _validate_condition_and_hashes(payload, key, context, factors, config, prompt)
    _validate_request_and_seed(
        key,
        context,
        factors,
        config,
        prompt,
        generation,
        request,
    )
    validate_serialized_generation_chain(generation, request)
    _validate_action_chain(payload, context, factors, generation)
    if float(payload["decision_latency_ms"]) < float(generation["latency_ms"]):
        raise ValueError("Decision latency excludes part of generation latency.")


def _validate_condition_and_hashes(
    payload: dict[str, Any],
    key: dict[str, Any],
    context: dict[str, Any],
    factors: dict[str, Any],
    config: dict[str, Any],
    prompt: dict[str, Any],
) -> None:
    expected_condition = "c" + "".join(
        _CONDITION_BITS[name][factors[name]]
        for name in ("policy_content", "output_enforcement", "execution_mode")
    )
    if key["condition_id"] != expected_condition or config["condition"] != factors:
        raise ValueError("Serialized condition identity is inconsistent.")
    if (
        context["generation_seed_master"]
        != config["transport"]["generation_seed_master"]
    ):
        raise ValueError("Serialized master seed drifted from the harness config.")
    if payload["config_sha256"] != "sha256:" + _canonical_digest(config):
        raise ValueError("Serialized harness config hash is invalid.")
    if (
        prompt["policy_content"] != factors["policy_content"]
        or prompt["output_enforcement"] != factors["output_enforcement"]
    ):
        raise ValueError("Serialized prompt factors drifted from the condition.")
    expected_prompt = build_prompt_artifact(
        PolicyContent(factors["policy_content"]),
        output_enforcement=OutputEnforcement(factors["output_enforcement"]),
    )
    expected_components = [
        {"name": name, "sha256": "sha256:" + digest}
        for name, digest in expected_prompt.component_hashes()
    ]
    if (
        prompt["provenance_scope"] != expected_prompt.provenance_scope
        or prompt["few_shot_num"] != 0
        or prompt["prompt_sha256"]
        != "sha256:"
        + hashlib.sha256(expected_prompt.system_prompt().encode("utf-8")).hexdigest()
        or prompt["component_sha256"] != expected_components
    ):
        raise ValueError("Serialized prompt provenance is invalid.")


def _validate_request_and_seed(
    key: dict[str, Any],
    context: dict[str, Any],
    factors: dict[str, Any],
    config: dict[str, Any],
    prompt: dict[str, Any],
    generation: dict[str, Any],
    request: dict[str, Any],
) -> None:
    transport = config["transport"]
    options = request["options"]
    expected_request = (
        factors["output_enforcement"],
        transport["think_mode"],
        float(transport["temperature"]),
        transport["context_tokens"],
        transport["max_output_tokens"],
        float(transport["timeout_sec"]),
    )
    observed_request = (
        request["output_enforcement"],
        request["think_mode"],
        float(options["temperature"]),
        options["num_ctx"],
        options["num_predict"],
        float(request["timeout_sec"]),
    )
    if observed_request != expected_request:
        raise ValueError("Serialized generation request drifted from the config.")
    validate_transport_evidence(
        generation["transport_evidence"],
        generation,
        transport,
        request,
    )
    first_message = request["messages"][0]
    if (
        len(request["messages"]) != 2
        or first_message["role"] != "system"
        or request["messages"][1]["role"] != "user"
    ):
        raise ValueError(
            "Scientific request must contain exactly system and user messages."
        )
    prompt_digest = (
        "sha256:" + hashlib.sha256(first_message["content"].encode("utf-8")).hexdigest()
    )
    if prompt["prompt_sha256"] != prompt_digest:
        raise ValueError("Serialized prompt hash does not bind the request.")

    master = context["generation_seed_master"]
    digest = request["model_digest"]
    if 1 not in context["available_action_ids"]:
        raise ValueError("Scientific availability must preserve fixed IDLE=1.")
    if (key["decision_index"] == 0) != (
        context["generation_seed_scope"] == "primary_snapshot"
    ):
        raise ValueError(
            "Serialized generation seed scope is invalid for the decision index."
        )
    if context["generation_seed_scope"] == "primary_snapshot":
        snapshot_id = context["decision_snapshot_id"]
        if snapshot_id is None:
            raise ValueError("Primary generation seed requires a snapshot ID.")
        expected_seed = primary_snapshot_generation_seed(
            master,
            digest,
            key["pair_id"],
            snapshot_id,
            key["replicate_id"],
        )
    else:
        if context["decision_snapshot_id"] is not None:
            raise ValueError("Post-divergence seed cannot carry a snapshot ID.")
        expected_seed = post_divergence_generation_seed(
            master,
            digest,
            key["case_id"],
            key["decision_index"],
            key["replicate_id"],
        )
    if options["seed"] != expected_seed:
        raise ValueError("Serialized generation seed is invalid.")


def _validate_action_chain(
    payload: dict[str, Any],
    context: dict[str, Any],
    factors: dict[str, Any],
    generation: dict[str, Any],
) -> None:
    resolution = payload["action_resolution"]
    stack = payload["shield_stack"]
    error_class = generation["error_class"]
    blocked = (
        error_class is not None and error_class not in _OPERATIONAL_GENERATION_FAILURES
    )
    expected_disposition = (
        "blocked_before_execution" if blocked else "ready_for_env_step"
    )
    if payload["disposition"] != expected_disposition:
        raise ValueError("Trace disposition does not match the generation outcome.")
    if blocked:
        if resolution is not None or stack is not None or payload["failure"] is None:
            raise ValueError("Blocked trace contains execution evidence.")
        expected_message = generation["error_message"] or error_class
        if (
            payload["failure"]["failure_class"] != error_class
            or payload["failure"]["message"] != expected_message
        ):
            raise ValueError("Blocked trace failure class is inconsistent.")
        return
    if resolution is None or stack is None:
        raise ValueError("Ready trace is missing action-stage evidence.")
    parser_input = generation["contract_text"] or generation["raw_output"] or ""
    if resolution["parser_input"] != parser_input:
        raise ValueError("Serialized parser input does not preserve model output.")
    available_ids = context["available_action_ids"]
    if available_ids != sorted(set(available_ids)) or any(
        isinstance(action_id, bool) or action_id not in range(5)
        for action_id in available_ids
    ):
        raise ValueError("Serialized available actions are not canonical.")
    available = set(available_ids)
    expected_resolution = resolve_action(
        parser_input,
        available_action_ids=context["available_action_ids"],
        timed_out=error_class == "generation_timeout",
        parser_mode=ParserMode(payload["harness_config"]["parser_mode"]),
        resolver_mode=ResolverMode(payload["harness_config"]["resolver_mode"]),
        fallback_policy=FallbackPolicy(payload["harness_config"]["fallback_policy"]),
    )
    expected_resolution_payload = {
        "parser_input": expected_resolution.raw_response,
        "syntax_status": expected_resolution.syntax_status.value,
        "strict_action": expected_resolution.strict_action,
        "recovered_action": expected_resolution.recovered_action,
        "recovery_stage": expected_resolution.recovery_stage.value,
        "violation": (
            None
            if expected_resolution.violation is None
            else expected_resolution.violation.value
        ),
        "action_available": expected_resolution.action_available.value,
        "fallback_action": expected_resolution.fallback_action,
        "used_fallback": expected_resolution.used_fallback,
        "final_resolved_action": expected_resolution.final_resolved_action,
    }
    if resolution != expected_resolution_payload:
        raise ValueError("Serialized action resolution is not deterministic.")
    strict_action = resolution["strict_action"]
    expected_availability = (
        "not_applicable"
        if strict_action is None
        else "available" if strict_action in available else "unavailable"
    )
    if resolution["action_available"] != expected_availability:
        raise ValueError("Serialized action availability is invalid.")
    fallback_action = resolution["fallback_action"]
    if fallback_action is not None and fallback_action not in available:
        raise ValueError("Serialized fallback action was unavailable.")
    proposal = (
        resolution["recovered_action"]
        if resolution["recovered_action"] is not None
        else strict_action
    )
    if (
        stack["proposed_action_id"] != proposal
        or stack["fallback_modified_action_id"] != resolution["final_resolved_action"]
        or stack["execution_mode"] != factors["execution_mode"]
    ):
        raise ValueError("Serialized shield stack does not preserve action resolution.")
    expected_input = stack["unshielded_action_id"]
    if expected_input != stack["fallback_modified_action_id"]:
        raise ValueError("Serialized unshielded action is invalid.")
    for stage in stack["stages"]:
        if stage["input_action_id"] != expected_input:
            raise ValueError("Serialized shield stages do not form one chain.")
        primitive = stage["primitive"]
        if stage["bypassed"]:
            if primitive is not None or stage["applied"]:
                raise ValueError("Bypassed shield stage contains primitive evidence.")
        else:
            if primitive is None or (
                primitive["original_action_id"] != stage["input_action_id"]
                or primitive["action_id"] != stage["output_action_id"]
                or primitive["applied"] != stage["applied"]
                or primitive["reason"] != stage["reason"]
                or primitive["shield_type"]
                != _PRIMITIVE_SHIELD_TYPE[stage["stage_name"]]
            ):
                raise ValueError("Shield primitive does not match its stage summary.")
            for field_name in primitive["nonfinite_values"]:
                if primitive.get(field_name) is not None:
                    raise ValueError(
                        "Non-finite shield evidence must serialize as null."
                    )
        expected_input = stage["output_action_id"]
    if factors["execution_mode"] == "unshielded_operational":
        valid_endpoint = (
            all(stage["bypassed"] and not stage["applied"] for stage in stack["stages"])
            and stack["shielded_action_id"] is None
            and stack["executed_action_id"] == stack["unshielded_action_id"]
        )
    else:
        valid_endpoint = (
            all(not stage["bypassed"] for stage in stack["stages"])
            and stack["shielded_action_id"] == expected_input
            and stack["executed_action_id"] == expected_input
        )
    if not valid_endpoint:
        raise ValueError("Serialized shield endpoint is invalid.")
    failure_class = error_class or resolution["violation"]
    failure = payload["failure"]
    if failure_class is None:
        if failure is not None:
            raise ValueError("Successful trace contains synthetic failure evidence.")
    else:
        expected_message = (
            generation["error_message"]
            if error_class is not None
            else "action_resolution_violation"
        )
        if (
            failure is None
            or failure["failure_class"] != failure_class
            or failure["message"] != expected_message
        ):
            raise ValueError("Serialized failure evidence is inconsistent.")


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["validate_serialized_trace_invariants"]
