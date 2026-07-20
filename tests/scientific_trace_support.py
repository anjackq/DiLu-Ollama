from __future__ import annotations

import dataclasses

from dilu.driver_agent.prompt_modules import build_prompt_artifact
from dilu.runtime.action_resolution import resolve_action
from dilu.runtime.harness_config import ExecutionMode, OutputEnforcement
from dilu.runtime.ollama_scientific_client import OllamaScientificClient
from dilu.runtime.scientific_trace import DecisionTraceRecord, TraceDisposition
from dilu.runtime.shield_stack import execute_shield_stack
from tests.scientific_transport_support import (
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_retry_policy,
    success_payload,
)
from tests.test_scientific_trace import _record


def blocked_record() -> DecisionTraceRecord:
    base = _record()
    payload = success_payload()
    payload["done"] = False
    generation = OllamaScientificClient(
        capabilities=make_capabilities(),
        retry_policy=make_retry_policy(),
        identity_inspector=identity_inspector_for(),
        post=lambda *args, **kwargs: FakeResponse(payload),
        sleep=lambda _: None,
    ).generate(base.generation.request)
    return DecisionTraceRecord(
        context=base.context,
        harness_config=base.harness_config,
        prompt_artifact=base.prompt_artifact,
        generation=generation,
        resolution=None,
        shield_stack=None,
        disposition=TraceDisposition.BLOCKED_BEFORE_EXECUTION,
        decision_latency_ms=generation.latency_ms,
    )


def action_resolution_failure_record() -> DecisionTraceRecord:
    base = _record()
    generation = OllamaScientificClient(
        capabilities=make_capabilities(),
        retry_policy=make_retry_policy(),
        identity_inspector=identity_inspector_for(),
        post=lambda *args, **kwargs: FakeResponse(success_payload("malformed")),
        sleep=lambda _: None,
    ).generate(base.generation.request)
    resolution = resolve_action(
        generation.contract_text or "",
        available_action_ids=base.context.available_action_ids,
    )
    stack = execute_shield_stack(
        scenario=object(),
        proposed_action_id=None,
        fallback_modified_action_id=resolution.final_resolved_action,
        execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
        shield_config=base.harness_config.shield,
    )
    return DecisionTraceRecord(
        context=base.context,
        harness_config=base.harness_config,
        prompt_artifact=base.prompt_artifact,
        generation=generation,
        resolution=resolution,
        shield_stack=stack,
        disposition=TraceDisposition.READY_FOR_ENV_STEP,
        decision_latency_ms=generation.latency_ms,
    )


def backend_schema_record() -> DecisionTraceRecord:
    base = _record()
    condition = dataclasses.replace(
        base.harness_config.condition,
        output_enforcement=OutputEnforcement.BACKEND_SCHEMA,
    )
    config = dataclasses.replace(base.harness_config, condition=condition)
    prompt = build_prompt_artifact(
        condition.policy_content,
        output_enforcement=condition.output_enforcement,
    )
    request = dataclasses.replace(
        base.generation.request,
        messages=(("system", prompt.system_prompt()), ("user", "scenario")),
        output_enforcement=OutputEnforcement.BACKEND_SCHEMA,
    )
    generation = OllamaScientificClient(
        capabilities=make_capabilities(),
        retry_policy=make_retry_policy(),
        identity_inspector=identity_inspector_for(),
        post=lambda *args, **kwargs: FakeResponse(
            success_payload('"Response to user:#### 3"')
        ),
        sleep=lambda _: None,
    ).generate(request)
    resolution = resolve_action(
        generation.contract_text or "",
        available_action_ids=base.context.available_action_ids,
    )
    stack = execute_shield_stack(
        scenario=object(),
        proposed_action_id=resolution.strict_action,
        fallback_modified_action_id=resolution.final_resolved_action,
        execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
        shield_config=config.shield,
    )
    context = dataclasses.replace(
        base.context,
        key=dataclasses.replace(
            base.context.key,
            condition_id=config.condition_id(),
        ),
    )
    return DecisionTraceRecord(
        context=context,
        harness_config=config,
        prompt_artifact=prompt,
        generation=generation,
        resolution=resolution,
        shield_stack=stack,
        disposition=TraceDisposition.READY_FOR_ENV_STEP,
        decision_latency_ms=generation.latency_ms,
    )


__all__ = [
    "action_resolution_failure_record",
    "backend_schema_record",
    "blocked_record",
]
