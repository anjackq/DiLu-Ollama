from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jsonschema
import requests

from dilu.driver_agent.prompt_modules import build_prompt_artifact
from dilu.runtime.action_resolution import resolve_action
from dilu.runtime.generation_seed import (
    post_divergence_generation_seed,
    primary_snapshot_generation_seed,
)
from dilu.runtime.harness_config import ExecutionMode
from dilu.runtime.ollama_scientific_client import OllamaScientificClient
from dilu.runtime.scientific_trace import (
    DecisionTraceContext,
    DecisionTraceKey,
    DecisionTraceRecord,
    GenerationSeedScope,
    ScientificTraceWriteError,
    ScientificTraceWriter,
    TraceDisposition,
    append_trace_before_step,
    trace_schema_path,
)
from dilu.runtime.shield_stack import execute_shield_stack
from tests.scientific_transport_support import (
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_request,
    make_retry_policy,
    success_payload,
)
from tests.test_scientific_driver_action_resolution import _scientific_config


def _generation(*, decision_index: int, timeout: bool = False):
    config = _scientific_config()
    prompt = build_prompt_artifact(
        config.condition.policy_content,
        output_enforcement=config.condition.output_enforcement,
    )
    if decision_index == 0:
        seed = primary_snapshot_generation_seed(
            config.transport.generation_seed_master,
            make_capabilities().model_digest,
            "pair-001",
            "snapshot-001",
            0,
        )
    else:
        seed = post_divergence_generation_seed(
            config.transport.generation_seed_master,
            make_capabilities().model_digest,
            "case-001",
            decision_index,
            0,
        )
    request = make_request()
    request = dataclasses.replace(
        request,
        request_id=f"req-case-001-step-{decision_index:03d}",
        messages=(("system", prompt.system_prompt()), ("user", "scenario")),
        options=dataclasses.replace(request.options, seed=seed),
    )
    post = (
        (lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("timeout")))
        if timeout
        else (lambda *args, **kwargs: FakeResponse(success_payload()))
    )
    return OllamaScientificClient(
        capabilities=make_capabilities(),
        retry_policy=make_retry_policy(),
        identity_inspector=identity_inspector_for(),
        post=post,
        sleep=lambda _: None,
    ).generate(request)


def _record(
    decision_index: int = 0,
    *,
    timeout: bool = False,
) -> DecisionTraceRecord:
    config = _scientific_config()
    generation = _generation(decision_index=decision_index, timeout=timeout)
    raw_for_resolution = generation.contract_text or generation.raw_response or ""
    resolution = resolve_action(
        raw_for_resolution,
        available_action_ids=(0, 1, 2, 3, 4),
        timed_out=timeout,
    )
    shield = execute_shield_stack(
        scenario=object(),
        proposed_action_id=resolution.strict_action,
        fallback_modified_action_id=resolution.final_resolved_action,
        execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
        shield_config=config.shield,
    )
    key = DecisionTraceKey(
        campaign_id="campaign-001",
        episode_attempt_id="episode-attempt-001",
        condition_id=config.condition_id(),
        case_id="case-001",
        pair_id="pair-001",
        template_id="template-001",
        replicate_id=0,
        decision_index=decision_index,
        env_step_index=decision_index,
    )
    context = DecisionTraceContext(
        key=key,
        benchmark_fingerprint="sha256:" + "b" * 64,
        code_revision="git:" + "d" * 40,
        simulator_seed=7,
        generation_seed_master=config.transport.generation_seed_master,
        generation_seed_scope=(
            GenerationSeedScope.PRIMARY_SNAPSHOT
            if decision_index == 0
            else GenerationSeedScope.POST_DIVERGENCE
        ),
        decision_snapshot_id="snapshot-001" if decision_index == 0 else None,
        available_action_ids=(0, 1, 2, 3, 4),
        event_phase="pre_step",
        applied_event_ids=(),
    )
    prompt = build_prompt_artifact(
        config.condition.policy_content,
        output_enforcement=config.condition.output_enforcement,
    )
    return DecisionTraceRecord(
        context=context,
        harness_config=config,
        prompt_artifact=prompt,
        generation=generation,
        resolution=resolution,
        shield_stack=shield,
        disposition=TraceDisposition.READY_FOR_ENV_STEP,
        decision_latency_ms=generation.latency_ms,
    )


class DecisionTraceRecordTests(unittest.TestCase):
    def test_record_is_frozen_and_schema_valid(self) -> None:
        record = _record()
        payload = record.to_dict()
        schema = json.loads(trace_schema_path().read_text(encoding="utf-8"))

        jsonschema.validate(payload, schema)
        self.assertEqual(payload["generation"]["raw_output"], "Response to user:#### 3")
        self.assertEqual(payload["action_resolution"]["strict_action"], 3)
        self.assertEqual(payload["shield_stack"]["executed_action_id"], 3)
        self.assertEqual(len(payload["generation"]["identity_checks"]), 2)
        self.assertEqual(len(payload["generation"]["attempts"]), 1)
        self.assertEqual(
            payload["generation"]["transport_evidence"],
            {
                "requested_profile": "ollama_native_chat",
                "effective_profile": "ollama_native_chat",
                "requested_think_mode": "no_think",
                "effective_think_mode": "no_think",
                "capability_model_tag": "qwen3:0.6b",
                "capability_model_digest": "sha256:" + "a" * 64,
                "capability_native_endpoint": "http://127.0.0.1:11434/api/chat",
                "capability_supported_think_modes": ["no_think"],
                "seed_verified": True,
                "schema_verified": True,
                "capability_probe_id": "s1-transport-probe-placeholder",
                "capability_artifact_sha256": "sha256:" + "c" * 64,
                "capability_snapshot_sha256": mock.ANY,
                "schema_mechanism": "ollama_api_chat_json_string_enum_v1",
            },
        )
        self.assertRegex(
            payload["generation"]["transport_evidence"]["capability_snapshot_sha256"],
            r"^sha256:[0-9a-f]{64}$",
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.decision_latency_ms = 1.0

    def test_cross_stage_or_seed_tampering_is_rejected(self) -> None:
        record = _record()
        invalid = (
            {
                "shield_stack": dataclasses.replace(
                    record.shield_stack, proposed_action_id=2
                )
            },
            {
                "context": dataclasses.replace(
                    record.context,
                    generation_seed_master=record.context.generation_seed_master + 1,
                )
            },
        )
        for replacement in invalid:
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    dataclasses.replace(record, **replacement)

    def test_timeout_never_serializes_synthetic_idle_as_model_output(self) -> None:
        record = _record(timeout=True)
        payload = record.to_dict()
        encoded = json.dumps(payload, sort_keys=True)

        self.assertIsNone(payload["generation"]["raw_output"])
        self.assertEqual(payload["action_resolution"]["parser_input"], "")
        self.assertIsNone(payload["shield_stack"]["proposed_action_id"])
        self.assertEqual(payload["shield_stack"]["fallback_modified_action_id"], 1)
        self.assertNotIn("Response to user:#### 1", encoded)


class ScientificTraceWriterTests(unittest.TestCase):
    def test_writer_appends_canonical_durable_jsonl_and_valid_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision_traces.jsonl"
            writer = ScientificTraceWriter(path, artifact_root=Path(tmp))
            first = writer.append(_record(0))
            second = writer.append(_record(1))
            resumed = ScientificTraceWriter(path, artifact_root=Path(tmp), resume=True)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual((first.line_number, second.line_number), (1, 2))
            self.assertEqual(first.relative_path, "decision_traces.jsonl")
            self.assertEqual(resumed.next_line_number, 3)
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_index_gap_and_truncated_resume_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision_traces.jsonl"
            writer = ScientificTraceWriter(path, artifact_root=Path(tmp))
            writer.append(_record(0))
            with self.assertRaises(ScientificTraceWriteError):
                writer.append(_record(2))

            with path.open("ab") as stream:
                stream.write(b'{"truncated":')
            with self.assertRaises(ScientificTraceWriteError):
                ScientificTraceWriter(path, artifact_root=Path(tmp), resume=True)

    def test_fsync_failure_prevents_environment_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ScientificTraceWriter(
                Path(tmp) / "decision_traces.jsonl",
                artifact_root=Path(tmp),
            )
            calls: list[str] = []

            def step(action: int) -> str:
                calls.append(f"step:{action}")
                return "stepped"

            with mock.patch.object(os, "fsync", side_effect=OSError("disk failure")):
                with self.assertRaises(ScientificTraceWriteError):
                    append_trace_before_step(writer, _record(), step)
            self.assertEqual(calls, [])

    def test_successful_order_is_append_fsync_then_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ScientificTraceWriter(
                Path(tmp) / "decision_traces.jsonl",
                artifact_root=Path(tmp),
            )
            calls: list[str] = []
            original_append = writer.append

            def append(record: DecisionTraceRecord):
                calls.append("append")
                reference = original_append(record)
                calls.append("fsynced")
                return reference

            writer.append = append

            def step(action: int) -> str:
                calls.append(f"step:{action}")
                return "stepped"

            reference, result = append_trace_before_step(writer, _record(), step)
            self.assertEqual(calls, ["append", "fsynced", "step:3"])
            self.assertEqual(reference.line_number, 1)
            self.assertEqual(result, "stepped")


if __name__ == "__main__":
    unittest.main()
