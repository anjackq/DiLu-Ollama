"""Coherent-rewrite rejection for completed runtime-lock evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from dilu.runtime._minimal_factorial_manifest import serialize_frozen_campaign
from dilu.runtime._runtime_lock_authoring_support import canonical_bytes
from dilu.runtime._runtime_lock_authoring_workflow import (
    build_capabilities,
    build_lock_plans,
    build_schedules,
)
from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest
from dilu.runtime.ollama_transport import OllamaModelIdentity
from tests.test_runtime_lock_authoring import (
    MANIFEST_PATH,
    ROOT,
    NativeFakes,
    fake_snapshot,
    run_authoring,
)

Mutation = Callable[[list[dict[str, object]]], None]


def _refresh_payload_evidence(records: list[dict[str, object]]) -> None:
    for record in records:
        payload = record["payload"]
        if not isinstance(payload, dict):
            raise AssertionError("test fixture payload must be a mapping")
        payload_bytes = canonical_bytes(payload)
        record["request_body"] = payload_bytes.decode("utf-8")
        record["payload_sha256"] = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()


def _coherently_rewrite_campaign(output: Path, mutation: Mutation) -> None:
    manifest = load_experiment_manifest(MANIFEST_PATH)
    case_set = json.loads((ROOT / manifest.case_path).read_text(encoding="utf-8"))
    snapshot = fake_snapshot()
    preflight_path = output / "s1" / "model_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    records = preflight["records"]
    if not isinstance(records, list):
        raise AssertionError("test fixture records must be a list")
    mutation(records)
    _refresh_payload_evidence(records)
    preflight_bytes = canonical_bytes(preflight)
    preflight_path.write_bytes(preflight_bytes)
    artifact_hash = "sha256:" + hashlib.sha256(preflight_bytes).hexdigest()

    bindings: dict[str, OllamaModelIdentity] = {}
    for model in manifest.models:
        record = next(item for item in records if item["model_slot"] == model.slot)
        identity = record["identity_before"]
        if not isinstance(identity, dict):
            raise AssertionError("test fixture identity must be a mapping")
        bindings[model.slot] = OllamaModelIdentity(
            identity["model_tag"],
            identity["model_digest"],
        )
    capabilities = build_capabilities(manifest, bindings, artifact_hash)
    smoke, union = build_schedules(manifest, case_set, snapshot, bindings)
    for lock in build_lock_plans(output, manifest, smoke, capabilities):
        lock.runtime_path.write_bytes(lock.runtime_bytes)
        lock.authorization_path.write_bytes(lock.authorization_bytes)
    (output / "smoke" / "campaign_manifest.json").write_bytes(
        serialize_frozen_campaign(manifest, snapshot, smoke, case_set)
    )
    (output / "llm_campaign" / "campaign_manifest.json").write_bytes(
        serialize_frozen_campaign(manifest, snapshot, union, case_set)
    )
    (output / "llm_campaign" / "union_schedule.json").write_bytes(
        canonical_bytes([row.to_payload() for row in union])
    )


def _request(records: list[dict[str, object]], index: int) -> dict[str, object]:
    request = records[index]["request"]
    if not isinstance(request, dict):
        raise AssertionError("test fixture request must be a mapping")
    return request


def _payload(records: list[dict[str, object]], index: int) -> dict[str, object]:
    payload = records[index]["payload"]
    if not isinstance(payload, dict):
        raise AssertionError("test fixture payload must be a mapping")
    return payload


def _mutate_endpoint(records: list[dict[str, object]]) -> None:
    for index in range(3):
        _request(records, index)["native_endpoint"] = (
            "http://localhost:11434/v1/chat/completions"
        )


def _mutate_prompt(records: list[dict[str, object]]) -> None:
    messages = [{"role": "user", "content": "Return a different action."}]
    for index in range(3):
        _request(records, index)["messages"] = messages
        _payload(records, index)["messages"] = messages


def _mutate_option(
    records: list[dict[str, object]],
    field: str,
    value: int | float,
) -> None:
    for index in range(3):
        request_options = _request(records, index)["options"]
        payload_options = _payload(records, index)["options"]
        if not isinstance(request_options, dict) or not isinstance(
            payload_options, dict
        ):
            raise AssertionError("test fixture options must be mappings")
        request_options[field] = value
        payload_options[field] = value


def _mutate_seed(records: list[dict[str, object]]) -> None:
    _mutate_option(records, "seed", 99)


def _mutate_temperature(records: list[dict[str, object]]) -> None:
    _mutate_option(records, "temperature", 0.75)


def _mutate_context(records: list[dict[str, object]]) -> None:
    _mutate_option(records, "num_ctx", 2048)


def _mutate_output_limit(records: list[dict[str, object]]) -> None:
    _mutate_option(records, "num_predict", 64)


def _mutate_schema(records: list[dict[str, object]]) -> None:
    _payload(records, 2)["format"] = {"type": "integer"}


def _mutate_think_mode(records: list[dict[str, object]]) -> None:
    for index in range(3):
        _request(records, index)["think_mode"] = "think"
        _payload(records, index)["think"] = True


def _mutate_timeout(records: list[dict[str, object]]) -> None:
    for index in range(3):
        _request(records, index)["timeout_sec"] = 999.0


def _mutate_probe_kind(records: list[dict[str, object]]) -> None:
    _request(records, 0)["request_id"] = "s1-qwen_06b-fallback"


def _mutate_order(records: list[dict[str, object]]) -> None:
    records[0], records[1] = records[1], records[0]


def _mutate_enforcement(records: list[dict[str, object]]) -> None:
    _request(records, 0)["output_enforcement"] = "backend_schema"
    _payload(records, 0)["format"] = {
        "type": "string",
        "enum": [f"Response to user:#### {index}" for index in range(5)],
    }


def _mutate_model_binding(records: list[dict[str, object]]) -> None:
    new_tag = "qwen3:0.6b-shadow"
    new_digest = "sha256:" + "c" * 64
    for record in records[:3]:
        record["identity_before"] = {
            "model_tag": new_tag,
            "model_digest": new_digest,
        }
        record["identity_after"] = {
            "model_tag": new_tag,
            "model_digest": new_digest,
        }
        _request(records, records.index(record))["model_tag"] = new_tag
        _request(records, records.index(record))["model_digest"] = new_digest
        payload = record["payload"]
        if not isinstance(payload, dict):
            raise AssertionError("test fixture payload must be a mapping")
        payload["model"] = new_tag


def _response_body(
    records: list[dict[str, object]],
    index: int,
) -> dict[str, object]:
    serialized = records[index]["response_body"]
    if not isinstance(serialized, str):
        raise AssertionError("test fixture response body must be text")
    body = json.loads(serialized)
    if not isinstance(body, dict):
        raise AssertionError("test fixture response body must be a mapping")
    return body


def _replace_response_body(
    records: list[dict[str, object]],
    index: int,
    body: dict[str, object],
) -> None:
    records[index]["response_body"] = canonical_bytes(body).decode("utf-8")


def _mutate_raw_response(records: list[dict[str, object]]) -> None:
    for index in range(2):
        records[index]["raw_response"] = "noncanonical response text"


def _mutate_body_content(records: list[dict[str, object]]) -> None:
    for index in range(2):
        body = _response_body(records, index)
        message = body["message"]
        if not isinstance(message, dict):
            raise AssertionError("test fixture message must be a mapping")
        message["content"] = "noncanonical response text"
        _replace_response_body(records, index, body)


def _mutate_decoded_action(records: list[dict[str, object]]) -> None:
    for index in range(2):
        records[index]["canonical_action"] = 4


def _coherently_mutate_action_to_zero(records: list[dict[str, object]]) -> None:
    action_text = "Response to user:#### 0"
    for index in range(3):
        body = _response_body(records, index)
        message = body["message"]
        if not isinstance(message, dict):
            raise AssertionError("test fixture message must be a mapping")
        raw_response = action_text if index < 2 else json.dumps(action_text)
        message["content"] = raw_response
        _replace_response_body(records, index, body)
        records[index]["raw_response"] = raw_response
        records[index]["canonical_action"] = 0


def _mutate_stop_reason(records: list[dict[str, object]]) -> None:
    for index in range(2):
        records[index]["stop_reason"] = "length"


def _mutate_tokens(records: list[dict[str, object]]) -> None:
    for index in range(2):
        records[index]["prompt_tokens"] = 20
        records[index]["completion_tokens"] = 8
        records[index]["total_tokens"] = 28


def _mutate_timing(records: list[dict[str, object]]) -> None:
    for index in range(2):
        timing = records[index]["backend_timing"]
        if not isinstance(timing, dict):
            raise AssertionError("test fixture timing must be a mapping")
        timing["total_duration_ns"] = 999_000_000


class ExistingRequestPlanTests(unittest.TestCase):
    def test_coherent_request_plan_rewrites_reject_without_posts(self) -> None:
        mutations = {
            "endpoint_fallback": _mutate_endpoint,
            "prompt": _mutate_prompt,
            "option_seed": _mutate_seed,
            "option_temperature": _mutate_temperature,
            "option_context": _mutate_context,
            "option_output_limit": _mutate_output_limit,
            "schema": _mutate_schema,
            "think_mode": _mutate_think_mode,
            "timeout": _mutate_timeout,
            "probe_kind": _mutate_probe_kind,
            "record_order": _mutate_order,
            "enforcement": _mutate_enforcement,
            "model_binding": _mutate_model_binding,
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "results"
                run_authoring(output, NativeFakes())
                _coherently_rewrite_campaign(output, mutation)
                rerun_fakes = NativeFakes()

                with self.assertRaises(ValueError):
                    run_authoring(output, rerun_fakes)

                self.assertEqual(rerun_fakes.post_calls, [])

    def test_coherent_response_evidence_rewrites_reject_without_posts(self) -> None:
        mutations = {
            "raw_response": _mutate_raw_response,
            "response_body_content": _mutate_body_content,
            "decoded_action": _mutate_decoded_action,
            "coherent_action_zero": _coherently_mutate_action_to_zero,
            "stop_reason": _mutate_stop_reason,
            "tokens": _mutate_tokens,
            "timing": _mutate_timing,
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "results"
                run_authoring(output, NativeFakes())
                _coherently_rewrite_campaign(output, mutation)
                rerun_fakes = NativeFakes()

                with self.assertRaises(ValueError):
                    run_authoring(output, rerun_fakes)

                self.assertEqual(rerun_fakes.post_calls, [])


if __name__ == "__main__":
    unittest.main()
