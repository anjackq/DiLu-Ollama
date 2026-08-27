from __future__ import annotations

import unittest

from dilu.runtime.harness_config import (
    ConditionSpec,
    ExecutionMode,
    OutputEnforcement,
    PolicyContent,
    ThinkMode,
)
from dilu.runtime.scientific_transport_types import (
    GenerationRequest,
    NativeGenerationOptions,
    build_native_chat_payload,
    grounded_action_text_schema,
)


MODEL_DIGEST = "sha256:" + "a" * 64


def make_request(
    *,
    output_enforcement: OutputEnforcement = OutputEnforcement.PROMPT_ONLY,
    available_action_ids: tuple[int, ...] | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        model_tag="qwen3:0.6b",
        model_digest=MODEL_DIGEST,
        request_id="req-grounded-schema-000",
        messages=(("system", "contract"), ("user", "scenario")),
        native_endpoint="http://127.0.0.1:11434/api/chat",
        options=NativeGenerationOptions(
            seed=1058710636,
            temperature=0.0,
            num_ctx=4096,
            num_predict=64,
        ),
        output_enforcement=output_enforcement,
        think_mode=ThinkMode.NO_THINK,
        timeout_sec=60.0,
        available_action_ids=available_action_ids,
    )


class GroundedActionTextSchemaTests(unittest.TestCase):
    def test_grounded_schema_restricts_enum(self) -> None:
        schema = grounded_action_text_schema((0, 1, 3))
        self.assertEqual(
            schema,
            {
                "type": "string",
                "enum": [
                    "Response to user:#### 0",
                    "Response to user:#### 1",
                    "Response to user:#### 3",
                ],
            },
        )

    def test_grounded_schema_rejects_empty_and_unknown(self) -> None:
        with self.assertRaises(ValueError):
            grounded_action_text_schema(())
        with self.assertRaises(ValueError):
            grounded_action_text_schema((0, 7))


class GroundedPayloadTests(unittest.TestCase):
    def test_payload_uses_grounded_enum(self) -> None:
        request = make_request(
            output_enforcement=OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
            available_action_ids=(1, 4),
        )
        payload = build_native_chat_payload(request)
        self.assertEqual(
            payload["format"]["enum"],
            ["Response to user:#### 1", "Response to user:#### 4"],
        )

    def test_payload_grounded_without_ids_fails(self) -> None:
        request = make_request(
            output_enforcement=OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
            available_action_ids=None,
        )
        with self.assertRaises(ValueError):
            build_native_chat_payload(request)

    def test_backend_schema_payload_unchanged(self) -> None:
        request = make_request(
            output_enforcement=OutputEnforcement.BACKEND_SCHEMA,
            available_action_ids=None,
        )
        payload = build_native_chat_payload(request)
        self.assertEqual(len(payload["format"]["enum"]), 5)


class ConditionIdGroundedMappingTests(unittest.TestCase):
    def test_condition_id_maps_grounded_output_enforcement_to_digit_two(self) -> None:
        unshielded = ConditionSpec(
            policy_content=PolicyContent.MODULAR_HARNESS,
            output_enforcement=OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
            execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
        )
        shielded = ConditionSpec(
            policy_content=PolicyContent.MODULAR_HARNESS,
            output_enforcement=OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
            execution_mode=ExecutionMode.SHIELDED,
        )

        self.assertEqual(unshielded.condition_id(), "c120")
        self.assertEqual(shielded.condition_id(), "c121")


if __name__ == "__main__":
    unittest.main()
