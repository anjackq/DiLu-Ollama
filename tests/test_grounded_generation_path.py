"""Task 3: thread per-decision action availability into the O2 generation path.

Covers the single request-construction site
(`DriverAgent._scientific_generation_request` / `_scientific_grounded_available_action_ids`
in dilu/driver_agent/driverAgent.py) that must pass `available_action_ids` into
`GenerationRequest` when, and only when, the condition is O2
(`OutputEnforcement.BACKEND_SCHEMA_GROUNDED`).
"""

from __future__ import annotations

import dataclasses
import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from dilu.runtime._scientific_contract_validation import (
    validate_output_contract_semantics,
)
from dilu.runtime.harness_config import OutputEnforcement
from dilu.runtime.ollama_scientific_client import OllamaScientificClient
from dilu.runtime.runtime_failures import (
    ProtocolInvariantCode,
    RuntimeFailureClass,
    RuntimeProtocolError,
)
from dilu.runtime.scientific_transport_types import SCHEMA_MECHANISM_GROUNDED
from tests.scientific_transport_support import (
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_request,
    make_retry_policy,
    success_payload,
)
from tests.test_scientific_driver_action_resolution import _forbid_legacy_helpers
from tests.test_scientific_driver_transport import _transport_agent


def _advance_generation_context(agent, *, request_id: str, seed: int) -> None:
    agent.scientific_generation_context = dataclasses.replace(
        agent.scientific_generation_context,
        request_id=request_id,
        generation_seed=seed,
    )


class GroundedGenerationPathTests(unittest.TestCase):
    def test_o2_decodes_the_json_string_returned_by_native_schema(self) -> None:
        raw_json_string = '"Response to user:#### 1"'
        capabilities = dataclasses.replace(
            make_capabilities(), schema_mechanism=SCHEMA_MECHANISM_GROUNDED
        )
        client = OllamaScientificClient(
            capabilities=capabilities,
            retry_policy=make_retry_policy(),
            identity_inspector=identity_inspector_for(),
            post=lambda *args, **kwargs: FakeResponse(
                success_payload(raw_json_string)
            ),
        )

        result = client.generate(
            make_request(
                OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
                available_action_ids=(1, 3),
            )
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.raw_response, raw_json_string)
        self.assertEqual(result.contract_text, "Response to user:#### 1")

    def test_o2_rejects_a_json_string_outside_the_grounded_enum(self) -> None:
        capabilities = dataclasses.replace(
            make_capabilities(), schema_mechanism=SCHEMA_MECHANISM_GROUNDED
        )
        client = OllamaScientificClient(
            capabilities=capabilities,
            retry_policy=make_retry_policy(),
            identity_inspector=identity_inspector_for(),
            post=lambda *args, **kwargs: FakeResponse(
                success_payload('"Response to user:#### 4"')
            ),
        )

        result = client.generate(
            make_request(
                OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
                available_action_ids=(1, 3),
            )
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.error_class, RuntimeFailureClass.SCHEMA_REJECTION)

    def test_o2_enum_follows_current_decision_availability(self) -> None:
        captured_payloads: list[dict] = []

        def post(*args, **kwargs):
            captured_payloads.append(kwargs["json"])
            return FakeResponse(success_payload('"Response to user:#### 1"'))

        agent = _transport_agent(
            post, output_enforcement=OutputEnforcement.BACKEND_SCHEMA_GROUNDED
        )
        _forbid_legacy_helpers(agent)

        agent.sce.available_action_ids = lambda: [0, 1, 2, 3, 4]
        agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        _advance_generation_context(
            agent, request_id="req-pair-001-step-001", seed=1058710637
        )
        agent.sce.available_action_ids = lambda: [1, 3]
        agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(len(captured_payloads), 2)
        first_enum = captured_payloads[0]["format"]["enum"]
        second_enum = captured_payloads[1]["format"]["enum"]
        self.assertEqual(len(first_enum), 5)
        self.assertEqual(
            second_enum,
            ["Response to user:#### 1", "Response to user:#### 3"],
        )

    def test_o1_enum_stays_static_regardless_of_availability(self) -> None:
        captured_payloads: list[dict] = []

        def post(*args, **kwargs):
            captured_payloads.append(kwargs["json"])
            # Native backend-schema (O1) responses are JSON-encoded strings;
            # the transport JSON-decodes them into contract text.
            return FakeResponse(success_payload('"Response to user:#### 1"'))

        agent = _transport_agent(
            post, output_enforcement=OutputEnforcement.BACKEND_SCHEMA
        )
        _forbid_legacy_helpers(agent)

        agent.sce.available_action_ids = lambda: [0, 1, 2, 3, 4]
        agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        _advance_generation_context(
            agent, request_id="req-pair-001-step-001", seed=1058710637
        )
        agent.sce.available_action_ids = lambda: [1, 3]
        agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(len(captured_payloads), 2)
        static_format = {
            "type": "string",
            "enum": [f"Response to user:#### {i}" for i in range(5)],
        }
        for payload in captured_payloads:
            self.assertEqual(payload["format"], static_format)

    def test_o2_fails_closed_without_resolved_decision_availability(self) -> None:
        agent = _transport_agent(
            lambda *args, **kwargs: FakeResponse(success_payload()),
            output_enforcement=OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
        )
        _forbid_legacy_helpers(agent)
        # Simulate a decision context that never resolved availability, i.e.
        # `few_shot_decision`'s own bookkeeping never ran (the availability
        # attribute was never populated for this generation call).
        agent.last_scientific_available_action_ids = None
        messages = [
            SystemMessage(content="contract"),
            HumanMessage(content="scenario"),
        ]

        with self.assertRaises(RuntimeProtocolError) as ctx:
            agent._invoke_response_with_diagnostics(
                messages, max_output_tokens_override=None
            )

        self.assertEqual(
            ctx.exception.invariant_code,
            ProtocolInvariantCode.ACTION_AVAILABILITY_UNRESOLVED,
        )


class GroundedContractValidationTests(unittest.TestCase):
    """`_scientific_contract_validation.py` predates O2 and only recognized
    `prompt_only`/`backend_schema`; any `backend_schema_grounded` attempt
    (success or failure) raised a bare `ValueError("Unknown output-enforcement
    mode.")` before ever reaching this task's availability threading. Fixing
    it was necessary to make any O2 decision (see
    `GroundedGenerationPathTests` above) survive `GenerationResult`
    construction at all.
    """

    def test_grounded_success_binds_json_decoded_contract_text(self) -> None:
        validate_output_contract_semantics(
            output_enforcement="backend_schema_grounded",
            think_mode="no_think",
            error_class=None,
            raw_output='"Response to user:#### 2"',
            contract_text="Response to user:#### 2",
            thinking_output="",
        )

    def test_grounded_success_rejects_noncanonical_or_mismatched_text(self) -> None:
        with self.assertRaises(ValueError):
            validate_output_contract_semantics(
                output_enforcement="backend_schema_grounded",
                think_mode="no_think",
                error_class=None,
                raw_output="Response to user:#### 2",
                contract_text="Response to user:#### 3",
                thinking_output="",
            )
        with self.assertRaises(ValueError):
            validate_output_contract_semantics(
                output_enforcement="backend_schema_grounded",
                think_mode="no_think",
                error_class=None,
                raw_output="not a canonical action",
                contract_text="not a canonical action",
                thinking_output="",
            )

    def test_grounded_failure_attempts_are_not_unknown_enforcement(self) -> None:
        # Failed attempts (any error_class) must short-circuit before the
        # contract-text checks, exactly like backend_schema already does.
        validate_output_contract_semantics(
            output_enforcement="backend_schema_grounded",
            think_mode="no_think",
            error_class="transport_drift",
            raw_output=None,
            contract_text=None,
            thinking_output="",
        )


if __name__ == "__main__":
    unittest.main()
