import dataclasses
import unittest
from typing import Any

import requests

from dilu.runtime.action_resolution import ActionSyntaxStatus
from dilu.runtime.harness_config import OutputEnforcement
from dilu.runtime.ollama_scientific_client import (
    OllamaScientificClient,
    PreAcceptTransportUnavailable,
    ScientificGenerationAbort,
    ScientificGenerationContext,
)
from dilu.runtime.runtime_failures import RuntimeFailureClass
from tests.scientific_transport_support import (
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_retry_policy,
    success_payload,
)
from tests.test_scientific_driver_action_resolution import (
    _forbid_legacy_helpers,
    _scientific_agent_response,
)


def _transport_agent(
    post: Any,
    *,
    output_enforcement: OutputEnforcement = OutputEnforcement.PROMPT_ONLY,
) -> Any:
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    agent = _scientific_agent_response("")
    del agent._invoke_response_with_diagnostics
    agent.oai_api_type = "ollama"
    agent.ollama_model_name = "qwen3:0.6b"
    agent.ollama_chat_url = "http://127.0.0.1:11434/api/chat"
    agent.ollama_use_native_chat_configured = "true"
    agent.ollama_use_native_chat = True
    agent.ollama_native_chat_resolution_reason = "scientific_explicit"
    agent.ollama_think_mode = "no_think"
    agent.temperature = 0.0
    agent.use_streaming = True
    agent.scientific_harness_config = dataclasses.replace(
        agent.scientific_harness_config,
        condition=dataclasses.replace(
            agent.scientific_harness_config.condition,
            output_enforcement=output_enforcement,
        ),
    )
    agent.scientific_transport_client = OllamaScientificClient(
        capabilities=make_capabilities(),
        retry_policy=make_retry_policy(),
        post=post,
        sleep=sleep,
        clock=clock,
        identity_inspector=identity_inspector_for(),
    )
    agent.scientific_generation_context = ScientificGenerationContext(
        request_id="req-pair-001-step-000",
        model_digest=make_capabilities().model_digest,
        generation_seed=1058710636,
    )
    return agent


class ScientificDriverTransportTests(unittest.TestCase):
    def test_schema_raw_json_is_retained_and_contract_text_enters_resolver(
        self,
    ) -> None:
        raw_json = '"Response to user:#### 3"'
        agent = _transport_agent(
            lambda *args, **kwargs: FakeResponse(success_payload(raw_json)),
            output_enforcement=OutputEnforcement.BACKEND_SCHEMA,
        )
        _forbid_legacy_helpers(agent)

        action, returned_text, _, _ = agent.few_shot_decision(
            fewshot_messages=[], fewshot_answers=[]
        )

        self.assertEqual((action, returned_text), (3, "Response to user:#### 3"))
        self.assertEqual(agent.last_generation_result.raw_response, raw_json)
        self.assertEqual(
            agent.last_generation_result.contract_text,
            "Response to user:#### 3",
        )
        self.assertEqual(
            agent.last_action_resolution.syntax_status,
            ActionSyntaxStatus.STRICT_VALID,
        )
        self.assertEqual(
            agent.last_decision_meta["scientific_request_id"],
            "req-pair-001-step-000",
        )
        self.assertEqual(agent.last_decision_meta["generation_seed"], 1058710636)
        self.assertEqual(
            tuple(
                item["phase"]
                for item in agent.last_decision_meta["scientific_identity_checks"]
            ),
            ("pre", "post"),
        )
        self.assertEqual(
            agent.last_decision_meta["scientific_backend_total_duration_ns"],
            120_000_000,
        )

    def test_timeout_and_empty_output_allow_only_fixed_idle_fallback(self) -> None:
        cases = (
            (
                "timeout",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    requests.Timeout("timeout")
                ),
                "",
                ActionSyntaxStatus.TIMEOUT,
                RuntimeFailureClass.GENERATION_TIMEOUT,
            ),
            (
                "empty",
                lambda *args, **kwargs: FakeResponse(
                    success_payload(" \t", eval_count=0)
                ),
                " \t",
                ActionSyntaxStatus.EMPTY,
                RuntimeFailureClass.MODEL_EMPTY_OUTPUT,
            ),
        )
        for name, post, expected_raw, expected_status, expected_failure in cases:
            with self.subTest(name=name):
                agent = _transport_agent(post)
                _forbid_legacy_helpers(agent)
                action, returned_text, _, _ = agent.few_shot_decision(
                    fewshot_messages=[], fewshot_answers=[]
                )
                self.assertEqual((action, returned_text), (1, expected_raw))
                self.assertEqual(
                    agent.last_action_resolution.syntax_status, expected_status
                )
                self.assertEqual(
                    agent.last_generation_result.error_class,
                    expected_failure,
                )

    def test_nonoperational_transport_failures_abort_without_action_evidence(
        self,
    ) -> None:
        failures = (
            (
                "schema_rejection",
                lambda *args, **kwargs: FakeResponse(
                    {"error": "invalid format schema"}, status_code=400
                ),
                OutputEnforcement.BACKEND_SCHEMA,
                RuntimeFailureClass.SCHEMA_REJECTION,
            ),
            (
                "transport_drift",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    requests.ConnectionError("accept boundary unknown")
                ),
                OutputEnforcement.PROMPT_ONLY,
                RuntimeFailureClass.TRANSPORT_DRIFT,
            ),
            (
                "retry_exhausted",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    PreAcceptTransportUnavailable("not accepted")
                ),
                OutputEnforcement.PROMPT_ONLY,
                RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT,
            ),
        )
        for name, post, enforcement, expected_failure in failures:
            with self.subTest(name=name):
                agent = _transport_agent(post, output_enforcement=enforcement)
                with self.assertRaises(ScientificGenerationAbort) as context:
                    agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])
                self.assertEqual(
                    context.exception.result.error_class,
                    expected_failure,
                )
                self.assertIsNone(agent.last_action_resolution)

    def test_syntax_and_availability_failures_never_retry_generation(self) -> None:
        cases = (
            ("Action id: 3", [0, 1, 2, 3, 4], ActionSyntaxStatus.INVALID),
            ("Response to user:#### 3", [1], ActionSyntaxStatus.STRICT_VALID),
        )
        for raw_response, available_actions, expected_status in cases:
            with self.subTest(raw_response=raw_response):
                calls: list[int] = []

                def post(*args, **kwargs):
                    calls.append(1)
                    return FakeResponse(success_payload(raw_response))

                agent = _transport_agent(post)
                agent.sce.available_action_ids = lambda: available_actions
                action, _, _, _ = agent.few_shot_decision(
                    fewshot_messages=[], fewshot_answers=[]
                )

                self.assertEqual(action, 1)
                self.assertEqual(calls, [1])
                self.assertEqual(len(agent.last_generation_result.attempts), 1)
                self.assertEqual(
                    agent.last_action_resolution.syntax_status,
                    expected_status,
                )

    def test_generic_timeout_cannot_reuse_previous_scientific_evidence(self) -> None:
        agent = _transport_agent(
            lambda *args, **kwargs: FakeResponse(success_payload())
        )
        agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])
        agent.scientific_generation_context = dataclasses.replace(
            agent.scientific_generation_context,
            request_id="req-pair-001-step-001",
            generation_seed=1058710637,
        )

        def generic_timeout(*_: Any, **__: Any) -> Any:
            raise TimeoutError("message conversion failed")

        agent._to_ollama_messages = generic_timeout
        with self.assertRaises(TimeoutError):
            agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertIsNone(agent.last_generation_result)
        self.assertIsNone(agent.last_action_resolution)


if __name__ == "__main__":
    unittest.main()
