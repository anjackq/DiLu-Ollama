import unittest
from typing import Any

from dilu.driver_agent.driverAgent import DriverAgent
from dilu.runtime.action_resolution import ActionSyntaxStatus
from dilu.runtime.harness_config import (
    ConditionSpec,
    ExecutionMode,
    FallbackPolicy,
    HarnessConfig,
    OutputEnforcement,
    ParserMode,
    PolicyContent,
    ResolverMode,
    RetryPolicy,
    ShieldConfig,
    ThinkMode,
    TraceLevel,
    TransportConfig,
    TransportProfile,
)
from dilu.runtime.runtime_failures import (
    ProtocolInvariantCode,
    RuntimeProtocolError,
)
from dilu.scenario.envScenario import EnvScenario


def _scientific_config() -> HarnessConfig:
    return HarnessConfig(
        condition=ConditionSpec(
            PolicyContent.HISTORICAL_DILU_2024,
            OutputEnforcement.PROMPT_ONLY,
            ExecutionMode.UNSHIELDED_OPERATIONAL,
        ),
        parser_mode=ParserMode.STRICT_ONLY,
        resolver_mode=ResolverMode.DISABLED,
        fallback_policy=FallbackPolicy.FIXED_IDLE,
        shield=ShieldConfig.implementation_defaults(),
        transport=TransportConfig(
            profile=TransportProfile.OLLAMA_NATIVE_CHAT,
            think_mode=ThinkMode.NO_THINK,
            temperature=0.0,
            context_tokens=4096,
            max_output_tokens=64,
            timeout_sec=60.0,
            generation_seed_master=20270713,
            allow_transport_fallback=False,
            adaptive_timeout=False,
        ),
        retry_policy=RetryPolicy(
            max_transport_unavailable_retries=1,
            retry_cooldown_sec=10.0,
            retry_on_timeout=False,
            retry_on_empty_output=False,
            retry_on_schema_rejection=False,
        ),
        trace_level=TraceLevel.MANDATORY_SCIENTIFIC,
    )


def _scientific_agent_response(text: str) -> DriverAgent:
    scenario = EnvScenario.__new__(EnvScenario)
    scenario._action_catalog = lambda: {
        0: "LANE_LEFT",
        1: "IDLE",
        2: "LANE_RIGHT",
        3: "FASTER",
        4: "SLOWER",
    }
    scenario.available_action_ids = lambda: [0, 1, 2, 3, 4]
    scenario.preferred_fallback_action_id = lambda: 4
    agent = DriverAgent.__new__(DriverAgent)
    agent.sce = scenario
    agent.scientific_harness_config = _scientific_config()
    agent.verbose = False
    agent.quiet_mode = True
    agent.oai_api_type = "openai"
    agent.decision_timeout_sec = 10.0
    agent.use_streaming = False
    agent.enable_checker_llm = False
    agent.enable_intent_resolver = False
    agent.runtime_max_output_tokens = 64
    agent.last_ollama_transport = "provider_default"
    agent.last_ollama_effective_think_mode = None
    agent.last_ollama_native_retry_used = False
    agent.last_ollama_native_timeout = False
    agent.last_ollama_native_timeout_short_circuit = False
    diagnostics = {
        "response_finish_reason": None,
        "response_model_provider": None,
        "response_visible_chars": len(text),
        "response_empty": not text,
        "reasoning_tokens": 0,
        "output_tokens": 0,
    }

    def fake_response(
        messages: list[Any], max_output_tokens_override: int | None = None
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        del messages, max_output_tokens_override
        return (
            text,
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "token_count_method": "test",
            },
            diagnostics,
        )

    agent._invoke_response_with_diagnostics = fake_response
    return agent


def _forbid_legacy_helpers(agent: DriverAgent) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError(
            "Scientific path invoked a legacy parse or recovery helper."
        )

    agent._runtime_response_contract_diagnostics = forbidden
    agent._extract_runtime_action_from_text = forbidden
    agent._extract_valid_action_from_text = forbidden
    agent._extract_semantic_action_from_text = forbidden
    agent._resolve_action_with_intent_resolver = forbidden


class ScenarioAndDriverIntegrationTests(unittest.TestCase):
    def test_action_token_lookup_preserves_legacy_fallback(self) -> None:
        scenario = EnvScenario.__new__(EnvScenario)
        scenario.envType = "highway-v0"
        scenario._action_catalog = lambda: {1: "IDLE", 4: "SLOWER"}
        scenario.available_action_ids = lambda: [0, 1, 2, 3, 4]

        self.assertEqual(scenario.action_id_for_token("IDLE"), 1)
        self.assertEqual(scenario.action_id_for_token("slower"), 4)
        self.assertEqual(scenario.preferred_fallback_action_id(), 4)

    def test_action_token_lookup_rejects_missing_or_duplicate_token(self) -> None:
        scenario = EnvScenario.__new__(EnvScenario)
        for catalog in ({4: "SLOWER"}, {1: "IDLE", 3: "idle"}):
            with self.subTest(catalog=catalog):
                scenario._action_catalog = lambda catalog=catalog: catalog
                with self.assertRaises(ValueError):
                    scenario.action_id_for_token("IDLE")

    def test_scientific_fallback_is_fixed_idle_not_legacy_slower(self) -> None:
        agent = _scientific_agent_response("invalid")

        self.assertEqual(agent._decision_fallback_action_id(), 1)
        self.assertEqual(agent._fallback_action_id(), 4)

    def test_scientific_fallback_fails_typed_when_idle_is_unavailable(self) -> None:
        agent = _scientific_agent_response("invalid")
        agent.sce.available_action_ids = lambda: [0, 2, 3, 4]

        with self.assertRaises(RuntimeProtocolError) as context:
            agent._decision_fallback_action_id()
        self.assertEqual(
            context.exception.invariant_code,
            ProtocolInvariantCode.FIXED_FALLBACK_UNAVAILABLE,
        )

    def test_scientific_availability_failure_never_invents_default_domain(self) -> None:
        agent = _scientific_agent_response("invalid")

        def unavailable() -> list[int]:
            raise RuntimeError("state unavailable")

        agent.sce.available_action_ids = unavailable
        with self.assertRaises(RuntimeProtocolError) as context:
            agent._decision_fallback_action_id()
        self.assertEqual(
            context.exception.invariant_code,
            ProtocolInvariantCode.ACTION_AVAILABILITY_UNRESOLVED,
        )

    def test_scientific_availability_is_canonicalized_at_ingress(self) -> None:
        class ActionId(int):
            pass

        agent = _scientific_agent_response("invalid")
        agent.sce.available_action_ids = lambda: [
            ActionId(1),
            ActionId(0),
            ActionId(2),
            ActionId(3),
            ActionId(4),
            ActionId(1),
        ]

        available_action_ids = agent._scientific_available_action_ids()

        self.assertEqual(available_action_ids, [0, 1, 2, 3, 4])
        self.assertTrue(
            all(type(action_id) is int for action_id in available_action_ids)
        )

    def test_scientific_availability_rejects_noncanonical_domain(self) -> None:
        agent = _scientific_agent_response("invalid")
        agent.sce.available_action_ids = lambda: [0, 1, 2, 3, 4, 5]

        with self.assertRaises(RuntimeProtocolError) as context:
            agent._scientific_available_action_ids()

        self.assertEqual(
            context.exception.invariant_code,
            ProtocolInvariantCode.ACTION_AVAILABILITY_UNRESOLVED,
        )

    def test_scientific_decisions_use_only_typed_action_resolution(self) -> None:
        cases = (
            ("Response to user:#### 3", 3, ActionSyntaxStatus.STRICT_VALID, None, 3),
            ("Action id: 3", 1, ActionSyntaxStatus.INVALID, "syntax_invalid", None),
            (" \t", 1, ActionSyntaxStatus.EMPTY, "model_empty_output", None),
        )
        for raw, expected_action, expected_status, expected_failure, proposal in cases:
            with self.subTest(raw=raw):
                agent = _scientific_agent_response(raw)
                _forbid_legacy_helpers(agent)
                action, returned_raw, _, _ = agent.few_shot_decision(
                    fewshot_messages=[], fewshot_answers=[]
                )
                self.assertEqual((action, returned_raw), (expected_action, raw))
                self.assertEqual(
                    agent.last_action_resolution.syntax_status, expected_status
                )
                self.assertEqual(
                    agent.last_decision_meta["fallback_reason"], expected_failure
                )
                self.assertEqual(
                    agent.last_decision_meta["original_selected_action"],
                    proposal,
                )

    def test_unavailable_strict_action_remains_the_model_proposal(self) -> None:
        agent = _scientific_agent_response("Response to user:#### 3")
        agent.sce.available_action_ids = lambda: [1, 4]

        action, _, _, _ = agent.few_shot_decision(
            fewshot_messages=[],
            fewshot_answers=[],
        )

        self.assertEqual(action, 1)
        self.assertEqual(agent.last_action_resolution.strict_action, 3)
        self.assertEqual(agent.last_decision_meta["original_selected_action"], 3)

    def test_untyped_timeout_aborts_instead_of_creating_action_evidence(self) -> None:
        agent = _scientific_agent_response("")
        _forbid_legacy_helpers(agent)

        def timeout_response(*_: Any, **__: Any) -> Any:
            raise TimeoutError

        agent._invoke_response_with_diagnostics = timeout_response
        with self.assertRaises(TimeoutError):
            agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertIsNone(agent.last_action_resolution)
        self.assertIsNone(agent.last_generation_result)


if __name__ == "__main__":
    unittest.main()
