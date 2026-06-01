import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from dilu.driver_agent.driverAgent import DriverAgent
from dilu.runtime.llm_env import configure_runtime_env, openai_compatible_default_headers_from_env
from dilu.runtime.ollama_transport import resolve_ollama_native_chat_mode


class _DummyScenario:
    def scenario_family(self):
        return "highway"

    def available_action_ids(self):
        return [0, 1, 2, 3, 4]

    def preferred_fallback_action_id(self):
        return 4

    def action_catalog_with_descriptions(self):
        return {
            0: {"token": "LANE_LEFT", "description": "Turn-left"},
            1: {"token": "IDLE", "description": "IDLE"},
            2: {"token": "LANE_RIGHT", "description": "Turn-right"},
            3: {"token": "FASTER", "description": "Acceleration"},
            4: {"token": "SLOWER", "description": "Deceleration"},
        }


def _make_agent_response(text, diagnostics=None):
    agent = DriverAgent.__new__(DriverAgent)
    agent.sce = _DummyScenario()
    agent.verbose = False
    agent.quiet_mode = True
    agent.oai_api_type = "openai"
    agent.decision_timeout_sec = 10.0
    agent.use_streaming = False
    agent.enable_checker_llm = False
    agent.enable_intent_resolver = False
    agent.intent_resolver_api_type = "ollama"
    agent.intent_resolver_model = ""
    agent.intent_resolver_timeout_sec = 5.0
    agent.intent_resolver_max_output_tokens = 32
    agent.intent_resolver_abstain_on_ambiguous = True
    agent.last_intent_resolver_prompt = None
    agent.max_tokens = 1024
    agent.runtime_max_output_tokens = 256
    agent.ollama_think_mode = "auto"
    agent.last_ollama_transport = "provider_default"
    agent.last_ollama_effective_think_mode = None
    agent.last_ollama_native_retry_used = False
    agent.last_ollama_native_timeout = False
    agent.last_ollama_native_timeout_short_circuit = False
    response_diagnostics = {
        "response_finish_reason": None,
        "response_model_provider": None,
        "response_visible_chars": len(str(text or "").strip()),
        "response_empty": len(str(text or "").strip()) == 0,
        "reasoning_tokens": 0,
        "output_tokens": 0,
    }
    response_diagnostics.update(diagnostics or {})
    agent.captured_messages = []

    def _fake_response(messages, max_output_tokens_override=None):
        agent.captured_messages = list(messages)
        return (
            text,
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "token_count_method": "test"},
            response_diagnostics,
        )

    agent._invoke_response_with_diagnostics = _fake_response
    return agent


class DriverAgentRuntimeContractTests(unittest.TestCase):
    def test_reactive_prompt_removes_copy_trap_and_adds_recovery_rules(self):
        agent = DriverAgent.__new__(DriverAgent)
        agent.sce = _DummyScenario()

        system_message = agent._build_system_message(fallback_action_id=4)

        self.assertNotIn("Safety fallback", system_message)
        self.assertNotIn("Example:", system_message)
        self.assertNotIn("choose the safest valid fallback", system_message)
        self.assertIn("60 km/h is a slow-flow floor", system_message)
        self.assertIn("prefer Acceleration over repeated IDLE", system_message)
        self.assertIn("HARD LANE-CHANGE SAFETY RULES", system_message)
        self.assertIn("within 15 m ahead or behind", system_message)

    def test_empty_fewshot_prompt_does_not_claim_examples_exist(self):
        agent = _make_agent_response("Response to user:#### 3\nReason: clear road.")

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 3)
        human_message = agent.captured_messages[-1].content
        self.assertNotIn("Above messages are some examples", human_message)
        self.assertIn("Use only the current scenario.", human_message)

    def test_nonempty_fewshot_prompt_keeps_example_guidance(self):
        agent = _make_agent_response("Response to user:#### 3\nReason: clear road.")

        action, *_ = agent.few_shot_decision(
            fewshot_messages=["Prior clear road."],
            fewshot_answers=["Response to user:#### 3\nReason: prior clear road."],
        )

        self.assertEqual(action, 3)
        human_message = agent.captured_messages[-1].content
        self.assertIn("Above messages are some examples", human_message)

    def test_strict_action_line_satisfies_runtime_contract(self):
        agent = _make_agent_response("Response to user:#### 3\nReason: clear road.")

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 3)
        self.assertTrue(agent.last_decision_meta["response_contract_satisfied"])
        self.assertFalse(agent.last_decision_meta["response_contract_recovered"])
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "strict_action_line")

    def test_missing_delimiter_recovers_but_marks_non_strict(self):
        agent = _make_agent_response("Response to user: 2\nReason: safe lane.")

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 2)
        self.assertFalse(agent.last_decision_meta["response_contract_satisfied"])
        self.assertTrue(agent.last_decision_meta["response_contract_recovered"])
        self.assertEqual(agent.last_decision_meta["response_recovery_reason"], "missing_delimiter_action_line")
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "missing_delimiter_recovered")

    def test_labeled_backup_recovers_but_marks_non_strict(self):
        agent = _make_agent_response("Reason: safe to cruise.\nAction id: 1")

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 1)
        self.assertFalse(agent.last_decision_meta["response_contract_satisfied"])
        self.assertTrue(agent.last_decision_meta["response_contract_recovered"])
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "labeled_backup")

    def test_semantic_label_recovery_extracts_clear_action_label(self):
        agent = _make_agent_response("Response to user:#### Acceleration")

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 3)
        self.assertFalse(agent.last_decision_meta["response_contract_satisfied"])
        self.assertTrue(agent.last_decision_meta["response_contract_recovered"])
        self.assertTrue(agent.last_decision_meta["semantic_recovery_used"])
        self.assertEqual(agent.last_decision_meta["semantic_recovery_label"], "FASTER")
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "semantic_label_recovered")
        self.assertEqual(agent.last_decision_meta["final_action_source"], "semantic_label_recovered")

    def test_semantic_label_recovery_respects_current_valid_actions(self):
        agent = _make_agent_response("Response to user:#### Turn-left Action_id:")
        agent.sce = _DummyScenario()
        agent.sce.available_action_ids = lambda: [1, 3, 4]

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 4)
        self.assertTrue(agent.last_decision_meta["used_fallback"])
        self.assertFalse(agent.last_decision_meta["semantic_recovery_used"])
        self.assertIn("unavailable", agent.last_decision_meta["semantic_recovery_reason"])
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "parse_fallback")

    def test_semantic_label_recovery_abstains_on_conflicting_labels(self):
        agent = _make_agent_response("Response to user:#### Acceleration or Deceleration")

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 4)
        self.assertTrue(agent.last_decision_meta["used_fallback"])
        self.assertFalse(agent.last_decision_meta["semantic_recovery_used"])
        self.assertIn("ambiguous", agent.last_decision_meta["semantic_recovery_reason"])
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "parse_fallback")

    def test_intent_resolver_recovers_high_confidence_json(self):
        agent = _make_agent_response("The best move is to speed up now.")
        agent.enable_intent_resolver = True
        agent.intent_resolver_model = "intent-decoder:latest"
        captured = {}

        def _fake_resolver(prompt):
            captured["prompt"] = prompt
            return '{"action_id": 3, "confidence": "high", "reason": "clear acceleration intent"}'

        agent._invoke_intent_resolver_model = _fake_resolver

        action, *_ = agent.few_shot_decision(
            scenario_description="SECRET_SCENARIO_TEXT",
            fewshot_messages=[],
            fewshot_answers=[],
        )

        self.assertEqual(action, 3)
        self.assertFalse(agent.last_decision_meta["used_fallback"])
        self.assertTrue(agent.last_decision_meta["intent_resolver_used"])
        self.assertFalse(agent.last_decision_meta["intent_resolver_abstained"])
        self.assertEqual(agent.last_decision_meta["intent_resolver_model"], "intent-decoder:latest")
        self.assertEqual(agent.last_decision_meta["intent_resolver_action_id"], 3)
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "intent_resolver_direct")
        self.assertEqual(agent.last_decision_meta["final_action_source"], "intent_resolver_direct")
        self.assertIn("The best move is to speed up now.", captured["prompt"])
        self.assertIn("| 3 | FASTER | Acceleration |", captured["prompt"])
        self.assertNotIn("SECRET_SCENARIO_TEXT", captured["prompt"])

    def test_intent_resolver_abstain_falls_back_safely(self):
        agent = _make_agent_response("Maybe do something, maybe not.")
        agent.enable_intent_resolver = True
        agent.intent_resolver_model = "intent-decoder:latest"
        agent._invoke_intent_resolver_model = lambda prompt: (
            '{"action_id": null, "confidence": "low", "reason": "ambiguous"}'
        )

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 4)
        self.assertTrue(agent.last_decision_meta["used_fallback"])
        self.assertTrue(agent.last_decision_meta["intent_resolver_used"])
        self.assertTrue(agent.last_decision_meta["intent_resolver_abstained"])
        self.assertEqual(agent.last_decision_meta["intent_resolver_reason"], "ambiguous")
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "parse_fallback")

    def test_intent_resolver_invalid_outputs_fall_back_safely(self):
        cases = [
            ("not-json", "invalid_response"),
            ('{"action_id": 99, "confidence": "high", "reason": "not valid"}', "not valid"),
        ]
        for resolver_output, expected_reason in cases:
            with self.subTest(resolver_output=resolver_output):
                agent = _make_agent_response("The intended move is unclear.")
                agent.enable_intent_resolver = True
                agent.intent_resolver_model = "intent-decoder:latest"
                agent._invoke_intent_resolver_model = lambda prompt, value=resolver_output: value

                action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

                self.assertEqual(action, 4)
                self.assertTrue(agent.last_decision_meta["used_fallback"])
                self.assertTrue(agent.last_decision_meta["intent_resolver_used"])
                self.assertIn(expected_reason, agent.last_decision_meta["intent_resolver_reason"])
                self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "parse_fallback")

    def test_intent_resolver_timeout_falls_back_safely(self):
        agent = _make_agent_response("The intended move is unclear.")
        agent.enable_intent_resolver = True
        agent.intent_resolver_model = "intent-decoder:latest"

        def _timeout(_prompt):
            raise TimeoutError("resolver timeout")

        agent._invoke_intent_resolver_model = _timeout

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 4)
        self.assertTrue(agent.last_decision_meta["used_fallback"])
        self.assertTrue(agent.last_decision_meta["intent_resolver_used"])
        self.assertEqual(agent.last_decision_meta["intent_resolver_reason"], "timeout")
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "parse_fallback")

    def test_max_token_empty_response_uses_specific_fallback_reason(self):
        agent = _make_agent_response("", {"response_finish_reason": "MAX_TOKENS", "response_empty": True})

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 4)
        self.assertTrue(agent.last_decision_meta["used_fallback"])
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "max_tokens_empty_response_fallback")
        self.assertTrue(agent.last_decision_meta["response_unparseable"])

    def test_unparseable_response_falls_back_safely(self):
        agent = _make_agent_response("I cannot decide.")

        action, *_ = agent.few_shot_decision(fewshot_messages=[], fewshot_answers=[])

        self.assertEqual(action, 4)
        self.assertTrue(agent.last_decision_meta["used_fallback"])
        self.assertEqual(agent.last_decision_meta["runtime_parse_path"], "parse_fallback")
        self.assertTrue(agent.last_decision_meta["response_unparseable"])

    def test_ollama_openai_compatible_direct_path_uses_max_tokens(self):
        agent = DriverAgent.__new__(DriverAgent)
        agent.oai_api_type = "ollama"
        agent.ollama_use_native_chat = False
        agent.ollama_model_name = "gemma3:1b"
        agent.ollama_api_key = "ollama"
        agent.ollama_chat_url = "http://localhost:11434/api/chat"
        agent.ollama_think_mode = "auto"
        agent.decision_timeout_sec = 10.0
        agent.last_ollama_transport = None
        agent.last_ollama_effective_think_mode = None
        agent.last_ollama_native_retry_used = False
        agent.last_ollama_native_timeout = False
        agent.last_ollama_native_timeout_short_circuit = False
        captured_payloads = []

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {"content": "Response to user:#### 3"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                    },
                }

        def _post(url, headers=None, json=None, timeout=None):
            captured_payloads.append(
                {
                    "url": url,
                    "headers": headers,
                    "json": json,
                    "timeout": timeout,
                }
            )
            return _Response()

        messages = [
            SystemMessage(content="System contract."),
            HumanMessage(content="Current scenario."),
        ]
        with patch("dilu.driver_agent.driverAgent.requests.post", side_effect=_post):
            content, usage, diagnostics = agent._invoke_response_with_diagnostics(
                messages,
                max_output_tokens_override=128,
            )

        payload = captured_payloads[0]["json"]
        self.assertEqual(content, "Response to user:#### 3")
        self.assertEqual(captured_payloads[0]["url"], "http://localhost:11434/v1/chat/completions")
        self.assertEqual(payload["model"], "gemma3:1b")
        self.assertEqual(payload["max_tokens"], 128)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(usage["completion_tokens"], 7)
        self.assertEqual(diagnostics["response_finish_reason"], "length")
        self.assertEqual(diagnostics["output_tokens"], 7)
        self.assertEqual(agent.last_ollama_transport, "openai_compat_direct")


class OpenRouterRuntimeEnvTests(unittest.TestCase):
    def test_openrouter_config_uses_openai_compatible_env_and_headers(self):
        config = {
            "OPENAI_API_TYPE": "openai",
            "OPENAI_CHAT_MODEL": "openai/gpt-5.2",
            "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
            "OPENROUTER_HTTP_REFERER": "https://example.test/dilu",
            "OPENROUTER_APP_TITLE": "DiLu-Ollama",
        }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
            selected = configure_runtime_env(config, mode="eval", quiet_override=True, progress_override=False)
            headers = openai_compatible_default_headers_from_env()
            env_snapshot = dict(os.environ)

        self.assertEqual(selected, "openai/gpt-5.2")
        self.assertEqual(env_snapshot["OPENAI_API_TYPE"], "openai")
        self.assertEqual(env_snapshot["OPENAI_API_KEY"], "sk-or-test")
        self.assertEqual(env_snapshot["OPENAI_BASE_URL"], "https://openrouter.ai/api/v1")
        self.assertEqual(env_snapshot["OPENAI_API_BASE"], "https://openrouter.ai/api/v1")
        self.assertEqual(headers["HTTP-Referer"], "https://example.test/dilu")
        self.assertEqual(headers["X-OpenRouter-Title"], "DiLu-Ollama")

    def test_openrouter_key_can_be_loaded_from_project_dotenv(self):
        config = {
            "OPENAI_API_TYPE": "openai",
            "OPENAI_CHAT_MODEL": "openai/gpt-5.4-mini",
            "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
        }

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            Path(tmp, ".env").write_text('OPENROUTER_API_KEY="sk-or-dotenv-test"\n', encoding="utf-8")
            try:
                os.chdir(tmp)
                selected = configure_runtime_env(
                    config,
                    mode="eval",
                    quiet_override=True,
                    progress_override=False,
                )
                env_snapshot = dict(os.environ)
            finally:
                os.chdir(cwd)

        self.assertEqual(selected, "openai/gpt-5.4-mini")
        self.assertEqual(env_snapshot["OPENAI_API_KEY"], "sk-or-dotenv-test")

    def test_intent_resolver_config_exports_runtime_env(self):
        config = {
            "OPENAI_API_TYPE": "ollama",
            "OLLAMA_CHAT_MODEL": "qwen3:4b",
            "OLLAMA_EMBED_MODEL": "qwen3-embedding:8b",
            "eval_enable_intent_resolver": True,
            "intent_resolver_api_type": "ollama",
            "intent_resolver_model": "llama3.2:3b",
            "intent_resolver_timeout_sec": 7,
            "intent_resolver_max_output_tokens": 24,
            "intent_resolver_abstain_on_ambiguous": True,
        }

        with patch.dict(os.environ, {}, clear=True):
            configure_runtime_env(config, mode="eval", quiet_override=True, progress_override=False)
            env_snapshot = dict(os.environ)

        self.assertEqual(env_snapshot["DILU_ENABLE_INTENT_RESOLVER"], "1")
        self.assertEqual(env_snapshot["DILU_INTENT_RESOLVER_API_TYPE"], "ollama")
        self.assertEqual(env_snapshot["DILU_INTENT_RESOLVER_MODEL"], "llama3.2:3b")
        self.assertEqual(env_snapshot["DILU_INTENT_RESOLVER_TIMEOUT_SEC"], "7")
        self.assertEqual(env_snapshot["DILU_INTENT_RESOLVER_MAX_OUTPUT_TOKENS"], "24")
        self.assertEqual(env_snapshot["DILU_INTENT_RESOLVER_ABSTAIN_ON_AMBIGUOUS"], "1")


class OllamaTransportResolverTests(unittest.TestCase):
    def test_auto_native_chat_enables_qwen_no_think(self):
        resolved = resolve_ollama_native_chat_mode("qwen3:4b", "auto", "no_think")

        self.assertTrue(resolved.effective_native_chat)
        self.assertEqual(resolved.configured_mode, "auto")
        self.assertEqual(resolved.reason, "thinking_family_no_think")

    def test_auto_native_chat_enables_deepseek_no_think(self):
        resolved = resolve_ollama_native_chat_mode("deepseek-r1:7b", "auto", "no_think")

        self.assertTrue(resolved.effective_native_chat)
        self.assertEqual(resolved.reason, "thinking_family_no_think")

    def test_auto_native_chat_leaves_llama_on_openai_compat(self):
        resolved = resolve_ollama_native_chat_mode("llama3.2:3b", "auto", "no_think")

        self.assertFalse(resolved.effective_native_chat)
        self.assertEqual(resolved.reason, "non_thinking_model")

    def test_explicit_false_overrides_thinking_family(self):
        resolved = resolve_ollama_native_chat_mode("qwen3:4b", False, "no_think")

        self.assertFalse(resolved.effective_native_chat)
        self.assertEqual(resolved.configured_mode, "false")
        self.assertEqual(resolved.reason, "manual_false")

    def test_explicit_true_overrides_non_thinking_family(self):
        resolved = resolve_ollama_native_chat_mode("llama3.2:3b", True, "auto")

        self.assertTrue(resolved.effective_native_chat)
        self.assertEqual(resolved.configured_mode, "true")
        self.assertEqual(resolved.reason, "manual_true")

    def test_unknown_auto_auto_uses_openai_compatible(self):
        resolved = resolve_ollama_native_chat_mode("custom-local:latest", "auto", "auto")

        self.assertFalse(resolved.effective_native_chat)
        self.assertEqual(resolved.reason, "non_thinking_model")

    def test_configure_runtime_env_exports_effective_auto_native_for_selected_model(self):
        config = {
            "OPENAI_API_TYPE": "ollama",
            "OLLAMA_CHAT_MODEL": "llama3.2:3b",
            "OLLAMA_EMBED_MODEL": "qwen3-embedding:8b",
            "OLLAMA_USE_NATIVE_CHAT": "auto",
            "OLLAMA_THINK_MODE": "no_think",
        }

        with patch.dict(os.environ, {}, clear=True):
            selected = configure_runtime_env(
                config,
                chat_model_override="qwen3:4b",
                mode="eval",
                quiet_override=True,
                progress_override=False,
            )
            env_snapshot = dict(os.environ)

        self.assertEqual(selected, "qwen3:4b")
        self.assertEqual(env_snapshot["OLLAMA_USE_NATIVE_CHAT_CONFIGURED"], "auto")
        self.assertEqual(env_snapshot["OLLAMA_USE_NATIVE_CHAT"], "1")
        self.assertEqual(env_snapshot["OLLAMA_USE_NATIVE_CHAT_EFFECTIVE"], "1")
        self.assertEqual(env_snapshot["OLLAMA_NATIVE_CHAT_RESOLUTION_REASON"], "thinking_family_no_think")

    def test_configure_runtime_env_keeps_auto_llama_on_openai_compatible(self):
        config = {
            "OPENAI_API_TYPE": "ollama",
            "OLLAMA_CHAT_MODEL": "llama3.2:3b",
            "OLLAMA_EMBED_MODEL": "qwen3-embedding:8b",
            "OLLAMA_USE_NATIVE_CHAT": "auto",
            "OLLAMA_THINK_MODE": "no_think",
        }

        with patch.dict(os.environ, {}, clear=True):
            configure_runtime_env(config, mode="eval", quiet_override=True, progress_override=False)
            env_snapshot = dict(os.environ)

        self.assertEqual(env_snapshot["OLLAMA_USE_NATIVE_CHAT_CONFIGURED"], "auto")
        self.assertEqual(env_snapshot["OLLAMA_USE_NATIVE_CHAT"], "0")
        self.assertEqual(env_snapshot["OLLAMA_USE_NATIVE_CHAT_EFFECTIVE"], "0")
        self.assertEqual(env_snapshot["OLLAMA_NATIVE_CHAT_RESOLUTION_REASON"], "non_thinking_model")


class _ShieldScenario:
    def __init__(self, ego, vehicles, available_actions=None, lane_count=3):
        self.ego = ego
        self.env = SimpleNamespace(
            unwrapped=SimpleNamespace(
                vehicle=ego,
                road=SimpleNamespace(vehicles=[ego, *vehicles]),
                action_type=SimpleNamespace(actions={0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}),
            )
        )
        self.network = SimpleNamespace(all_side_lanes=lambda lane_index: [("a", "b", idx) for idx in range(lane_count)])
        self._available_actions = list(available_actions or [0, 1, 2, 3, 4])

    def available_action_ids(self):
        return list(self._available_actions)

    def preferred_fallback_action_id(self):
        return 4 if 4 in self._available_actions else self._available_actions[0]

    def scenario_family(self):
        return "highway"


def _vehicle(lane_rank, speed, x):
    return SimpleNamespace(lane_index=("a", "b", lane_rank), speed=float(speed), position=np.array([float(x), 0.0]))


class SafetyShieldTests(unittest.TestCase):
    def test_lane_change_shield_blocks_unsafe_target_rear_gap(self):
        from dilu.runtime.safety_shields import apply_lane_change_safety_shield

        ego = _vehicle(1, 25.0, 0.0)
        rear_blocker = _vehicle(0, 28.0, -5.0)
        scenario = _ShieldScenario(ego, [rear_blocker])

        result = apply_lane_change_safety_shield(scenario, 0)

        self.assertTrue(result.applied)
        self.assertEqual(result.action_id, 1)
        self.assertEqual(result.reason, "target_rear_gap_below_required")
        metadata = result.to_metadata("lane_change")
        self.assertEqual(metadata["lane_change_target_rear_gap_m"], 5.0)
        self.assertAlmostEqual(metadata["lane_change_target_rear_ttc_sec"], 1.6667)
        self.assertEqual(metadata["lane_change_required_front_gap_m"], 14.0)
        self.assertEqual(metadata["lane_change_required_rear_gap_m"], 10.0)
        self.assertEqual(metadata["lane_change_required_rear_ttc_sec"], 2.5)

    def test_longitudinal_shield_uses_slower_for_severe_acceleration_front_risk(self):
        from dilu.runtime.safety_shields import apply_longitudinal_safety_shield

        ego = _vehicle(1, 25.0, 0.0)
        lead = _vehicle(1, 12.0, 10.0)
        scenario = _ShieldScenario(ego, [lead])

        result = apply_longitudinal_safety_shield(scenario, 3)

        self.assertTrue(result.applied)
        self.assertEqual(result.action_id, 4)
        self.assertIn(result.reason, {"front_clearance_blocks_accelerate", "projected_front_gap_blocks_accelerate"})
        metadata = result.to_metadata("longitudinal_safety")
        self.assertEqual(metadata["longitudinal_safety_original_action_id"], 3)
        self.assertEqual(metadata["longitudinal_safety_final_action_id"], 4)
        self.assertEqual(metadata["longitudinal_safety_current_front_gap_m"], 10.0)
        self.assertEqual(metadata["longitudinal_safety_required_front_gap_m"], 12.0)
        self.assertIsNotNone(metadata["longitudinal_safety_projected_front_gap_m"])
        self.assertIsNotNone(metadata["longitudinal_safety_projected_front_ttc_sec"])
        self.assertEqual(metadata["longitudinal_safety_projection_horizon_sec"], 1.0)

    def test_longitudinal_shield_keeps_caution_acceleration_at_idle(self):
        from dilu.runtime.safety_shields import apply_longitudinal_safety_shield

        ego = _vehicle(1, 20.0, 0.0)
        lead = _vehicle(1, 19.0, 14.0)
        scenario = _ShieldScenario(ego, [lead])

        result = apply_longitudinal_safety_shield(scenario, 3)

        self.assertTrue(result.applied)
        self.assertEqual(result.action_id, 1)
        self.assertEqual(result.reason, "projected_front_gap_blocks_accelerate")

    def test_longitudinal_shield_falls_back_to_idle_when_slower_unavailable(self):
        from dilu.runtime.safety_shields import apply_longitudinal_safety_shield

        ego = _vehicle(1, 25.0, 0.0)
        lead = _vehicle(1, 12.0, 10.0)
        scenario = _ShieldScenario(ego, [lead], available_actions=[0, 1, 2, 3])

        result = apply_longitudinal_safety_shield(scenario, 3)

        self.assertTrue(result.applied)
        self.assertEqual(result.action_id, 1)

    def test_longitudinal_shield_brakes_critical_idle_front_risk(self):
        from dilu.runtime.safety_shields import apply_longitudinal_safety_shield

        ego = _vehicle(1, 20.0, 0.0)
        lead = _vehicle(1, 10.0, 7.0)
        scenario = _ShieldScenario(ego, [lead])

        result = apply_longitudinal_safety_shield(scenario, 1)

        self.assertTrue(result.applied)
        self.assertEqual(result.action_id, 4)
        self.assertEqual(result.reason, "front_clearance_blocks_idle")

    def test_longitudinal_shield_brakes_projected_idle_front_risk(self):
        from dilu.runtime.safety_shields import apply_longitudinal_safety_shield

        ego = _vehicle(1, 25.0, 0.0)
        lead = _vehicle(1, 19.0, 12.0)
        scenario = _ShieldScenario(ego, [lead])

        result = apply_longitudinal_safety_shield(scenario, 1)

        self.assertTrue(result.applied)
        self.assertEqual(result.action_id, 4)
        self.assertEqual(result.reason, "projected_front_clearance_blocks_idle")
        metadata = result.to_metadata("longitudinal_safety")
        self.assertEqual(metadata["longitudinal_safety_current_front_gap_m"], 12.0)
        self.assertEqual(metadata["longitudinal_safety_current_front_ttc_sec"], 2.0)
        self.assertEqual(metadata["longitudinal_safety_projected_front_gap_m"], 6.0)
        self.assertEqual(metadata["longitudinal_safety_projected_front_ttc_sec"], 1.0)
        self.assertEqual(metadata["longitudinal_safety_required_front_gap_m"], 8.0)
        self.assertEqual(metadata["longitudinal_safety_required_front_ttc_sec"], 2.0)

    def test_longitudinal_shield_projected_idle_uses_idle_when_slower_unavailable(self):
        from dilu.runtime.safety_shields import apply_longitudinal_safety_shield

        ego = _vehicle(1, 25.0, 0.0)
        lead = _vehicle(1, 19.0, 12.0)
        scenario = _ShieldScenario(ego, [lead], available_actions=[0, 1, 2, 3])

        result = apply_longitudinal_safety_shield(scenario, 1)

        self.assertTrue(result.applied)
        self.assertEqual(result.action_id, 1)
        self.assertEqual(result.reason, "projected_front_clearance_blocks_idle")

    def test_safe_acceleration_without_front_vehicle_is_not_changed(self):
        from dilu.runtime.safety_shields import apply_longitudinal_safety_shield

        ego = _vehicle(1, 20.0, 0.0)
        scenario = _ShieldScenario(ego, [])

        result = apply_longitudinal_safety_shield(scenario, 3)

        self.assertFalse(result.applied)
        self.assertEqual(result.action_id, 3)
        self.assertEqual(result.reason, "safe")

    def test_low_speed_recovery_turns_safe_idle_into_faster(self):
        from dilu.runtime.safety_shields import apply_low_speed_recovery_shield

        ego = _vehicle(1, 10.0, 0.0)
        scenario = _ShieldScenario(ego, [])

        result = apply_low_speed_recovery_shield(scenario, 1)

        self.assertTrue(result.applied)
        self.assertEqual(result.action_id, 3)
        self.assertEqual(result.reason, "low_speed_recovery_after_front_risk")
        self.assertEqual(result.shield_type, "flow_recovery")

    def test_low_speed_recovery_skips_step_with_safety_shield(self):
        from dilu.runtime.safety_shields import apply_low_speed_recovery_shield

        ego = _vehicle(1, 10.0, 0.0)
        scenario = _ShieldScenario(ego, [])

        result = apply_low_speed_recovery_shield(scenario, 1, safety_shield_applied=True)

        self.assertFalse(result.applied)
        self.assertEqual(result.action_id, 1)
        self.assertEqual(result.reason, "safety_shield_already_applied")

    def test_low_speed_recovery_requires_safe_front_gap(self):
        from dilu.runtime.safety_shields import apply_low_speed_recovery_shield

        ego = _vehicle(1, 10.0, 0.0)
        close_lead = _vehicle(1, 9.0, 18.0)
        scenario = _ShieldScenario(ego, [close_lead])

        result = apply_low_speed_recovery_shield(scenario, 1)

        self.assertFalse(result.applied)
        self.assertEqual(result.action_id, 1)
        self.assertEqual(result.reason, "front_gap_not_safe_for_recovery")


class EvalShieldIntegrationTests(unittest.TestCase):
    def test_eval_response_metrics_accept_new_two_line_contract(self):
        from evaluate_models_ollama import _response_format_metrics

        metrics = _response_format_metrics("Response to user:#### 3\nReason: clear road.")

        self.assertTrue(metrics["has_delimiter"])
        self.assertTrue(metrics["strict_format_match"])
        self.assertTrue(metrics["direct_action_parseable"])
        self.assertEqual(metrics["strict_action"], 3)
        self.assertEqual(metrics["direct_parsed_action"], 3)

    def test_eval_helper_records_shield_override_metadata(self):
        from evaluate_models_ollama import _apply_reactive_safety_shields

        ego = _vehicle(1, 25.0, 0.0)
        rear_blocker = _vehicle(0, 28.0, -5.0)
        scenario = _ShieldScenario(ego, [rear_blocker])
        decision_meta = {"selected_action": 0}

        action, shield_meta = _apply_reactive_safety_shields(0, scenario, decision_meta)

        self.assertEqual(action, 1)
        self.assertTrue(shield_meta["lane_change_shield_applied"])
        self.assertEqual(shield_meta["lane_change_original_action_id"], 0)
        self.assertEqual(shield_meta["lane_change_final_action_id"], 1)
        self.assertEqual(shield_meta["lane_change_target_rear_gap_m"], 5.0)
        self.assertAlmostEqual(shield_meta["lane_change_target_rear_ttc_sec"], 1.6667)
        self.assertEqual(shield_meta["lane_change_required_rear_ttc_sec"], 2.5)
        self.assertEqual(decision_meta["runtime_override_reason_class"], "safety_shield")

    def test_eval_helper_records_flow_recovery_metadata_separately(self):
        from evaluate_models_ollama import _apply_reactive_safety_shields

        ego = _vehicle(1, 10.0, 0.0)
        scenario = _ShieldScenario(ego, [])
        decision_meta = {"selected_action": 1}

        action, shield_meta = _apply_reactive_safety_shields(1, scenario, decision_meta)

        self.assertEqual(action, 3)
        self.assertFalse(shield_meta["lane_change_shield_applied"])
        self.assertFalse(shield_meta["longitudinal_safety_shield_applied"])
        self.assertTrue(shield_meta["flow_recovery_shield_applied"])
        self.assertEqual(shield_meta["flow_recovery_original_action_id"], 1)
        self.assertEqual(shield_meta["flow_recovery_final_action_id"], 3)
        self.assertEqual(
            shield_meta["flow_recovery_shield_reason"],
            "low_speed_recovery_after_front_risk",
        )
        self.assertEqual(shield_meta["flow_recovery_reason"], "low_speed_recovery_after_front_risk")
        self.assertEqual(decision_meta["runtime_override_reason_class"], "flow_recovery_shield")

    def test_eval_helper_and_trace_record_exact_braked_final_action(self):
        from evaluate_models_ollama import _apply_reactive_safety_shields, _decision_trace_item

        ego = _vehicle(1, 25.0, 0.0)
        lead = _vehicle(1, 12.0, 10.0)
        scenario = _ShieldScenario(ego, [lead])
        decision_meta = {"selected_action": 3}

        action, shield_meta = _apply_reactive_safety_shields(3, scenario, decision_meta)
        item = _decision_trace_item(
            step_idx=1,
            action_id=action,
            response_text="Response to user:#### 3\nReason: accelerate.",
            decision_meta=decision_meta,
        )

        self.assertEqual(action, 4)
        self.assertEqual(decision_meta["selected_action"], 4)
        self.assertEqual(shield_meta["longitudinal_safety_original_action_id"], 3)
        self.assertEqual(shield_meta["longitudinal_safety_final_action_id"], 4)
        self.assertEqual(item["model_action_id"], 3)
        self.assertEqual(item["final_action_id"], 4)
        self.assertEqual(item["longitudinal_safety_original_action_id"], 3)
        self.assertEqual(item["longitudinal_safety_final_action_id"], 4)

    def test_eval_helper_and_trace_record_projected_idle_braking(self):
        from evaluate_models_ollama import _apply_reactive_safety_shields, _decision_trace_item

        ego = _vehicle(1, 25.0, 0.0)
        lead = _vehicle(1, 19.0, 12.0)
        scenario = _ShieldScenario(ego, [lead])
        decision_meta = {"selected_action": 1}

        action, shield_meta = _apply_reactive_safety_shields(1, scenario, decision_meta)
        item = _decision_trace_item(
            step_idx=5,
            action_id=action,
            response_text="Response to user:#### 1\nReason: maintain speed.",
            decision_meta=decision_meta,
        )

        self.assertEqual(action, 4)
        self.assertEqual(decision_meta["selected_action"], 4)
        self.assertTrue(shield_meta["longitudinal_safety_shield_applied"])
        self.assertEqual(
            shield_meta["longitudinal_safety_shield_reason"],
            "projected_front_clearance_blocks_idle",
        )
        self.assertEqual(item["model_action_id"], 1)
        self.assertEqual(item["final_action_id"], 4)
        self.assertEqual(
            item["longitudinal_safety_shield_reason"],
            "projected_front_clearance_blocks_idle",
        )

    def test_decision_trace_item_records_runtime_and_shield_diagnostics(self):
        from evaluate_models_ollama import _decision_trace_item

        item = _decision_trace_item(
            step_idx=7,
            action_id=3,
            response_text="Response to user:#### 3\nReason: recover speed.",
            decision_meta={
                "original_selected_action": 1,
                "selected_action": 3,
                "runtime_parse_path": "strict_action_line",
                "fallback_reason": None,
                "semantic_recovery_used": True,
                "semantic_recovery_label": "FASTER",
                "intent_resolver_used": True,
                "intent_resolver_model": "intent-decoder:latest",
                "intent_resolver_action_id": 3,
                "intent_resolver_abstained": False,
                "intent_resolver_reason": "clear acceleration intent",
                "final_action_source": "intent_resolver_direct",
                "response_contract_satisfied": True,
                "response_contract_recovered": False,
                "ego_speed_mps": 10.0,
                "flow_recovery_shield_applied": True,
                "flow_recovery_reason": "low_speed_recovery_after_front_risk",
            },
        )

        self.assertEqual(item["model_action_id"], 1)
        self.assertEqual(item["final_action_id"], 3)
        self.assertEqual(item["runtime_parse_path"], "strict_action_line")
        self.assertTrue(item["semantic_recovery_used"])
        self.assertEqual(item["intent_resolver_action_id"], 3)
        self.assertEqual(item["final_action_source"], "intent_resolver_direct")
        self.assertTrue(item["response_contract_satisfied"])
        self.assertEqual(item["ego_speed_mps"], 10.0)
        self.assertTrue(item["flow_recovery_shield_applied"])
        self.assertEqual(item["response_first_line"], "Response to user:#### 3")

    def test_aggregate_reports_semantic_and_intent_resolver_counts(self):
        from evaluate_models_ollama import aggregate_results

        episode = {
            "seed": 1,
            "crashed": False,
            "error": None,
            "success_no_collision": True,
            "truncated": False,
            "terminated": True,
            "steps": 5,
            "episode_runtime_sec": 1.0,
            "decisions_made": 5,
            "decision_calls_total": 5,
            "decision_timeout_count": 0,
            "fallback_action_count": 1,
            "responses_with_delimiter": 5,
            "responses_strict_format": 3,
            "responses_direct_parseable": 3,
            "format_failure_count": 2,
            "fallback_reason_counts": {"parse_fallback": 1},
            "runtime_parse_path_counts": {
                "strict_action_line": 3,
                "semantic_label_recovered": 1,
                "intent_resolver_direct": 1,
            },
            "semantic_recovery_count": 1,
            "semantic_recovery_label_counts": {"FASTER": 1},
            "intent_resolver_used_count": 2,
            "intent_resolver_recovery_count": 1,
            "intent_resolver_abstain_count": 1,
            "episode_reward_sum": 4.0,
            "avg_ego_speed_mps": 20.0,
            "ttc_danger_rate": 0.0,
            "headway_violation_rate": 0.0,
            "rear_ttc_danger_rate": 0.0,
            "rear_headway_violation_rate": 0.0,
            "low_speed_blocking_rate": 0.0,
            "lane_change_rate": 0.0,
            "lane_change_shield_count": 0,
            "longitudinal_safety_shield_count": 0,
            "flow_recovery_shield_count": 0,
            "flap_accel_decel_rate": 0.0,
            "decision_latency_ms_avg": 100.0,
        }

        summary = aggregate_results("parser_model", [episode])

        self.assertEqual(summary["semantic_recoveries_total"], 1)
        self.assertEqual(summary["semantic_recovery_rate_mean"], 0.2)
        self.assertEqual(summary["semantic_recovery_label_counts"]["FASTER"], 1)
        self.assertEqual(summary["intent_resolver_used_total"], 2)
        self.assertEqual(summary["intent_resolver_recoveries_total"], 1)
        self.assertEqual(summary["intent_resolver_abstains_total"], 1)
        self.assertEqual(summary["intent_resolver_recovery_rate_mean"], 0.2)


if __name__ == "__main__":
    unittest.main()
