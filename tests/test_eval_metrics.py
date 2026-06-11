import unittest
from unittest.mock import patch

from dilu.runtime import load_runtime_config

from evaluate_models_ollama import (
    _annotate_aggregate_with_ollama_preflight_status,
    _build_skipped_model_aggregate,
    _build_measurement_integrity_summary,
    _classify_ollama_preflight_failure,
    _ollama_case_unload_policy,
    _ollama_preflight_probe,
    _ollama_runtime_controls_report,
    _resolve_eval_artifact_modes,
    _resolve_simulation_duration,
    _run_ollama_preflight,
    _sanitize_process_output,
    _stop_ollama_model_after_case,
    _summarize_decision_latency_samples,
    aggregate_results,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ArtifactArgs:
    def __init__(
        self,
        *,
        save_run_artifacts=False,
        no_save_run_artifacts=False,
        record_video=False,
        no_record_video=False,
    ):
        self.save_run_artifacts = save_run_artifacts
        self.no_save_run_artifacts = no_save_run_artifacts
        self.record_video = record_video
        self.no_record_video = no_record_video


class EvalMetricsTests(unittest.TestCase):
    def test_artifact_mode_does_not_record_video_from_save_artifacts(self):
        save_run_artifacts, record_video = _resolve_eval_artifact_modes(
            {"eval_save_run_artifacts": True},
            _ArtifactArgs(),
            measurement_mode=False,
        )

        self.assertTrue(save_run_artifacts)
        self.assertFalse(record_video)

    def test_artifact_mode_record_video_implies_run_artifacts(self):
        save_run_artifacts, record_video = _resolve_eval_artifact_modes(
            {"eval_save_run_artifacts": False},
            _ArtifactArgs(record_video=True),
            measurement_mode=False,
        )

        self.assertTrue(save_run_artifacts)
        self.assertTrue(record_video)

    def test_artifact_mode_cli_overrides_can_disable_config_artifacts(self):
        save_run_artifacts, record_video = _resolve_eval_artifact_modes(
            {"eval_save_run_artifacts": True, "eval_record_video": True},
            _ArtifactArgs(no_save_run_artifacts=True),
            measurement_mode=False,
        )

        self.assertFalse(save_run_artifacts)
        self.assertFalse(record_video)

    def test_artifact_mode_measurement_disables_artifacts_and_video(self):
        save_run_artifacts, record_video = _resolve_eval_artifact_modes(
            {"eval_save_run_artifacts": True, "eval_record_video": True},
            _ArtifactArgs(),
            measurement_mode=True,
        )

        self.assertFalse(save_run_artifacts)
        self.assertFalse(record_video)

    def test_resolve_simulation_duration_uses_env_snapshot_when_config_omits_field(self):
        duration = _resolve_simulation_duration(
            config={"OPENAI_API_TYPE": "ollama"},
            env_config_snapshot={"duration": 30},
        )
        self.assertEqual(duration, 30)

    def test_resolve_simulation_duration_prefers_explicit_config_override(self):
        duration = _resolve_simulation_duration(
            config={"simulation_duration": 45},
            env_config_snapshot={"duration": 30},
        )
        self.assertEqual(duration, 45)

    def test_summarize_decision_latency_samples_uses_actual_decision_latencies(self):
        stats = _summarize_decision_latency_samples([1.0, 2.0, 3.0])
        self.assertEqual(stats["decision_latency_ms_avg"], 2000.0)
        self.assertAlmostEqual(stats["p95_decision_latency_sec"], 2.9, places=4)

    def test_summarize_decision_latency_samples_returns_none_when_no_decisions_exist(self):
        stats = _summarize_decision_latency_samples([])
        self.assertIsNone(stats["decision_latency_ms_avg"])
        self.assertIsNone(stats["p95_decision_latency_sec"])

    def test_annotate_aggregate_with_ollama_preflight_status_surfaces_failure_fields(self):
        aggregate = {"model": "qwen3:4b"}
        annotated = _annotate_aggregate_with_ollama_preflight_status(
            aggregate,
            {
                "qwen3:4b": {
                    "model": "qwen3:4b",
                    "ok": False,
                    "transport": "openai_compat_v1",
                    "elapsed_sec": None,
                    "error": "HTTPError: 404 Client Error",
                }
            },
        )
        self.assertFalse(annotated["ollama_preflight_ok"])
        self.assertEqual(annotated["ollama_preflight_transport"], "openai_compat_v1")
        self.assertEqual(annotated["ollama_preflight_error"], "HTTPError: 404 Client Error")

    def test_ollama_preflight_auto_uses_native_for_qwen_no_think(self):
        config = {
            "OLLAMA_USE_NATIVE_CHAT": "auto",
            "OLLAMA_THINK_MODE": "no_think",
            "OLLAMA_API_BASE": "http://localhost:11434/v1",
        }

        with patch("evaluate_models_ollama.requests.post") as post_mock:
            post_mock.return_value = _FakeResponse({"message": {"content": "4"}})
            result = _ollama_preflight_probe(config, "qwen3:4b", 15.0)

        self.assertEqual(result["transport"], "native_api_chat")
        self.assertEqual(result["ollama_native_chat_configured"], "auto")
        self.assertTrue(result["ollama_native_chat_effective"])
        self.assertEqual(result["ollama_native_chat_resolution_reason"], "thinking_family_no_think")
        self.assertEqual(post_mock.call_args.kwargs["json"]["think"], False)
        self.assertEqual(post_mock.call_args.kwargs["json"]["options"]["num_predict"], 8)
        self.assertIn("/api/chat", post_mock.call_args.args[0])

    def test_ollama_preflight_native_uses_safe_context_and_keep_alive_when_configured(self):
        config = {
            "OLLAMA_USE_NATIVE_CHAT": "auto",
            "OLLAMA_THINK_MODE": "no_think",
            "OLLAMA_API_BASE": "http://localhost:11434/v1",
            "ollama_runtime_num_ctx": 4096,
            "ollama_runtime_keep_alive": "0",
        }

        with patch("evaluate_models_ollama.requests.post") as post_mock:
            post_mock.return_value = _FakeResponse({"message": {"content": "4"}})
            _ollama_preflight_probe(config, "qwen3:4b", 15.0)

        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["num_predict"], 8)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["keep_alive"], "0")

    def test_ollama_preflight_safe_config_uses_native_for_llama(self):
        config = load_runtime_config("config.llm_full_safe.yaml")

        with patch("evaluate_models_ollama.requests.post") as post_mock:
            post_mock.return_value = _FakeResponse({"message": {"content": "4"}})
            result = _ollama_preflight_probe(config, "llama3.2:3b", 15.0)

        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(result["transport"], "native_api_chat")
        self.assertTrue(result["ollama_native_chat_effective"])
        self.assertEqual(payload["model"], "llama3.2:3b")
        self.assertEqual(payload["options"]["num_predict"], 8)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["keep_alive"], "10m")

    def test_ollama_runtime_controls_report_includes_config_and_env(self):
        config = {
            "ollama_runtime_num_ctx": 4096,
            "ollama_runtime_keep_alive": "0",
        }

        with patch.dict("os.environ", {"DILU_OLLAMA_NUM_CTX": "4096"}, clear=True):
            report = _ollama_runtime_controls_report(config)

        self.assertEqual(report["configured"]["ollama_runtime_num_ctx"], 4096)
        self.assertEqual(report["configured"]["ollama_runtime_keep_alive"], "0")
        self.assertEqual(report["exported_env"]["DILU_OLLAMA_NUM_CTX"], "4096")
        self.assertEqual(report["effective_env"]["DILU_OLLAMA_NUM_CTX"], "4096")
        self.assertIn("server environment variables", report["server_env_notice"])

    def test_ollama_case_unload_policy_defaults_disabled(self):
        policy = _ollama_case_unload_policy({})

        self.assertFalse(policy["enabled"])
        self.assertIsNone(policy["method"])

    def test_ollama_case_unload_stops_model_when_enabled(self):
        config = {
            "ollama_unload_after_case": True,
            "ollama_unload_after_case_timeout_sec": 15,
        }

        with patch("evaluate_models_ollama.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = "\x1b[?25ldone\x1b[0m"
            run_mock.return_value.stderr = "\x1b[31mwarning\x1b[0m"
            result = _stop_ollama_model_after_case(config, "llama3.2:3b")

        self.assertTrue(result["attempted"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "llama3.2:3b")
        self.assertEqual(result["stdout"], "done")
        self.assertEqual(result["stderr"], "warning")
        self.assertEqual(run_mock.call_args.args[0], ["ollama", "stop", "llama3.2:3b"])
        self.assertEqual(run_mock.call_args.kwargs["timeout"], 15.0)

    def test_sanitize_process_output_removes_ansi_and_control_codes(self):
        text = "\x1b[?25l\x1b[31merror\x1b[0m\x00\nok"

        cleaned = _sanitize_process_output(text)

        self.assertEqual(cleaned, "error\nok")

    def test_ollama_preflight_unloads_model_after_success_when_enabled(self):
        config = {
            "ollama_unload_after_case": True,
            "ollama_unload_after_case_timeout_sec": 15,
        }

        with patch("evaluate_models_ollama._ollama_preflight_probe") as probe_mock, patch(
            "evaluate_models_ollama.subprocess.run"
        ) as run_mock:
            probe_mock.return_value = {
                "model": "llama3.2:3b",
                "ok": True,
                "transport": "native_api_chat",
                "elapsed_sec": 0.1,
                "response_preview": "4",
            }
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = ""
            run_mock.return_value.stderr = ""
            results = _run_ollama_preflight(config, ["llama3.2:3b"], 15.0, quiet_mode=True)

        self.assertTrue(results[0]["ok"])
        self.assertTrue(results[0]["ollama_preflight_unload"]["attempted"])
        self.assertTrue(results[0]["ollama_preflight_unload"]["ok"])
        self.assertEqual(results[0]["ollama_preflight_unload"]["phase"], "preflight")
        self.assertEqual(run_mock.call_args.args[0], ["ollama", "stop", "llama3.2:3b"])

    def test_ollama_preflight_unloads_model_after_failure_when_enabled(self):
        config = {
            "OLLAMA_USE_NATIVE_CHAT": True,
            "OLLAMA_THINK_MODE": "auto",
            "ollama_unload_after_case": True,
            "ollama_unload_after_case_timeout_sec": 15,
        }

        with patch("evaluate_models_ollama._ollama_preflight_probe") as probe_mock, patch(
            "evaluate_models_ollama.subprocess.run"
        ) as run_mock:
            probe_mock.side_effect = TimeoutError("preflight timeout")
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = ""
            run_mock.return_value.stderr = ""
            results = _run_ollama_preflight(config, ["llama3.2:3b"], 15.0, quiet_mode=True)

        self.assertFalse(results[0]["ok"])
        self.assertIn("preflight timeout", results[0]["error"])
        self.assertTrue(results[0]["ollama_preflight_unload"]["attempted"])
        self.assertTrue(results[0]["ollama_preflight_unload"]["ok"])
        self.assertEqual(results[0]["ollama_preflight_unload"]["phase"], "preflight")

    def test_aggregate_results_includes_ollama_case_unload_counts(self):
        episodes = [
            {
                "crashed": False,
                "error": None,
                "success_no_collision": True,
                "truncated": False,
                "terminated": False,
                "steps": 2,
                "episode_runtime_sec": 1.0,
                "ollama_case_unload": {"attempted": True, "ok": True},
            },
            {
                "crashed": False,
                "error": None,
                "success_no_collision": True,
                "truncated": False,
                "terminated": False,
                "steps": 2,
                "episode_runtime_sec": 1.0,
                "ollama_case_unload": {"attempted": True, "ok": False},
            },
        ]

        aggregate = aggregate_results("llama3.2:3b", episodes)

        self.assertEqual(aggregate["ollama_case_unload_attempts"], 2)
        self.assertEqual(aggregate["ollama_case_unload_successes"], 1)
        self.assertEqual(aggregate["ollama_case_unload_failures"], 1)
        self.assertEqual(aggregate["ollama_case_unload_success_rate"], 0.5)

    def test_ollama_preflight_auto_uses_openai_compat_for_llama(self):
        config = {
            "OLLAMA_USE_NATIVE_CHAT": "auto",
            "OLLAMA_THINK_MODE": "no_think",
            "OLLAMA_API_BASE": "http://localhost:11434/v1",
        }

        with patch("evaluate_models_ollama.requests.post") as post_mock:
            post_mock.return_value = _FakeResponse(
                {"choices": [{"message": {"content": "4"}}]}
            )
            result = _ollama_preflight_probe(config, "llama3.2:3b", 15.0)

        self.assertEqual(result["transport"], "openai_compat_v1")
        self.assertEqual(result["ollama_native_chat_configured"], "auto")
        self.assertFalse(result["ollama_native_chat_effective"])
        self.assertEqual(result["ollama_native_chat_resolution_reason"], "non_thinking_model")
        self.assertNotIn("think", post_mock.call_args.kwargs["json"])
        self.assertIn("/v1/chat/completions", post_mock.call_args.args[0])

    def test_classify_ollama_preflight_failure_distinguishes_hard_and_soft_failures(self):
        self.assertEqual(
            _classify_ollama_preflight_failure(
                {
                    "ok": False,
                    "error": "HTTPError: 404 Client Error: Not Found",
                    "status_code": 404,
                }
            ),
            "hard",
        )
        self.assertEqual(
            _classify_ollama_preflight_failure(
                {
                    "ok": False,
                    "error": "ReadTimeout: request timed out",
                    "status_code": None,
                }
            ),
            "soft",
        )

    def test_build_skipped_model_aggregate_marks_model_incomplete(self):
        agg = _build_skipped_model_aggregate(
            model_name="qwen3:4b",
            planned_episode_count=5,
            reason="hard_ollama_preflight_failure",
            preflight_probe={
                "model": "qwen3:4b",
                "ok": False,
                "transport": "openai_compat_v1",
                "error": "HTTPError: 404 Client Error",
            },
            benchmark_mode=True,
        )
        self.assertEqual(agg["model"], "qwen3:4b")
        self.assertEqual(agg["planned_episode_count"], 5)
        self.assertEqual(agg["executed_episode_count"], 0)
        self.assertEqual(agg["skipped_episode_count"], 5)
        self.assertFalse(agg["episode_execution_complete"])
        self.assertTrue(agg["model_skipped_due_to_preflight"])
        self.assertEqual(agg["model_skipped_reason"], "hard_ollama_preflight_failure")
        self.assertFalse(agg["benchmark_result_valid"])
        self.assertIn("incomplete_episode_set", agg["benchmark_result_invalid_reason"])
        self.assertFalse(agg["ollama_preflight_ok"])

    def test_build_measurement_integrity_summary_lists_preflight_failures(self):
        summary = _build_measurement_integrity_summary(
            [
                {"model": "llama3.2:1b", "ok": True, "transport": "openai_compat_v1", "elapsed_sec": 0.5},
                {"model": "qwen3:4b", "ok": False, "transport": "openai_compat_v1", "error": "HTTPError: 404 Client Error", "timeout_sec": 15.0},
            ],
            "Ollama preflight failed before evaluation.",
        )
        self.assertEqual(summary["measurement_integrity_warnings"], ["Ollama preflight failed before evaluation."])
        self.assertEqual(len(summary["ollama_preflight_failed_models"]), 1)
        self.assertEqual(summary["ollama_preflight_failed_models"][0]["model"], "qwen3:4b")
        self.assertEqual(summary["skipped_models_due_to_preflight"], [])
        self.assertEqual(summary["quarantined_models_due_to_timeout_collapse"], [])


if __name__ == "__main__":
    unittest.main()
