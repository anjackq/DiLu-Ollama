import unittest
from unittest.mock import patch

from evaluate_models_ollama import (
    _annotate_aggregate_with_ollama_preflight_status,
    _build_skipped_model_aggregate,
    _build_measurement_integrity_summary,
    _classify_ollama_preflight_failure,
    _ollama_preflight_probe,
    _resolve_eval_artifact_modes,
    _resolve_simulation_duration,
    _summarize_decision_latency_samples,
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
        self.assertIn("/api/chat", post_mock.call_args.args[0])

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
