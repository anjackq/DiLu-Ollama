import tempfile
import unittest
from unittest.mock import patch

from plot_eval_compare import (
    _display_model_label,
    _plot_grid,
    _shorten_model_name,
    _should_use_horizontal_layout,
    plot_aggregates,
)


class PlotEvalCompareTests(unittest.TestCase):
    def test_shorten_model_name_humanizes_local_and_base_tags(self):
        self.assertEqual(
            _shorten_model_name("dilu-phi4-mini-3_8b-instruct-v1"),
            "dilu-phi4-mini:3.8-instruct",
        )
        self.assertEqual(
            _shorten_model_name("dilu-deepseek-r1-1_5b-v3"),
            "dilu-deepseek-r1:1.5",
        )
        self.assertEqual(
            _shorten_model_name("dilu-qwen3-1_7b-v1"),
            "dilu-qwen3:1.7",
        )
        self.assertEqual(_shorten_model_name("phi4-mini:3.8b"), "phi4-mini:3.8")
        self.assertEqual(_shorten_model_name("qwen3:4b"), "qwen3:4")

    def test_display_model_label_drops_invalid_and_keeps_other_status_tags(self):
        label = _display_model_label(
            {
                "model": "dilu-phi4-mini-3_8b-instruct-v1",
                "benchmark_result_valid": False,
                "model_skipped_due_to_preflight": True,
                "model_quarantined_due_to_timeout_collapse": True,
                "episode_execution_complete": False,
            }
        )

        self.assertIn("dilu-phi4-mini:3.8-instruct", label)
        self.assertNotIn("INVALID", label)
        self.assertIn("PREFLIGHT_SKIP", label)
        self.assertIn("QUARANTINED", label)
        self.assertIn("INCOMPLETE", label)

    def test_plot_grid_rotates_xticks_90_degrees(self):
        charts = [
            {
                "values": [0.1, 0.2],
                "title": "Task Completion Rate",
                "ylim": (0, 1),
                "color": "#1f78b4",
            }
        ]
        captured = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"

            def _capture_savefig(*args, **kwargs):
                fig = __import__("matplotlib.pyplot").pyplot.gcf()
                captured["rotations"] = [label.get_rotation() for label in fig.axes[0].get_xticklabels()]

            with patch("matplotlib.pyplot.savefig", side_effect=_capture_savefig), patch("matplotlib.pyplot.close"):
                _plot_grid(["model_a", "model_b"], charts, "Test Plot", out)

        self.assertEqual(captured["rotations"], [90.0, 90.0])

    def test_dense_layout_switches_to_horizontal_bars(self):
        self.assertFalse(_should_use_horizontal_layout(8))
        self.assertTrue(_should_use_horizontal_layout(9))

    def test_default_benchmark_plot_splits_into_task_behavior_and_efficiency_figures(self):
        report = {
            "aggregates": [
                {
                    "model": "base-model",
                    "task_completion_rate": 0.2,
                    "driving_score_v2": 0.1,
                    "ttc_score_mean": 0.9,
                    "time_efficiency_score_mean": 0.4,
                    "stop_episode_rate": 0.8,
                    "near_stop_rate_mean": 0.6,
                    "decision_latency_ms_avg_mean": 1200.0,
                    "tokens_per_second_mean": 80.0,
                    "completion_tokens_total_mean": 200.0,
                },
                {
                    "model": "ft-model",
                    "task_completion_rate": 0.5,
                    "driving_score_v2": 0.3,
                    "ttc_score_mean": 0.95,
                    "time_efficiency_score_mean": 0.7,
                    "stop_episode_rate": 0.3,
                    "near_stop_rate_mean": 0.2,
                    "decision_latency_ms_avg_mean": 900.0,
                    "tokens_per_second_mean": 110.0,
                    "completion_tokens_total_mean": 180.0,
                },
            ]
        }
        captured = []

        def _capture(models, charts, title, output_path):
            captured.append(
                {
                    "models": list(models),
                    "titles": [chart["title"] for chart in charts],
                    "title": title,
                    "output_path": output_path,
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                result = plot_aggregates(report, out)

        self.assertEqual(result["primary_path"], out)
        self.assertEqual(result["companion_paths"], [f"{tmpdir}\\plot_behavior.png", f"{tmpdir}\\plot_efficiency.png"])
        self.assertEqual([item["output_path"] for item in captured], [out, f"{tmpdir}\\plot_behavior.png", f"{tmpdir}\\plot_efficiency.png"])
        self.assertEqual(captured[0]["titles"], ["Driving Score v2", "Task Completion Rate", "TTC Score Mean", "Time Efficiency Score Mean"])
        self.assertEqual(captured[1]["titles"], ["Stop Episode Rate", "Near-Stop Rate Mean"])
        self.assertIn("Decision Latency Mean (ms)", captured[2]["titles"])
        self.assertEqual(captured[0]["models"][0], "ft-model")

    def test_default_benchmark_plot_adds_energy_companion_when_energy_metrics_exist(self):
        report = {
            "aggregates": [
                {
                    "model": "energy-model",
                    "task_completion_rate": 0.2,
                    "driving_score_v2": 0.1,
                    "ttc_score_mean": 0.9,
                    "time_efficiency_score_mean": 0.4,
                    "stop_episode_rate": 0.8,
                    "near_stop_rate_mean": 0.6,
                    "net_energy_j_mean": 25.0,
                    "energy_per_decision_j_mean": 2.0,
                }
            ]
        }
        captured = []

        def _capture(models, charts, title, output_path):
            captured.append(output_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                result = plot_aggregates(report, out)

        self.assertIn(f"{tmpdir}\\plot_energy.png", result["companion_paths"])
        self.assertIn(f"{tmpdir}\\plot_energy.png", captured)

    def test_extended_benchmark_emits_runtime_companion(self):
        report = {
            "aggregates": [
                {
                    "model": "qwen3:1.7b",
                    "task_completion_rate": 0.3,
                    "driving_score_v2": 0.08,
                    "ttc_score_mean": 0.94,
                    "time_efficiency_score_mean": 0.13,
                    "stop_episode_rate": 0.55,
                    "near_stop_rate_mean": 0.24,
                    "decision_timeout_rate_mean": 0.3,
                    "fallback_action_rate_mean": 0.4,
                    "timeout_episode_rate": 0.2,
                    "ttc_danger_rate_mean": 0.1,
                }
            ]
        }
        captured = []

        def _capture(models, charts, title, output_path):
            captured.append({"titles": [chart["title"] for chart in charts], "path": output_path})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                result = plot_aggregates(report, out, extended=True)

        self.assertIn(f"{tmpdir}\\plot_runtime.png", result["companion_paths"])
        runtime_call = next(item for item in captured if item["path"].endswith("_runtime.png"))
        self.assertIn("Decision Timeout Rate", runtime_call["titles"])
        self.assertIn("Fallback Action Rate", runtime_call["titles"])

    def test_nonbenchmark_default_plot_emits_runtime_summary_and_efficiency_companion(self):
        report = {
            "aggregates": [
                {
                    "model": "phi4-mini:3.8b",
                    "crash_rate": 0.0,
                    "no_collision_rate": 1.0,
                    "avg_steps": 30.0,
                    "avg_episode_runtime_sec": 620.0,
                    "decision_latency_ms_avg_mean": 20000.0,
                    "tokens_per_second_mean": 40.0,
                }
            ]
        }
        captured = []

        def _capture(models, charts, title, output_path):
            captured.append({"titles": [chart["title"] for chart in charts], "path": output_path})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                result = plot_aggregates(report, out)

        self.assertEqual(result["primary_path"], out)
        self.assertIn(f"{tmpdir}\\plot_efficiency.png", result["companion_paths"])
        self.assertEqual(captured[0]["titles"], ["Crash Rate", "No-Collision Rate", "Average Steps", "Avg Episode Runtime (s)"])

    def test_invalid_and_quarantined_status_labels_remain_visible_without_invalid_tag(self):
        report = {
            "aggregates": [
                {
                    "model": "phi4-mini:3.8b",
                    "task_completion_rate": 0.0,
                    "driving_score_v2": 0.0,
                    "ttc_score_mean": 0.0,
                    "time_efficiency_score_mean": 0.0,
                    "benchmark_result_valid": False,
                    "model_skipped_due_to_preflight": True,
                    "model_quarantined_due_to_timeout_collapse": True,
                    "episode_execution_complete": False,
                }
            ]
        }
        captured = {}

        def _capture(models, charts, title, output_path):
            captured["models"] = list(models)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                plot_aggregates(report, out)

        self.assertIn("phi4-mini:3.8", captured["models"][0])
        self.assertNotIn("INVALID", captured["models"][0])
        self.assertIn("PREFLIGHT_SKIP", captured["models"][0])
        self.assertIn("QUARANTINED", captured["models"][0])
        self.assertIn("INCOMPLETE", captured["models"][0])

    def test_all_metrics_paginates_and_omits_ollama_diagnostic_charts(self):
        report = {
            "aggregates": [
                {
                    "model": "qwen3:1.7b",
                    "task_completion_rate": 0.3,
                    "driving_score": 0.19,
                    "driving_score_v2": 0.08,
                    "ttc_score_mean": 0.94,
                    "speed_variance_score_mean": 0.02,
                    "time_efficiency_score_mean": 0.13,
                    "overall_score_mean": 0.5,
                    "overall_score_v2_mean": 0.22,
                    "stop_episode_rate": 0.55,
                    "near_stop_rate_mean": 0.24,
                    "behavior_penalty_factor_v2_mean": 0.42,
                    "decision_latency_ms_avg_mean": 1700.0,
                    "tokens_per_second_mean": 12.0,
                    "prompt_tokens_total_mean": 150.0,
                    "completion_tokens_total_mean": 220.0,
                    "total_tokens_mean": 370.0,
                    "decision_timeout_rate_mean": 0.1,
                    "fallback_action_rate_mean": 0.2,
                    "timeout_episode_rate": 0.05,
                    "ttc_danger_rate_mean": 0.01,
                    "ollama_native_retry_rate_mean": 0.5,
                    "ollama_openai_fallback_rate_mean": 0.2,
                }
            ]
        }
        captured_titles = []

        def _capture(models, charts, title, output_path):
            captured_titles.extend(chart["title"] for chart in charts)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                result = plot_aggregates(report, out, all_metrics=True)

        self.assertIn(f"{tmpdir}\\plot_runtime.png", result["companion_paths"])
        self.assertIn("Prompt Tokens Mean", captured_titles)
        self.assertIn("Driving Score v2", captured_titles)
        self.assertNotIn("Ollama Native Retry Rate", captured_titles)
        self.assertNotIn("Ollama /v1 Fallback Rate", captured_titles)


if __name__ == "__main__":
    unittest.main()
