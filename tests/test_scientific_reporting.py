import tempfile
import unittest
from pathlib import Path

from dilu.runtime.scientific_reporting import (
    annotate_aggregate_with_scientific_reporting,
    bootstrap_ci95,
    build_primary_metric_spec,
    continuous_metric_summary,
    wilson_ci95,
    write_scientific_analysis_artifacts,
)
from evaluate_models_ollama import aggregate_results


def _clean_episode(seed=1, **updates):
    episode = {
        "seed": seed,
        "crashed": False,
        "error": None,
        "success_no_collision": True,
        "truncated": False,
        "terminated": True,
        "steps": 30,
        "episode_runtime_sec": 3.0,
        "decisions_made": 30,
        "decision_calls_total": 30,
        "decision_timeout_count": 0,
        "fallback_action_count": 0,
        "responses_with_delimiter": 30,
        "responses_strict_format": 30,
        "responses_direct_parseable": 30,
        "format_failure_count": 0,
        "episode_reward_sum": 24.0,
        "episode_reward_avg": 0.8,
        "avg_ego_speed_mps": 24.0,
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
        "decision_latency_ms_avg": 1000.0,
    }
    episode.update(updates)
    return episode


class ScientificReportingTests(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic(self):
        first = bootstrap_ci95([1.0, 2.0, 3.0], iterations=200, seed=7)
        second = bootstrap_ci95([1.0, 2.0, 3.0], iterations=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_wilson_ci_handles_binary_rate(self):
        summary = wilson_ci95(successes=0, total=10)
        self.assertEqual(summary, [0.0, summary[1]])
        self.assertGreater(summary[1], 0.0)

    def test_continuous_summary_handles_empty_and_single_sample(self):
        empty = continuous_metric_summary([])
        self.assertIsNone(empty["mean"])
        self.assertIn("missing_sample_warning", empty["warnings"])

        single = continuous_metric_summary([4.0])
        self.assertEqual(single["ci95"], [4.0, 4.0])
        self.assertIn("low_n_warning", single["warnings"])

    def test_primary_metric_prefers_balanced_benchmark_driving_score(self):
        aggregate = {
            "model": "test-model",
            "episodes": 30,
            "crashes": 0,
            "no_collision_rate": 1.0,
            "ttc_danger_rate_mean": 0.0,
            "headway_violation_rate_mean": 0.0,
            "rear_ttc_danger_rate_mean": 0.0,
            "rear_headway_violation_rate_mean": 0.0,
            "decision_timeout_rate_mean": 0.0,
            "fallback_action_rate_mean": 0.0,
            "response_strict_format_rate": 1.0,
            "driving_score_balanced_v1": 0.76,
            "driving_task_score_v2": 0.72,
            "driving_score_v2": 0.72,
            "driving_score_behavior_v1": 0.81,
            "llm_driver_score_v1": 0.91,
            "dilu_joint_score_v1": 0.8587,
            "avg_reward_per_step": 1.0,
        }
        annotated = annotate_aggregate_with_scientific_reporting(aggregate, [_clean_episode() for _ in range(30)])
        self.assertEqual(annotated["primary_metric_name"], "driving_score_balanced_v1")
        self.assertEqual(annotated["primary_metric_value"], 0.76)
        self.assertEqual(annotated["primary_llm_metric_name"], "llm_driver_score_v1")
        self.assertEqual(annotated["primary_llm_metric_value"], 0.91)
        self.assertEqual(annotated["secondary_joint_metric_name"], "dilu_joint_score_v1")
        self.assertTrue(annotated["primary_metric_valid"])
        self.assertEqual(annotated["scientific_validity_status"], "valid_for_claim")

    def test_seed_mode_primary_metric_uses_speed_not_reward(self):
        episodes = [_clean_episode(seed=i) for i in range(30)]
        aggregate = aggregate_results("seed-model", episodes)
        self.assertEqual(aggregate["primary_metric_name"], "avg_ego_speed_mps")
        self.assertNotEqual(aggregate["primary_metric_name"], "avg_reward_per_step")
        self.assertEqual(aggregate["scientific_validity_status"], "valid_for_claim")
        self.assertIn("scientific_stats", aggregate)
        self.assertFalse(aggregate["scientific_stats"]["hypothesis_tests_enabled"])

    def test_smoke_n3_is_exploratory_only(self):
        episodes = [_clean_episode(seed=i) for i in range(3)]
        aggregate = aggregate_results("smoke-model", episodes)
        self.assertTrue(aggregate["primary_metric_valid"])
        self.assertEqual(aggregate["scientific_validity_status"], "exploratory_only")
        self.assertIn("n_episodes<30", aggregate["scientific_validity_reasons"])

    def test_timeout_heavy_run_is_exploratory_only(self):
        episodes = [
            _clean_episode(
                seed=i,
                decision_timeout_count=3,
                fallback_action_count=3,
                responses_strict_format=20,
            )
            for i in range(30)
        ]
        aggregate = aggregate_results("timeout-model", episodes)
        self.assertFalse(aggregate["primary_metric_valid"])
        self.assertEqual(aggregate["scientific_validity_status"], "exploratory_only")
        self.assertIn("decision_timeout_rate_mean>0.05", aggregate["scientific_validity_reasons"])

    def test_safety_failure_is_failure_analysis_only(self):
        episodes = [_clean_episode(seed=i) for i in range(30)]
        episodes[0]["crashed"] = True
        episodes[0]["success_no_collision"] = False
        aggregate = aggregate_results("unsafe-model", episodes)
        self.assertFalse(aggregate["primary_metric_valid"])
        self.assertEqual(aggregate["scientific_validity_status"], "failure_analysis_only")
        self.assertIn("crashes>0", aggregate["scientific_validity_reasons"])

    def test_write_scientific_artifacts_outputs_expected_files(self):
        spec = build_primary_metric_spec()
        report = {
            "experiment_id": "unit_test",
            "metrics_config": {"primary_metric_spec": spec},
            "aggregates": [
                annotate_aggregate_with_scientific_reporting(
                    {
                        "model": "test-model",
                        "episodes": 3,
                        "crashes": 0,
                        "no_collision_rate": 1.0,
                        "ttc_danger_rate_mean": 0.0,
                        "headway_violation_rate_mean": 0.0,
                        "rear_ttc_danger_rate_mean": 0.0,
                        "rear_headway_violation_rate_mean": 0.0,
                        "decision_timeout_rate_mean": 0.0,
                        "fallback_action_rate_mean": 0.0,
                        "response_strict_format_rate": 1.0,
                        "avg_ego_speed_mps": 24.0,
                        "driving_score_balanced_v1": 0.77,
                        "driving_task_score_v2": 0.74,
                        "driving_behavior_task_gap_v1": 0.06,
                        "driving_score_behavior_v1": 0.8,
                        "llm_driver_score_v1": 0.9,
                        "llm_flow_recovery_independence_score_v1": 1.0,
                        "llm_intervention_independence_score_v1": 1.0,
                        "dilu_joint_score_v1": 0.8485,
                    },
                    [_clean_episode(seed=i) for i in range(3)],
                    spec,
                )
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            outputs = write_scientific_analysis_artifacts(report, tmp)
            self.assertTrue(Path(outputs["scientific_summary_md"]).is_file())
            self.assertTrue(Path(outputs["stats_appendix_md"]).is_file())
            self.assertTrue(Path(outputs["metrics_table_csv"]).is_file())
            table_text = Path(outputs["metrics_table_csv"]).read_text(encoding="utf-8")
            self.assertIn("driving_score_balanced_v1", table_text)
            self.assertIn("driving_task_score_v2", table_text)
            self.assertIn("driving_behavior_task_gap_v1", table_text)
            self.assertIn("driving_score_behavior_v1", table_text)
            self.assertIn("llm_driver_score_v1", table_text)
            self.assertIn("llm_flow_recovery_independence_score_v1", table_text)
            self.assertIn("llm_intervention_independence_score_v1", table_text)

    def test_scientific_stats_include_llm_intervention_diagnostics(self):
        episodes = [
            _clean_episode(
                seed=i,
                driving_score_balanced_v1=0.8,
                driving_task_score_v2=0.75,
                driving_behavior_task_gap_v1=0.05,
                llm_flow_recovery_independence_score_v1=1.0,
                llm_safety_intervention_independence_score_v1=1.0,
                llm_parser_independence_score_v1=1.0,
                llm_intervention_independence_score_v1=1.0,
            )
            for i in range(3)
        ]
        aggregate = aggregate_results("diagnostic-model", episodes)
        continuous = aggregate["scientific_stats"]["continuous_metrics"]

        self.assertIn("driving_score_balanced_v1", continuous)
        self.assertIn("driving_task_score_v2", continuous)
        self.assertIn("driving_behavior_task_gap_v1", continuous)
        self.assertIn("llm_flow_recovery_independence_score_v1", continuous)
        self.assertIn("llm_safety_intervention_independence_score_v1", continuous)
        self.assertIn("llm_parser_independence_score_v1", continuous)
        self.assertIn("llm_intervention_independence_score_v1", continuous)


if __name__ == "__main__":
    unittest.main()
