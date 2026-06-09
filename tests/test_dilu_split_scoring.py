import unittest

from dilu.runtime.dilu_scoring import (
    SPLIT_SCORING_POLICY_VERSION,
    compute_split_scores_for_episode,
)
from evaluate_models_ollama import aggregate_results


def _episode(**updates):
    base = {
        "crashed": False,
        "success_no_collision": True,
        "task_completed": True,
        "completion_rate": 1.0,
        "ttc_danger_rate": 0.0,
        "headway_violation_rate": 0.0,
        "rear_ttc_danger_rate": 0.0,
        "rear_headway_violation_rate": 0.0,
        "unsafe_lane_change_attempt_count": 0,
        "unsafe_longitudinal_action_attempt_count": 0,
        "time_efficiency_score": 0.8,
        "avg_ego_speed_mps": 24.0,
        "benchmark_max_progress_m": 120.0,
        "speed_variance_score": 0.9,
        "low_speed_blocking_rate": 0.0,
        "stop_rate": 0.0,
        "near_stop_rate": 0.0,
        "flap_accel_decel_rate": 0.0,
        "lane_change_rate": 0.05,
        "responses_strict_format": 10,
        "responses_with_delimiter": 10,
        "responses_direct_parseable": 10,
        "format_failure_count": 0,
        "decisions_made": 10,
        "decision_calls_total": 10,
        "decision_timeout_count": 0,
        "fallback_action_count": 0,
        "episode_stop_reason": "completed",
        "lane_change_shield_count": 0,
        "longitudinal_safety_shield_count": 0,
        "semantic_recovery_count": 0,
        "intent_resolver_used_count": 0,
        "decision_latency_ms_avg": 800.0,
        "p95_decision_latency_sec": 1.0,
        "completion_tokens_total": 120,
        "total_tokens": 600,
        "tokens_per_second": 20.0,
        "energy_per_decision_j": None,
        "energy_per_token_j": None,
    }
    base.update(updates)
    return base


class DiluSplitScoringTests(unittest.TestCase):
    def test_clean_episode_has_high_driving_and_llm_scores(self):
        scored = compute_split_scores_for_episode(
            _episode(driving_score_v2=0.81, driving_score=0.75)
        )

        self.assertEqual(
            scored["split_scoring_policy_version"],
            SPLIT_SCORING_POLICY_VERSION,
        )
        self.assertGreater(scored["driving_score_behavior_v1"], 0.8)
        self.assertEqual(scored["driving_task_score_v2"], 0.81)
        self.assertEqual(scored["driving_task_score_source"], "driving_score_v2")
        self.assertGreater(scored["driving_score_balanced_v1"], 0.8)
        self.assertGreater(scored["llm_driver_score_v1"], 0.8)
        self.assertGreater(scored["dilu_joint_score_v1"], 0.8)

    def test_balanced_driving_score_uses_geometric_mean(self):
        scored = compute_split_scores_for_episode(_episode(driving_score_v2=0.25))
        expected = round(
            (scored["driving_score_behavior_v1"] * scored["driving_task_score_v2"]) ** 0.5,
            4,
        )

        self.assertEqual(scored["driving_score_balanced_v1"], expected)
        self.assertAlmostEqual(
            scored["driving_behavior_task_gap_v1"],
            round(scored["driving_score_behavior_v1"] - 0.25, 4),
            places=4,
        )

    def test_zero_task_score_zeros_balanced_driving_score(self):
        scored = compute_split_scores_for_episode(_episode(driving_score_v2=0.0))

        self.assertGreater(scored["driving_score_behavior_v1"], 0.0)
        self.assertEqual(scored["driving_task_score_v2"], 0.0)
        self.assertEqual(scored["driving_score_balanced_v1"], 0.0)

    def test_balanced_driving_score_prefers_v2_and_falls_back_to_legacy(self):
        preferred = compute_split_scores_for_episode(
            _episode(driving_score_v2=0.64, driving_score=0.16)
        )
        fallback = compute_split_scores_for_episode(_episode(driving_score=0.49))

        self.assertEqual(preferred["driving_task_score_v2"], 0.64)
        self.assertEqual(preferred["driving_task_score_source"], "driving_score_v2")
        self.assertEqual(fallback["driving_task_score_v2"], 0.49)
        self.assertEqual(fallback["driving_task_score_source"], "driving_score")

    def test_missing_task_score_leaves_balanced_driving_score_null(self):
        scored = compute_split_scores_for_episode(_episode())

        self.assertIsNone(scored["driving_task_score_v2"])
        self.assertIsNone(scored["driving_task_score_source"])
        self.assertIsNone(scored["driving_score_balanced_v1"])
        self.assertIsNone(scored["driving_behavior_task_gap_v1"])

    def test_crash_reduces_but_does_not_zero_soft_driving_score(self):
        clean = compute_split_scores_for_episode(_episode())
        crashed = compute_split_scores_for_episode(
            _episode(crashed=True, success_no_collision=False)
        )

        self.assertLess(crashed["driving_safety_score_v1"], clean["driving_safety_score_v1"])
        self.assertLess(crashed["driving_score_behavior_v1"], clean["driving_score_behavior_v1"])
        self.assertGreater(crashed["driving_score_behavior_v1"], 0.0)

    def test_safety_and_comfort_penalties_affect_driving_subscores(self):
        risky = compute_split_scores_for_episode(
            _episode(
                ttc_danger_rate=0.4,
                headway_violation_rate=0.3,
                rear_ttc_danger_rate=0.2,
                low_speed_blocking_rate=0.5,
                stop_rate=0.2,
                near_stop_rate=0.4,
                flap_accel_decel_rate=0.3,
            )
        )
        clean = compute_split_scores_for_episode(_episode())

        self.assertLess(risky["driving_safety_score_v1"], clean["driving_safety_score_v1"])
        self.assertLess(risky["driving_comfort_score_v1"], clean["driving_comfort_score_v1"])

    def test_llm_score_penalizes_runtime_and_action_validity_failures(self):
        clean = compute_split_scores_for_episode(_episode())
        bad = compute_split_scores_for_episode(
            _episode(
                responses_strict_format=4,
                responses_direct_parseable=5,
                format_failure_count=6,
                decision_timeout_count=3,
                fallback_action_count=4,
                episode_stop_reason="episode_timeout_cap",
                lane_change_shield_count=2,
                longitudinal_safety_shield_count=1,
                semantic_recovery_count=2,
                intent_resolver_used_count=2,
                decision_latency_ms_avg=8000.0,
                p95_decision_latency_sec=12.0,
                completion_tokens_total=800,
                total_tokens=4000,
                tokens_per_second=2.0,
            )
        )

        self.assertLess(bad["llm_output_contract_score_v1"], clean["llm_output_contract_score_v1"])
        self.assertLess(bad["llm_runtime_reliability_score_v1"], clean["llm_runtime_reliability_score_v1"])
        self.assertLess(bad["llm_action_validity_score_v1"], clean["llm_action_validity_score_v1"])
        self.assertLess(
            bad["llm_intervention_independence_score_v1"],
            clean["llm_intervention_independence_score_v1"],
        )
        self.assertLess(bad["llm_latency_score_v1"], clean["llm_latency_score_v1"])
        self.assertLess(bad["llm_driver_score_v1"], clean["llm_driver_score_v1"])

    def test_flow_recovery_dependence_penalizes_llm_action_validity(self):
        clean = compute_split_scores_for_episode(_episode())
        recovered = compute_split_scores_for_episode(
            _episode(
                flow_recovery_shield_count=5,
                decisions_made=10,
                decision_calls_total=10,
            )
        )

        self.assertEqual(clean["llm_flow_recovery_independence_score_v1"], 1.0)
        self.assertEqual(recovered["llm_flow_recovery_independence_score_v1"], 0.0)
        self.assertLess(recovered["llm_action_validity_score_v1"], clean["llm_action_validity_score_v1"])
        self.assertLess(recovered["llm_driver_score_v1"], clean["llm_driver_score_v1"])

    def test_llm_score_does_not_penalize_driving_failure_without_interventions(self):
        clean = compute_split_scores_for_episode(_episode())
        poor_driving = compute_split_scores_for_episode(
            _episode(
                task_completed=False,
                completion_rate=0.0,
                avg_ego_speed_mps=3.0,
                low_speed_blocking_rate=0.8,
                stop_rate=0.5,
                near_stop_rate=0.8,
                flow_recovery_shield_count=0,
                lane_change_shield_count=0,
                longitudinal_safety_shield_count=0,
                unsafe_lane_change_attempt_count=0,
                unsafe_longitudinal_action_attempt_count=0,
                semantic_recovery_count=0,
                intent_resolver_used_count=0,
            )
        )

        self.assertLess(poor_driving["driving_score_behavior_v1"], clean["driving_score_behavior_v1"])
        self.assertEqual(poor_driving["llm_action_validity_score_v1"], clean["llm_action_validity_score_v1"])
        self.assertEqual(poor_driving["llm_driver_score_v1"], clean["llm_driver_score_v1"])

    def test_gemma_like_flow_recovery_pattern_lowers_llm_score(self):
        scored = compute_split_scores_for_episode(
            _episode(
                flow_recovery_shield_count=9,
                decisions_made=22,
                decision_calls_total=22,
                responses_strict_format=22,
                responses_with_delimiter=22,
                responses_direct_parseable=22,
                format_failure_count=0,
                decision_timeout_count=0,
                fallback_action_count=0,
                decision_latency_ms_avg=800.0,
                p95_decision_latency_sec=1.0,
            )
        )

        self.assertLess(scored["llm_driver_score_v1"], 0.85)
        self.assertEqual(scored["llm_flow_recovery_independence_score_v1"], 0.0)

    def test_llama_like_minor_parse_recovery_stays_high(self):
        scored = compute_split_scores_for_episode(
            _episode(
                semantic_recovery_count=1,
                decisions_made=22,
                decision_calls_total=22,
                responses_strict_format=21,
                responses_with_delimiter=22,
                responses_direct_parseable=22,
                format_failure_count=1,
                flow_recovery_shield_count=0,
            )
        )

        self.assertGreater(scored["llm_driver_score_v1"], 0.95)

    def test_missing_energy_is_not_penalized_when_token_latency_exist(self):
        without_energy = compute_split_scores_for_episode(
            _episode(energy_per_decision_j=None, energy_per_token_j=None)
        )
        with_good_energy = compute_split_scores_for_episode(
            _episode(energy_per_decision_j=0.1, energy_per_token_j=0.001)
        )

        self.assertAlmostEqual(
            without_energy["llm_resource_efficiency_score_v1"],
            with_good_energy["llm_resource_efficiency_score_v1"],
            places=4,
        )

    def test_aggregate_results_contains_split_score_means(self):
        episodes = [
            compute_split_scores_for_episode(
                _episode(
                    seed=i,
                    driving_score_v2=0.8,
                    error=None,
                    truncated=False,
                    terminated=True,
                    steps=10,
                    episode_runtime_sec=2.0,
                    episode_reward_sum=8.0,
                    episode_reward_avg=0.8,
                )
            )
            for i in range(3)
        ]
        aggregate = aggregate_results("split-model", episodes)

        self.assertIn("driving_score_behavior_v1", aggregate)
        self.assertIn("driving_score_balanced_v1", aggregate)
        self.assertIn("driving_task_score_v2", aggregate)
        self.assertIn("driving_behavior_task_gap_v1", aggregate)
        self.assertIn("llm_driver_score_v1", aggregate)
        self.assertIn("dilu_joint_score_v1", aggregate)
        self.assertIn("llm_flow_recovery_independence_score_v1", aggregate)
        self.assertIn("llm_intervention_independence_score_v1", aggregate)
        self.assertIn("driving_score_balanced_v1_ci95", aggregate)
        self.assertIn("driving_task_score_v2_ci95", aggregate)
        self.assertIn("driving_score_behavior_v1_ci95", aggregate)
        self.assertIn("llm_driver_score_v1_ci95", aggregate)
        self.assertIn("llm_flow_recovery_independence_score_v1_ci95", aggregate)

    def test_aggregate_balanced_score_is_mean_of_episode_balanced_scores(self):
        episodes = [
            _episode(
                seed=1,
                error=None,
                truncated=False,
                terminated=True,
                steps=10,
                episode_runtime_sec=1.0,
                episode_reward_sum=1.0,
                episode_reward_avg=0.1,
                driving_score_behavior_v1=1.0,
                driving_task_score_v2=0.0,
                driving_score_balanced_v1=0.0,
                driving_behavior_task_gap_v1=1.0,
            ),
            _episode(
                seed=2,
                error=None,
                truncated=False,
                terminated=True,
                steps=10,
                episode_runtime_sec=1.0,
                episode_reward_sum=1.0,
                episode_reward_avg=0.1,
                driving_score_behavior_v1=0.0,
                driving_task_score_v2=1.0,
                driving_score_balanced_v1=0.0,
                driving_behavior_task_gap_v1=-1.0,
            ),
        ]
        aggregate = aggregate_results("balanced-aggregate-model", episodes)

        self.assertEqual(aggregate["driving_score_balanced_v1"], 0.0)
        self.assertNotEqual(
            aggregate["driving_score_balanced_v1"],
            round((0.5 * 0.5) ** 0.5, 4),
        )


if __name__ == "__main__":
    unittest.main()
