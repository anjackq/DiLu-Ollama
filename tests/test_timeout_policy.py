import unittest

from dilu.runtime.model_policy import (
    build_decision_timeout_penalty_state,
    decision_timeout_penalty_snapshot,
    update_decision_timeout_penalty_state,
)
from evaluate_models_ollama import (
    _should_early_stop_timeout_episode,
    aggregate_results,
)


def _base_episode(**overrides):
    episode = {
        "crashed": False,
        "error": None,
        "success_no_collision": True,
        "truncated": False,
        "terminated": True,
        "steps": 10,
        "episode_runtime_sec": 1.0,
        "decisions_made": 10,
        "decision_calls_total": 10,
        "decision_timeout_count": 0,
        "fallback_action_count": 0,
        "ollama_native_retry_count": 0,
        "ollama_openai_fallback_count": 0,
        "ollama_native_decision_count": 0,
        "ollama_native_timeout_count": 0,
        "ollama_native_timeout_short_circuit_count": 0,
        "ollama_downgrade_triggered": False,
        "responses_with_delimiter": 0,
        "responses_strict_format": 0,
        "responses_direct_parseable": 0,
        "format_failure_count": 0,
        "episode_reward_sum": 0.0,
        "avg_ego_speed_mps": 20.0,
        "ttc_danger_rate": 0.0,
        "headway_violation_rate": 0.0,
        "rear_ttc_danger_rate": 0.0,
        "rear_headway_violation_rate": 0.0,
        "low_speed_blocking_rate": 0.0,
        "stop_rate": 0.0,
        "near_stop_rate": 0.0,
        "stopped_ever": False,
        "lane_change_rate": 0.0,
        "flap_accel_decel_rate": 0.0,
        "decision_latency_ms_avg": 100.0,
        "timeout_penalty_events": 0,
        "timeout_penalty_timeout_triggers": 0,
        "timeout_penalty_slow_triggers": 0,
        "timeout_penalty_stage_max": 0,
        "timeout_penalty_final_decision_timeout_sec": 15.0,
        "timeout_policy_mode": "laddered",
        "timeout_level_initial_sec": 15.0,
        "timeout_level_final_sec": 15.0,
        "timeout_level_max_sec": 15.0,
        "timeout_escalation_count": 0,
        "timeout_recovery_count": 0,
        "timeout_level_15_rate": 1.0,
        "timeout_level_20_rate": 0.0,
        "timeout_level_30_rate": 0.0,
        "timeout_phase_counts": {"active": 10},
        "warmup_active": False,
        "warmup_success_observed": True,
        "warmup_timeout_failures": 0,
        "fallback_reason_counts": {},
        "runtime_parse_path_counts": {},
        "lane_change_shield_reason_counts": {},
        "longitudinal_safety_shield_reason_counts": {},
        "flow_recovery_reason_counts": {},
        "episode_stop_reason": "completed",
        "timeout_early_stop_triggered": False,
    }
    episode.update(overrides)
    return episode


class TimeoutPolicyTests(unittest.TestCase):
    def test_eval_ladder_starts_at_15_seconds(self):
        state = build_decision_timeout_penalty_state(
            config={},
            provider="ollama",
            mode="eval",
            baseline_decision_timeout_sec=120.0,
        )
        snapshot = decision_timeout_penalty_snapshot(state)
        self.assertEqual(snapshot["policy_mode"], "laddered")
        self.assertEqual(snapshot["baseline_decision_timeout_sec"], 15.0)
        self.assertEqual(snapshot["effective_decision_timeout_sec"], 15.0)
        self.assertEqual(snapshot["timeout_ladder_sec"], [15.0, 20.0, 30.0])
        self.assertFalse(snapshot["warmup_active"])
        self.assertEqual(snapshot["timeout_phase"], "active")

    def test_eval_timeout_policy_can_be_disabled(self):
        state = build_decision_timeout_penalty_state(
            config={"eval_timeout_policy_mode": "disabled"},
            provider="openai",
            mode="eval",
            baseline_decision_timeout_sec=45.0,
        )
        snapshot = decision_timeout_penalty_snapshot(state)
        self.assertFalse(snapshot["enabled"])
        self.assertEqual(snapshot["policy_mode"], "disabled")
        self.assertEqual(snapshot["effective_decision_timeout_sec"], 45.0)
        self.assertEqual(snapshot["timeout_phase"], "disabled")

        result = update_decision_timeout_penalty_state(
            state,
            timed_out=True,
            decision_elapsed_sec=45.0,
            slow_threshold_sec=5.0,
        )
        self.assertFalse(result["escalated"])
        self.assertEqual(result["effective_decision_timeout_sec"], 45.0)

    def test_remote_ladder_starts_in_first_response_warmup(self):
        state = build_decision_timeout_penalty_state(
            config={
                "eval_first_response_warmup_timeout_sec": 75,
                "eval_first_response_warmup_max_failures": 3,
            },
            provider="openai",
            mode="eval",
            baseline_decision_timeout_sec=60.0,
        )
        snapshot = decision_timeout_penalty_snapshot(state)
        self.assertEqual(snapshot["policy_mode"], "laddered")
        self.assertTrue(snapshot["warmup_active"])
        self.assertFalse(snapshot["warmup_success_observed"])
        self.assertEqual(snapshot["effective_decision_timeout_sec"], 75.0)
        self.assertEqual(snapshot["warmup_timeout_sec"], 75.0)
        self.assertEqual(snapshot["warmup_max_failures"], 3)
        self.assertEqual(snapshot["timeout_phase"], "warmup")

    def test_local_ollama_ladder_skips_first_response_warmup(self):
        state = build_decision_timeout_penalty_state(
            config={"eval_first_response_warmup_timeout_sec": 75},
            provider="ollama",
            mode="eval",
            baseline_decision_timeout_sec=60.0,
        )
        snapshot = decision_timeout_penalty_snapshot(state)
        self.assertFalse(snapshot["warmup_active"])
        self.assertTrue(snapshot["warmup_success_observed"])
        self.assertEqual(snapshot["effective_decision_timeout_sec"], 15.0)
        self.assertEqual(snapshot["timeout_phase"], "active")

    def test_remote_warmup_timeout_counts_failures_until_success(self):
        state = build_decision_timeout_penalty_state(
            config={
                "eval_first_response_warmup_timeout_sec": 75,
                "eval_first_response_warmup_max_failures": 2,
            },
            provider="gemini",
            mode="eval",
            baseline_decision_timeout_sec=60.0,
        )
        result = update_decision_timeout_penalty_state(
            state,
            timed_out=True,
            decision_elapsed_sec=75.0,
            slow_threshold_sec=5.0,
        )
        self.assertEqual(result["reason"], "warmup_timeout")
        self.assertTrue(result["warmup_active"])
        self.assertEqual(result["warmup_timeout_failures"], 1)
        self.assertEqual(result["effective_decision_timeout_sec"], 75.0)

        result = update_decision_timeout_penalty_state(
            state,
            timed_out=False,
            decision_elapsed_sec=2.0,
            slow_threshold_sec=5.0,
        )
        self.assertTrue(result["warmup_completed"])
        self.assertFalse(result["warmup_active"])
        self.assertEqual(result["effective_decision_timeout_sec"], 15.0)
        snapshot = decision_timeout_penalty_snapshot(state)
        self.assertTrue(snapshot["warmup_success_observed"])
        self.assertEqual(snapshot["timeout_phase"], "active")

    def test_eval_ladder_escalates_only_on_timeout(self):
        state = build_decision_timeout_penalty_state(
            config={},
            provider="ollama",
            mode="eval",
            baseline_decision_timeout_sec=60.0,
        )
        result = update_decision_timeout_penalty_state(
            state,
            timed_out=False,
            decision_elapsed_sec=25.0,
            slow_threshold_sec=5.0,
        )
        self.assertFalse(result["escalated"])
        self.assertEqual(result["effective_decision_timeout_sec"], 15.0)

        result = update_decision_timeout_penalty_state(
            state,
            timed_out=True,
            decision_elapsed_sec=15.0,
            slow_threshold_sec=5.0,
        )
        self.assertTrue(result["escalated"])
        self.assertEqual(result["effective_decision_timeout_sec"], 20.0)

        result = update_decision_timeout_penalty_state(
            state,
            timed_out=True,
            decision_elapsed_sec=20.0,
            slow_threshold_sec=5.0,
        )
        self.assertEqual(result["effective_decision_timeout_sec"], 30.0)

        result = update_decision_timeout_penalty_state(
            state,
            timed_out=True,
            decision_elapsed_sec=30.0,
            slow_threshold_sec=5.0,
        )
        self.assertEqual(result["effective_decision_timeout_sec"], 30.0)

    def test_eval_ladder_recovers_after_three_clean_decisions(self):
        state = build_decision_timeout_penalty_state(
            config={},
            provider="ollama",
            mode="eval",
            baseline_decision_timeout_sec=60.0,
        )
        update_decision_timeout_penalty_state(state, timed_out=True, decision_elapsed_sec=15.0, slow_threshold_sec=5.0)
        update_decision_timeout_penalty_state(state, timed_out=True, decision_elapsed_sec=20.0, slow_threshold_sec=5.0)
        self.assertEqual(decision_timeout_penalty_snapshot(state)["effective_decision_timeout_sec"], 30.0)

        for _ in range(3):
            result = update_decision_timeout_penalty_state(
                state,
                timed_out=False,
                decision_elapsed_sec=1.0,
                slow_threshold_sec=5.0,
            )
        self.assertTrue(result["recovered"])
        self.assertEqual(result["effective_decision_timeout_sec"], 20.0)

        for _ in range(3):
            result = update_decision_timeout_penalty_state(
                state,
                timed_out=False,
                decision_elapsed_sec=1.0,
                slow_threshold_sec=5.0,
            )
        self.assertTrue(result["recovered"])
        self.assertEqual(result["effective_decision_timeout_sec"], 15.0)

    def test_runtime_mode_keeps_legacy_shrink_behavior(self):
        state = build_decision_timeout_penalty_state(
            config={
                "adaptive_timeout_penalty_enabled": True,
                "adaptive_timeout_halving_factor": 0.5,
                "adaptive_timeout_min_sec": 4.0,
            },
            provider="ollama",
            mode="runtime",
            baseline_decision_timeout_sec=60.0,
        )
        snapshot = decision_timeout_penalty_snapshot(state)
        self.assertEqual(snapshot["policy_mode"], "legacy")
        self.assertEqual(snapshot["effective_decision_timeout_sec"], 60.0)

        result = update_decision_timeout_penalty_state(
            state,
            timed_out=True,
            decision_elapsed_sec=60.0,
            slow_threshold_sec=5.0,
        )
        self.assertTrue(result["escalated"])
        self.assertEqual(result["effective_decision_timeout_sec"], 30.0)

    def test_aggregate_includes_timeout_ladder_means(self):
        summary = aggregate_results(
            "timeout_model",
            [
                _base_episode(
                    timeout_level_max_sec=30.0,
                    timeout_escalation_count=2,
                    timeout_recovery_count=1,
                    timeout_level_15_rate=0.4,
                    timeout_level_20_rate=0.2,
                    timeout_level_30_rate=0.4,
                ),
                _base_episode(
                    timeout_level_max_sec=20.0,
                    timeout_escalation_count=1,
                    timeout_recovery_count=1,
                    timeout_level_15_rate=0.6,
                    timeout_level_20_rate=0.4,
                    timeout_level_30_rate=0.0,
                ),
            ],
        )
        self.assertEqual(summary["timeout_level_max_sec_mean"], 25.0)
        self.assertEqual(summary["timeout_escalation_count_mean"], 1.5)
        self.assertEqual(summary["timeout_recovery_count_mean"], 1.0)
        self.assertEqual(summary["timeout_level_15_rate_mean"], 0.5)
        self.assertEqual(summary["timeout_level_20_rate_mean"], 0.3)
        self.assertEqual(summary["timeout_level_30_rate_mean"], 0.2)

    def test_aggregate_includes_runtime_reason_counts_and_warmup_totals(self):
        summary = aggregate_results(
            "diagnostic_model",
            [
                _base_episode(
                    fallback_reason_counts={"parse_fallback": 2},
                    runtime_parse_path_counts={"strict_action_line": 3, "parse_fallback": 2},
                    lane_change_shield_reason_counts={"target_rear_gap_below_required": 1},
                    longitudinal_safety_shield_reason_counts={"front_clearance_blocks_accelerate": 2},
                    flow_recovery_reason_counts={"low_speed_recovery_after_front_risk": 1},
                    timeout_phase_counts={"warmup": 1, "active": 4},
                    warmup_timeout_failures=1,
                    warmup_active=True,
                    warmup_success_observed=False,
                ),
                _base_episode(
                    fallback_reason_counts={"decision_timeout": 1},
                    runtime_parse_path_counts={"timeout_fallback": 1, "strict_action_line": 4},
                    timeout_phase_counts={"active": 5},
                    warmup_timeout_failures=0,
                    warmup_active=False,
                    warmup_success_observed=True,
                ),
            ],
        )

        self.assertEqual(summary["fallback_reason_counts"]["parse_fallback"], 2)
        self.assertEqual(summary["fallback_reason_counts"]["decision_timeout"], 1)
        self.assertEqual(summary["runtime_parse_path_counts"]["strict_action_line"], 7)
        self.assertEqual(summary["lane_change_shield_reason_counts"]["target_rear_gap_below_required"], 1)
        self.assertEqual(
            summary["longitudinal_safety_shield_reason_counts"]["front_clearance_blocks_accelerate"],
            2,
        )
        self.assertEqual(summary["flow_recovery_reason_counts"]["low_speed_recovery_after_front_risk"], 1)
        self.assertEqual(summary["timeout_phase_counts"]["active"], 9)
        self.assertEqual(summary["timeout_phase_counts"]["warmup"], 1)
        self.assertEqual(summary["warmup_timeout_failures_total"], 1)
        self.assertEqual(summary["warmup_active_episode_count"], 1)
        self.assertEqual(summary["warmup_success_observed_episode_count"], 1)

    def test_early_stop_policy_triggers_after_three_consecutive_timeout_fallbacks_at_max_level(self):
        should_stop, reason = _should_early_stop_timeout_episode(
            decision_calls_total=5,
            decision_timeout_count=3,
            fallback_action_count=3,
            consecutive_timeout_fallbacks=3,
            timeout_policy_mode="laddered",
            active_timeout_level=30.0,
            max_timeout_level=30.0,
            policy={
                "enabled": True,
                "min_decisions": 5,
                "consecutive_timeout_fallbacks": 3,
                "collapse_rate_threshold": 0.8,
                "require_max_timeout_level": True,
            },
        )
        self.assertTrue(should_stop)
        self.assertEqual(reason, "consecutive_timeout_fallback_collapse")

    def test_early_stop_policy_respects_min_decisions_and_max_level_gate(self):
        should_stop, reason = _should_early_stop_timeout_episode(
            decision_calls_total=4,
            decision_timeout_count=4,
            fallback_action_count=4,
            consecutive_timeout_fallbacks=4,
            timeout_policy_mode="laddered",
            active_timeout_level=20.0,
            max_timeout_level=30.0,
            policy={
                "enabled": True,
                "min_decisions": 5,
                "consecutive_timeout_fallbacks": 3,
                "collapse_rate_threshold": 0.8,
                "require_max_timeout_level": True,
            },
        )
        self.assertFalse(should_stop)
        self.assertIsNone(reason)

    def test_aggregate_marks_incomplete_quarantined_benchmark_runs_invalid(self):
        summary = aggregate_results(
            "timeout_model",
            [
                _base_episode(
                    task_completed=False,
                    driving_score=0.0,
                    driving_score_v2=0.0,
                    overall_score=0.0,
                    overall_score_v2=0.0,
                    episode_stop_reason="episode_timeout_cap",
                    timeout_early_stop_triggered=True,
                ),
                _base_episode(
                    task_completed=True,
                    driving_score=0.5,
                    driving_score_v2=0.5,
                    overall_score=0.5,
                    overall_score_v2=0.5,
                ),
            ],
            planned_episode_count=4,
            model_quarantined_due_to_timeout_collapse=True,
            model_quarantine_reason="timeout_collapse_quarantine",
        )
        self.assertEqual(summary["planned_episode_count"], 4)
        self.assertEqual(summary["executed_episode_count"], 2)
        self.assertEqual(summary["skipped_episode_count"], 2)
        self.assertFalse(summary["episode_execution_complete"])
        self.assertEqual(summary["episodes_stopped_by_timeout_cap"], 1)
        self.assertTrue(summary["model_quarantined_due_to_timeout_collapse"])
        self.assertFalse(summary["benchmark_result_valid"])
        self.assertIn("incomplete_episode_set", summary["benchmark_result_invalid_reason"])
        self.assertIn("timeout_collapse_quarantine", summary["benchmark_result_invalid_reason"])


if __name__ == "__main__":
    unittest.main()
