import unittest

import numpy as np

from dilu.runtime.task_benchmark import (
    BenchmarkEpisodeEvaluator,
    benchmark_metric_config,
    validate_benchmark_case,
)


class _DummyVehicle:
    def __init__(self, *, lane_rank: int = 1, speed: float = 24.0, x: float = 0.0) -> None:
        self.lane_index = ("a", "b", lane_rank)
        self.speed = float(speed)
        self.position = np.array([float(x), float(lane_rank) * 4.0], dtype=float)


class _DummyRoad:
    def __init__(self, ego: _DummyVehicle) -> None:
        self.vehicles = [ego]

    def neighbour_vehicles(self, vehicle, lane_index):
        return None, None


class _DummyUnwrapped:
    def __init__(self, ego: _DummyVehicle) -> None:
        self.vehicle = ego
        self.road = _DummyRoad(ego)
        self.config = {"policy_frequency": 1}

    def get_available_actions(self):
        return [0, 1, 2, 3, 4]


class _DummyEnv:
    def __init__(self, ego: _DummyVehicle) -> None:
        self.unwrapped = _DummyUnwrapped(ego)


def _safe_metrics(*, front_gap_m=40.0, ttc_sec=5.0):
    return {
        "front_gap_m": front_gap_m,
        "ttc_sec": ttc_sec,
        "ttc_danger": False,
        "headway_violation": False,
    }


def _recovery_case(*, requires_event: bool, events=None):
    case = {
        "case_id": "completion_recovery_test",
        "category": "cut_in_then_recover",
        "instruction": "Brake for the hazard, then recover after it clears.",
        "time_limit_sec": 20.0,
        "success_criteria": {
            "type": "cut_in_then_recover",
            "clear_front_gap_m": 25.0,
            "clear_front_ttc_sec": 4.0,
            "min_recovery_speed_mps": 20.0,
            "requires_event": requires_event,
            "requires_brake_action": True,
            "hold_steps": 1,
        },
    }
    if events is not None:
        case["scenario_spec"] = {
            "vehicles": [
                {
                    "id": "hazard",
                    "role": "lead",
                    "lane_offset": 0,
                    "x_offset_m": 40.0,
                    "speed_mps": 20.0,
                }
            ],
            "events": list(events),
        }
    return case


class TaskCompletionSemanticsTests(unittest.TestCase):
    def test_speed_band_validation_uses_runtime_float_tolerance(self) -> None:
        case = {
            "case_id": "speed_roundoff_validation",
            "category": "speed_increase",
            "instruction": "Reach the target speed band.",
            "time_limit_sec": 20.0,
            "success_criteria": {
                "type": "speed_band",
                "min_speed_mps": 20.0,
                "max_speed_mps": 30.0,
            },
        }
        initial_state = {
            "initial_speed_mps": 19.999999999999996,
        }

        reasons = validate_benchmark_case(case, initial_state)

        self.assertIn("initial_speed_inside_target_band", reasons)

    def test_speed_band_accepts_float_roundoff_at_both_boundaries(self) -> None:
        case = {
            "case_id": "speed_roundoff",
            "category": "speed_increase",
            "instruction": "Stay inside the target speed band.",
            "time_limit_sec": 20.0,
            "success_criteria": {
                "type": "speed_band",
                "min_speed_mps": 20.0,
                "max_speed_mps": 30.0,
                "hold_steps": 1,
            },
        }
        for speed in (19.999999999999996, 30.000000000000004):
            with self.subTest(speed=speed):
                ego = _DummyVehicle(speed=speed)
                env = _DummyEnv(ego)
                evaluator = BenchmarkEpisodeEvaluator(case, env)
                evaluator.update(
                    env,
                    1,
                    _safe_metrics(),
                    crashed=False,
                    action_context={"final_action_id": 1},
                )

                result = evaluator.finalize(
                    crashed=False,
                    episode_stop_reason="completed",
                )

                self.assertTrue(result["task_completed"])

    def test_recovery_clear_waits_for_last_scheduled_event(self) -> None:
        ego = _DummyVehicle(speed=24.0)
        env = _DummyEnv(ego)
        evaluator = BenchmarkEpisodeEvaluator(
            _recovery_case(
                requires_event=True,
                events=[
                    {
                        "id": "hazard_enters",
                        "step": 3,
                        "type": "set_target_speed",
                        "vehicle_id": "hazard",
                        "target_speed_mps": 15.0,
                    },
                    {
                        "id": "hazard_clears",
                        "step": 10,
                        "type": "set_target_speed",
                        "vehicle_id": "hazard",
                        "target_speed_mps": 24.0,
                    },
                ],
            ),
            env,
        )

        evaluator.update(
            env,
            1,
            _safe_metrics(),
            crashed=False,
            action_context={"final_action_id": 1},
        )

        self.assertIsNone(evaluator.recovery_clear_step)
        evaluator.update(
            env,
            10,
            _safe_metrics(),
            crashed=False,
            action_context={"final_action_id": 1},
        )
        self.assertIsNone(evaluator.recovery_clear_step)
        evaluator.update(
            env,
            11,
            _safe_metrics(),
            crashed=False,
            action_context={
                "final_action_id": 4,
                "benchmark_events_applied": True,
                "benchmark_event_ids": ["hazard_clears"],
                "benchmark_event_types": ["set_target_speed"],
                "benchmark_event_step": 10,
            },
        )
        self.assertEqual(evaluator.recovery_clear_step, 11)

    def test_completion_semantics_have_a_new_scoring_policy_version(self) -> None:
        config = benchmark_metric_config()

        self.assertEqual(
            config["benchmark_scoring_policy_version"],
            "v2_behavior_aware_completion_v2",
        )

    def test_recovery_speed_accepts_float_roundoff_at_threshold(self) -> None:
        ego = _DummyVehicle(speed=15.0)
        env = _DummyEnv(ego)
        evaluator = BenchmarkEpisodeEvaluator(
            _recovery_case(requires_event=False),
            env,
        )
        evaluator.update(
            env,
            1,
            _safe_metrics(front_gap_m=15.0, ttc_sec=2.0),
            crashed=False,
            action_context={"final_action_id": 4},
        )
        ego.speed = 19.999999999999996
        evaluator.update(
            env,
            2,
            _safe_metrics(),
            crashed=False,
            action_context={"final_action_id": 3},
        )

        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")

        self.assertTrue(result["task_completed"])
        self.assertTrue(result["benchmark_criteria_status"]["recovery_speed_satisfied"])

    def test_crash_after_completion_is_not_task_completed(self) -> None:
        ego = _DummyVehicle(speed=24.0)
        env = _DummyEnv(ego)
        evaluator = BenchmarkEpisodeEvaluator(
            {
                "case_id": "completion_then_crash",
                "category": "false_opening_stability",
                "instruction": "Keep a stable lane.",
                "time_limit_sec": 20.0,
                "success_criteria": {
                    "type": "false_opening_stability",
                    "requires_event": False,
                    "min_survival_steps": 1,
                    "min_avg_speed_mps": 20.0,
                    "max_lane_changes": 0,
                    "hold_steps": 1,
                },
            },
            env,
        )
        evaluator.update(
            env,
            1,
            _safe_metrics(),
            crashed=False,
            action_context={"final_action_id": 1},
        )
        self.assertTrue(evaluator.task_completed)
        evaluator.update(
            env,
            2,
            _safe_metrics(front_gap_m=2.0, ttc_sec=0.5),
            crashed=True,
            action_context={"final_action_id": 1},
        )

        result = evaluator.finalize(crashed=True, episode_stop_reason="crash")

        self.assertFalse(result["task_completed"])
        self.assertEqual(result["benchmark_failure_reason"], "crash")

    def test_shield_intervention_is_diagnostic_not_a_stress_task_gate(self) -> None:
        ego = _DummyVehicle(speed=24.0)
        env = _DummyEnv(ego)
        evaluator = BenchmarkEpisodeEvaluator(
            {
                "case_id": "shield_diagnostic_only",
                "category": "dense_dynamic_flow",
                "instruction": "Maintain safe flow.",
                "time_limit_sec": 20.0,
                "success_criteria": {
                    "type": "dense_dynamic_flow",
                    "requires_event": False,
                    "min_survival_steps": 1,
                    "max_unsafe_lane_change_attempts": 0,
                    "hold_steps": 1,
                },
            },
            env,
        )
        evaluator.update(
            env,
            1,
            _safe_metrics(),
            crashed=False,
            action_context={"final_action_id": 1, "lane_change_shield_applied": True},
        )

        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")

        self.assertTrue(result["task_completed"])
        self.assertEqual(result["benchmark_unsafe_lane_change_attempts"], 1)
        self.assertFalse(result["benchmark_criteria_status"]["unsafe_attempt_satisfied"])


if __name__ == "__main__":
    unittest.main()
