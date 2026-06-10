import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from dilu.runtime.highway_env_config import resolve_simulation_env_bundle
from dilu.runtime.non_llm_baselines import (
    EXPERT_BASELINE_NAME,
    BaselinePolicy,
    baseline_names_for_levels,
    get_baseline_spec,
    resolve_baseline_names,
)
from dilu.runtime.safety_shields import (
    FASTER_ACTION_ID,
    IDLE_ACTION_ID,
    LANE_LEFT_ACTION_ID,
    SLOWER_ACTION_ID,
)
from dilu.runtime.task_benchmark import load_benchmark_case_set
from evaluate_non_llm_baselines import run_baseline_episode


class _DummyVehicle:
    def __init__(self, lane_rank, speed, x):
        self.lane_index = ("0", "1", int(lane_rank))
        self.speed = float(speed)
        self.position = np.array([float(x), float(lane_rank) * 4.0], dtype=float)


class _DummyRoad:
    def __init__(self, vehicles):
        self.vehicles = list(vehicles)


class _DummyUnwrapped:
    def __init__(self, ego_vehicle, vehicles, available_actions):
        self.vehicle = ego_vehicle
        self.road = _DummyRoad(vehicles)
        self.config = {"lanes_count": 3}
        self._available_actions = list(available_actions)

    def get_available_actions(self):
        return list(self._available_actions)


class _DummyEnv:
    def __init__(self, ego_vehicle, vehicles, available_actions=(0, 1, 2, 3, 4)):
        self.unwrapped = _DummyUnwrapped(ego_vehicle, vehicles, available_actions)


def _make_env(ego_speed=20.0, front_gap=None, front_speed=16.0, left_rear_gap=None):
    ego = _DummyVehicle(1, ego_speed, 100.0)
    vehicles = [ego]
    if front_gap is not None:
        vehicles.append(_DummyVehicle(1, front_speed, 100.0 + float(front_gap)))
    if left_rear_gap is not None:
        vehicles.append(_DummyVehicle(0, 28.0, 100.0 - float(left_rear_gap)))
    return _DummyEnv(ego, vehicles)


class NonLlmBaselineRegistryTests(unittest.TestCase):
    def _run_cli(self, *args, benchmark_case_set="dilu_highway_reactive_v1", baselines="safe_stop"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            command = [
                sys.executable,
                "evaluate_non_llm_baselines.py",
                "--config",
                "config.yaml",
                "--benchmark-case-set",
                benchmark_case_set,
                "--limit",
                "1",
                "--baselines",
                baselines,
                "--output-root",
                tmp_dir,
                *args,
            ]
            result = subprocess.run(
                command,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                text=True,
                capture_output=True,
            )
            report_path = os.path.join(tmp_dir, "non_llm_baseline_report.json")
            return result, os.path.exists(report_path)

    def test_resolve_all_and_level_selection(self):
        level_one = baseline_names_for_levels([1])
        self.assertIn("always_faster", level_one)
        self.assertIn("speed_hold_25", level_one)
        self.assertNotIn("defensive_rule_driver", level_one)
        self.assertNotIn(EXPERT_BASELINE_NAME, level_one)

        level_one_two = baseline_names_for_levels([1, 2])
        self.assertIn("defensive_rule_driver", level_one_two)
        self.assertIn("scenario_aware_rule_driver", level_one_two)
        self.assertNotIn(EXPERT_BASELINE_NAME, level_one_two)

        all_names = resolve_baseline_names("all", levels=None)
        self.assertIn(EXPERT_BASELINE_NAME, all_names)

    def test_unknown_baseline_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown baselines"):
            resolve_baseline_names("does_not_exist", levels=None)

    def test_scenario_aware_baseline_is_task_aware_not_oracle_aware(self):
        spec = get_baseline_spec("scenario_aware_rule_driver")

        self.assertTrue(spec.uses_case_category)
        self.assertTrue(spec.uses_success_criteria)
        self.assertFalse(spec.uses_hidden_scenario_spec)
        self.assertFalse(spec.uses_future_events)

    def test_cli_list_baselines_prints_new_registry(self):
        result = subprocess.run(
            [sys.executable, "evaluate_non_llm_baselines.py", "--list-baselines"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("always_faster", result.stdout)
        self.assertIn("defensive_rule_driver", result.stdout)
        self.assertIn(EXPERT_BASELINE_NAME, result.stdout)

    def test_cli_no_progress_smoke(self):
        result, report_exists = self._run_cli("--no-progress")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report_exists)

    def test_cli_progress_smoke(self):
        result, report_exists = self._run_cli("--progress")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report_exists)

    def test_cli_rejects_conflicting_progress_flags(self):
        result, _ = self._run_cli("--progress", "--no-progress")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Use only one of --progress or --no-progress", result.stderr)

    def test_cli_stress_v2_speed_hold_smoke(self):
        result, report_exists = self._run_cli(
            "--no-progress",
            "--benchmark-categories",
            "traffic_jam_escape",
            benchmark_case_set="dilu_highway_reactive_stress_v2",
            baselines="speed_hold_20",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report_exists)


class NonLlmBaselinePolicyTests(unittest.TestCase):
    def test_fixed_action_baselines_use_requested_action_when_available(self):
        config = {"reward_speed_range": [20, 30]}
        env = _make_env()

        self.assertEqual(BaselinePolicy("always_faster", config).decide(env, {}, 1).action_id, FASTER_ACTION_ID)
        self.assertEqual(BaselinePolicy("always_slower", config).decide(env, {}, 1).action_id, SLOWER_ACTION_ID)
        self.assertEqual(BaselinePolicy("always_left", config).decide(env, {}, 1).action_id, LANE_LEFT_ACTION_ID)

    def test_safe_stop_prefers_slower(self):
        decision = BaselinePolicy("safe_stop", {}).decide(_make_env(ego_speed=12.0), {}, 1)

        self.assertEqual(decision.action_id, SLOWER_ACTION_ID)
        self.assertEqual(decision.reason, "safe_stop_brake")

    def test_speed_hold_tracks_target_speed(self):
        policy = BaselinePolicy("speed_hold_25", {})

        self.assertEqual(policy.decide(_make_env(ego_speed=20.0), {}, 1).action_id, FASTER_ACTION_ID)
        self.assertEqual(policy.decide(_make_env(ego_speed=29.0), {}, 1).action_id, SLOWER_ACTION_ID)
        self.assertEqual(policy.decide(_make_env(ego_speed=25.2), {}, 1).action_id, IDLE_ACTION_ID)

    def test_defensive_rule_brakes_under_front_risk(self):
        decision = BaselinePolicy("defensive_rule_driver", {}).decide(
            _make_env(ego_speed=24.0, front_gap=8.0, front_speed=12.0),
            {},
            1,
        )

        self.assertEqual(decision.action_id, SLOWER_ACTION_ID)
        self.assertEqual(decision.reason, "defensive_front_risk_brake")

    def test_overtake_rule_uses_safe_target_lane_for_slow_lead(self):
        case = {
            "category": "slow_lead_overtake",
            "success_criteria": {"type": "safe_overtake", "target_lane_offset": -1},
        }
        decision = BaselinePolicy("overtake_rule_driver", {}).decide(
            _make_env(ego_speed=25.0, front_gap=24.0, front_speed=18.0),
            case,
            1,
        )

        self.assertEqual(decision.action_id, LANE_LEFT_ACTION_ID)
        self.assertEqual(decision.reason, "overtake_safe_target_lane")

    def test_scenario_aware_rule_switches_by_category(self):
        overtake_case = {
            "category": "slow_lead_overtake",
            "instruction": "Pass safely.",
            "success_criteria": {"type": "safe_overtake", "target_lane_offset": -1},
            "scenario_spec": {"events": [{"id": "hidden", "step": 2, "type": "set_speed"}]},
        }
        cut_in_case = {
            "category": "cut_in_brake_response",
            "instruction": "React to the cut-in.",
            "success_criteria": {"type": "cut_in_brake_response"},
        }

        overtake = BaselinePolicy("scenario_aware_rule_driver", {}).decide(
            _make_env(ego_speed=25.0, front_gap=24.0, front_speed=18.0),
            overtake_case,
            1,
        )
        cut_in = BaselinePolicy("scenario_aware_rule_driver", {}).decide(
            _make_env(ego_speed=24.0, front_gap=8.0, front_speed=12.0),
            cut_in_case,
            1,
        )

        self.assertEqual(overtake.action_id, LANE_LEFT_ACTION_ID)
        self.assertEqual(cut_in.action_id, SLOWER_ACTION_ID)
        self.assertFalse(overtake.metadata["uses_hidden_scenario_spec"])
        self.assertFalse(overtake.metadata["uses_future_events"])


class TrueIdmEgoBaselineTests(unittest.TestCase):
    def test_true_idm_ego_runs_one_highway_case_as_expert_vehicle(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(repo_root, "config.example.yaml"), "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        case_set = load_benchmark_case_set("dilu_highway_reactive_v1")
        case = dict(case_set["cases"][0])
        case["time_limit_sec"] = 1.0
        env_bundle = resolve_simulation_env_bundle(
            config,
            show_trajectories=False,
            render_agent=False,
            env_id_override=case_set["target_env_id"],
            env_config_overrides=(case_set.get("defaults") or {}).get("env_overrides") or {},
            require_discrete_meta_action=True,
        )

        episode = run_baseline_episode(
            config=config,
            env_config_map=env_bundle["env_config_map"],
            env_type=env_bundle["env_id"],
            case=case,
            policy=BaselinePolicy(EXPERT_BASELINE_NAME, config),
            safety_shields_enabled=True,
        )

        self.assertEqual(episode["baseline_policy"], EXPERT_BASELINE_NAME)
        self.assertEqual(episode["baseline_control_mode"], "expert_vehicle")
        self.assertEqual(episode["baseline_claim_scope"], "expert_behavior_reference_only")
        self.assertIsNone(episode["llm_driver_score_v1"])
        self.assertGreater(episode["steps"], 0)


if __name__ == "__main__":
    unittest.main()
