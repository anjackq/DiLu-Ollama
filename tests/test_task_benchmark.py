import copy
import os
import unittest

import gymnasium as gym
import numpy as np
import yaml

from dilu.runtime.highway_scenario_spec import (
    _set_vehicle_state,
    _vehicle_x,
    apply_highway_scenario_events,
    apply_highway_scenario_spec,
    normalize_scenario_spec,
)
from dilu.runtime.highway_env_config import resolve_simulation_env_bundle
from dilu.runtime.task_benchmark import (
    BenchmarkEpisodeEvaluator,
    augment_behavior_aware_benchmark_episode,
    benchmark_result_validity,
    bootstrap_ci95,
    compute_benchmark_case_scores,
    load_benchmark_case_set,
    summarize_benchmark_episodes,
    validate_benchmark_case_set,
)
from evaluate_models_ollama import aggregate_results
from evaluate_models_ollama import _decision_trace_item
from evaluate_models_ollama import extract_step_traffic_metrics


class _DummyVehicle:
    def __init__(self, lane_rank, speed, x):
        self.lane_index = ("a", "b", int(lane_rank))
        self.speed = float(speed)
        self.position = np.array([float(x), float(lane_rank) * 4.0], dtype=float)


class _DummyRoad:
    def __init__(self, ego_vehicle, front_vehicle):
        self.ego_vehicle = ego_vehicle
        self.front_vehicle = front_vehicle
        self.vehicles = [ego_vehicle]
        if front_vehicle is not None:
            self.vehicles.append(front_vehicle)

    def neighbour_vehicles(self, vehicle, lane_index):
        return self.front_vehicle, None


class _DummyUnwrapped:
    def __init__(self, ego_vehicle, front_vehicle, available_actions):
        self.vehicle = ego_vehicle
        self.road = _DummyRoad(ego_vehicle, front_vehicle)
        self.config = {"policy_frequency": 1}
        self._available_actions = list(available_actions)

    def get_available_actions(self):
        return list(self._available_actions)


class _DummyEnv:
    def __init__(self, ego_vehicle, front_vehicle, available_actions):
        self.unwrapped = _DummyUnwrapped(ego_vehicle, front_vehicle, available_actions)


class TaskBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(repo_root, "config.example.yaml")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        cls.env_bundle = resolve_simulation_env_bundle(
            config,
            show_trajectories=False,
            render_agent=False,
        )

    def test_builtin_case_set_validates(self):
        case_set = load_benchmark_case_set("lampilot_highway_v1")
        result = validate_benchmark_case_set(
            case_set,
            self.env_bundle["env_config_map"],
            self.env_bundle["env_id"],
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["total_cases"], 40)
        self.assertEqual(result["summary"]["invalid_case_count"], 0)

    def test_invalid_case_set_fails_prevalidation(self):
        case_set = load_benchmark_case_set("lampilot_highway_v1")
        bad_case = copy.deepcopy(case_set["cases"][0])
        bad_case["success_criteria"]["min_speed_mps"] = 24.0
        bad_case["success_criteria"]["max_speed_mps"] = 26.0
        custom_case_set = {
            "benchmark_name": "invalid_speed_case",
            "cases": [bad_case],
        }
        result = validate_benchmark_case_set(
            custom_case_set,
            self.env_bundle["env_config_map"],
            self.env_bundle["env_id"],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["summary"]["invalid_case_count"], 1)
        self.assertIn(
            "initial_speed_inside_target_band",
            result["invalid_cases"][0]["reasons"],
        )

    def test_native_env_bundle_preserves_current_default_target_speeds_without_override(self):
        target_speeds = self.env_bundle["env_config_snapshot"]["action"]["target_speeds"]
        self.assertEqual(list(target_speeds), [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        self.assertEqual(self.env_bundle["env_profile_label"], "default_stop_capable")

    def test_native_env_bundle_accepts_stop_capable_target_speed_override(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(repo_root, "config.example.yaml")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["sim_action_target_speeds"] = [0, 5, 10, 15, 20, 25, 30]
        bundle = resolve_simulation_env_bundle(
            config,
            show_trajectories=False,
            render_agent=False,
        )
        target_speeds = bundle["env_config_snapshot"]["action"]["target_speeds"]
        self.assertEqual(list(target_speeds), [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

    def test_highway_scenario_spec_repositions_ego_and_places_vehicles(self):
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        try:
            env.unwrapped.configure(self.env_bundle["env_config_map"][self.env_bundle["env_id"]])
            env.reset(seed=123)
            case = {
                "scenario_spec": {
                    "clear_existing_vehicles": True,
                    "ego": {"lane_rank": 1, "x_m": 100.0, "speed_mps": 25.0},
                    "vehicles": [
                        {
                            "id": "slow_lead",
                            "role": "lead",
                            "lane_offset": 0,
                            "x_offset_m": 28.0,
                            "speed_mps": 18.0,
                            "target_speed_mps": 18.0,
                        },
                        {
                            "id": "left_rear",
                            "role": "left_rear",
                            "lane_offset": -1,
                            "x_offset_m": -45.0,
                            "speed_mps": 25.0,
                        },
                    ],
                }
            }

            meta = apply_highway_scenario_spec(env, case)
            ego_vehicle = env.unwrapped.vehicle
            front, _rear = env.unwrapped.road.neighbour_vehicles(ego_vehicle, ego_vehicle.lane_index)

            self.assertTrue(meta["benchmark_scenario_spec_applied"])
            self.assertEqual(ego_vehicle.lane_index[2], 1)
            self.assertAlmostEqual(float(ego_vehicle.position[0]), 100.0, places=3)
            self.assertAlmostEqual(float(ego_vehicle.speed), 25.0, places=3)
            self.assertEqual(len(env.unwrapped.road.vehicles), 3)
            self.assertIsNotNone(front)
            self.assertAlmostEqual(float(ego_vehicle.lane_distance_to(front)), 28.0, places=3)
        finally:
            env.close()

    def test_highway_scenario_spec_rejects_invalid_specs(self):
        with self.assertRaisesRegex(ValueError, "duplicate vehicle id"):
            normalize_scenario_spec(
                {
                    "vehicles": [
                        {"id": "dup", "x_offset_m": 20, "speed_mps": 20},
                        {"id": "dup", "x_offset_m": 40, "speed_mps": 20},
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "front-role x_offset_m must be positive"):
            normalize_scenario_spec(
                {"vehicles": [{"id": "bad_front", "role": "lead", "x_offset_m": -5, "speed_mps": 20}]}
            )

    def test_highway_scenario_events_reposition_update_and_spawn_vehicle(self):
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        try:
            env.unwrapped.configure(self.env_bundle["env_config_map"][self.env_bundle["env_id"]])
            env.reset(seed=321)
            case = {
                "scenario_spec": {
                    "clear_existing_vehicles": True,
                    "ego": {"lane_rank": 1, "x_m": 100.0, "speed_mps": 25.0},
                    "vehicles": [
                        {
                            "id": "lead",
                            "role": "lead",
                            "lane_offset": 0,
                            "x_offset_m": 40.0,
                            "speed_mps": 20.0,
                            "target_speed_mps": 20.0,
                        }
                    ],
                    "events": [
                        {
                            "id": "lead_cut",
                            "step": 2,
                            "type": "reposition_vehicle",
                            "vehicle_id": "lead",
                            "lane_offset": 0,
                            "x_offset_m": 18.0,
                            "speed_mps": 15.0,
                            "target_speed_mps": 14.0,
                        },
                        {
                            "id": "spawn_rear",
                            "step": 2,
                            "type": "spawn_vehicle",
                            "vehicle": {
                                "id": "rear",
                                "role": "rear",
                                "lane_offset": 0,
                                "x_offset_m": -30.0,
                                "speed_mps": 24.0,
                                "target_speed_mps": 24.0,
                            },
                        },
                    ],
                }
            }
            apply_highway_scenario_spec(env, case)
            applied_ids = set()

            meta = apply_highway_scenario_events(env, case, step_idx=2, applied_event_ids=applied_ids)

            ego_vehicle = env.unwrapped.vehicle
            vehicles_by_id = {
                getattr(vehicle, "dilu_benchmark_id", ""): vehicle
                for vehicle in env.unwrapped.road.vehicles
            }
            self.assertTrue(meta["benchmark_events_applied"])
            self.assertEqual(meta["benchmark_event_ids"], ["lead_cut", "spawn_rear"])
            self.assertIn("lead_cut", applied_ids)
            self.assertIn("spawn_rear", applied_ids)
            self.assertAlmostEqual(float(ego_vehicle.lane_distance_to(vehicles_by_id["lead"])), 18.0, places=3)
            self.assertAlmostEqual(float(vehicles_by_id["lead"].speed), 15.0, places=3)
            self.assertAlmostEqual(float(vehicles_by_id["lead"].target_speed), 14.0, places=3)
            self.assertAlmostEqual(float(ego_vehicle.lane_distance_to(vehicles_by_id["rear"])), -30.0, places=3)
        finally:
            env.close()

    def test_highway_scenario_reposition_lane_offset_defaults_to_scenario_ego_reference(self):
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        try:
            env.unwrapped.configure(self.env_bundle["env_config_map"][self.env_bundle["env_id"]])
            env.reset(seed=654)
            case = {
                "scenario_spec": {
                    "clear_existing_vehicles": True,
                    "ego": {"lane_rank": 1, "x_m": 100.0, "speed_mps": 25.0},
                    "vehicles": [
                        {
                            "id": "left_car",
                            "role": "left_front",
                            "lane_offset": -1,
                            "x_offset_m": 25.0,
                            "speed_mps": 20.0,
                            "target_speed_mps": 20.0,
                        }
                    ],
                    "events": [
                        {
                            "id": "left_car_reposition",
                            "step": 2,
                            "type": "reposition_vehicle",
                            "vehicle_id": "left_car",
                            "lane_offset": -1,
                            "x_offset_m": 70.0,
                            "speed_mps": 26.0,
                        }
                    ],
                }
            }
            apply_highway_scenario_spec(env, case)
            ego_vehicle = env.unwrapped.vehicle
            _set_vehicle_state(
                ego_vehicle,
                env,
                0,
                _vehicle_x(ego_vehicle) or 100.0,
                float(ego_vehicle.speed),
            )

            meta = apply_highway_scenario_events(env, case, step_idx=2, applied_event_ids=set())

            vehicles_by_id = {
                getattr(vehicle, "dilu_benchmark_id", ""): vehicle
                for vehicle in env.unwrapped.road.vehicles
            }
            self.assertTrue(meta["benchmark_events_applied"])
            self.assertEqual(vehicles_by_id["left_car"].lane_index[2], 0)
            self.assertAlmostEqual(float(ego_vehicle.lane_distance_to(vehicles_by_id["left_car"])), 70.0, places=3)
            self.assertEqual(meta["benchmark_events"][0]["resolved_lane_reference"], "scenario_ego")
            self.assertEqual(meta["benchmark_events"][0]["resolved_lane_rank"], 0)
        finally:
            env.close()

    def test_highway_scenario_reposition_lane_offset_can_use_current_ego_reference(self):
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        try:
            env.unwrapped.configure(self.env_bundle["env_config_map"][self.env_bundle["env_id"]])
            env.reset(seed=655)
            case = {
                "scenario_spec": {
                    "clear_existing_vehicles": True,
                    "ego": {"lane_rank": 1, "x_m": 100.0, "speed_mps": 25.0},
                    "vehicles": [
                        {
                            "id": "same_lane_car",
                            "role": "lead",
                            "lane_offset": 0,
                            "x_offset_m": 35.0,
                            "speed_mps": 20.0,
                            "target_speed_mps": 20.0,
                        }
                    ],
                    "events": [
                        {
                            "id": "current_ego_cut_in",
                            "step": 2,
                            "type": "reposition_vehicle",
                            "vehicle_id": "same_lane_car",
                            "lane_offset": 0,
                            "lane_reference": "current_ego",
                            "x_offset_m": 18.0,
                            "speed_mps": 18.0,
                        }
                    ],
                }
            }
            apply_highway_scenario_spec(env, case)
            ego_vehicle = env.unwrapped.vehicle
            _set_vehicle_state(
                ego_vehicle,
                env,
                2,
                _vehicle_x(ego_vehicle) or 100.0,
                float(ego_vehicle.speed),
            )

            meta = apply_highway_scenario_events(env, case, step_idx=2, applied_event_ids=set())

            moved = next(
                vehicle
                for vehicle in env.unwrapped.road.vehicles
                if getattr(vehicle, "dilu_benchmark_id", "") == "same_lane_car"
            )
            self.assertEqual(moved.lane_index[2], 2)
            self.assertEqual(meta["benchmark_events"][0]["resolved_lane_reference"], "current_ego")
            self.assertEqual(meta["benchmark_events"][0]["resolved_lane_rank"], 2)
        finally:
            env.close()

    def test_highway_scenario_reposition_lane_offset_can_use_vehicle_current_reference(self):
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        try:
            env.unwrapped.configure(self.env_bundle["env_config_map"][self.env_bundle["env_id"]])
            env.reset(seed=656)
            case = {
                "scenario_spec": {
                    "clear_existing_vehicles": True,
                    "ego": {"lane_rank": 1, "x_m": 100.0, "speed_mps": 25.0},
                    "vehicles": [
                        {
                            "id": "left_car",
                            "role": "left_front",
                            "lane_offset": -1,
                            "x_offset_m": 30.0,
                            "speed_mps": 20.0,
                            "target_speed_mps": 20.0,
                        }
                    ],
                    "events": [
                        {
                            "id": "vehicle_relative_move",
                            "step": 2,
                            "type": "reposition_vehicle",
                            "vehicle_id": "left_car",
                            "lane_offset": 1,
                            "lane_reference": "vehicle_current",
                            "x_offset_m": 45.0,
                        }
                    ],
                }
            }
            apply_highway_scenario_spec(env, case)

            meta = apply_highway_scenario_events(env, case, step_idx=2, applied_event_ids=set())

            moved = next(
                vehicle
                for vehicle in env.unwrapped.road.vehicles
                if getattr(vehicle, "dilu_benchmark_id", "") == "left_car"
            )
            self.assertEqual(moved.lane_index[2], 1)
            self.assertEqual(meta["benchmark_events"][0]["resolved_lane_reference"], "vehicle_current")
            self.assertEqual(meta["benchmark_events"][0]["resolved_lane_rank"], 1)
        finally:
            env.close()

    def test_stress_reposition_events_apply_after_ego_moves_to_edge_lanes(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_stress_v1")
        cases_by_id = {case["case_id"]: case for case in case_set["cases"]}

        for case_id, step_idx, forced_lane in [
            ("delayed_overtake_gap_001", 6, 0),
            ("closing_rear_lane_change_001", 4, 0),
            ("right_lane_opening_discipline_001", 6, 2),
        ]:
            env = gym.make("highway-fast-v0", render_mode="rgb_array")
            try:
                env.unwrapped.configure(self.env_bundle["env_config_map"][self.env_bundle["env_id"]])
                env.reset(seed=int(cases_by_id[case_id]["seed"]))
                apply_highway_scenario_spec(env, cases_by_id[case_id])
                ego_vehicle = env.unwrapped.vehicle
                _set_vehicle_state(
                    ego_vehicle,
                    env,
                    forced_lane,
                    _vehicle_x(ego_vehicle) or 100.0,
                    float(ego_vehicle.speed),
                )

                meta = apply_highway_scenario_events(
                    env,
                    cases_by_id[case_id],
                    step_idx=step_idx,
                    applied_event_ids=set(),
                )

                self.assertTrue(meta["benchmark_events_applied"], case_id)
                for event in meta["benchmark_events"]:
                    self.assertGreaterEqual(event["resolved_lane_rank"], 0, case_id)
                    self.assertLess(event["resolved_lane_rank"], 3, case_id)
            finally:
                env.close()

    def test_highway_scenario_events_reject_invalid_event_specs(self):
        with self.assertRaisesRegex(ValueError, "duplicate event id"):
            normalize_scenario_spec(
                {
                    "vehicles": [{"id": "lead", "role": "lead", "x_offset_m": 40, "speed_mps": 20}],
                    "events": [
                        {"id": "dup", "step": 1, "type": "set_speed", "vehicle_id": "lead", "speed_mps": 18},
                        {"id": "dup", "step": 2, "type": "set_speed", "vehicle_id": "lead", "speed_mps": 19},
                    ],
                }
            )

    def test_action_trace_records_benchmark_event_metadata(self):
        trace = _decision_trace_item(
            step_idx=3,
            action_id=4,
            response_text="Response to user:#### 4",
            decision_meta={
                "original_selected_action": 1,
                "selected_action": 4,
                "benchmark_events_applied": True,
                "benchmark_event_ids": ["cut_in"],
                "benchmark_event_types": ["reposition_vehicle"],
                "benchmark_event_step": 3,
            },
        )

        self.assertTrue(trace["benchmark_events_applied"])
        self.assertEqual(trace["benchmark_event_ids"], ["cut_in"])
        self.assertEqual(trace["benchmark_event_types"], ["reposition_vehicle"])
        self.assertEqual(trace["final_action_id"], 4)
        with self.assertRaisesRegex(ValueError, "references unknown vehicle"):
            normalize_scenario_spec(
                {
                    "vehicles": [{"id": "lead", "role": "lead", "x_offset_m": 40, "speed_mps": 20}],
                    "events": [
                        {"id": "bad_ref", "step": 1, "type": "set_speed", "vehicle_id": "missing", "speed_mps": 18}
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "unsupported lane_reference"):
            normalize_scenario_spec(
                {
                    "vehicles": [{"id": "lead", "role": "lead", "x_offset_m": 40, "speed_mps": 20}],
                    "events": [
                        {
                            "id": "bad_lane_reference",
                            "step": 1,
                            "type": "reposition_vehicle",
                            "vehicle_id": "lead",
                            "lane_offset": 0,
                            "lane_reference": "future_ego",
                        }
                    ],
                }
            )

    def test_dilu_highway_reactive_case_set_validates_and_uses_stop_capable_speeds(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_v1")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(repo_root, "config.example.yaml")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        bundle = resolve_simulation_env_bundle(
            config,
            show_trajectories=False,
            render_agent=False,
            env_id_override=case_set["target_env_id"],
            native_env_defaults_override=True,
            env_config_overrides=case_set["defaults"]["env_overrides"],
            require_discrete_meta_action=True,
        )
        result = validate_benchmark_case_set(
            case_set,
            bundle["env_config_map"],
            bundle["env_id"],
        )

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["summary"]["total_cases"], 60)
        self.assertEqual(sorted(case_set["categories"]), [
            "blocked_lane_patience",
            "dense_traffic_flow",
            "free_flow_cruise",
            "lane_discipline",
            "post_brake_recovery",
            "slow_lead_overtake",
        ])
        self.assertEqual(
            list(bundle["env_config_snapshot"]["action"]["target_speeds"]),
            [0, 5, 10, 15, 20, 25, 30],
        )

    def test_dilu_highway_reactive_action_target_speed_cli_override_still_wins(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_v1")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(repo_root, "config.example.yaml")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        bundle = resolve_simulation_env_bundle(
            config,
            show_trajectories=False,
            render_agent=False,
            env_id_override=case_set["target_env_id"],
            native_env_defaults_override=True,
            env_config_overrides=case_set["defaults"]["env_overrides"],
            action_target_speeds_override="20,25,30",
            require_discrete_meta_action=True,
        )

        self.assertEqual(list(bundle["env_config_snapshot"]["action"]["target_speeds"]), [20, 25, 30])

    def test_dilu_highway_reactive_stress_case_set_validates(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_stress_v1")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(repo_root, "config.example.yaml")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        bundle = resolve_simulation_env_bundle(
            config,
            show_trajectories=False,
            render_agent=False,
            env_id_override=case_set["target_env_id"],
            native_env_defaults_override=True,
            env_config_overrides=case_set["defaults"]["env_overrides"],
            require_discrete_meta_action=True,
        )
        result = validate_benchmark_case_set(
            case_set,
            bundle["env_config_map"],
            bundle["env_id"],
        )

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["summary"]["total_cases"], 80)
        self.assertEqual(len(case_set["categories"]), 8)
        self.assertEqual(
            list(bundle["env_config_snapshot"]["action"]["target_speeds"]),
            [0, 5, 10, 15, 20, 25, 30],
        )
        self.assertTrue(result["summary"]["scheduled_event_validation_enabled"])
        self.assertEqual(result["summary"]["scheduled_event_validated_case_count"], 80)

    def test_dilu_highway_reactive_stress_v2_case_set_validates(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_stress_v2")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(repo_root, "config.example.yaml")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        bundle = resolve_simulation_env_bundle(
            config,
            show_trajectories=False,
            render_agent=False,
            env_id_override=case_set["target_env_id"],
            native_env_defaults_override=True,
            env_config_overrides=case_set["defaults"]["env_overrides"],
            require_discrete_meta_action=True,
        )
        result = validate_benchmark_case_set(
            case_set,
            bundle["env_config_map"],
            bundle["env_id"],
        )

        self.assertTrue(result["passed"], result)
        self.assertEqual(case_set["benchmark_name"], "dilu_highway_reactive_stress_v2")
        self.assertEqual(case_set["target_env_id"], "highway-fast-v0")
        self.assertEqual(result["summary"]["total_cases"], 120)
        self.assertEqual(len(case_set["categories"]), 10)
        category_counts = {
            category: sum(1 for case in case_set["cases"] if case["category"] == category)
            for category in case_set["categories"]
        }
        self.assertTrue(all(count == 12 for count in category_counts.values()), category_counts)
        self.assertEqual(
            list(bundle["env_config_snapshot"]["action"]["target_speeds"]),
            [0, 5, 10, 15, 20, 25, 30],
        )
        self.assertEqual(result["summary"]["scheduled_event_validated_case_count"], 120)
        self.assertTrue(any((case.get("env_overrides") or {}).get("lanes_count") == 4 for case in case_set["cases"]))

    def test_benchmark_validation_checks_scheduled_events(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_stress_v1")
        bad_case = copy.deepcopy(case_set["cases"][0])
        bad_case["case_id"] = "bad_scheduled_event_case"
        bad_case["scenario_spec"]["events"] = [
            {
                "id": "bad_lane",
                "step": 2,
                "type": "reposition_vehicle",
                "vehicle_id": bad_case["scenario_spec"]["vehicles"][0]["id"],
                "lane_rank": 99,
                "x_offset_m": 20.0,
            }
        ]
        custom_case_set = {
            "benchmark_name": "bad_scheduled_event_suite",
            "target_env_id": "highway-fast-v0",
            "cases": [bad_case],
        }

        result = validate_benchmark_case_set(
            custom_case_set,
            self.env_bundle["env_config_map"],
            self.env_bundle["env_id"],
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["summary"]["scheduled_event_validated_case_count"], 1)
        self.assertTrue(
            any("scheduled_event_validation_error" in reason for reason in result["invalid_cases"][0]["reasons"])
        )

    def test_step_metrics_detect_stop_and_near_stop_independently_from_low_speed_blocking(self):
        ego = _DummyVehicle(lane_rank=1, speed=0.1, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=0.0, x=120.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        step_metrics = extract_step_traffic_metrics(
            env=env,
            ttc_threshold_sec=2.0,
            headway_threshold_m=15.0,
            rear_ttc_threshold_sec=2.5,
            rear_headway_threshold_m=12.0,
            low_speed_blocking_threshold_mps=8.5,
            blocking_front_gap_safe_m=25.0,
            blocking_front_ttc_safe_sec=4.0,
            stop_threshold_mps=0.5,
            near_stop_threshold_mps=2.0,
        )
        self.assertTrue(step_metrics["stopped"])
        self.assertTrue(step_metrics["near_stop"])
        self.assertTrue(step_metrics["low_speed_blocking"])

    def test_score_semantics_make_completion_decisive(self):
        crash_case = compute_benchmark_case_scores(
            task_completed=True,
            crashed=True,
            min_positive_ttc_sec=4.0,
            speed_history=[25.0, 25.0, 25.0],
            completion_time_sec=2.0,
            time_limit_sec=10.0,
        )
        incomplete_case = compute_benchmark_case_scores(
            task_completed=False,
            crashed=False,
            min_positive_ttc_sec=4.0,
            speed_history=[25.0, 25.0, 25.0],
            completion_time_sec=None,
            time_limit_sec=10.0,
        )
        weak_case = compute_benchmark_case_scores(
            task_completed=True,
            crashed=False,
            min_positive_ttc_sec=1.0,
            speed_history=[20.0, 30.0, 20.0, 30.0],
            completion_time_sec=8.0,
            time_limit_sec=10.0,
        )
        strong_case = compute_benchmark_case_scores(
            task_completed=True,
            crashed=False,
            min_positive_ttc_sec=3.0,
            speed_history=[28.0, 28.0, 29.0, 28.0],
            completion_time_sec=4.0,
            time_limit_sec=10.0,
        )
        self.assertEqual(crash_case["driving_score"], 0.0)
        self.assertEqual(incomplete_case["driving_score"], 0.0)
        self.assertGreater(strong_case["driving_score"], weak_case["driving_score"])

    def test_behavior_aware_v2_penalizes_qwen_like_conservative_timeout_behavior(self):
        episode = {
            "category": "speed_increase",
            "task_completed": True,
            "crashed": False,
            "overall_score": 0.58,
            "driving_score": 0.58,
            "stop_rate": 0.45,
            "near_stop_rate": 0.65,
            "low_speed_blocking_rate": 0.55,
            "decision_timeout_rate": 0.16,
            "fallback_action_rate": 0.16,
        }
        scored = augment_behavior_aware_benchmark_episode(episode)
        self.assertLess(scored["behavior_penalty_factor_v2"], 1.0)
        self.assertLess(scored["overall_score_v2"], scored["overall_score"])
        self.assertLess(scored["driving_score_v2"], scored["driving_score"])

    def test_behavior_aware_v2_is_task_aware_for_defensive_categories(self):
        defensive_episode = {
            "category": "speed_decrease",
            "task_completed": True,
            "crashed": False,
            "overall_score": 0.58,
            "driving_score": 0.58,
            "stop_rate": 0.12,
            "near_stop_rate": 0.22,
            "low_speed_blocking_rate": 0.02,
            "decision_timeout_rate": 0.0,
            "fallback_action_rate": 0.0,
        }
        assertive_episode = dict(defensive_episode)
        assertive_episode["category"] = "speed_increase"
        defensive_scored = augment_behavior_aware_benchmark_episode(defensive_episode)
        assertive_scored = augment_behavior_aware_benchmark_episode(assertive_episode)
        self.assertGreater(defensive_scored["behavior_penalty_factor_v2"], assertive_scored["behavior_penalty_factor_v2"])
        self.assertGreater(defensive_scored["driving_score_v2"], assertive_scored["driving_score_v2"])

    def test_bootstrap_ci_is_deterministic(self):
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        first = bootstrap_ci95(values, iterations=500, seed=123)
        second = bootstrap_ci95(values, iterations=500, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertLessEqual(first[0], first[1])

    def test_speed_band_predicate_requires_hold_steps(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=20.0, x=80.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "speed_band_test",
            "category": "speed_increase",
            "instruction": "speed test",
            "time_limit_sec": 10,
            "success_criteria": {
                "type": "speed_band",
                "min_speed_mps": 27.0,
                "max_speed_mps": 30.0,
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        ego.speed = 28.0
        evaluator.update(env, 1, {"front_gap_m": 80.0, "ttc_sec": 5.0}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        evaluator.update(env, 2, {"front_gap_m": 80.0, "ttc_sec": 5.0}, crashed=False)
        self.assertTrue(evaluator.task_completed)

    def test_front_gap_band_predicate_requires_hold_steps(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=20.0, x=40.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "front_gap_test",
            "category": "follow_gap_increase",
            "instruction": "gap test",
            "time_limit_sec": 12,
            "success_criteria": {
                "type": "front_gap_band",
                "min_gap_m": 70.0,
                "max_gap_m": 110.0,
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        front.position[0] = 85.0
        evaluator.update(env, 1, {"front_gap_m": 85.0, "ttc_sec": 6.0}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        evaluator.update(env, 2, {"front_gap_m": 85.0, "ttc_sec": 6.0}, crashed=False)
        self.assertTrue(evaluator.task_completed)

    def test_lane_change_predicate_requires_hold_steps(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=20.0, x=60.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "lane_change_test",
            "category": "lane_change_left",
            "instruction": "lane test",
            "time_limit_sec": 10,
            "success_criteria": {
                "type": "lane_change",
                "direction": "left",
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        ego.lane_index = ("a", "b", 0)
        ego.position[1] = 0.0
        evaluator.update(env, 1, {"front_gap_m": 60.0, "ttc_sec": 4.0}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        evaluator.update(env, 2, {"front_gap_m": 60.0, "ttc_sec": 4.0}, crashed=False)
        self.assertTrue(evaluator.task_completed)

    def test_overtake_predicate_requires_lane_use_and_pass_margin(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=20.0, x=45.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "overtake_test",
            "category": "overtake_left",
            "instruction": "overtake test",
            "time_limit_sec": 16,
            "success_criteria": {
                "type": "overtake",
                "direction": "left",
                "pass_margin_m": 5.0,
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        ego.lane_index = ("a", "b", 0)
        ego.position[1] = 0.0
        ego.position[0] = 60.0
        evaluator.update(env, 1, {"front_gap_m": None, "ttc_sec": 5.0}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        ego.position[0] = 65.0
        evaluator.update(env, 2, {"front_gap_m": None, "ttc_sec": 5.0}, crashed=False)
        self.assertTrue(evaluator.task_completed)

    def test_safe_overtake_tracks_latency_and_requires_no_unsafe_attempts(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=18.0, x=28.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "safe_overtake_test",
            "category": "slow_lead_overtake",
            "instruction": "overtake test",
            "time_limit_sec": 16,
            "success_criteria": {
                "type": "safe_overtake",
                "direction": "left",
                "pass_margin_m": 8.0,
                "min_final_speed_mps": 22.0,
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        ego.lane_index = ("a", "b", 0)
        ego.position[1] = 0.0
        ego.position[0] = 40.0
        evaluator.update(
            env,
            1,
            {"front_gap_m": None, "ttc_sec": 5.0},
            crashed=False,
            action_context={"final_action_id": 0},
        )
        self.assertFalse(evaluator.task_completed)
        ego.position[0] = 42.0
        evaluator.update(
            env,
            2,
            {"front_gap_m": None, "ttc_sec": 5.0},
            crashed=False,
            action_context={"final_action_id": 1},
        )
        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")

        self.assertTrue(result["task_completed"])
        self.assertEqual(result["benchmark_overtake_latency_steps"], 1)
        self.assertEqual(result["benchmark_unsafe_lane_change_attempts"], 0)

    def test_safe_overtake_counts_missed_open_target_lane_opportunity(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=18.0, x=40.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "safe_overtake_missed_opportunity_test",
            "category": "slow_lead_overtake",
            "instruction": "overtake test",
            "time_limit_sec": 16,
            "success_criteria": {
                "type": "safe_overtake",
                "direction": "left",
                "pass_margin_m": 8.0,
                "min_final_speed_mps": 22.0,
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)

        evaluator.update(
            env,
            1,
            {"front_gap_m": 40.0, "ttc_sec": 5.0},
            crashed=False,
            action_context={"final_action_id": 1},
        )
        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")

        self.assertEqual(result["benchmark_safe_overtake_opportunity_steps"], 1)
        self.assertEqual(result["benchmark_missed_overtake_opportunity_steps"], 1)
        self.assertEqual(result["benchmark_first_safe_overtake_opportunity_step"], 1)
        self.assertIsNone(result["benchmark_first_lane_change_attempt_step"])
        self.assertEqual(result["benchmark_missed_overtake_opportunity_rate"], 1.0)

    def test_safe_overtake_records_lane_change_attempt_without_missed_opportunity(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=18.0, x=40.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "safe_overtake_attempt_test",
            "category": "slow_lead_overtake",
            "instruction": "overtake test",
            "time_limit_sec": 16,
            "success_criteria": {
                "type": "safe_overtake",
                "direction": "left",
                "pass_margin_m": 8.0,
                "min_final_speed_mps": 22.0,
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)

        evaluator.update(
            env,
            1,
            {"front_gap_m": 40.0, "ttc_sec": 5.0},
            crashed=False,
            action_context={"final_action_id": 0, "lane_change_shield_applied": True},
        )
        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")

        self.assertEqual(result["benchmark_safe_overtake_opportunity_steps"], 1)
        self.assertEqual(result["benchmark_missed_overtake_opportunity_steps"], 0)
        self.assertEqual(result["benchmark_first_lane_change_attempt_step"], 1)
        self.assertEqual(result["benchmark_unsafe_lane_change_attempts"], 1)

    def test_benchmark_summary_aggregates_missed_overtake_opportunities(self):
        episodes = [
            {
                "category": "slow_lead_overtake",
                "task_completed": False,
                "ttc_score": 1.0,
                "speed_variance_score": 1.0,
                "time_efficiency_score": 0.0,
                "overall_score": 0.8,
                "driving_score": 0.0,
                "benchmark_failure_reason": "task_not_completed",
                "benchmark_success_criteria": {"type": "safe_overtake"},
                "benchmark_safe_overtake_opportunity_steps": 5,
                "benchmark_missed_overtake_opportunity_steps": 3,
            },
            {
                "category": "slow_lead_overtake",
                "task_completed": True,
                "ttc_score": 1.0,
                "speed_variance_score": 0.9,
                "time_efficiency_score": 0.8,
                "overall_score": 0.9,
                "driving_score": 0.9,
                "benchmark_failure_reason": "",
                "benchmark_success_criteria": {"type": "safe_overtake"},
                "benchmark_safe_overtake_opportunity_steps": 5,
                "benchmark_missed_overtake_opportunity_steps": 1,
            },
        ]

        summary = summarize_benchmark_episodes(episodes)

        self.assertEqual(summary["safe_overtake_opportunity_steps_total"], 10)
        self.assertEqual(summary["missed_overtake_opportunity_steps_total"], 4)
        self.assertEqual(summary["missed_overtake_opportunity_rate"], 0.4)
        category = summary["benchmark_by_category"]["slow_lead_overtake"]
        self.assertEqual(category["safe_overtake_opportunity_steps_total"], 10)
        self.assertEqual(category["missed_overtake_opportunity_steps_total"], 4)
        self.assertEqual(category["missed_overtake_opportunity_rate"], 0.4)

    def test_benchmark_summary_exposes_stress_success_rates(self):
        episodes = [
            {
                "category": "cut_in_brake_response",
                "task_completed": True,
                "ttc_score": 1.0,
                "speed_variance_score": 1.0,
                "time_efficiency_score": 0.8,
                "overall_score": 0.9,
                "driving_score": 0.9,
                "benchmark_success_criteria": {"type": "cut_in_brake_response"},
            },
            {
                "category": "cut_in_brake_response",
                "task_completed": False,
                "ttc_score": 0.6,
                "speed_variance_score": 0.8,
                "time_efficiency_score": 0.0,
                "overall_score": 0.5,
                "driving_score": 0.0,
                "benchmark_failure_reason": "task_not_completed",
                "benchmark_success_criteria": {"type": "cut_in_brake_response"},
            },
            {
                "category": "dense_dynamic_flow",
                "task_completed": True,
                "ttc_score": 1.0,
                "speed_variance_score": 0.9,
                "time_efficiency_score": 0.8,
                "overall_score": 0.9,
                "driving_score": 0.9,
                "benchmark_success_criteria": {"type": "dense_dynamic_flow"},
            },
        ]

        summary = summarize_benchmark_episodes(episodes)

        self.assertEqual(summary["cut_in_response_success_rate"], 0.5)
        self.assertEqual(summary["dynamic_dense_flow_success_rate"], 1.0)

    def test_benchmark_summary_exposes_stress_v2_success_and_pressure_rates(self):
        episodes = [
            {
                "category": "timed_gap_overtake",
                "task_completed": True,
                "ttc_score": 1.0,
                "speed_variance_score": 0.9,
                "time_efficiency_score": 0.8,
                "overall_score": 0.9,
                "driving_score": 0.9,
                "benchmark_success_criteria": {"type": "timed_gap_overtake"},
                "benchmark_valid_opportunity_step": 5,
                "benchmark_maneuver_in_window": True,
                "benchmark_passive_trap_failed": False,
            },
            {
                "category": "timed_gap_overtake",
                "task_completed": False,
                "ttc_score": 0.8,
                "speed_variance_score": 0.9,
                "time_efficiency_score": 0.0,
                "overall_score": 0.7,
                "driving_score": 0.0,
                "benchmark_success_criteria": {"type": "timed_gap_overtake"},
                "benchmark_valid_opportunity_step": 5,
                "benchmark_maneuver_in_window": False,
                "benchmark_passive_trap_failed": True,
            },
            {
                "category": "traffic_jam_escape",
                "task_completed": True,
                "ttc_score": 1.0,
                "speed_variance_score": 0.8,
                "time_efficiency_score": 0.8,
                "overall_score": 0.85,
                "driving_score": 0.85,
                "benchmark_success_criteria": {"type": "traffic_jam_escape"},
                "benchmark_valid_opportunity_step": 4,
                "benchmark_maneuver_in_window": True,
                "benchmark_passive_trap_failed": False,
            },
        ]

        summary = summarize_benchmark_episodes(episodes)

        self.assertEqual(summary["timed_gap_overtake_success_rate"], 0.5)
        self.assertEqual(summary["traffic_jam_escape_success_rate"], 1.0)
        self.assertEqual(summary["passive_trap_failure_rate"], 0.3333)
        self.assertEqual(summary["timely_maneuver_opportunity_count"], 3)
        self.assertEqual(summary["timely_maneuver_success_rate"], 0.6667)
        category = summary["benchmark_by_category"]["timed_gap_overtake"]
        self.assertEqual(category["passive_trap_failure_rate"], 0.5)
        self.assertEqual(category["timely_maneuver_success_rate"], 0.5)

    def test_blocked_lane_patience_fails_if_lane_change_shield_fires(self):
        ego = _DummyVehicle(lane_rank=1, speed=22.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=18.0, x=28.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "blocked_patience_test",
            "category": "blocked_lane_patience",
            "instruction": "patience test",
            "time_limit_sec": 12,
            "success_criteria": {
                "type": "blocked_lane_patience",
                "min_survival_steps": 2,
                "min_speed_mps": 18.0,
                "max_unsafe_lane_change_attempts": 0,
                "hold_steps": 1,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        evaluator.update(
            env,
            1,
            {"front_gap_m": 28.0, "ttc_sec": 5.0},
            crashed=False,
            action_context={"final_action_id": 1, "lane_change_shield_applied": True},
        )
        evaluator.update(
            env,
            2,
            {"front_gap_m": 30.0, "ttc_sec": 5.0},
            crashed=False,
            action_context={"final_action_id": 1},
        )

        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")
        self.assertFalse(result["task_completed"])
        self.assertEqual(result["benchmark_unsafe_lane_change_attempts"], 1)
        self.assertFalse(result["benchmark_criteria_status"]["unsafe_attempt_satisfied"])

    def test_post_brake_recovery_completes_after_clear_front_gap_and_speed_recovery(self):
        ego = _DummyVehicle(lane_rank=1, speed=15.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=15.0, x=16.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "post_brake_recovery_test",
            "category": "post_brake_recovery",
            "instruction": "recovery test",
            "time_limit_sec": 12,
            "success_criteria": {
                "type": "post_brake_recovery",
                "clear_front_gap_m": 25.0,
                "clear_front_ttc_sec": 4.0,
                "min_recovery_speed_mps": 22.0,
                "hold_steps": 1,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        evaluator.update(env, 1, {"front_gap_m": 16.0, "ttc_sec": 2.0}, crashed=False)
        ego.speed = 23.0
        evaluator.update(env, 2, {"front_gap_m": None, "ttc_sec": None}, crashed=False)
        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")

        self.assertTrue(result["task_completed"])
        self.assertEqual(result["benchmark_recovery_clear_step"], 2)
        self.assertEqual(result["benchmark_recovery_time_steps"], 0)

    def test_flow_dense_and_lane_discipline_reactive_criteria(self):
        ego = _DummyVehicle(lane_rank=1, speed=24.0, x=0.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[0, 1, 2, 3, 4])

        flow_case = {
            "case_id": "flow_case",
            "category": "free_flow_cruise",
            "instruction": "flow test",
            "time_limit_sec": 10,
            "success_criteria": {
                "type": "flow_cruise",
                "min_speed_mps": 20.0,
                "max_speed_mps": 30.0,
                "min_survival_steps": 2,
                "max_lane_changes": 0,
                "hold_steps": 1,
            },
        }
        flow_eval = BenchmarkEpisodeEvaluator(flow_case, env)
        flow_eval.update(env, 1, {"front_gap_m": None, "ttc_sec": None}, crashed=False, action_context={"final_action_id": 1})
        flow_eval.update(env, 2, {"front_gap_m": None, "ttc_sec": None}, crashed=False, action_context={"final_action_id": 1})
        self.assertTrue(flow_eval.finalize(False, "completed")["task_completed"])

        dense_case = {
            "case_id": "dense_case",
            "category": "dense_traffic_flow",
            "instruction": "dense test",
            "time_limit_sec": 10,
            "success_criteria": {
                "type": "dense_flow",
                "min_survival_steps": 2,
                "min_avg_speed_mps": 18.0,
                "max_ttc_danger_rate": 0.5,
                "max_headway_violation_rate": 0.5,
                "max_flap_accel_decel_count": 0,
                "hold_steps": 1,
            },
        }
        dense_eval = BenchmarkEpisodeEvaluator(dense_case, env)
        dense_eval.update(env, 1, {"front_gap_m": 40.0, "ttc_sec": 5.0}, crashed=False, action_context={"final_action_id": 1})
        dense_eval.update(env, 2, {"front_gap_m": 42.0, "ttc_sec": 5.0}, crashed=False, action_context={"final_action_id": 1})
        self.assertTrue(dense_eval.finalize(False, "completed")["task_completed"])

        lane_case = {
            "case_id": "lane_case",
            "category": "lane_discipline",
            "instruction": "lane discipline test",
            "time_limit_sec": 10,
            "success_criteria": {
                "type": "lane_discipline",
                "direction": "right",
                "expect_move": True,
                "min_speed_mps": 20.0,
                "hold_steps": 1,
            },
        }
        lane_eval = BenchmarkEpisodeEvaluator(lane_case, env)
        ego.lane_index = ("a", "b", 2)
        lane_eval.update(env, 1, {"front_gap_m": None, "ttc_sec": None}, crashed=False, action_context={"final_action_id": 2})
        self.assertTrue(lane_eval.finalize(False, "completed")["task_completed"])

    def test_stress_cut_in_and_false_alarm_criteria(self):
        ego = _DummyVehicle(lane_rank=1, speed=22.0, x=0.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[0, 1, 2, 3, 4])
        cut_in_case = {
            "case_id": "cut_in_test",
            "category": "cut_in_brake_response",
            "instruction": "cut in test",
            "time_limit_sec": 20,
            "success_criteria": {
                "type": "cut_in_brake_response",
                "min_survival_steps": 2,
                "max_ttc_danger_rate": 0.5,
                "max_headway_violation_rate": 0.5,
                "max_flap_accel_decel_count": 1,
                "min_avg_speed_mps": 10.0,
                "hold_steps": 1,
            },
        }
        cut_eval = BenchmarkEpisodeEvaluator(cut_in_case, env)
        cut_eval.update(
            env,
            1,
            {"front_gap_m": 20.0, "ttc_sec": 3.0, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={
                "final_action_id": 4,
                "benchmark_events_applied": True,
                "benchmark_event_ids": ["cut_in"],
                "benchmark_event_types": ["reposition_vehicle"],
                "benchmark_event_step": 1,
            },
        )
        cut_eval.update(
            env,
            2,
            {"front_gap_m": 24.0, "ttc_sec": 4.0, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 1},
        )
        self.assertTrue(cut_eval.finalize(False, "completed")["task_completed"])

        false_alarm_case = {
            "case_id": "false_alarm_test",
            "category": "false_alarm_stability",
            "instruction": "false alarm test",
            "time_limit_sec": 20,
            "success_criteria": {
                "type": "false_alarm_stability",
                "min_survival_steps": 2,
                "min_speed_mps": 18.0,
                "max_speed_mps": 30.0,
                "min_avg_speed_mps": 18.0,
                "max_lane_changes": 0,
                "max_flap_accel_decel_count": 0,
                "hold_steps": 1,
            },
        }
        false_eval = BenchmarkEpisodeEvaluator(false_alarm_case, env)
        false_eval.update(
            env,
            1,
            {"front_gap_m": 80.0, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={
                "final_action_id": 2,
                "benchmark_events_applied": True,
                "benchmark_event_ids": ["safe_motion"],
                "benchmark_event_types": ["set_lane_change"],
                "benchmark_event_step": 1,
            },
        )
        false_eval.update(
            env,
            2,
            {"front_gap_m": 80.0, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 1},
        )
        result = false_eval.finalize(False, "completed")
        self.assertFalse(result["task_completed"])
        self.assertFalse(result["benchmark_criteria_status"]["lane_change_satisfied"])
        self.assertEqual(result["benchmark_event_count_applied"], 1)

    def test_delayed_overtake_reports_shield_without_failing_safe_pass(self):
        ego = _DummyVehicle(lane_rank=1, speed=24.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=18.0, x=35.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "delayed_overtake_test",
            "category": "delayed_overtake_gap",
            "instruction": "delayed overtake test",
            "time_limit_sec": 20,
            "success_criteria": {
                "type": "delayed_overtake_gap",
                "direction": "left",
                "pass_margin_m": 8.0,
                "min_final_speed_mps": 20.0,
                "hold_steps": 1,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        evaluator.update(
            env,
            1,
            {"front_gap_m": 35.0, "ttc_sec": 5.0, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 0, "lane_change_shield_applied": True},
        )
        ego.lane_index = ("a", "b", 0)
        ego.position[0] = 45.0
        evaluator.update(
            env,
            2,
            {"front_gap_m": None, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={
                "final_action_id": 0,
                "benchmark_events_applied": True,
                "benchmark_event_ids": ["gap_open"],
                "benchmark_event_types": ["reposition_vehicle"],
                "benchmark_event_step": 2,
            },
        )
        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")

        self.assertTrue(result["task_completed"])
        self.assertEqual(result["benchmark_unsafe_lane_change_attempts"], 1)
        self.assertFalse(
            result["benchmark_criteria_status"]["unsafe_attempt_satisfied"]
        )
        self.assertTrue(result["benchmark_criteria_status"]["safety_satisfied"])

    def test_stress_v2_timed_gap_overtake_requires_in_window_safe_pass(self):
        ego = _DummyVehicle(lane_rank=1, speed=24.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=18.0, x=35.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "timed_gap_v2_test",
            "category": "timed_gap_overtake",
            "instruction": "timed gap test",
            "time_limit_sec": 20,
            "success_criteria": {
                "type": "timed_gap_overtake",
                "direction": "left",
                "opportunity_start_step": 5,
                "opportunity_end_step": 8,
                "pass_margin_m": 8.0,
                "min_final_speed_mps": 20.0,
                "min_progress_m": 35.0,
                "requires_event": False,
                "hold_steps": 1,
                "passive_trap": True,
            },
        }

        no_maneuver_eval = BenchmarkEpisodeEvaluator(copy.deepcopy(case), env)
        no_maneuver_eval.update(
            env,
            5,
            {"front_gap_m": 35.0, "ttc_sec": 5.0, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 1},
        )
        no_maneuver_result = no_maneuver_eval.finalize(False, "completed")
        self.assertFalse(no_maneuver_result["task_completed"])
        self.assertTrue(no_maneuver_result["benchmark_passive_trap_failed"])

        ego = _DummyVehicle(lane_rank=1, speed=24.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=18.0, x=35.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        early_eval = BenchmarkEpisodeEvaluator(copy.deepcopy(case), env)
        early_eval.update(
            env,
            3,
            {"front_gap_m": 35.0, "ttc_sec": 5.0, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 0, "lane_change_shield_applied": True},
        )
        ego.lane_index = ("a", "b", 0)
        ego.position[0] = 50.0
        early_eval.update(
            env,
            6,
            {"front_gap_m": None, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 1},
        )
        early_result = early_eval.finalize(False, "completed")
        self.assertFalse(early_result["task_completed"])
        self.assertEqual(early_result["benchmark_unsafe_lane_change_attempts"], 1)

        ego = _DummyVehicle(lane_rank=1, speed=24.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=18.0, x=35.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        pass_eval = BenchmarkEpisodeEvaluator(copy.deepcopy(case), env)
        ego.lane_index = ("a", "b", 0)
        ego.position[0] = 50.0
        pass_eval.update(
            env,
            6,
            {"front_gap_m": None, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 0},
        )
        pass_result = pass_eval.finalize(False, "completed")
        self.assertTrue(pass_result["task_completed"])
        self.assertTrue(pass_result["benchmark_maneuver_in_window"])
        self.assertTrue(pass_result["benchmark_criteria_status"]["maneuver_in_window_satisfied"])

    def test_stress_v2_traffic_jam_escape_fails_passive_and_passes_lane_escape(self):
        case = {
            "case_id": "jam_escape_v2_test",
            "category": "traffic_jam_escape",
            "instruction": "jam escape test",
            "time_limit_sec": 20,
            "success_criteria": {
                "type": "traffic_jam_escape",
                "direction": "left",
                "opportunity_start_step": 3,
                "opportunity_end_step": 10,
                "min_final_speed_mps": 18.0,
                "min_progress_m": 40.0,
                "requires_event": False,
                "hold_steps": 1,
                "passive_trap": True,
            },
        }
        ego = _DummyVehicle(lane_rank=1, speed=12.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=8.0, x=24.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        passive_eval = BenchmarkEpisodeEvaluator(copy.deepcopy(case), env)
        passive_eval.update(
            env,
            4,
            {"front_gap_m": 24.0, "ttc_sec": 6.0, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 1},
        )
        passive_result = passive_eval.finalize(False, "completed")
        self.assertFalse(passive_result["task_completed"])
        self.assertTrue(passive_result["benchmark_passive_trap_failed"])

        ego = _DummyVehicle(lane_rank=1, speed=22.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=8.0, x=24.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        escape_eval = BenchmarkEpisodeEvaluator(copy.deepcopy(case), env)
        ego.lane_index = ("a", "b", 0)
        ego.position[0] = 45.0
        escape_eval.update(
            env,
            4,
            {"front_gap_m": None, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 0},
        )
        escape_result = escape_eval.finalize(False, "completed")
        self.assertTrue(escape_result["task_completed"])
        self.assertEqual(escape_result["benchmark_jam_exit_step"], 4)

    def test_stress_v2_patience_rejects_early_lane_change_and_accepts_waiting(self):
        case = {
            "case_id": "jam_patience_v2_test",
            "category": "traffic_jam_patience",
            "instruction": "jam patience test",
            "time_limit_sec": 20,
            "success_criteria": {
                "type": "traffic_jam_patience",
                "safe_window_start_step": 8,
                "min_progress_m": 20.0,
                "max_lane_changes": 1,
                "requires_event": False,
                "hold_steps": 1,
            },
        }
        ego = _DummyVehicle(lane_rank=1, speed=18.0, x=25.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[0, 1, 2, 3, 4])
        early_eval = BenchmarkEpisodeEvaluator(copy.deepcopy(case), env)
        early_eval.update(
            env,
            4,
            {"front_gap_m": 60.0, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 0, "lane_change_shield_applied": True},
        )
        early_result = early_eval.finalize(False, "completed")
        self.assertFalse(early_result["task_completed"])
        self.assertFalse(early_result["benchmark_criteria_status"]["no_early_maneuver_satisfied"])

        ego = _DummyVehicle(lane_rank=1, speed=19.0, x=25.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[0, 1, 2, 3, 4])
        wait_eval = BenchmarkEpisodeEvaluator(copy.deepcopy(case), env)
        ego.position[0] = 50.0
        wait_eval.update(
            env,
            8,
            {"front_gap_m": 60.0, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 1},
        )
        wait_result = wait_eval.finalize(False, "completed")
        self.assertTrue(wait_result["task_completed"])
        self.assertTrue(wait_result["benchmark_criteria_status"]["no_early_maneuver_satisfied"])

    def test_stress_v2_cut_in_and_stop_go_require_braking_and_recovery(self):
        cut_case = {
            "case_id": "cut_in_recover_v2_test",
            "category": "cut_in_then_recover",
            "instruction": "cut-in recover test",
            "time_limit_sec": 20,
            "success_criteria": {
                "type": "cut_in_then_recover",
                "clear_front_gap_m": 25.0,
                "clear_front_ttc_sec": 4.0,
                "min_recovery_speed_mps": 20.0,
                "requires_event": False,
                "requires_brake_action": True,
                "hold_steps": 1,
            },
        }
        ego = _DummyVehicle(lane_rank=1, speed=16.0, x=0.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[0, 1, 2, 3, 4])
        missing_brake_eval = BenchmarkEpisodeEvaluator(copy.deepcopy(cut_case), env)
        ego.speed = 22.0
        missing_brake_eval.update(
            env,
            10,
            {"front_gap_m": None, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 1},
        )
        self.assertFalse(missing_brake_eval.finalize(False, "completed")["task_completed"])

        ego = _DummyVehicle(lane_rank=1, speed=16.0, x=0.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[0, 1, 2, 3, 4])
        recover_eval = BenchmarkEpisodeEvaluator(copy.deepcopy(cut_case), env)
        recover_eval.update(
            env,
            3,
            {"front_gap_m": 16.0, "ttc_sec": 2.0, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 4},
        )
        ego.speed = 22.0
        recover_eval.update(
            env,
            10,
            {"front_gap_m": None, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 3},
        )
        recover_result = recover_eval.finalize(False, "completed")
        self.assertTrue(recover_result["task_completed"])
        self.assertTrue(recover_result["benchmark_recovery_after_wave"])

        wave_case = copy.deepcopy(cut_case)
        wave_case["case_id"] = "stop_go_wave_v2_test"
        wave_case["category"] = "stop_go_wave_response"
        wave_case["success_criteria"].update(
            {
                "type": "stop_go_wave_response",
                "min_recovery_speed_mps": 18.0,
                "min_progress_m": 25.0,
            }
        )
        ego = _DummyVehicle(lane_rank=1, speed=14.0, x=0.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[0, 1, 2, 3, 4])
        wave_eval = BenchmarkEpisodeEvaluator(wave_case, env)
        wave_eval.update(
            env,
            3,
            {"front_gap_m": 18.0, "ttc_sec": 3.0, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 4},
        )
        ego.speed = 20.0
        ego.position[0] = 30.0
        wave_eval.update(
            env,
            12,
            {"front_gap_m": None, "ttc_sec": None, "ttc_danger": False, "headway_violation": False},
            crashed=False,
            action_context={"final_action_id": 3},
        )
        wave_result = wave_eval.finalize(False, "completed")
        self.assertTrue(wave_result["task_completed"])
        self.assertTrue(wave_result["benchmark_criteria_status"]["recovery_speed_satisfied"])

    def test_merge_complete_requires_progress_speed_and_hold_steps(self):
        ego = _DummyVehicle(lane_rank=1, speed=10.0, x=0.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[0, 1, 3, 4])
        case = {
            "case_id": "merge_speed_test",
            "category": "decisive_merge",
            "instruction": "merge test",
            "time_limit_sec": 12,
            "success_criteria": {
                "type": "merge_complete",
                "target_lane_offset": -1,
                "hold_steps": 2,
                "min_progress_m": 30.0,
                "min_speed_mps": 12.0,
                "max_speed_mps": 28.0,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        ego.lane_index = ("a", "b", 0)
        ego.position[0] = 35.0
        evaluator.update(env, 1, {"front_gap_m": None, "ttc_sec": None}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        self.assertFalse(evaluator.last_criteria_status["speed_band_satisfied"])

        ego.speed = 14.0
        evaluator.update(env, 2, {"front_gap_m": None, "ttc_sec": None}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        evaluator.update(env, 3, {"front_gap_m": None, "ttc_sec": None}, crashed=False)

        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")
        self.assertTrue(result["task_completed"])
        self.assertEqual(result["benchmark_completion_speed_mps"], 14.0)
        self.assertEqual(result["benchmark_completion_progress_m"], 35.0)
        self.assertTrue(result["benchmark_criteria_status"]["merge_progress_satisfied"])

    def test_arrive_with_required_yield_fails_without_slowing(self):
        ego = _DummyVehicle(lane_rank=1, speed=8.0, x=0.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[1, 3, 4])
        case = {
            "case_id": "yield_missing_test",
            "category": "yield_required",
            "instruction": "yield test",
            "time_limit_sec": 12,
            "success_criteria": {
                "type": "arrive",
                "hold_steps": 1,
                "requires_yield": True,
                "yield_speed_mps": 4.0,
                "min_yield_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        evaluator.update(
            env,
            1,
            {"front_gap_m": None, "ttc_sec": None},
            crashed=False,
            info={"is_success": True},
        )

        self.assertFalse(evaluator.task_completed)
        self.assertEqual(evaluator.yield_observed_steps, 0)
        self.assertTrue(evaluator.last_criteria_status["arrived"])
        self.assertFalse(evaluator.last_criteria_status["yield_satisfied"])

    def test_arrive_with_required_yield_passes_after_enough_yield_steps(self):
        ego = _DummyVehicle(lane_rank=1, speed=8.0, x=0.0)
        env = _DummyEnv(ego, front_vehicle=None, available_actions=[1, 3, 4])
        case = {
            "case_id": "yield_pass_test",
            "category": "yield_required",
            "instruction": "yield test",
            "time_limit_sec": 12,
            "success_criteria": {
                "type": "arrive",
                "hold_steps": 1,
                "requires_yield": True,
                "yield_speed_mps": 4.0,
                "min_yield_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        ego.speed = 3.0
        evaluator.update(
            env,
            1,
            {"front_gap_m": None, "ttc_sec": None},
            crashed=False,
            info={"is_success": False},
        )
        evaluator.update(
            env,
            2,
            {"front_gap_m": None, "ttc_sec": None},
            crashed=False,
            info={"is_success": True},
        )

        result = evaluator.finalize(crashed=False, episode_stop_reason="completed")
        self.assertTrue(result["task_completed"])
        self.assertEqual(result["benchmark_yield_observed_steps"], 2)
        self.assertTrue(result["benchmark_criteria_status"]["yield_satisfied"])

    def test_intersection_v1_env_override_preserves_discrete_target_speeds(self):
        bundle = resolve_simulation_env_bundle(
            {"sim_env_id": "intersection-v1", "sim_use_native_env_defaults": True},
            show_trajectories=False,
            render_agent=False,
            env_config_overrides={
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": [0, 5, 10, 15, 20, 25, 30],
                }
            },
            require_discrete_meta_action=True,
        )

        self.assertEqual(bundle["env_id"], "intersection-v1")
        self.assertEqual(
            list(bundle["env_config_snapshot"]["action"]["target_speeds"]),
            [0, 5, 10, 15, 20, 25, 30],
        )

    def test_benchmark_aggregate_includes_category_breakdown_and_ci(self):
        episodes = [
            {
                "category": "speed_increase",
                "task_completed": True,
                "ttc_score": 1.0,
                "speed_variance_score": 0.9,
                "time_efficiency_score": 0.8,
                "overall_score": 0.93,
                "driving_score": 0.93,
                "overall_score_v2": 0.75,
                "driving_score_v2": 0.75,
                "behavior_penalty_factor_v2": 0.8,
                "conservative_penalty_severity_v2": 0.1,
                "runtime_penalty_severity_v2": 0.125,
                "benchmark_failure_reason": "",
            },
            {
                "category": "speed_increase",
                "task_completed": False,
                "ttc_score": 0.9,
                "speed_variance_score": 0.8,
                "time_efficiency_score": 0.0,
                "overall_score": 0.69,
                "driving_score": 0.0,
                "overall_score_v2": 0.2,
                "driving_score_v2": 0.0,
                "behavior_penalty_factor_v2": 0.29,
                "conservative_penalty_severity_v2": 0.5,
                "runtime_penalty_severity_v2": 0.42,
                "benchmark_failure_reason": "task_not_completed",
            },
            {
                "category": "lane_change_left",
                "task_completed": True,
                "ttc_score": 0.8,
                "speed_variance_score": 0.7,
                "time_efficiency_score": 0.6,
                "overall_score": 0.73,
                "driving_score": 0.73,
                "overall_score_v2": 0.65,
                "driving_score_v2": 0.65,
                "behavior_penalty_factor_v2": 0.89,
                "conservative_penalty_severity_v2": 0.08,
                "runtime_penalty_severity_v2": 0.03,
                "benchmark_failure_reason": "",
            },
        ]
        summary = summarize_benchmark_episodes(episodes)
        self.assertEqual(summary["benchmark_case_count"], 3)
        self.assertAlmostEqual(summary["task_completion_rate"], 0.6667, places=4)
        self.assertIsNotNone(summary["task_completion_rate_ci95"])
        self.assertIsNotNone(summary["driving_score_ci95"])
        self.assertIn("driving_score_v2", summary)
        self.assertIn("driving_score_v2_ci95", summary)
        self.assertIn("behavior_penalty_factor_v2_mean", summary)
        self.assertIn("speed_increase", summary["benchmark_by_category"])
        self.assertIn("lane_change_left", summary["benchmark_by_category"])
        self.assertIn("driving_score_v2", summary["benchmark_by_category"]["speed_increase"])
        self.assertEqual(
            summary["benchmark_by_category"]["speed_increase"]["benchmark_case_count"],
            2,
        )

    def test_benchmark_result_validity_thresholds(self):
        valid, reason = benchmark_result_validity(
            decision_timeout_rate_mean=0.49,
            fallback_action_rate_mean=0.1,
            timeout_episode_rate=0.2,
        )
        self.assertTrue(valid)
        self.assertIsNone(reason)

        valid, reason = benchmark_result_validity(
            decision_timeout_rate_mean=0.5,
            fallback_action_rate_mean=0.2,
            timeout_episode_rate=0.1,
        )
        self.assertFalse(valid)
        self.assertIn("decision_timeout_rate_mean>=0.5", reason)

    def test_benchmark_finalize_preserves_timeout_cap_failure_reason(self):
        case = {
            "case_id": "speed_increase_case",
            "instruction": "Speed up safely.",
            "category": "speed_increase",
            "seed": 1,
            "success_criteria": {"type": "speed_band", "min_speed_mps": 20.0, "max_speed_mps": 30.0, "hold_steps": 2},
            "time_limit_sec": 8,
        }

        class _DummyEnv:
            config = {"policy_frequency": 1}
            unwrapped = None

        env = _DummyEnv()
        env.unwrapped = env
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        metrics = evaluator.finalize(crashed=False, episode_stop_reason="episode_timeout_cap")
        self.assertEqual(metrics["benchmark_failure_reason"], "episode_timeout_cap")

    def test_seed_mode_aggregate_has_no_benchmark_fields(self):
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
        }
        summary = aggregate_results("seed_only_model", [episode])
        self.assertNotIn("task_completion_rate", summary)
        self.assertNotIn("benchmark_result_valid", summary)

    def test_aggregate_includes_stop_and_near_stop_metrics(self):
        episodes = [
            {
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
                "avg_ego_speed_mps": 0.2,
                "ttc_danger_rate": 0.0,
                "headway_violation_rate": 0.0,
                "rear_ttc_danger_rate": 0.0,
                "rear_headway_violation_rate": 0.0,
                "low_speed_blocking_rate": 1.0,
                "lane_change_rate": 0.0,
                "flap_accel_decel_rate": 0.0,
                "decision_latency_ms_avg": 5.0,
                "timeout_penalty_events": 0,
                "timeout_penalty_timeout_triggers": 0,
                "timeout_penalty_slow_triggers": 0,
                "timeout_penalty_stage_max": 0,
                "min_ego_speed_mps": 0.0,
                "stopped_ever": True,
                "stop_steps": 8,
                "stop_rate": 0.8,
                "near_stop_steps": 10,
                "near_stop_rate": 1.0,
            },
            {
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
                "avg_ego_speed_mps": 12.0,
                "ttc_danger_rate": 0.0,
                "headway_violation_rate": 0.0,
                "rear_ttc_danger_rate": 0.0,
                "rear_headway_violation_rate": 0.0,
                "low_speed_blocking_rate": 0.0,
                "lane_change_rate": 0.0,
                "flap_accel_decel_rate": 0.0,
                "decision_latency_ms_avg": 5.0,
                "timeout_penalty_events": 0,
                "timeout_penalty_timeout_triggers": 0,
                "timeout_penalty_slow_triggers": 0,
                "timeout_penalty_stage_max": 0,
                "min_ego_speed_mps": 10.0,
                "stopped_ever": False,
                "stop_steps": 0,
                "stop_rate": 0.0,
                "near_stop_steps": 0,
                "near_stop_rate": 0.0,
            },
        ]
        summary = aggregate_results("stop_metrics_model", episodes)
        self.assertEqual(summary["min_ego_speed_mps_mean"], 5.0)
        self.assertEqual(summary["stop_episode_rate"], 0.5)
        self.assertEqual(summary["stop_rate_mean"], 0.4)
        self.assertEqual(summary["near_stop_episode_rate"], 0.5)
        self.assertEqual(summary["near_stop_rate_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
