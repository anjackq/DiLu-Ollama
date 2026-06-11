import unittest
import os
import tempfile
import json

import yaml
from dilu.runtime import (
    DEFAULT_DILU_SEEDS,
    load_benchmark_case_set,
    load_runtime_config,
)
from evaluate_models_ollama import (
    _apply_measurement_runtime_overrides,
    _filter_benchmark_cases_by_category,
    _parse_benchmark_category_filter,
    main as eval_main,
    parse_seeds,
    resolve_eval_seeds,
)
from merge_eval_reports import (
    _compat_profile,
    _compare_profiles,
    _discover_available_model_names,
    _discover_model_artifacts,
    _infer_compare_metadata_from_summary,
    _read_episodes,
)


class _Args:
    def __init__(
        self,
        *,
        ollama_think_mode=None,
        ollama_use_native_chat=False,
        ollama_disable_native_chat=False,
    ):
        self.ollama_think_mode = ollama_think_mode
        self.ollama_use_native_chat = ollama_use_native_chat
        self.ollama_disable_native_chat = ollama_disable_native_chat


class CliMergeTests(unittest.TestCase):
    def test_eval_seed_bank_accepts_config_list_and_cli_override(self):
        config = {"eval_seed_bank": list(DEFAULT_DILU_SEEDS)}

        self.assertEqual(len(resolve_eval_seeds(config, None)), 100)
        self.assertEqual(resolve_eval_seeds(config, None, seed_count=3), [4091, 2125, 9293])
        self.assertEqual(resolve_eval_seeds(config, "1,2,3"), [1, 2, 3])
        self.assertEqual(resolve_eval_seeds(config, "1,2,3", seed_count=2), [1, 2])
        self.assertEqual(parse_seeds([10, "20", 30]), [10, 20, 30])
        with self.assertRaises(ValueError):
            resolve_eval_seeds(config, None, seed_count=101)

    def test_active_config_falls_back_to_builtin_100_seed_bank(self):
        cfg = load_runtime_config("config.yaml")

        self.assertNotIn("eval_seed_bank", cfg)
        self.assertEqual(cfg["OLLAMA_USE_NATIVE_CHAT"], "auto")
        self.assertEqual(len(resolve_eval_seeds(cfg, None)), 100)
        self.assertEqual(resolve_eval_seeds(cfg, None, seed_count=5), [4091, 2125, 9293, 8030, 1620])

    def test_merge_eval_reports_discovers_measurement_mode_energy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_root = os.path.join(tmpdir, "measurement_exp")
            energy_dir = os.path.join(experiment_root, "models", "qwen3_14b", "energy")
            os.makedirs(energy_dir, exist_ok=True)

            summary_path = os.path.join(energy_dir, "energy_summary_20260412_120000.json")
            episodes_path = os.path.join(energy_dir, "energy_episodes_20260412_120000.json")
            summary_payload = {
                "model": "qwen3:14b",
                "aggregate": {"model": "qwen3:14b", "driving_score_v2": 0.0},
                "episodes": [{"seed": 101, "max_steps": 12}],
                "metrics_config": {
                    "few_shot_num": 0,
                    "decision_timeout_sec": 20.0,
                },
            }
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary_payload, handle)
            with open(episodes_path, "w", encoding="utf-8") as handle:
                json.dump({"episodes": summary_payload["episodes"]}, handle)

            discovered = _discover_model_artifacts(experiment_root, "qwen3:14b")
            available_models = _discover_available_model_names(experiment_root)
            episodes = _read_episodes(summary_payload, discovered["episodes_path"])
            profile = _compat_profile("qwen3:14b", discovered, summary_payload, episodes)

        self.assertIsNotNone(discovered)
        self.assertIn("qwen3:14b", available_models)
        self.assertEqual(os.path.abspath(summary_path), discovered["summary_path"])
        self.assertEqual(os.path.abspath(episodes_path), discovered["episodes_path"])
        self.assertEqual(profile["few_shot_num"], 0)
        self.assertEqual(profile["simulation_duration"], 12)

    def test_merge_eval_reports_infers_benchmark_metadata_from_measurement_summary(self):
        summary_payload = {
            "aggregate": {"driving_score_v2": 0.1},
            "metrics_config": {
                "benchmark_mode": True,
                "benchmark_case_set": "lampilot_highway_v1",
                "benchmark_case_set_path": "benchmarks/lampilot_highway_v1/cases.json",
                "benchmark_categories": ["speed_increase", "speed_decrease"],
                "benchmark_variant": "legacy_direct_action",
                "execution_mode": "direct_action_loop",
                "benchmark_fingerprint": "lampilot_highway_v1:abc123",
                "benchmark_metric_config": {
                    "recommended_headline_metric": "driving_score_v2",
                },
            },
        }

        metadata = _infer_compare_metadata_from_summary(summary_payload)

        self.assertTrue(metadata["benchmark_mode"])
        self.assertEqual(metadata["benchmark_case_set"], "lampilot_highway_v1")
        self.assertEqual(metadata["benchmark_variant"], "legacy_direct_action")
        self.assertEqual(metadata["execution_mode"], "direct_action_loop")
        self.assertEqual(metadata["benchmark_fingerprint"], "lampilot_highway_v1:abc123")
        self.assertEqual(metadata["headline_task_metric"], "driving_score_v2")
        self.assertEqual(metadata["benchmark_categories"], ["speed_increase", "speed_decrease"])

    def test_merge_eval_reports_rejects_cross_benchmark_variant_mixing(self):
        legacy_summary = {
            "aggregate": {"driving_score_v2": 0.1},
            "metrics_config": {
                "few_shot_num": 0,
                "benchmark_mode": True,
                "benchmark_case_set": "lampilot_highway_v1",
                "benchmark_variant": "legacy_direct_action",
                "execution_mode": "direct_action_loop",
                "benchmark_fingerprint": "lampilot_highway_v1:legacy",
            },
        }
        port_summary = {
            "aggregate": {"task_success_rate": 0.5},
            "metrics_config": {
                "few_shot_num": 0,
                "benchmark_mode": True,
                "benchmark_case_set": "lampilot_highway_port_v1",
                "benchmark_variant": "port_policy_exec",
                "execution_mode": "programmatic_policy_exec",
                "benchmark_fingerprint": "lampilot_highway_port_v1:port",
            },
        }
        legacy_episodes = [{"seed": 1, "max_steps": 12}]
        port_episodes = [{"seed": 1, "max_steps": 12}]

        legacy_profile = _compat_profile("legacy", {"manifest": {}}, legacy_summary, legacy_episodes)
        port_profile = _compat_profile("port", {"manifest": {}}, port_summary, port_episodes)
        diffs = _compare_profiles("legacy", legacy_profile, "port", port_profile)

        self.assertTrue(any("benchmark_case_set mismatch" in diff for diff in diffs))
        self.assertTrue(any("benchmark_variant mismatch" in diff for diff in diffs))
        self.assertTrue(any("benchmark_fingerprint mismatch" in diff for diff in diffs))

    def test_load_runtime_config_supports_relative_base_config_inheritance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base.yaml")
            child_path = os.path.join(tmpdir, "child.yaml")
            with open(base_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "OPENAI_API_TYPE": "ollama",
                        "sim_env_id": "highway-fast-v0",
                        "sim_action_target_speeds": [20, 25, 30],
                        "nested": {
                            "keep": 1,
                            "override": 2,
                        },
                    },
                    handle,
                    sort_keys=False,
                )
            with open(child_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "base_config": "base.yaml",
                        "sim_action_target_speeds": [0, 5, 10, 15, 20, 25, 30],
                        "nested": {
                            "override": 3,
                        },
                    },
                    handle,
                    sort_keys=False,
                )

            loaded = load_runtime_config(child_path)

        self.assertEqual(loaded["OPENAI_API_TYPE"], "ollama")
        self.assertEqual(loaded["sim_env_id"], "highway-fast-v0")
        self.assertEqual(loaded["sim_action_target_speeds"], [0, 5, 10, 15, 20, 25, 30])
        self.assertEqual(loaded["nested"]["keep"], 1)
        self.assertEqual(loaded["nested"]["override"], 3)

    def test_llm_full_safe_config_inherits_active_config_with_resource_overrides(self):
        active = load_runtime_config("config.yaml")
        safe = load_runtime_config("config.llm_full_safe.yaml")

        self.assertEqual(safe["OPENAI_API_TYPE"], active["OPENAI_API_TYPE"])
        self.assertEqual(safe["OLLAMA_CHAT_MODEL"], active["OLLAMA_CHAT_MODEL"])
        self.assertEqual(safe["sim_action_target_speeds"], active["sim_action_target_speeds"])
        self.assertTrue(safe["OLLAMA_USE_NATIVE_CHAT"])
        self.assertFalse(safe["eval_record_video"])
        self.assertFalse(safe["eval_save_run_artifacts"])
        self.assertFalse(safe["eval_enable_intent_resolver"])
        self.assertEqual(safe["ollama_runtime_num_ctx"], 4096)
        self.assertEqual(safe["ollama_runtime_keep_alive"], "10m")
        self.assertTrue(safe["ollama_unload_after_case"])
        self.assertEqual(safe["ollama_unload_after_case_timeout_sec"], 15)
        self.assertEqual(safe["ollama_runtime_max_loaded_models"], 1)
        self.assertEqual(safe["ollama_runtime_num_parallel"], 1)
        self.assertEqual(safe["ollama_runtime_max_queue"], 1)

    def test_measurement_mode_applies_benchmark_ollama_overrides(self):
        config = {
            "OPENAI_API_TYPE": "ollama",
            "OLLAMA_THINK_MODE": "auto",
            "OLLAMA_USE_NATIVE_CHAT": False,
        }
        updated = _apply_measurement_runtime_overrides(
            config,
            _Args(ollama_think_mode="no_think"),
            energy_mode="latency_only",
        )
        self.assertEqual(updated["OLLAMA_THINK_MODE"], "no_think")
        self.assertTrue(updated["OLLAMA_USE_NATIVE_CHAT"])
        self.assertTrue(updated["_benchmark_ollama_runtime_overrides"]["auto_forced_native_chat"])

    def test_standard_eval_mode_leaves_ollama_runtime_config_unchanged(self):
        config = {
            "OPENAI_API_TYPE": "ollama",
            "OLLAMA_THINK_MODE": "auto",
            "OLLAMA_USE_NATIVE_CHAT": False,
        }
        updated = _apply_measurement_runtime_overrides(
            config,
            _Args(ollama_think_mode="no_think", ollama_use_native_chat=True),
            energy_mode="none",
        )
        self.assertEqual(updated["OLLAMA_THINK_MODE"], "auto")
        self.assertFalse(updated["OLLAMA_USE_NATIVE_CHAT"])
        self.assertNotIn("_benchmark_ollama_runtime_overrides", updated)

    def test_benchmark_category_filter_parser_accepts_single_and_multiple_categories(self):
        self.assertEqual(
            _parse_benchmark_category_filter("slow_lead_overtake"),
            ["slow_lead_overtake"],
        )
        self.assertEqual(
            _parse_benchmark_category_filter("slow_lead_overtake, lane_discipline"),
            ["slow_lead_overtake", "lane_discipline"],
        )
        self.assertEqual(_parse_benchmark_category_filter(None), [])

    def test_benchmark_category_filter_parser_rejects_empty_tokens(self):
        with self.assertRaisesRegex(ValueError, "empty category"):
            _parse_benchmark_category_filter("slow_lead_overtake,,lane_discipline")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _parse_benchmark_category_filter(" ")

    def test_benchmark_category_filter_selects_expected_highway_reactive_cases(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_v1")

        filtered_case_set, filtered_cases = _filter_benchmark_cases_by_category(
            case_set,
            ["slow_lead_overtake"],
        )

        self.assertEqual(len(filtered_cases), 10)
        self.assertEqual(filtered_case_set["categories"], ["slow_lead_overtake"])
        self.assertTrue(all(case["category"] == "slow_lead_overtake" for case in filtered_cases))

    def test_benchmark_category_filter_selects_expected_stress_cases(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_stress_v1")

        filtered_case_set, filtered_cases = _filter_benchmark_cases_by_category(
            case_set,
            ["cut_in_brake_response"],
        )

        self.assertEqual(len(filtered_cases), 10)
        self.assertEqual(filtered_case_set["categories"], ["cut_in_brake_response"])
        self.assertTrue(all(case["category"] == "cut_in_brake_response" for case in filtered_cases))

    def test_benchmark_category_filter_selects_expected_stress_v2_cases(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_stress_v2")

        filtered_case_set, filtered_cases = _filter_benchmark_cases_by_category(
            case_set,
            ["traffic_jam_escape"],
        )

        self.assertEqual(len(filtered_cases), 12)
        self.assertEqual(filtered_case_set["categories"], ["traffic_jam_escape"])
        self.assertTrue(all(case["category"] == "traffic_jam_escape" for case in filtered_cases))

    def test_benchmark_category_filter_supports_multiple_categories_in_original_order(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_v1")

        filtered_case_set, filtered_cases = _filter_benchmark_cases_by_category(
            case_set,
            ["slow_lead_overtake", "lane_discipline"],
        )

        self.assertEqual(len(filtered_cases), 20)
        self.assertEqual(
            filtered_case_set["categories"],
            ["lane_discipline", "slow_lead_overtake"],
        )
        self.assertEqual(filtered_cases[0]["case_id"], "slow_lead_overtake_001")
        self.assertEqual(filtered_cases[9]["case_id"], "slow_lead_overtake_010")
        self.assertEqual(filtered_cases[10]["case_id"], "lane_discipline_001")

    def test_benchmark_category_filter_rejects_unknown_category(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_v1")

        with self.assertRaisesRegex(ValueError, "Available categories"):
            _filter_benchmark_cases_by_category(case_set, ["missing_category"])

    def test_benchmark_categories_requires_benchmark_mode(self):
        with self.assertRaisesRegex(ValueError, "--benchmark-categories requires --benchmark-case-set"):
            eval_main(["--models", "llama3.2:3b", "--benchmark-categories", "slow_lead_overtake"])

    def test_limit_after_benchmark_category_filter_selects_first_cases_inside_category(self):
        case_set = load_benchmark_case_set("dilu_highway_reactive_v1")
        _, filtered_cases = _filter_benchmark_cases_by_category(
            case_set,
            ["slow_lead_overtake"],
        )

        limited_cases = filtered_cases[:3]

        self.assertEqual(
            [case["case_id"] for case in limited_cases],
            ["slow_lead_overtake_001", "slow_lead_overtake_002", "slow_lead_overtake_003"],
        )


if __name__ == "__main__":
    unittest.main()
