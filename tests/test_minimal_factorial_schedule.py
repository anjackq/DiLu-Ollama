from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"
CASE_PATH = ROOT / "benchmarks" / "dilu_highway_reactive_stress_v2" / "cases.json"


class MinimalFactorialScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

        self.manifest = load_experiment_manifest(MANIFEST_PATH)
        self.case_set = json.loads(CASE_PATH.read_text(encoding="utf-8"))
        self.digests = {
            "qwen_06b": "a" * 64,
            "llama_1b": "b" * 64,
        }

    @staticmethod
    def _clean_git(command, **_kwargs):
        action = command[1]
        stdout = "27c2cd9\n" if action == "rev-parse" else command[-1] + "\n" if action == "ls-files" else ""
        return type("Completed", (), {"stdout": stdout, "returncode": 0})()

    def test_manifest_is_strict_and_freezes_required_constants(self) -> None:
        self.assertEqual(self.manifest.campaign_id, "iclr2027-minimal-factorial-v1")
        self.assertEqual(self.manifest.smoke_campaign_id, "iclr2027-minimal-factorial-smoke-v1")
        self.assertEqual(self.manifest.case_path, "benchmarks/dilu_highway_reactive_stress_v2/cases.json")
        self.assertEqual(tuple(model.slot for model in self.manifest.models), ("qwen_06b", "llama_1b"))
        self.assertEqual(tuple(model.tag for model in self.manifest.models), ("qwen3:0.6b", "llama3.2:1b"))
        self.assertEqual(self.manifest.transport.native_endpoint, "http://localhost:11434/api/chat")
        self.assertEqual(self.manifest.transport.generation_seed_master, 20270728)
        self.assertEqual(self.manifest.selection.categories, 10)
        self.assertEqual(self.manifest.selection.stage1_cases_per_category, 3)
        self.assertEqual(self.manifest.selection.stage2_cases_per_category, 12)
        with self.assertRaises(ValueError):
            type(self.manifest).from_mapping({})

    def test_conditions_and_hash_selected_cases_are_deterministic(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_harness_config,
            select_smoke_case,
            select_stage1_cases,
        )

        conditions = tuple(build_harness_config(self.manifest, condition) for condition in range(8))
        self.assertEqual(tuple(item.condition_id() for item in conditions), tuple(f"c{i:03b}" for i in range(8)))
        smoke = select_smoke_case(self.case_set, self.manifest.campaign_id)
        expected_smoke = min(
            self.case_set["cases"],
            key=lambda case: hashlib.sha256(
                f"{self.manifest.campaign_id}|smoke|{case['case_id']}".encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(smoke["case_id"], expected_smoke["case_id"])
        selected = select_stage1_cases(self.case_set, self.manifest.campaign_id)
        self.assertEqual(len(selected), 30)
        self.assertEqual(len({case["case_id"] for case in selected}), 30)
        for category in {case["category"] for case in self.case_set["cases"]}:
            self.assertEqual(sum(case["category"] == category for case in selected), 3)

    def test_union_schedule_has_exact_rows_reused_endpoints_and_stable_identities(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import build_union_schedule

        schedule = build_union_schedule(
            self.manifest, self.case_set, self.digests, code_revision="27c2cd9"
        )
        stage1 = tuple(item for item in schedule if item.stage == "s1")
        stage2 = tuple(item for item in schedule if item.stage == "s2_additional")
        self.assertEqual(len(stage1), 480)
        self.assertEqual(len(stage2), 360)
        self.assertEqual(len(schedule), 840)
        self.assertEqual({item.condition_id for item in stage2}, {"c000", "c111"})
        endpoint_stage1_cases = {
            item.case_id for item in stage1 if item.condition_id in {"c000", "c111"}
        }
        self.assertEqual(len(endpoint_stage1_cases), 30)
        self.assertFalse(endpoint_stage1_cases & {item.case_id for item in stage2})
        episode = schedule[0]
        self.assertEqual(episode.replicate_id, 0)
        self.assertEqual(episode.pair_id, "pair-" + hashlib.sha256(
            f"{self.manifest.campaign_id}|{episode.case_id}|{episode.simulator_seed}".encode("utf-8")
        ).hexdigest())
        self.assertTrue(episode.template_id.startswith("stress-v2-"))
        self.assertTrue(episode.primary_snapshot_id.startswith("snapshot-"))
        self.assertTrue(episode.episode_attempt_id.startswith("episode-"))
        self.assertEqual(schedule, build_union_schedule(
            self.manifest, self.case_set, self.digests, code_revision="27c2cd9"
        ))

    def test_runtime_snapshot_is_pre_execution_and_fails_closed_on_runtime_drift(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import build_runtime_snapshot

        with patch("dilu.runtime.minimal_factorial_schedule.subprocess.run", self._clean_git):
            snapshot = build_runtime_snapshot(self.manifest, self.case_set)
        required = {
            "code_revision", "source_sha256", "runtime_config", "environment_config",
            "primary_metric_spec", "shield_config", "scoring_fingerprint",
            "predicate_fingerprint", "simulator_versions", "trace_schema_sha256",
        }
        self.assertTrue(required <= set(snapshot.payload))
        self.assertEqual(snapshot.payload["code_revision"], "27c2cd9")
        self.assertEqual(len(snapshot.sha256), 64)
        changed = dict(self.case_set)
        changed["cases"] = list(changed["cases"][:-1])
        with self.assertRaises(ValueError):
            build_runtime_snapshot(self.manifest, changed)

    def test_smoke_and_union_identities_use_runtime_full_fingerprint_and_exact_formulas(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_smoke_schedule,
            build_union_schedule,
        )

        smoke = build_smoke_schedule(
            self.manifest, self.case_set, self.digests, code_revision="27c2cd9"
        )[0]
        union = build_union_schedule(
            self.manifest, self.case_set, self.digests, code_revision="27c2cd9"
        )[0]
        for episode, campaign_id in ((smoke, self.manifest.smoke_campaign_id), (union, self.manifest.campaign_id)):
            self.assertEqual(episode.identity().campaign_id, campaign_id)
            self.assertEqual(episode.benchmark_fingerprint[:7], "sha256:")
            self.assertEqual(len(episode.benchmark_fingerprint), 71)
            self.assertEqual(episode.pair_id, "pair-" + hashlib.sha256(
                f"{campaign_id}|{episode.case_id}|{episode.simulator_seed}".encode()
            ).hexdigest())
            self.assertEqual(episode.episode_attempt_id, "episode-" + hashlib.sha256(
                f"{campaign_id}|{episode.model_tag}|{episode.model_digest}|{episode.condition_id}|{episode.case_id}|{episode.simulator_seed}|0".encode()
            ).hexdigest())
            self.assertEqual(episode.template_id, "stress-v2-" + hashlib.sha256(
                f"{episode.benchmark_fingerprint}|{episode.case_id}".encode()
            ).hexdigest())
            self.assertEqual(episode.primary_snapshot_id, "snapshot-" + hashlib.sha256(
                f"{episode.benchmark_fingerprint}|{episode.case_id}|{episode.simulator_seed}".encode()
            ).hexdigest())

    def test_manifest_and_snapshot_are_deeply_immutable_and_round_trip_supporting_specs(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_runtime_snapshot,
            write_frozen_campaign_manifest,
        )

        with self.assertRaises(TypeError):
            self.manifest.fixed_harness.retry_policy["retry_on_timeout"] = True
        self.assertEqual(self.manifest.bootstrap.draws, 20000)
        self.assertEqual(self.manifest.bootstrap.version, "bootstrap-v1")
        self.assertEqual(self.manifest.outputs.analysis, "analysis")
        with patch("dilu.runtime.minimal_factorial_schedule.subprocess.run", self._clean_git):
            snapshot = build_runtime_snapshot(self.manifest, self.case_set)
        with self.assertRaises(TypeError):
            snapshot.payload["code_revision"] = "drift"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frozen.json"
            write_frozen_campaign_manifest(path, self.manifest, snapshot, ())
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["manifest"]["bootstrap"], {"draws": 20000, "version": "bootstrap-v1"})
        self.assertEqual(saved["manifest"]["outputs"]["s1"], "s1")

    def test_snapshot_rejects_porcelain_dirty_and_untracked_and_fingerprint_drift(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import build_runtime_snapshot

        dirty = type("Completed", (), {"stdout": " M config.example.yaml\n", "returncode": 0})()
        with patch("dilu.runtime.minimal_factorial_schedule.subprocess.run", return_value=dirty):
            with self.assertRaises(ValueError):
                build_runtime_snapshot(self.manifest, self.case_set)
        untracked = type("Completed", (), {"stdout": "?? unexpected.txt\n", "returncode": 0})()
        with patch("dilu.runtime.minimal_factorial_schedule.subprocess.run", return_value=untracked):
            with self.assertRaises(ValueError):
                build_runtime_snapshot(self.manifest, self.case_set)
        staged = type("Completed", (), {"stdout": "M  config.example.yaml\n", "returncode": 0})()
        with patch("dilu.runtime.minimal_factorial_schedule.subprocess.run", return_value=staged):
            with self.assertRaises(ValueError):
                build_runtime_snapshot(self.manifest, self.case_set)
        with patch("dilu.runtime.minimal_factorial_schedule.subprocess.run", self._clean_git), patch(
            "dilu.runtime.minimal_factorial_schedule._file_sha", return_value="0" * 64
        ):
            with self.assertRaises(ValueError):
                build_runtime_snapshot(self.manifest, self.case_set)
        def missing_source(command, **_kwargs):
            if command[1] == "ls-files":
                return type("Completed", (), {"stdout": "", "returncode": 1})()
            return self._clean_git(command)
        with patch("dilu.runtime.minimal_factorial_schedule.subprocess.run", missing_source):
            with self.assertRaises(ValueError):
                build_runtime_snapshot(self.manifest, self.case_set)


if __name__ == "__main__":
    unittest.main()
