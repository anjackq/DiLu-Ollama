from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.grounded_decoding_schedule_support import (
    FROZEN_DIGESTS,
    MANIFEST_PATH,
    ROOT,
    V5_EPISODES_PATH,
    V5_MANIFEST_PATH,
    V7_EPISODES_PATH,
    V7_MANIFEST_PATH,
    fake_git,
    frozen_bindings,
    read_jsonl,
)


class GroundedDecodingScheduleTests(unittest.TestCase):
    """Covers all six numbered requirements from task-4-brief.md."""

    def setUp(self) -> None:
        from dilu.runtime.grounded_decoding_schedule import (
            build_runtime_snapshot,
            load_grounded_decoding_manifest,
        )
        from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

        self.manifest = load_grounded_decoding_manifest(MANIFEST_PATH)
        self.cases = json.loads((ROOT / self.manifest.case_path).read_text())
        self.v5_manifest = load_experiment_manifest(V5_MANIFEST_PATH)
        self.v7_manifest = load_experiment_manifest(V7_MANIFEST_PATH)
        self.bindings = frozen_bindings()
        with patch("dilu.runtime._minimal_factorial_manifest.subprocess.run", fake_git):
            self.snapshot = build_runtime_snapshot(self.manifest, self.cases)

    def _schedule(self, **kwargs):
        from dilu.runtime.grounded_decoding_schedule import build_v8_schedule

        return build_v8_schedule(
            self.manifest,
            self.cases,
            self.bindings,
            runtime_snapshot=self.snapshot,
            **kwargs,
        )

    def _contract(self):
        from dilu.runtime.grounded_decoding_schedule import build_comparator_contract

        return build_comparator_contract(
            self.manifest, self.cases, self.v5_manifest, self.v7_manifest
        )

    # -- Requirement 1: exactly the two P1 O2 cells, c120/c121 -------------

    def test_grounded_condition_cells_are_exactly_c120_and_c121(self) -> None:
        schedule = self._schedule()
        self.assertEqual({row.condition_id for row in schedule.stage1}, {"c120", "c121"})
        self.assertEqual({row.condition_id for row in schedule.stage2_additional}, {"c121"})
        # Pin the harness's own derivation, independent of the schedule builder:
        # P1=modular_harness -> "1", O2=backend_schema_grounded -> "2", E0/E1 -> "0"/"1".
        from dilu.runtime._harness_config_support import (
            ConditionSpec,
            ExecutionMode,
            OutputEnforcement,
            PolicyContent,
        )

        e0 = ConditionSpec(
            PolicyContent.MODULAR_HARNESS,
            OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
            ExecutionMode.UNSHIELDED_OPERATIONAL,
        )
        e1 = ConditionSpec(
            PolicyContent.MODULAR_HARNESS,
            OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
            ExecutionMode.SHIELDED,
        )
        self.assertEqual(e0.condition_id(), "c120")
        self.assertEqual(e1.condition_id(), "c121")

    # -- Requirement 2: Stage-1 selection equals the frozen 30 case ids ----

    def test_stage1_selection_equals_frozen_campaign_case_ids(self) -> None:
        v7_rows = read_jsonl(V7_EPISODES_PATH)
        frozen_stage1_case_ids = {
            row["case_id"] for row in v7_rows if row["stage"] == "stage1"
        }
        self.assertEqual(len(frozen_stage1_case_ids), 30)

        schedule = self._schedule()
        v8_stage1_case_ids = {row.case_id for row in schedule.stage1}
        self.assertEqual(v8_stage1_case_ids, frozen_stage1_case_ids)

        v5_rows = read_jsonl(V5_EPISODES_PATH)
        v5_stage1_case_ids = {row["case_id"] for row in v5_rows if row["stage"] == "stage1"}
        self.assertEqual(v5_stage1_case_ids, frozen_stage1_case_ids)

    # -- Requirement 3: 300 + 180 = 480 unique, identical identity recipe --

    def test_300_plus_180_equals_480_unique_with_identical_identity_recipe(self) -> None:
        schedule = self._schedule()
        self.assertEqual(len(schedule.stage1), 300)
        self.assertEqual(len(schedule.stage2_additional), 180)
        self.assertEqual(len(schedule.all_claim_bearing), 480)
        self.assertEqual(
            len({row.episode_attempt_id for row in schedule.all_claim_bearing}), 480
        )
        self.assertTrue(all(row.stage == "stage1" for row in schedule.stage1))
        self.assertTrue(
            all(row.stage == "stage2_additional" for row in schedule.stage2_additional)
        )
        self.assertEqual(len({row.case_id for row in schedule.stage1}), 30)
        self.assertEqual(len({row.case_id for row in schedule.stage2_additional}), 90)
        self.assertFalse(
            {row.case_id for row in schedule.stage1}
            & {row.case_id for row in schedule.stage2_additional}
        )
        self.assertEqual(
            {row.model_slot for row in schedule.stage2_additional},
            {"qwen_06b", "llama_1b"},
        )
        self.assertTrue(
            all(
                row.campaign_id == "iclr2027-grounded-decoding-v8"
                for row in schedule.all_claim_bearing
            )
        )

        row = schedule.stage1[0]
        expected_attempt_id = "episode-" + hashlib.sha256(
            f"{row.campaign_id}|{row.model_tag}|{row.model_digest}|"
            f"{row.condition_id}|{row.case_id}|{row.simulator_seed}|0".encode()
        ).hexdigest()
        self.assertEqual(row.episode_attempt_id, expected_attempt_id)
        expected_pair_id = "pair-" + hashlib.sha256(
            f"{row.campaign_id}|{row.case_id}|{row.simulator_seed}".encode()
        ).hexdigest()
        self.assertEqual(row.pair_id, expected_pair_id)

    # -- Requirement 4: comparator contract pairs every row exactly once ---

    def test_comparator_contract_pairs_every_row_to_exactly_one_frozen_row(self) -> None:
        from dilu.runtime.grounded_decoding_schedule import pair_v8_row

        schedule = self._schedule()
        contract = self._contract()

        for row in schedule.stage1:
            paired = pair_v8_row(contract, row)
            self.assertEqual(paired.model_tag, row.model_tag)
            self.assertEqual(paired.case_id, row.case_id)
            self.assertEqual(paired.simulator_seed, row.simulator_seed)
            self.assertEqual(
                paired.condition_id, {"c120": "c110", "c121": "c111"}[row.condition_id]
            )

        for row in schedule.stage2_additional:
            self.assertEqual(row.condition_id, "c121")
            paired = pair_v8_row(contract, row)
            self.assertEqual(paired.condition_id, "c111")
            self.assertEqual(paired.campaign_id, self.v5_manifest.campaign_id)
            self.assertIn(row.model_tag, {"qwen3:0.6b", "llama3.2:1b"})

    def test_validate_comparator_pairing_succeeds_over_the_full_schedule(self) -> None:
        from dilu.runtime.grounded_decoding_schedule import validate_comparator_pairing

        schedule = self._schedule()
        contract = self._contract()
        validate_comparator_pairing(schedule, contract)  # must not raise

    # -- Requirement 5: fail-closed typed errors ----------------------------

    def test_missing_comparator_row_fails_closed(self) -> None:
        from dilu.runtime.grounded_decoding_schedule import MissingComparatorRowError

        contract = self._contract()
        with self.assertRaises(MissingComparatorRowError):
            contract.resolve(
                model_tag="qwen3:0.6b",
                model_digest=FROZEN_DIGESTS["qwen_06b"],
                condition_id="c110",
                case_id="no_such_case",
                simulator_seed=1,
            )

    def test_comparator_digest_mismatch_fails_closed(self) -> None:
        from dilu.runtime.grounded_decoding_schedule import ComparatorDigestMismatchError

        contract = self._contract()
        with self.assertRaises(ComparatorDigestMismatchError):
            contract.resolve(
                model_tag="qwen3:0.6b",
                model_digest="sha256:" + "0" * 64,
                condition_id="c110",
                case_id="bottleneck_merge_pressure_005",
                simulator_seed=38005,
            )

    def test_case_set_fingerprint_drift_fails_closed(self) -> None:
        from dilu.runtime.grounded_decoding_schedule import (
            CaseSetFingerprintDriftError,
            ComparatorPaths,
            build_comparator_contract,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake_v5.jsonl"
            row = self._fake_row(benchmark_fingerprint="sha256:" + "1" * 64)
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            drifted_manifest = replace(
                self.manifest,
                comparators=ComparatorPaths(
                    {
                        "v5_manifest": str(path),
                        "v5_episodes": str(path),
                        "v7_manifest": str(path),
                        "v7_episodes": str(path),
                    }
                ),
            )
            with self.assertRaises(CaseSetFingerprintDriftError):
                build_comparator_contract(
                    drifted_manifest, self.cases, self.v5_manifest, self.v7_manifest
                )

    def test_scoring_version_drift_fails_closed(self) -> None:
        from dilu.runtime.grounded_decoding_schedule import (
            ComparatorPaths,
            ScoringVersionDriftError,
            build_comparator_contract,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake_v5.jsonl"
            row = self._fake_row(split_scoring_policy_version="dilu_split_score_v9.9")
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            drifted_manifest = replace(
                self.manifest,
                comparators=ComparatorPaths(
                    {
                        "v5_manifest": str(path),
                        "v5_episodes": str(path),
                        "v7_manifest": str(path),
                        "v7_episodes": str(path),
                    }
                ),
            )
            with self.assertRaises(ScoringVersionDriftError):
                build_comparator_contract(
                    drifted_manifest, self.cases, self.v5_manifest, self.v7_manifest
                )

    def _fake_row(self, **overrides) -> dict:
        row = {
            "campaign_id": self.v5_manifest.campaign_id,
            "model_tag": "qwen3:0.6b",
            "model_digest": FROZEN_DIGESTS["qwen_06b"],
            "condition_id": "c110",
            "case_id": "bottleneck_merge_pressure_005",
            "simulator_seed": 38005,
            "benchmark_fingerprint": (
                "sha256:bd6d65d694a1452e0770e9854e478bb463be8302168e8c17396e86786401fd33"
            ),
            "split_scoring_policy_version": "dilu_split_score_v1.2",
            "balanced_driving_score_policy_version": "dilu_balanced_driving_score_v1",
        }
        row.update(overrides)
        return row

    # -- Requirement 6: registered contingency ------------------------------

    def test_rerun_comparators_for_adds_within_v8_o1_rows(self) -> None:
        schedule = self._schedule(rerun_comparators_for=frozenset({"qwen_06b"}))
        rerun = schedule.rerun_rows
        self.assertEqual(len(rerun), 150)  # 30 (c110) + 30 (c111) + 90 (c111 endpoint)

        stage1_shaped = [row for row in rerun if row.stage == "comparator_rerun_stage1"]
        stage2_shaped = [row for row in rerun if row.stage == "comparator_rerun_stage2"]
        self.assertEqual(len(stage1_shaped), 60)
        self.assertEqual(len(stage2_shaped), 90)
        self.assertEqual({row.condition_id for row in stage1_shaped}, {"c110", "c111"})
        self.assertEqual({row.condition_id for row in stage2_shaped}, {"c111"})
        self.assertTrue(all(row.model_slot == "qwen_06b" for row in rerun))
        self.assertTrue(
            all(row.campaign_id == "iclr2027-grounded-decoding-v8" for row in rerun)
        )

        base_schedule = self._schedule()
        stage1_case_ids = {row.case_id for row in base_schedule.stage1}
        self.assertEqual(
            {row.case_id for row in stage1_shaped if row.condition_id == "c110"},
            stage1_case_ids,
        )
        remaining_case_ids = {row.case_id for row in base_schedule.stage2_additional}
        self.assertEqual({row.case_id for row in stage2_shaped}, remaining_case_ids)

        self.assertEqual(len(schedule.all_claim_bearing), 480 + 150)

    def test_rerun_comparators_for_skips_cross_campaign_pairing_for_that_model(self) -> None:
        from dilu.runtime.grounded_decoding_schedule import (
            ComparatorDigestMismatchError,
            build_v8_schedule,
            validate_comparator_pairing,
        )

        # Drift qwen_06b's live digest so cross-campaign pairing would fail...
        drifted_bindings = dict(self.bindings)
        drifted_bindings["qwen_06b"] = replace(
            self.bindings["qwen_06b"], model_digest="sha256:" + "9" * 64
        )
        drifted_schedule = build_v8_schedule(
            self.manifest,
            self.cases,
            drifted_bindings,
            runtime_snapshot=self.snapshot,
        )
        contract = self._contract()
        # ... without the contingency, pairing fails closed.
        with self.assertRaises(ComparatorDigestMismatchError):
            validate_comparator_pairing(drifted_schedule, contract)

        # With the contingency registered, qwen_06b's rows are skipped, so
        # pairing succeeds even though its live digest no longer matches the
        # frozen comparator.
        rerun_schedule = build_v8_schedule(
            self.manifest,
            self.cases,
            drifted_bindings,
            runtime_snapshot=self.snapshot,
            rerun_comparators_for=frozenset({"qwen_06b"}),
        )
        validate_comparator_pairing(
            rerun_schedule, contract, rerun_comparators_for=frozenset({"qwen_06b"})
        )  # must not raise

    def test_rerun_comparators_for_rejects_unknown_model_slot(self) -> None:
        with self.assertRaises(ValueError):
            self._schedule(rerun_comparators_for=frozenset({"not_a_model"}))


if __name__ == "__main__":
    unittest.main()
