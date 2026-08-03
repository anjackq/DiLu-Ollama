from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dilu.runtime.ollama_transport import OllamaModelIdentity

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"


class MinimalFactorialScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

        self.manifest = load_experiment_manifest(MANIFEST)
        self.cases = json.loads((ROOT / self.manifest.case_path).read_text())
        self.digests = {
            "qwen_06b": "sha256:" + "a" * 64,
            "llama_1b": "sha256:" + "b" * 64,
        }

    @staticmethod
    def _git(command, **_kwargs):
        action = command[1]
        stdout = (
            "dfe6c9a97ea6ef6cd9edd845b21395fb7d7cc003\n"
            if action == "rev-parse"
            else command[-1] + "\n"
            if action == "ls-files"
            else ""
        )
        return type("Completed", (), {"stdout": stdout, "returncode": 0})()

    def _snapshot(self):
        from dilu.runtime.minimal_factorial_schedule import build_runtime_snapshot

        with patch(
            "dilu.runtime._minimal_factorial_manifest.subprocess.run", self._git
        ):
            return build_runtime_snapshot(self.manifest, self.cases)

    def test_selection_and_condition_factorial_are_deterministic(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_harness_config,
            select_smoke_case,
            select_stage1_cases,
        )

        self.assertEqual(
            [build_harness_config(self.manifest, i).condition_id() for i in range(8)],
            [f"c{i:03b}" for i in range(8)],
        )
        smoke = select_smoke_case(self.cases, self.manifest.selection.smoke_hash_prefix)
        expected = min(
            self.cases["cases"],
            key=lambda case: hashlib.sha256(
                f"{self.manifest.selection.smoke_hash_prefix}|{case['case_id']}".encode()
            ).hexdigest(),
        )
        self.assertEqual(smoke["case_id"], expected["case_id"])
        self.assertEqual(
            len(
                select_stage1_cases(
                    self.cases, self.manifest.selection.stage1_hash_prefix
                )
            ),
            30,
        )
        selected = select_stage1_cases(
            self.cases, self.manifest.selection.stage1_hash_prefix
        )
        for category in {case["category"] for case in self.cases["cases"]}:
            self.assertEqual(
                sum(case["category"] == category for case in selected),
                3,
            )

    def test_union_and_smoke_have_exact_rows_and_identities(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_smoke_schedule,
            build_union_schedule,
        )

        snapshot = self._snapshot()
        union = build_union_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        smoke = build_smoke_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        self.assertEqual(
            (
                sum(row.stage == "stage1" for row in union),
                sum(row.stage == "stage2_additional" for row in union),
            ),
            (480, 360),
        )
        self.assertEqual(len(union), 840)
        self.assertEqual(len(smoke), 16)
        self.assertEqual(
            len({row.episode_attempt_id for row in union}),
            840,
        )
        self.assertEqual(
            {row.condition_id for row in smoke}, {f"c{i:03b}" for i in range(8)}
        )
        stage1_cases = {row.case_id for row in union if row.stage == "stage1"}
        stage2_rows = [row for row in union if row.stage == "stage2_additional"]
        self.assertEqual(len(stage1_cases), 30)
        self.assertEqual({row.condition_id for row in stage2_rows}, {"c000", "c111"})
        self.assertEqual(len({row.case_id for row in stage2_rows}), 90)
        self.assertFalse(stage1_cases & {row.case_id for row in stage2_rows})
        for slot in self.digests:
            for endpoint in ("c000", "c111"):
                endpoint_rows = [
                    row
                    for row in union
                    if row.model_slot == slot and row.condition_id == endpoint
                ]
                self.assertEqual(len(endpoint_rows), 120)
                self.assertEqual(
                    sum(row.stage == "stage1" for row in endpoint_rows),
                    30,
                )
        row = smoke[0]
        identity = OllamaModelIdentity(row.model_tag, row.model_digest)
        self.assertEqual(row.identity().campaign_id, self.manifest.smoke_campaign_id)
        self.assertEqual(row.model_digest, identity.model_digest)
        self.assertEqual(
            row.pair_id,
            "pair-"
            + hashlib.sha256(
                f"{row.campaign_id}|{row.case_id}|{row.simulator_seed}".encode()
            ).hexdigest(),
        )

    def test_v2_preserves_v1_case_selection_but_versions_evidence_ids(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_smoke_schedule,
            build_union_schedule,
        )

        snapshot = self._snapshot()
        v1_manifest = replace(
            self.manifest,
            campaign_id="iclr2027-minimal-factorial-v1",
            smoke_campaign_id="iclr2027-minimal-factorial-smoke-v1",
        )
        v1_smoke = build_smoke_schedule(
            v1_manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        v2_smoke = build_smoke_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        v1_union = build_union_schedule(
            v1_manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        v2_union = build_union_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )

        self.assertEqual(
            {row.case_id for row in v1_smoke},
            {row.case_id for row in v2_smoke},
        )
        self.assertEqual(
            {row.case_id for row in v1_union if row.stage == "stage1"},
            {row.case_id for row in v2_union if row.stage == "stage1"},
        )
        self.assertEqual(
            [(row.stage, row.case_id) for row in v1_union],
            [(row.stage, row.case_id) for row in v2_union],
        )
        self.assertTrue(
            {row.pair_id for row in v1_union}.isdisjoint(
                {row.pair_id for row in v2_union}
            )
        )
        self.assertTrue(
            {row.episode_attempt_id for row in v1_union}.isdisjoint(
                {row.episode_attempt_id for row in v2_union}
            )
        )
        self.assertNotEqual(v1_smoke[0].pair_id, v2_smoke[0].pair_id)
        self.assertNotEqual(
            v1_smoke[0].episode_attempt_id,
            v2_smoke[0].episode_attempt_id,
        )
        self.assertEqual(v2_smoke[0].case_id, "mandatory_overtake_slow_lead_002")


if __name__ == "__main__":
    unittest.main()
