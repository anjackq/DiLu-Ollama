from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
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

    @staticmethod
    def _selection_sha(rows) -> str:
        selection = sorted(
            {(row.stage, row.case_id, row.simulator_seed) for row in rows}
        )
        payload = json.dumps(selection, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _request_id(row: Any, decision_index: int = 0) -> str:
        payload = "|".join(
            (
                row.campaign_id,
                row.episode_attempt_id,
                row.case_id,
                str(decision_index),
            )
        ).encode("utf-8")
        return "req-" + hashlib.sha256(payload).hexdigest()

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

    def test_v4_preserves_v3_scientific_schedule_but_versions_evidence_ids(
        self,
    ) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_smoke_schedule,
            build_union_schedule,
        )
        from dilu.runtime._minimal_factorial_schedule_support import OutputSpec
        from tests.runtime_factorization_support import runtime

        snapshot = self._snapshot()
        v3_outputs = self.manifest.outputs.to_dict()
        v3_outputs["root"] = "results/iclr2027_minimal_factorial_v3"
        v3_manifest = replace(
            self.manifest,
            campaign_id="iclr2027-minimal-factorial-v3",
            smoke_campaign_id="iclr2027-minimal-factorial-smoke-v3",
            outputs=OutputSpec(v3_outputs),
        )
        v3_smoke = build_smoke_schedule(
            v3_manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        v4_smoke = build_smoke_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        v3_union = build_union_schedule(
            v3_manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        v4_union = build_union_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )

        self.assertEqual(
            [
                (
                    row.stage,
                    row.case_id,
                    row.simulator_seed,
                    row.condition.to_canonical_dict(),
                )
                for row in v3_smoke
            ],
            [
                (
                    row.stage,
                    row.case_id,
                    row.simulator_seed,
                    row.condition.to_canonical_dict(),
                )
                for row in v4_smoke
            ],
        )
        self.assertEqual(
            [
                (
                    row.stage,
                    row.case_id,
                    row.simulator_seed,
                    row.condition.to_canonical_dict(),
                )
                for row in v3_union
            ],
            [
                (
                    row.stage,
                    row.case_id,
                    row.simulator_seed,
                    row.condition.to_canonical_dict(),
                )
                for row in v4_union
            ],
        )
        self.assertEqual(self.manifest.transport.generation_seed_master, 20270728)
        self.assertEqual(
            self._selection_sha(v4_smoke),
            "ec5f202c2f05cee83d5df0527ab818d724aee4ed773491dd91fc28efaf018883",
        )
        self.assertEqual(
            self._selection_sha(v4_union),
            "237cbe106386cde5acfbe1531353a3e0b7afade59900a2b4827761ddfb6673b1",
        )
        for v3_rows, v4_rows in ((v3_smoke, v4_smoke), (v3_union, v4_union)):
            self.assertTrue(
                {row.pair_id for row in v3_rows}.isdisjoint(
                    {row.pair_id for row in v4_rows}
                )
            )
            self.assertTrue(
                {row.episode_attempt_id for row in v3_rows}.isdisjoint(
                    {row.episode_attempt_id for row in v4_rows}
                )
            )
            self.assertTrue(
                {self._request_id(row) for row in v3_rows}.isdisjoint(
                    {self._request_id(row) for row in v4_rows}
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_ids: dict[str, str] = {}
            for label, row in (
                ("v3-smoke", v3_smoke[0]),
                ("v3-union", v3_union[0]),
                ("v4-smoke", v4_smoke[0]),
                ("v4-union", v4_union[0]),
            ):
                scientific_runtime = runtime(
                    root / label,
                    config=row.condition,
                    episode_identity=row.identity(),
                )
                scientific_runtime.begin_attempt()
                request_ids[label] = (
                    scientific_runtime.generation_context(0).request_id
                )
                self.assertEqual(request_ids[label], self._request_id(row))
            self.assertTrue(
                {request_ids["v3-smoke"], request_ids["v3-union"]}.isdisjoint(
                    {request_ids["v4-smoke"], request_ids["v4-union"]}
                )
            )
        self.assertEqual(v4_smoke[0].case_id, "mandatory_overtake_slow_lead_002")


if __name__ == "__main__":
    unittest.main()
