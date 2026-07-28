from __future__ import annotations

import hashlib
import json
import unittest
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
        smoke = select_smoke_case(self.cases, self.manifest.campaign_id)
        expected = min(
            self.cases["cases"],
            key=lambda case: hashlib.sha256(
                f"{self.manifest.campaign_id}|smoke|{case['case_id']}".encode()
            ).hexdigest(),
        )
        self.assertEqual(smoke["case_id"], expected["case_id"])
        self.assertEqual(
            len(select_stage1_cases(self.cases, self.manifest.campaign_id)), 30
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
                sum(row.stage == "s1" for row in union),
                sum(row.stage == "s2_additional" for row in union),
            ),
            (480, 360),
        )
        self.assertEqual(len(union), 840)
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


if __name__ == "__main__":
    unittest.main()
