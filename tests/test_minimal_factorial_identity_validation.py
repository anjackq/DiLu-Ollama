from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class MinimalFactorialIdentityValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

        self.manifest = load_experiment_manifest(
            ROOT / "configs/iclr2027/minimal_factorial.yaml"
        )
        self.cases = json.loads((ROOT / self.manifest.case_path).read_text())
        self.digests = {
            "qwen_06b": "sha256:" + "a" * 64,
            "llama_1b": "sha256:" + "b" * 64,
        }

    @staticmethod
    def _git(command, **_kwargs):
        action = command[1]
        stdout = (
            "1e80682a758fef5da59e848a616cb1df1521f0f6\n"
            if action == "rev-parse"
            else command[-1] + "\n"
            if action == "ls-files"
            else ""
        )
        return type("Completed", (), {"stdout": stdout, "returncode": 0})()

    def test_every_identity_field_tamper_is_rejected_before_publish(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_harness_config,
            build_runtime_snapshot,
            build_union_schedule,
            write_frozen_campaign_manifest,
        )

        with patch(
            "dilu.runtime._minimal_factorial_manifest.subprocess.run", self._git
        ):
            snapshot = build_runtime_snapshot(self.manifest, self.cases)
        schedule = build_union_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        row = schedule[0]
        changes = {
            "campaign_id": self.manifest.smoke_campaign_id,
            "model_slot": "missing_slot",
            "model_tag": "wrong:model",
            "model_digest": "not-a-digest",
            "condition_id": "c111",
            "condition": build_harness_config(self.manifest, 7),
            "replicate_id": 1,
            "pair_id": "pair-" + "0" * 64,
            "template_id": "stress-v2-" + "0" * 64,
            "primary_snapshot_id": "snapshot-" + "0" * 64,
            "episode_attempt_id": "episode-" + "0" * 64,
            "code_revision": "0" * 40,
            "benchmark_fingerprint": "sha256:" + "0" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            for field, value in changes.items():
                mutated = (replace(row, **{field: value}), *schedule[1:])
                with self.subTest(field=field), self.assertRaises(ValueError):
                    write_frozen_campaign_manifest(
                        Path(directory) / f"{field}.json",
                        self.manifest,
                        snapshot,
                        mutated,
                        case_set=self.cases,
                    )


if __name__ == "__main__":
    unittest.main()
