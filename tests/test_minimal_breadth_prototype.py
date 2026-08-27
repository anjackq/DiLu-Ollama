from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime._minimal_factorial_manifest import ModelSpec, RuntimeSnapshot
from dilu.runtime._minimal_factorial_schedule_support import OutputSpec, SelectionSpec

ROOT = Path(__file__).resolve().parents[1]
V5_MANIFEST = ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"
V6_MANIFEST = ROOT / "configs" / "iclr2027" / "minimal_breadth_oe_v6.yaml"


class MinimalBreadthPrototypeTests(unittest.TestCase):
    def test_v6_manifest_registers_three_models_and_four_conditions(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

        manifest = load_experiment_manifest(V6_MANIFEST)

        self.assertEqual(
            [(model.slot, model.tag) for model in manifest.models],
            [
                ("llama_3b", "llama3.2:3b"),
                ("gemma_4b", "gemma3:4b"),
                ("qwen_8b", "qwen3:8b"),
            ],
        )
        self.assertEqual(tuple(manifest.selection.condition_indexes), (4, 5, 6, 7))
        self.assertFalse(manifest.selection.include_stage2)

    def test_v6_builds_exact_stage1_only_schedule(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_smoke_schedule,
            build_union_schedule,
            load_experiment_manifest,
        )

        manifest = load_experiment_manifest(V5_MANIFEST)
        outputs = manifest.outputs.to_dict()
        outputs["root"] = "results/iclr2027_minimal_breadth_oe_v6"
        selection = manifest.selection.to_dict()
        selection.update(condition_indexes=[4, 5, 6, 7], include_stage2=False)
        prototype = replace(
            manifest,
            campaign_id="iclr2027-minimal-breadth-oe-v6",
            smoke_campaign_id="iclr2027-minimal-breadth-oe-smoke-v6",
            models=(
                ModelSpec("llama_3b", "llama3.2:3b"),
                ModelSpec("gemma_4b", "gemma3:4b"),
                ModelSpec("qwen_8b", "qwen3:8b"),
            ),
            selection=SelectionSpec(selection),
            outputs=OutputSpec(outputs),
        )
        cases = json.loads((ROOT / prototype.case_path).read_text(encoding="utf-8"))
        snapshot = RuntimeSnapshot.create(
            {
                "case_set_fingerprint": (
                    "sha256:bd6d65d694a1452e0770e9854e478bb463be8302168e8c17396e86786401fd33"
                ),
                "code_revision": "a" * 40,
            }
        )
        digests = {
            "llama_3b": "sha256:" + "a" * 64,
            "gemma_4b": "sha256:" + "b" * 64,
            "qwen_8b": "sha256:" + "c" * 64,
        }

        claim = build_union_schedule(
            prototype, cases, digests, runtime_snapshot=snapshot
        )
        smoke = build_smoke_schedule(
            prototype, cases, digests, runtime_snapshot=snapshot
        )

        self.assertEqual(len(claim), 360)
        self.assertEqual(len(smoke), 12)
        self.assertEqual({row.stage for row in claim}, {"stage1"})
        self.assertEqual(
            {row.condition_id for row in claim}, {"c100", "c101", "c110", "c111"}
        )
        self.assertEqual(len({row.case_id for row in claim}), 30)
        self.assertEqual(len({row.episode_attempt_id for row in claim}), 360)

    def test_probe_lock_uses_requested_manifest(self) -> None:
        from dilu.runtime import minimal_factorial_runner as runner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "configs" / "iclr2027" / "breadth.yaml"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("placeholder", encoding="utf-8")
            authored = SimpleNamespace(preflight_path=root / "preflight.json")
            with mock.patch.object(
                runner, "author_verified_runtime_locks", return_value=authored
            ) as author:
                result = runner.run_probe_lock(manifest_path)

        self.assertEqual(result, authored.preflight_path)
        author.assert_called_once_with(root, manifest_path=manifest_path.resolve())


if __name__ == "__main__":
    unittest.main()
