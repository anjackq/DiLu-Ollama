from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import yaml

from dilu.runtime.ollama_transport import OllamaModelIdentity

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"


class MinimalFactorialManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

        self.manifest = load_experiment_manifest(MANIFEST)
        self.cases = json.loads((ROOT / self.manifest.case_path).read_text())
        self.digests = {
            "qwen_06b": "sha256:" + "a" * 64,
            "llama_1b": "sha256:" + "b" * 64,
        }
        self.bindings = {
            model.slot: OllamaModelIdentity(model.tag, self.digests[model.slot])
            for model in self.manifest.models
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

    def test_typed_specs_and_snapshot_are_deeply_immutable(self) -> None:
        from dilu.runtime._minimal_factorial_schedule_support import (
            FrozenSpec,
            RetrySpec,
            TransportSpec,
        )
        from dilu.runtime.minimal_factorial_schedule import RuntimeSnapshot

        self.assertIsInstance(self.manifest.transport, TransportSpec)
        self.assertIsInstance(self.manifest.fixed_harness.retry_policy, RetrySpec)
        with self.assertRaises(TypeError):
            FrozenSpec({"nested": {"value": 1}}).nested["value"] = 2
        with self.assertRaises(AttributeError):
            _ = FrozenSpec({}).missing
        with self.assertRaises(TypeError):
            RuntimeSnapshot(MappingProxyType({}), "x")
        with self.assertRaises(TypeError):
            self._snapshot().payload["code_revision"] = "drift"

    def test_exact_constants_and_output_roots_survive_frozen_round_trip(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_smoke_schedule,
            write_frozen_campaign_manifest,
        )

        self.assertEqual(
            {
                "schema_version": self.manifest.schema_version,
                "campaign_id": self.manifest.campaign_id,
                "smoke_campaign_id": self.manifest.smoke_campaign_id,
                "case_path": self.manifest.case_path,
                "models": tuple(
                    (model.slot, model.tag) for model in self.manifest.models
                ),
                "endpoint": self.manifest.transport.native_endpoint,
                "seed": self.manifest.transport.generation_seed_master,
                "categories": self.manifest.selection.categories,
                "stage1_per_category": (
                    self.manifest.selection.stage1_cases_per_category
                ),
                "stage2_per_category": (
                    self.manifest.selection.stage2_cases_per_category
                ),
                "smoke_hash_prefix": self.manifest.selection.smoke_hash_prefix,
                "stage1_hash_prefix": self.manifest.selection.stage1_hash_prefix,
            },
            {
                "schema_version": "iclr2027_minimal_factorial_manifest_v1",
                "campaign_id": "iclr2027-minimal-factorial-v5",
                "smoke_campaign_id": "iclr2027-minimal-factorial-smoke-v5",
                "case_path": ("benchmarks/dilu_highway_reactive_stress_v2/cases.json"),
                "models": (
                    ("qwen_06b", "qwen3:0.6b"),
                    ("llama_1b", "llama3.2:1b"),
                ),
                "endpoint": "http://localhost:11434/api/chat",
                "seed": 20270728,
                "categories": 10,
                "stage1_per_category": 3,
                "stage2_per_category": 12,
                "smoke_hash_prefix": "iclr2027-minimal-factorial-v1|smoke",
                "stage1_hash_prefix": "iclr2027-minimal-factorial-v1",
            },
        )
        self.assertEqual(
            self.manifest.bootstrap.to_dict(),
            {"draws": 20000, "version": "bootstrap-v1"},
        )
        outputs = {
            "root": "results/iclr2027_minimal_factorial_v5",
            "s1": "s1",
            "smoke": "smoke",
            "llm_campaign": "llm_campaign",
            "baselines": "baselines",
            "analysis": "analysis",
        }
        self.assertEqual(self.manifest.outputs.to_dict(), outputs)
        self.assertEqual(len(set(outputs.values())), len(outputs))
        snapshot = self._snapshot()
        schedule = build_smoke_schedule(
            self.manifest,
            self.cases,
            self.digests,
            runtime_snapshot=snapshot,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            write_frozen_campaign_manifest(
                path,
                self.manifest,
                snapshot,
                schedule,
                case_set=self.cases,
                model_bindings=self.bindings,
            )
            saved = json.loads(path.read_text())
        self.assertEqual(
            saved["manifest"]["bootstrap"], self.manifest.bootstrap.to_dict()
        )
        self.assertEqual(saved["manifest"]["outputs"], outputs)

    def test_v5_changes_only_the_three_registered_identity_fields(self) -> None:
        from dilu.runtime._minimal_factorial_schedule_support import canonical_sha256

        raw_manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            (
                raw_manifest["campaign_id"],
                raw_manifest["smoke_campaign_id"],
                raw_manifest["outputs"]["root"],
            ),
            (
                "iclr2027-minimal-factorial-v5",
                "iclr2027-minimal-factorial-smoke-v5",
                "results/iclr2027_minimal_factorial_v5",
            ),
        )
        v4_comparison = json.loads(json.dumps(raw_manifest))
        v4_comparison["campaign_id"] = "iclr2027-minimal-factorial-v4"
        v4_comparison["smoke_campaign_id"] = (
            "iclr2027-minimal-factorial-smoke-v4"
        )
        v4_comparison["outputs"]["root"] = (
            "results/iclr2027_minimal_factorial_v4"
        )
        self.assertEqual(
            canonical_sha256(v4_comparison),
            "a19f35bb4ad8c4cb245fd59046ae58628d8d2074ca4c0bfaafba6312d623f8b3",
        )

    def test_git_drift_and_atomic_write_once(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_runtime_snapshot,
            build_union_schedule,
            write_frozen_campaign_manifest,
        )

        failed = type("Completed", (), {"stdout": "", "returncode": 128})()
        with patch(
            "dilu.runtime._minimal_factorial_manifest.subprocess.run",
            return_value=failed,
        ):
            with self.assertRaises(ValueError):
                build_runtime_snapshot(self.manifest, self.cases)
        snapshot = self._snapshot()
        schedule = build_union_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_frozen_campaign_manifest(
                path,
                self.manifest,
                snapshot,
                schedule,
                case_set=self.cases,
                model_bindings=self.bindings,
            )
            content = path.read_bytes()
            write_frozen_campaign_manifest(
                path,
                self.manifest,
                snapshot,
                schedule,
                case_set=self.cases,
                model_bindings=self.bindings,
            )
            self.assertEqual(path.read_bytes(), content)
            with self.assertRaises(ValueError):
                write_frozen_campaign_manifest(
                    path,
                    self.manifest,
                    snapshot,
                    schedule[:-1],
                    case_set=self.cases,
                    model_bindings=self.bindings,
                )

    def test_racing_writer_keeps_first_artifact(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_union_schedule,
            write_frozen_campaign_manifest,
        )

        snapshot = self._snapshot()
        schedule = build_union_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_frozen_campaign_manifest(
                path,
                self.manifest,
                snapshot,
                schedule,
                case_set=self.cases,
                model_bindings=self.bindings,
            )
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                write_frozen_campaign_manifest(
                    path,
                    self.manifest,
                    snapshot,
                    schedule[:-1],
                    case_set=self.cases,
                    model_bindings=self.bindings,
                )
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
