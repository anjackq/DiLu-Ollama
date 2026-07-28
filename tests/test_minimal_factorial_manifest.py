from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class MinimalFactorialManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

        self.manifest = load_experiment_manifest(
            ROOT / "configs/iclr2027/minimal_factorial.yaml"
        )
        self.cases = json.loads((ROOT / self.manifest.case_path).read_text())

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
            TransportSpec,
        )
        from dilu.runtime.minimal_factorial_schedule import RuntimeSnapshot

        self.assertIsInstance(self.manifest.transport, TransportSpec)
        with self.assertRaises(TypeError):
            FrozenSpec({"nested": {"value": 1}}).nested["value"] = 2
        with self.assertRaises(AttributeError):
            _ = FrozenSpec({}).missing
        with self.assertRaises(TypeError):
            RuntimeSnapshot(MappingProxyType({}), "x")
        with self.assertRaises(TypeError):
            self._snapshot().payload["code_revision"] = "drift"

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
        digests = {"qwen_06b": "sha256:" + "a" * 64, "llama_1b": "sha256:" + "b" * 64}
        schedule = build_union_schedule(
            self.manifest, self.cases, digests, runtime_snapshot=snapshot
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_frozen_campaign_manifest(path, self.manifest, snapshot, schedule)
            content = path.read_bytes()
            write_frozen_campaign_manifest(path, self.manifest, snapshot, schedule)
            self.assertEqual(path.read_bytes(), content)
            with self.assertRaises(ValueError):
                write_frozen_campaign_manifest(
                    path, self.manifest, snapshot, schedule[:-1]
                )

    def test_racing_writer_keeps_first_artifact(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_union_schedule,
            write_frozen_campaign_manifest,
        )

        snapshot = self._snapshot()
        digests = {"qwen_06b": "sha256:" + "a" * 64, "llama_1b": "sha256:" + "b" * 64}
        schedule = build_union_schedule(
            self.manifest, self.cases, digests, runtime_snapshot=snapshot
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_frozen_campaign_manifest(path, self.manifest, snapshot, schedule)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                write_frozen_campaign_manifest(
                    path, self.manifest, snapshot, schedule[:-1]
                )
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
