from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dilu.runtime.ollama_transport import OllamaModelIdentity

ROOT = Path(__file__).resolve().parents[1]


class MinimalFactorialProvenanceTests(unittest.TestCase):
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
        self.bindings = {
            model.slot: OllamaModelIdentity(model.tag, self.digests[model.slot])
            for model in self.manifest.models
        }

    @staticmethod
    def _git(command, **_kwargs):
        action = command[1]
        stdout = (
            "2d7d8492dc260832e68348933c9928616a6edccb\n"
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

    def test_schedule_rejects_tampered_snapshot_hash_and_nonhex_revision(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            RuntimeSnapshot,
            build_union_schedule,
        )

        snapshot = self._snapshot()
        object.__setattr__(snapshot, "sha256", "0" * 64)
        with self.assertRaises(ValueError):
            build_union_schedule(
                self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
            )
        snapshot = RuntimeSnapshot.create(
            {**self._snapshot().payload, "code_revision": "x" * 40}
        )
        with self.assertRaises(ValueError):
            build_union_schedule(
                self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
            )

    def test_writer_rejects_invalid_digest_before_publish(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_union_schedule,
            write_frozen_campaign_manifest,
        )

        snapshot = self._snapshot()
        schedule = build_union_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        invalid = (replace(schedule[0], model_digest="not-a-digest"), *schedule[1:])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_frozen_campaign_manifest(
                    Path(directory) / "frozen.json",
                    self.manifest,
                    snapshot,
                    invalid,
                    case_set=self.cases,
                    model_bindings=self.bindings,
                )

    def test_barrier_writers_preserve_first_distinct_artifact(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_union_schedule,
            write_frozen_campaign_manifest,
        )

        snapshot = self._snapshot()
        schedule = build_union_schedule(
            self.manifest, self.cases, self.digests, runtime_snapshot=snapshot
        )
        barrier = threading.Barrier(2)
        outcomes = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"

            def write(rows):
                barrier.wait()
                try:
                    write_frozen_campaign_manifest(
                        path,
                        self.manifest,
                        snapshot,
                        rows,
                        case_set=self.cases,
                        model_bindings=self.bindings,
                    )
                    outcomes.append("ok")
                except ValueError:
                    outcomes.append("different")

            first = threading.Thread(target=write, args=(schedule,))
            second = threading.Thread(target=write, args=(schedule[:-1],))
            first.start()
            second.start()
            first.join()
            second.join()
        self.assertEqual(sorted(outcomes), ["different", "ok"])


if __name__ == "__main__":
    unittest.main()
