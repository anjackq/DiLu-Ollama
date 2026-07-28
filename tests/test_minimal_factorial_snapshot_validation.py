from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from dilu.runtime import _minimal_factorial_manifest as manifest_module

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"


class MinimalFactorialSnapshotValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

        self.manifest = load_experiment_manifest(MANIFEST)
        self.cases = json.loads((ROOT / self.manifest.case_path).read_text())

    @staticmethod
    def _completed(stdout: str = "", returncode: int = 0):
        return type(
            "Completed",
            (),
            {"stdout": stdout, "returncode": returncode},
        )()

    @classmethod
    def _clean_git(cls, command, **_kwargs):
        action = command[1]
        stdout = (
            "46367dae59ddf349bacf39b36efb1bf464befcb5\n"
            if action == "rev-parse"
            else command[-1] + "\n"
            if action == "ls-files"
            else ""
        )
        return cls._completed(stdout)

    def _build(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import build_runtime_snapshot

        build_runtime_snapshot(self.manifest, self.cases)

    def test_rejects_staged_and_untracked_worktree_state(self) -> None:
        for label, status in (
            ("staged", "M  config.example.yaml\n"),
            ("untracked", "?? unexpected.txt\n"),
        ):
            with self.subTest(label=label):

                def dirty_git(command, **kwargs):
                    if command[1] == "status":
                        return self._completed(status)
                    return self._clean_git(command, **kwargs)

                with patch(
                    "dilu.runtime._minimal_factorial_manifest.subprocess.run",
                    dirty_git,
                ):
                    with self.assertRaises(ValueError):
                        self._build()

    def test_rejects_untracked_runtime_source_and_git_command_failure(self) -> None:
        def untracked_source(command, **kwargs):
            if (
                command[1] == "ls-files"
                and command[-1] == self.manifest.runtime_sources.runtime_config
            ):
                return self._completed(returncode=1)
            return self._clean_git(command, **kwargs)

        with patch(
            "dilu.runtime._minimal_factorial_manifest.subprocess.run",
            untracked_source,
        ):
            with self.assertRaises(ValueError):
                self._build()

        def failed_status(command, **kwargs):
            if command[1] == "status":
                return self._completed(returncode=128)
            return self._clean_git(command, **kwargs)

        with patch(
            "dilu.runtime._minimal_factorial_manifest.subprocess.run",
            failed_status,
        ):
            with self.assertRaises(ValueError):
                self._build()

    def test_rejects_source_scoring_and_predicate_fingerprint_drift(self) -> None:
        original_file_sha = manifest_module._file_sha
        original_canonical = manifest_module.canonical_sha256

        for label, drifted_name in (
            ("source", "minimal_factorial_runtime.yaml"),
            ("scoring", "dilu_scoring.py"),
        ):
            with self.subTest(label=label):

                def drift_file(path, *, target=drifted_name):
                    return "0" * 64 if path.name == target else original_file_sha(path)

                with (
                    patch(
                        "dilu.runtime._minimal_factorial_manifest.subprocess.run",
                        self._clean_git,
                    ),
                    patch(
                        "dilu.runtime._minimal_factorial_manifest._file_sha",
                        drift_file,
                    ),
                ):
                    with self.assertRaises(ValueError):
                        self._build()

        def drift_predicate(value):
            if (
                isinstance(value, list)
                and len(value) == 120
                and all(isinstance(item, dict) for item in value)
            ):
                return "0" * 64
            return original_canonical(value)

        with (
            patch(
                "dilu.runtime._minimal_factorial_manifest.subprocess.run",
                self._clean_git,
            ),
            patch(
                "dilu.runtime._minimal_factorial_manifest.canonical_sha256",
                drift_predicate,
            ),
        ):
            with self.assertRaises(ValueError):
                self._build()

    def test_rejects_case_fingerprint_drift(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import build_runtime_snapshot

        drifted = copy.deepcopy(self.cases)
        drifted["cases"][0]["seed"] += 1
        with self.assertRaises(ValueError):
            build_runtime_snapshot(self.manifest, drifted)


if __name__ == "__main__":
    unittest.main()
