from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.test_runtime_lock_authoring import NativeFakes, run_authoring


def rename_case_only(path: Path, new_name: str) -> Path:
    intermediate = path.with_name("case-rename-intermediate")
    path.rename(intermediate)
    renamed = path.with_name(new_name)
    intermediate.rename(renamed)
    return renamed


class RuntimeLockAuthoringFilesystemTests(unittest.TestCase):
    def test_destination_root_physical_casing_must_match_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            physical = sandbox / "RESULTS"
            physical.mkdir()
            requested = sandbox / "results"

            with self.assertRaisesRegex(ValueError, "casing"):
                run_authoring(requested, NativeFakes())

    def test_intermediate_output_directory_casing_must_match(self) -> None:
        for requested_name, physical_name in (
            ("s1", "S1"),
            ("smoke", "SMOKE"),
            ("llm_campaign", "LLM_CAMPAIGN"),
        ):
            with (
                self.subTest(requested_name=requested_name),
                tempfile.TemporaryDirectory() as tmp,
            ):
                output = Path(tmp) / "results"
                output.mkdir()
                output.joinpath(physical_name).mkdir()

                with self.assertRaisesRegex(ValueError, "casing"):
                    run_authoring(output, NativeFakes())

    def test_lock_root_casing_drift_blocks_exact_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            run_authoring(output, NativeFakes())
            rename_case_only(output / "s1" / "locks", "LOCKS")

            with self.assertRaisesRegex(ValueError, "casing"):
                run_authoring(output, NativeFakes())

    def test_artifact_filename_casing_drift_blocks_exact_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            run_authoring(output, NativeFakes())
            manifest = output / "smoke" / "campaign_manifest.json"
            rename_case_only(manifest, "CAMPAIGN_MANIFEST.JSON")

            with self.assertRaisesRegex(ValueError, "casing"):
                run_authoring(output, NativeFakes())


if __name__ == "__main__":
    unittest.main()
