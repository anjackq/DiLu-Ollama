from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime import runtime_lock_authoring
from tests.test_runtime_lock_authoring import NativeFakes, run_authoring

PUBLICATION_BOUNDARIES = 36


def create_directory_redirect(link: Path, target: Path) -> None:
    if os.name != "nt":
        link.symlink_to(target, target_is_directory=True)
        return
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not create test junction: {result.stderr}")


class RuntimeLockAuthoringTransactionTests(unittest.TestCase):
    def test_fresh_authoring_creates_missing_destination_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "missing-parent" / "results"

            result = run_authoring(output, NativeFakes())

            self.assertEqual(len(result.lock_artifacts), 16)
            self.assertTrue(output.is_dir())

    def test_completed_exact_rerun_loads_without_native_posts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            fresh_fakes = NativeFakes()
            first = run_authoring(output, fresh_fakes)
            rerun_fakes = NativeFakes()

            second = run_authoring(output, rerun_fakes)

            self.assertEqual(len(fresh_fakes.post_calls), 6)
            self.assertEqual(rerun_fakes.post_calls, [])
            self.assertEqual(first.preflight_sha256, second.preflight_sha256)
            self.assertEqual(
                [item.verified_binding.to_dict() for item in first.lock_artifacts],
                [item.verified_binding.to_dict() for item in second.lock_artifacts],
            )

    def test_preexisting_invalid_or_different_destination_rejects_without_posts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            output.mkdir()
            marker = output / "unexpected.txt"
            marker.write_text("preserve", encoding="utf-8")
            fakes = NativeFakes()

            with self.assertRaises(ValueError):
                run_authoring(output, fakes)
            self.assertEqual(fakes.post_calls, [])
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(list(output.iterdir()), [marker])

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            run_authoring(output, NativeFakes())
            preflight = output / "s1" / "model_preflight.json"
            preflight.write_bytes(b"{}")
            fakes = NativeFakes()

            with self.assertRaises(ValueError):
                run_authoring(output, fakes)
            self.assertEqual(fakes.post_calls, [])
            self.assertEqual(preflight.read_bytes(), b"{}")

    def test_destination_swap_after_validation_cannot_escape_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            output = sandbox / "requested" / "results"
            output.parent.mkdir()
            external = sandbox / "external"
            external.mkdir()
            original_validate = (
                runtime_lock_authoring.validate_unredirected_artifact_paths
            )
            swapped = False

            def validate_then_swap(paths: tuple[Path, ...]) -> None:
                nonlocal swapped
                original_validate(paths)
                if not swapped:
                    create_directory_redirect(output, external)
                    swapped = True

            with (
                mock.patch.object(
                    runtime_lock_authoring,
                    "validate_unredirected_artifact_paths",
                    side_effect=validate_then_swap,
                ),
                self.assertRaises((OSError, ValueError)),
            ):
                run_authoring(output, NativeFakes())

            self.assertEqual(list(external.iterdir()), [])

    def test_parent_swap_during_probe_cannot_write_or_clean_external(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            parent = sandbox / "requested"
            parent.mkdir()
            moved_parent = sandbox / "requested-original"
            output = parent / "results"
            external = sandbox / "external"
            external.mkdir()
            fakes = NativeFakes()
            original_get = fakes.get
            swapped = False
            external_write_seen = False

            def get_then_swap(
                url: str,
                *,
                timeout: float,
                allow_redirects: bool,
            ) -> object:
                nonlocal swapped
                response = original_get(
                    url,
                    timeout=timeout,
                    allow_redirects=allow_redirects,
                )
                if not swapped:
                    parent.rename(moved_parent)
                    create_directory_redirect(parent, external)
                    swapped = True
                return response

            def observe_boundary(index: int, _path: Path) -> None:
                nonlocal external_write_seen
                if index == 1:
                    external_write_seen = any(external.rglob("*"))

            fakes.get = get_then_swap  # type: ignore[method-assign]
            try:
                with self.assertRaises((OSError, ValueError)):
                    run_authoring(
                        output,
                        fakes,
                        publication_boundary_hook=observe_boundary,
                    )
                self.assertFalse(external_write_seen)
                self.assertEqual(list(external.iterdir()), [])
            finally:
                if os.path.lexists(parent):
                    os.rmdir(parent)
                if moved_parent.exists():
                    moved_parent.rename(parent)

    def test_failure_at_every_publication_boundary_leaves_no_destination(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            for fail_at in range(PUBLICATION_BOUNDARIES):
                output = sandbox / f"results-{fail_at:02d}"
                observed: list[int] = []

                def fail_boundary(
                    index: int,
                    _path: Path,
                    *,
                    observed_indices: list[int] = observed,
                    failure_index: int = fail_at,
                ) -> None:
                    observed_indices.append(index)
                    if index == failure_index:
                        raise RuntimeError(f"injected publication failure {index}")

                with self.assertRaisesRegex(
                    RuntimeError,
                    f"injected publication failure {fail_at}",
                ):
                    run_authoring(
                        output,
                        NativeFakes(),
                        publication_boundary_hook=fail_boundary,
                    )
                self.assertFalse(output.exists())
                self.assertEqual(observed, list(range(fail_at + 1)))

            clean_output = sandbox / "clean-rerun"
            result = run_authoring(clean_output, NativeFakes())
            self.assertEqual(len(result.lock_artifacts), 16)


if __name__ == "__main__":
    unittest.main()
