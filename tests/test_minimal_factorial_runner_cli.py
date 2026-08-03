from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime import minimal_factorial_runner as runner


class MinimalFactorialStatusTests(unittest.TestCase):
    def test_status_counts_are_grouped_without_writing_artifacts(self) -> None:
        rows = (
            {
                "stage": "stage1",
                "model_slot": "qwen",
                "condition_id": "c000",
                "episode_attempt_id": "episode-1",
            },
            {
                "stage": "stage1",
                "model_slot": "qwen",
                "condition_id": "c000",
                "episode_attempt_id": "episode-2",
            },
            {
                "stage": "stage2_additional",
                "model_slot": "qwen",
                "condition_id": "c111",
                "episode_attempt_id": "episode-3",
            },
        )
        statuses = {
            "episode-1": runner.AttemptStatus.COMPLETED,
            "episode-2": runner.AttemptStatus.STARTED,
            "episode-3": runner.AttemptStatus.FAILED,
        }

        report = runner._summarize_status(rows, statuses)

        groups = {
            (item["stage"], item["model_slot"], item["condition_id"]): item
            for item in report["groups"]
        }
        stage1 = groups[("stage1", "qwen", "c000")]
        self.assertEqual(stage1["scheduled"], 2)
        self.assertEqual(stage1["completed"], 1)
        self.assertEqual(stage1["resumable"], 1)
        self.assertEqual(stage1["pending"], 0)
        stage2 = groups[("stage2_additional", "qwen", "c111")]
        self.assertEqual(stage2["failed"], 1)


class MinimalFactorialCliTests(unittest.TestCase):
    def test_cli_smoke_and_status_are_thin_delegations(self) -> None:
        from scripts import run_iclr2027_minimal_factorial as cli

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.yaml"
            summary = runner.RunSummary(
                "smoke",
                Path(tmp),
                16,
                16,
                0,
                0,
                0,
                0,
                0,
                True,
            )
            with (
                mock.patch.object(
                    cli,
                    "run_smoke",
                    return_value=summary,
                ) as run_smoke,
                mock.patch.object(
                    cli,
                    "campaign_status",
                    return_value={"groups": [], "totals": {}},
                ) as status,
            ):
                self.assertEqual(
                    cli.main(["--manifest", str(manifest), "smoke", "--resume"]),
                    0,
                )
                self.assertEqual(
                    cli.main(["--manifest", str(manifest), "status"]),
                    0,
                )

        run_smoke.assert_called_once_with(manifest, resume=True)
        status.assert_called_once_with(manifest)


if __name__ == "__main__":
    unittest.main()
