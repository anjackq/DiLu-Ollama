from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime import _minimal_factorial_runner_status as runner_status
from dilu.runtime import minimal_factorial_runner as runner
from dilu.runtime._campaign_attempt_state import AttemptLedgerSnapshot
from dilu.runtime.campaign_attempts import ScientificAttemptLedger
from dilu.runtime.scientific_trace import ScientificTraceValidationError, TraceReference


def _artifact_inventory(root: Path) -> tuple[tuple[object, ...], ...]:
    inventory: list[tuple[object, ...]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            inventory.append((relative, digest, path.stat().st_mtime_ns))
        else:
            inventory.append((relative, "directory"))
    return tuple(inventory)


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

    def test_artifact_validation_reports_corruption_without_raising(self) -> None:
        prepared, row = self._prepared_claim(1)
        completed = AttemptLedgerSnapshot(
            {row.episode_attempt_id: runner.AttemptStatus.COMPLETED},
            frozenset(),
        )
        good_reference = self._reference(1)
        good_summary = {
            "episode_attempt_id": row.episode_attempt_id,
            "scientific_trace_references": [good_reference],
        }
        corruptions = (
            (
                ValueError("Episode summary record integrity hash is invalid."),
                (),
                {},
                None,
                "integrity hash",
            ),
            (None, (good_summary, good_summary), {}, None, "duplicate"),
            (None, (), {}, None, "completed attempt IDs"),
            (
                None,
                (good_summary,),
                {(row.campaign_id, row.episode_attempt_id): ()},
                None,
                "ordered trace references",
            ),
            (
                None,
                (good_summary,),
                {},
                ScientificTraceValidationError("trace hash mismatch"),
                "scientific trace evidence invalid",
            ),
        )

        for (
            summary_error,
            summaries,
            traces,
            trace_error,
            expected_error,
        ) in corruptions:
            load_effect: object = summaries
            if summary_error is not None:
                load_effect = summary_error
            with (
                self.subTest(expected_error=expected_error),
                mock.patch.object(
                    runner_status,
                    "_read_attempt_snapshot",
                    return_value=completed,
                ),
                mock.patch.object(
                    runner_status,
                    "load_summary_records",
                    side_effect=load_effect if isinstance(load_effect, Exception) else None,
                    return_value=load_effect if not isinstance(load_effect, Exception) else None,
                ),
                mock.patch.object(
                    runner_status,
                    "_read_validated_trace_references",
                    side_effect=trace_error,
                    return_value=traces,
                ),
            ):
                report = runner_status.campaign_status((prepared,))

            validation = report["artifact_validation"]
            self.assertFalse(validation["valid"])
            self.assertFalse(validation["claim_promotion_allowed"])
            self.assertTrue(
                any(expected_error in error for error in validation["errors"]),
                validation["errors"],
            )

    def test_claim_promotion_requires_exactly_840_valid_joined_completions(
        self,
    ) -> None:
        prepared, _ = self._prepared_claim(840)
        statuses = {
            row.episode_attempt_id: runner.AttemptStatus.COMPLETED
            for row in prepared.schedule
        }
        summaries = tuple(
            {
                "episode_attempt_id": row.episode_attempt_id,
                "scientific_trace_references": [self._reference(index + 1)],
            }
            for index, row in enumerate(prepared.schedule)
        )
        traces = {
            (row.campaign_id, row.episode_attempt_id): (self._reference(index + 1),)
            for index, row in enumerate(prepared.schedule)
        }
        with (
            mock.patch.object(
                runner_status,
                "_read_attempt_snapshot",
                return_value=AttemptLedgerSnapshot(statuses, frozenset()),
            ),
            mock.patch.object(
                runner_status,
                "load_summary_records",
                return_value=summaries,
            ),
            mock.patch.object(
                runner_status,
                "_read_validated_trace_references",
                return_value=traces,
            ),
        ):
            report = runner_status.campaign_status((prepared,))

        self.assertTrue(report["artifact_validation"]["valid"])
        self.assertTrue(
            report["artifact_validation"]["claim_promotion_allowed"]
        )
        self.assertEqual(report["totals"]["scheduled"], 840)
        self.assertEqual(report["totals"]["completed"], 840)

        statuses["out-of-schedule"] = runner.AttemptStatus.FAILED
        with (
            mock.patch.object(
                runner_status,
                "_read_attempt_snapshot",
                return_value=AttemptLedgerSnapshot(statuses, frozenset()),
            ),
            mock.patch.object(
                runner_status,
                "load_summary_records",
                return_value=summaries,
            ),
            mock.patch.object(
                runner_status,
                "_read_validated_trace_references",
                return_value=traces,
            ),
        ):
            report_with_bad_attempt = runner_status.campaign_status((prepared,))

        self.assertFalse(
            report_with_bad_attempt["artifact_validation"][
                "claim_promotion_allowed"
            ]
        )

    def test_status_validation_does_not_create_artifacts(self) -> None:
        for existing_trace in (False, True):
            with self.subTest(existing_trace=existing_trace), tempfile.TemporaryDirectory() as tmp:
                output_root = Path(tmp)
                prepared, _ = self._prepared_claim(1, output_root=output_root)
                if existing_trace:
                    trace_path = output_root / "traces" / "decision_traces.jsonl"
                    trace_path.parent.mkdir()
                    trace_path.write_bytes(b"")
                before = _artifact_inventory(output_root)
                report = runner_status.campaign_status((prepared,))
                after = _artifact_inventory(output_root)
            self.assertEqual(before, after)
            self.assertTrue(report["artifact_validation"]["valid"])
            self.assertFalse(report["artifact_validation"]["claim_promotion_allowed"])

    def test_status_rejects_terminal_ledger_trace_mismatch_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            prepared, row = self._prepared_claim(1, output_root=output_root)
            ledger_path = output_root / "campaign_attempts.jsonl"
            ledger = ScientificAttemptLedger(ledger_path, campaign_id=row.campaign_id)
            reference = TraceReference(**self._reference(1))
            ledger.append_started(row.episode_attempt_id)
            ledger.append_terminal(
                row.episode_attempt_id,
                status=runner.AttemptStatus.COMPLETED,
                decision_count=1,
                trace_references=(reference,),
            )
            before = _artifact_inventory(output_root)
            report = runner_status.campaign_status((prepared,))
            after = _artifact_inventory(output_root)
        validation = report["artifact_validation"]
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["claim_promotion_allowed"])
        errors = validation["errors"]
        self.assertTrue(any("ledger trace evidence" in error for error in errors), errors)
        self.assertEqual(before, after)

    @staticmethod
    def _reference(line_number: int) -> dict[str, object]:
        return {
            "relative_path": "traces/decision_traces.jsonl",
            "line_number": line_number,
            "record_sha256": "sha256:" + f"{line_number:064x}",
            "schema_version": "iclr2027.scientific_trace.v1",
            "schema_sha256": "sha256:" + "e" * 64,
        }

    @staticmethod
    def _prepared_claim(
        count: int,
        *,
        output_root: Path = Path("claim"),
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        rows = tuple(
            SimpleNamespace(
                stage="stage1" if index < 480 else "stage2_additional",
                campaign_id="campaign-claim",
                episode_attempt_id=f"episode-{index:03d}",
                model_slot="qwen",
                condition_id="c000",
                to_payload=lambda index=index: {
                    "stage": "stage1" if index < 480 else "stage2_additional",
                    "campaign_id": "campaign-claim",
                    "episode_attempt_id": f"episode-{index:03d}",
                    "model_slot": "qwen",
                    "condition_id": "c000",
                },
            )
            for index in range(count)
        )
        return (
            SimpleNamespace(
                schedule=rows,
                output_root=output_root,
                snapshot=SimpleNamespace(sha256="a" * 64),
            ),
            rows[0],
        )


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

    def test_cli_forwards_positive_batch_limit_and_enables_faulthandler(self) -> None:
        from scripts import run_iclr2027_minimal_factorial as cli

        with (
            mock.patch.object(cli.faulthandler, "is_enabled", return_value=False),
            mock.patch.object(cli.faulthandler, "enable") as enable,
            mock.patch.object(
                cli,
                "run_claim_stage",
                return_value=mock.sentinel.summary,
            ) as run_claim,
        ):
            result = cli.main(
                [
                    "--manifest",
                    "manifest.yaml",
                    "run",
                    "--stage",
                    "stage2",
                    "--resume",
                    "--max-episodes",
                    "20",
                ]
            )

        self.assertEqual(result, 0)
        enable.assert_called_once_with()
        run_claim.assert_called_once_with(
            Path("manifest.yaml"),
            stage="stage2",
            resume=True,
            max_episodes=20,
        )

    def test_cli_rejects_non_positive_and_non_integer_batch_limits(self) -> None:
        from scripts import run_iclr2027_minimal_factorial as cli

        for invalid in ("0", "-1", "True", "1.5"):
            with (
                self.subTest(max_episodes=invalid),
                mock.patch.object(cli, "run_claim_stage") as run_claim,
                self.assertRaises(SystemExit),
            ):
                cli.main(
                    [
                        "run",
                        "--stage",
                        "stage2",
                        "--max-episodes",
                        invalid,
                    ]
                )
            run_claim.assert_not_called()


if __name__ == "__main__":
    unittest.main()
