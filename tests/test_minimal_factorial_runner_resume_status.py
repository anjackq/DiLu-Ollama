from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime import _minimal_factorial_runner_status as runner_status
from dilu.runtime import _minimal_factorial_runner_execution as execution
from dilu.runtime import minimal_factorial_runner as runner
from dilu.runtime._append_intent_io import append_intent_path_for
from dilu.runtime._campaign_attempt_state import (
    AttemptLedgerSnapshot,
    read_validated_attempt_snapshot,
)
from dilu.runtime.campaign_attempts import (
    ScientificAttemptLedger,
    ScientificAttemptWriteError,
)


class MinimalFactorialResumeApprovalTests(unittest.TestCase):
    def test_started_attempt_with_registered_request_is_not_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ScientificAttemptLedger(
                Path(tmp) / "campaign_attempts.jsonl",
                campaign_id="campaign-001",
            )
            ledger.append_started("episode-001")
            ledger.register_request_id("request-001", "episode-001")

            self.assertFalse(ledger.can_resume("episode-001"))

    def test_run_counts_only_ledger_approved_started_attempt_as_resumable(
        self,
    ) -> None:
        row = SimpleNamespace(episode_attempt_id="episode-001")
        statuses = {"episode-001": runner.AttemptStatus.STARTED}
        ledger = mock.Mock()
        ledger.can_resume.return_value = False

        resumable, ambiguous = execution._started_resume_counts(
            (row,),
            statuses,
            ledger,
        )

        self.assertEqual((resumable, ambiguous), (0, 1))

    def test_resume_scans_streamingly_and_rebuilds_request_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "campaign_attempts.jsonl"
            ledger = ScientificAttemptLedger(
                ledger_path,
                campaign_id="campaign-001",
            )
            ledger.append_started("episode-001")
            ledger.register_request_id("request-001", "episode-001")

            with (
                mock.patch.object(
                    Path,
                    "read_bytes",
                    side_effect=AssertionError("whole-file byte read"),
                ),
                mock.patch.object(
                    Path,
                    "read_text",
                    side_effect=AssertionError("whole-file text read"),
                ),
            ):
                resumed = ScientificAttemptLedger(
                    ledger_path,
                    campaign_id="campaign-001",
                    resume=True,
                )

            self.assertFalse(resumed.can_resume("episode-001"))
            with self.assertRaises(ScientificAttemptWriteError):
                resumed.register_request_id("request-001", "episode-001")
            resumed.register_request_id("request-002", "episode-001")


class MinimalFactorialStatusEvidenceTests(unittest.TestCase):
    def test_status_rejects_attempt_ledger_ids_outside_frozen_schedule(
        self,
    ) -> None:
        cases = (
            ("started", None),
            ("failed", runner.AttemptStatus.FAILED),
            ("blocked", runner.AttemptStatus.BLOCKED),
        )
        for name, terminal_status in cases:
            with self.subTest(status=name), tempfile.TemporaryDirectory() as tmp:
                prepared, row = self._prepared(Path(tmp))
                ledger = ScientificAttemptLedger(
                    prepared.output_root / "campaign_attempts.jsonl",
                    campaign_id=row.campaign_id,
                )
                ledger.append_started("unexpected-episode")
                if terminal_status is not None:
                    ledger.append_terminal(
                        "unexpected-episode",
                        status=terminal_status,
                        decision_count=0,
                        failure_class="runtime_protocol_error",
                        failure_message="unexpected scheduled identity",
                        trace_absence_reason="aborted_before_first_decision",
                    )

                report = runner_status.campaign_status((prepared,))

            validation = report["artifact_validation"]
            self.assertFalse(validation["valid"])
            self.assertFalse(validation["claim_promotion_allowed"])
            self.assertTrue(
                any(
                    "unexpected attempt ledger IDs" in error
                    for error in validation["errors"]
                ),
                validation["errors"],
            )

    def test_attempt_snapshot_includes_terminal_records_with_default_compatibility(
        self,
    ) -> None:
        self.assertEqual(
            AttemptLedgerSnapshot({}, frozenset()).terminal_records,
            (),
        )
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "campaign_attempts.jsonl"
            ledger = ScientificAttemptLedger(
                ledger_path,
                campaign_id="campaign-001",
            )
            ledger.append_started("episode-001")
            ledger.append_terminal(
                "episode-001",
                status=runner.AttemptStatus.FAILED,
                decision_count=0,
                failure_class="infrastructure_exception",
                failure_message="offline failure",
                trace_absence_reason="aborted_before_first_decision",
            )

            snapshot = read_validated_attempt_snapshot(
                ledger_path,
                campaign_id="campaign-001",
            )

        self.assertEqual(len(snapshot.terminal_records), 1)
        self.assertEqual(snapshot.terminal_records[0].episode_attempt_id, "episode-001")

    def test_status_rejects_orphan_trace_episode_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared, row = self._prepared(Path(tmp))
            reference = ({"line_number": 1},)
            cases = (
                (
                    {
                        row.episode_attempt_id: runner.AttemptStatus.STARTED,
                        "orphan": runner.AttemptStatus.STARTED,
                    },
                    (row.campaign_id, "orphan"),
                ),
                ({}, (row.campaign_id, row.episode_attempt_id)),
            )
            for statuses, trace_key in cases:
                with (
                    self.subTest(trace_key=trace_key),
                    mock.patch.object(
                        runner_status,
                        "_read_attempt_snapshot",
                        return_value=AttemptLedgerSnapshot(statuses, frozenset()),
                    ),
                    mock.patch.object(
                        runner_status,
                        "_read_validated_trace_references",
                        return_value={trace_key: reference},
                    ),
                ):
                    report = runner_status.campaign_status((prepared,))

                validation = report["artifact_validation"]
                self.assertFalse(validation["valid"])
                self.assertFalse(validation["claim_promotion_allowed"])
                self.assertTrue(
                    any("orphan trace episode" in error for error in validation["errors"]),
                    validation["errors"],
                )

    def test_status_reports_attempt_ledger_oserror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared, _ = self._prepared(Path(tmp))
            with mock.patch.object(
                runner_status,
                "_read_attempt_snapshot",
                side_effect=OSError("ledger unavailable"),
            ):
                report = runner_status.campaign_status((prepared,))

        validation = report["artifact_validation"]
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["claim_promotion_allowed"])
        self.assertTrue(
            any("ledger unavailable" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_status_reader_rejects_tampered_ledger_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "campaign_attempts.jsonl"
            ledger = ScientificAttemptLedger(
                ledger_path,
                campaign_id="campaign-001",
            )
            ledger.append_started("episode-001")
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            payload["record_sha256"] = "sha256:" + "0" * 64
            ledger_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "status validation"):
                runner_status._read_attempt_statuses(
                    ledger_path,
                    campaign_id="campaign-001",
                )

    def test_status_reader_rejects_append_started_during_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "campaign_attempts.jsonl"
            ledger = ScientificAttemptLedger(
                ledger_path,
                campaign_id="campaign-001",
            )
            ledger.append_started("episode-001")
            original_scan = ScientificAttemptLedger._scan_existing

            def scan_then_publish_intent(active: ScientificAttemptLedger) -> None:
                original_scan(active)
                append_intent_path_for(active.path).write_text(
                    "pending\n",
                    encoding="utf-8",
                )

            with (
                mock.patch.object(
                    ScientificAttemptLedger,
                    "_scan_existing",
                    autospec=True,
                    side_effect=scan_then_publish_intent,
                ),
                self.assertRaisesRegex(ValueError, "status validation"),
            ):
                runner_status._read_attempt_statuses(
                    ledger_path,
                    campaign_id="campaign-001",
                )

    def test_public_status_revalidates_frozen_campaign_before_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = SimpleNamespace(
                root="artifacts",
                smoke="smoke",
                llm_campaign="claim",
            )
            manifest = SimpleNamespace(outputs=outputs)
            validated = SimpleNamespace(repo_root=root, manifest=manifest)
            smoke = SimpleNamespace(output_root=root / "artifacts" / "smoke")
            (smoke.output_root / "campaign_manifest.json").parent.mkdir(parents=True)
            (smoke.output_root / "campaign_manifest.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            report = {"groups": [], "totals": {}}

            with (
                mock.patch.object(runner, "_repo_root", return_value=root),
                mock.patch.object(
                    runner,
                    "load_experiment_manifest",
                    return_value=manifest,
                ),
                mock.patch.object(
                    runner,
                    "_validate_live_snapshot",
                    return_value=validated,
                ) as validate,
                mock.patch.object(
                    runner,
                    "_open_frozen_campaign",
                    return_value=smoke,
                ) as open_frozen,
                mock.patch.object(
                    runner,
                    "_campaign_status_impl",
                    return_value=report,
                ) as status_impl,
            ):
                actual = runner.campaign_status(Path("manifest.yaml"))

            self.assertEqual(actual, report)
            validate.assert_called_once_with(Path("manifest.yaml"))
            open_frozen.assert_called_once_with(validated, "smoke")
            status_impl.assert_called_once_with((smoke,))

    @staticmethod
    def _prepared(root: Path) -> tuple[SimpleNamespace, SimpleNamespace]:
        row = SimpleNamespace(
            stage="stage1",
            campaign_id="campaign-claim",
            episode_attempt_id="episode-001",
            model_slot="qwen",
            condition_id="c000",
            to_payload=lambda: {
                "stage": "stage1",
                "campaign_id": "campaign-claim",
                "episode_attempt_id": "episode-001",
                "model_slot": "qwen",
                "condition_id": "c000",
            },
        )
        prepared = SimpleNamespace(
            schedule=(row,),
            output_root=root,
            snapshot=SimpleNamespace(sha256="a" * 64),
        )
        return prepared, row


if __name__ == "__main__":
    unittest.main()
