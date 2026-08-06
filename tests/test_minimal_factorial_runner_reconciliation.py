from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime import _minimal_factorial_runner_execution as execution
from dilu.runtime import minimal_factorial_runner as runner
from dilu.runtime._minimal_factorial_runner_summaries import append_summary_record
from tests.test_minimal_factorial_runner_summary_evidence import (
    _FakeLedger,
    _FakeTrace,
    _row,
)


class MinimalFactorialReconciliationTests(unittest.TestCase):
    def test_duplicate_outbox_is_rejected_before_any_terminal_append(self) -> None:
        row = _row()
        snapshot_sha256 = "f" * 64

        class AtomicLedger(_FakeLedger):
            latest: AtomicLedger | None = None

            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.statuses[row.episode_attempt_id] = runner.AttemptStatus.STARTED
                self.completed_appends = 0
                type(self).latest = self

            def append_terminal(
                self,
                attempt_id: str,
                *,
                status: runner.AttemptStatus,
                **_kwargs: object,
            ) -> None:
                self.completed_appends += 1
                self.statuses[attempt_id] = status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_campaign_summary(root, row, snapshot_sha256)
            self._write_campaign_summary(root, row, snapshot_sha256)
            prepared = SimpleNamespace(
                output_root=root,
                schedule=(row,),
                capabilities={},
                snapshot=SimpleNamespace(sha256=snapshot_sha256),
            )
            build_clients = mock.Mock()

            with self.assertRaisesRegex(ValueError, "duplicate"):
                execution.execute_campaign(
                    prepared,
                    scheduled_rows=(row,),
                    denominator_rows=(row,),
                    resume=True,
                    stage="smoke",
                    ledger_type=AtomicLedger,
                    trace_type=_FakeTrace,
                    client_builder=build_clients,
                    episode_runner=mock.Mock(),
                    pending_selector=runner._select_pending_rows,
                    summary_appender=mock.Mock(),
                    failure_recorder=mock.Mock(),
                    completion_checker=runner._completion_errors,
                )

        self.assertEqual(AtomicLedger.latest.completed_appends, 0)
        self.assertIs(
            AtomicLedger.latest.attempt_status(row.episode_attempt_id),
            runner.AttemptStatus.STARTED,
        )
        build_clients.assert_not_called()

    def test_terminal_trace_mismatch_fails_before_clients_and_pending_episode(
        self,
    ) -> None:
        completed = _row()
        pending = SimpleNamespace(
            campaign_id=completed.campaign_id,
            episode_attempt_id="episode-002",
            model_slot="qwen-2",
            model_tag="qwen:test-2",
            model_digest="sha256:" + "2" * 64,
            condition=SimpleNamespace(retry_policy=mock.sentinel.retry),
            to_payload=lambda: {"episode_attempt_id": "episode-002"},
        )
        rows = (completed, pending)
        snapshot_sha256 = "f" * 64
        campaign_hash = execution._campaign_provenance_sha256(rows, snapshot_sha256)

        class CorruptTerminalLedger(_FakeLedger):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.statuses[completed.episode_attempt_id] = (
                    runner.AttemptStatus.COMPLETED
                )

            def validate_trace_evidence(self, _writer: object) -> None:
                raise ValueError("ledger trace evidence mismatch")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_summary_record(
                root / "episodes.jsonl",
                {
                    "task_completed": True,
                    **completed.to_payload(),
                    "runtime_snapshot_sha256": "sha256:" + snapshot_sha256,
                    "campaign_provenance_sha256": campaign_hash,
                    "scientific_trace_references": [
                        reference.to_dict()
                        for reference in _FakeTrace().references_for_attempt(
                            completed.campaign_id,
                            completed.episode_attempt_id,
                        )
                    ],
                },
            )
            prepared = SimpleNamespace(
                output_root=root,
                schedule=rows,
                capabilities={
                    pending.model_slot: SimpleNamespace(
                        model_tag=pending.model_tag,
                        model_digest=pending.model_digest,
                    )
                },
                snapshot=SimpleNamespace(sha256=snapshot_sha256),
            )
            build_clients = mock.Mock(
                return_value={pending.model_slot: mock.sentinel.client}
            )
            episode_runner = mock.Mock(return_value={})

            with self.assertRaisesRegex(ValueError, "ledger trace evidence mismatch"):
                execution.execute_campaign(
                    prepared,
                    scheduled_rows=(pending,),
                    denominator_rows=rows,
                    resume=True,
                    stage="stage2",
                    ledger_type=CorruptTerminalLedger,
                    trace_type=_FakeTrace,
                    client_builder=build_clients,
                    episode_runner=episode_runner,
                    pending_selector=runner._select_pending_rows,
                    summary_appender=mock.Mock(),
                    failure_recorder=mock.Mock(),
                    completion_checker=runner._completion_errors,
                )

        build_clients.assert_not_called()
        episode_runner.assert_not_called()

    def _write_campaign_summary(
        self,
        root: Path,
        row: SimpleNamespace,
        snapshot_sha256: str,
        *,
        trace_references: list[dict[str, object]] | None = None,
    ) -> str:
        campaign_hash = execution._campaign_provenance_sha256(
            (row,),
            snapshot_sha256,
        )
        append_summary_record(
            root / "episodes.jsonl",
            {
                "task_completed": True,
                **row.to_payload(),
                "runtime_snapshot_sha256": "sha256:" + snapshot_sha256,
                "campaign_provenance_sha256": campaign_hash,
                "scientific_trace_references": trace_references
                or [
                    reference.to_dict()
                    for reference in _FakeTrace().references_for_attempt(
                        row.campaign_id,
                        row.episode_attempt_id,
                    )
                ],
            },
        )
        return campaign_hash

    def test_resume_finalizes_summary_first_attempt_before_clients(self) -> None:
        row = _row()
        snapshot_sha256 = "f" * 64

        class ResumeLedger(_FakeLedger):
            latest: ResumeLedger | None = None
            persisted_status = runner.AttemptStatus.STARTED
            completed_appends = 0

            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.statuses[row.episode_attempt_id] = type(self).persisted_status
                type(self).latest = self

            def can_resume(self, attempt_id: str) -> bool:
                return self.statuses.get(attempt_id) is runner.AttemptStatus.STARTED

            def append_terminal(
                self,
                attempt_id: str,
                *,
                status: runner.AttemptStatus,
                **_kwargs: object,
            ) -> None:
                type(self).completed_appends += 1
                type(self).persisted_status = status
                self.statuses[attempt_id] = status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_hash = self._write_campaign_summary(root, row, snapshot_sha256)
            prepared = SimpleNamespace(
                output_root=root,
                schedule=(row,),
                capabilities={},
                snapshot=SimpleNamespace(sha256=snapshot_sha256),
            )
            build_clients = mock.Mock(
                side_effect=AssertionError("clients built during reconciliation")
            )
            episode_runner = mock.Mock(
                side_effect=AssertionError("episode reran during reconciliation")
            )

            result = execution.execute_campaign(
                prepared,
                scheduled_rows=(row,),
                denominator_rows=(row,),
                resume=True,
                stage="smoke",
                ledger_type=ResumeLedger,
                trace_type=_FakeTrace,
                client_builder=build_clients,
                episode_runner=episode_runner,
                pending_selector=runner._select_pending_rows,
                summary_appender=mock.Mock(),
                failure_recorder=mock.Mock(),
                completion_checker=runner._completion_errors,
            )
            repeated = execution.execute_campaign(
                prepared,
                scheduled_rows=(row,),
                denominator_rows=(row,),
                resume=True,
                stage="smoke",
                ledger_type=ResumeLedger,
                trace_type=_FakeTrace,
                client_builder=build_clients,
                episode_runner=episode_runner,
                pending_selector=runner._select_pending_rows,
                summary_appender=mock.Mock(),
                failure_recorder=mock.Mock(),
                completion_checker=runner._completion_errors,
            )

        build_clients.assert_not_called()
        episode_runner.assert_not_called()
        self.assertEqual(ResumeLedger.completed_appends, 1)
        self.assertEqual(result.completed, 1)
        self.assertEqual(repeated.completed, 1)
        self.assertTrue(repeated.promotion_allowed)
        self.assertTrue(result.promotion_allowed)
        self.assertEqual(result.campaign_provenance_sha256, campaign_hash)

    def test_completed_without_summary_fails_before_clients(self) -> None:
        row = _row()

        class CompletedLedger(_FakeLedger):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.statuses[row.episode_attempt_id] = runner.AttemptStatus.COMPLETED

        with tempfile.TemporaryDirectory() as tmp:
            prepared = SimpleNamespace(
                output_root=Path(tmp),
                schedule=(row,),
                capabilities={},
                snapshot=SimpleNamespace(sha256="f" * 64),
            )
            build_clients = mock.Mock()

            with self.assertRaisesRegex(ValueError, "Completed.*summary"):
                execution.execute_campaign(
                    prepared,
                    scheduled_rows=(row,),
                    denominator_rows=(row,),
                    resume=True,
                    stage="smoke",
                    ledger_type=CompletedLedger,
                    trace_type=_FakeTrace,
                    client_builder=build_clients,
                    episode_runner=mock.Mock(),
                    pending_selector=runner._select_pending_rows,
                    summary_appender=mock.Mock(),
                    failure_recorder=mock.Mock(),
                    completion_checker=runner._completion_errors,
                )

        build_clients.assert_not_called()

    def test_resume_rejects_trace_mismatch_and_failed_summary_state(self) -> None:
        row = _row()
        snapshot_sha256 = "f" * 64
        cases = (
            (runner.AttemptStatus.STARTED, [{"relative_path": "wrong"}], "trace"),
            (runner.AttemptStatus.FAILED, None, "status"),
            (runner.AttemptStatus.BLOCKED, None, "status"),
            (runner.AttemptStatus.WRITE_AMBIGUOUS, None, "status"),
            (None, None, "status"),
        )
        for status, references, message in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_campaign_summary(
                    root,
                    row,
                    snapshot_sha256,
                    trace_references=references,
                )

                class InvalidLedger(_FakeLedger):
                    def __init__(self, *args: object, **kwargs: object) -> None:
                        super().__init__(*args, **kwargs)
                        if status is not None:
                            self.statuses[row.episode_attempt_id] = status

                prepared = SimpleNamespace(
                    output_root=root,
                    schedule=(row,),
                    capabilities={},
                    snapshot=SimpleNamespace(sha256=snapshot_sha256),
                )
                build_clients = mock.Mock()

                with self.assertRaisesRegex(ValueError, message):
                    execution.execute_campaign(
                        prepared,
                        scheduled_rows=(row,),
                        denominator_rows=(row,),
                        resume=True,
                        stage="smoke",
                        ledger_type=InvalidLedger,
                        trace_type=_FakeTrace,
                        client_builder=build_clients,
                        episode_runner=mock.Mock(),
                        pending_selector=runner._select_pending_rows,
                        summary_appender=mock.Mock(),
                        failure_recorder=mock.Mock(),
                        completion_checker=runner._completion_errors,
                    )

                build_clients.assert_not_called()


if __name__ == "__main__":
    unittest.main()
