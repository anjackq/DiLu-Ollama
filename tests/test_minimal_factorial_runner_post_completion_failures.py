from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime import _minimal_factorial_runner_execution as execution
from dilu.runtime import minimal_factorial_runner as runner
from tests.test_minimal_factorial_runner_execution import (
    _FakeLedger,
    _FakeTraceWriter,
    _row,
)


class PostCompletionFailureTests(unittest.TestCase):
    def test_cleanup_failure_after_completion_does_not_mutate_summary_evidence(
        self,
    ) -> None:
        row = _row(0)

        class RecordingLedger(_FakeLedger):
            latest: RecordingLedger | None = None

            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.summary_failures = 0
                self.event_types: list[str] = []
                type(self).latest = self

            def append_terminal(
                self,
                attempt_id: str,
                *,
                status: runner.AttemptStatus,
                **kwargs: object,
            ) -> None:
                super().append_terminal(attempt_id, status=status, **kwargs)
                self.event_types.append("attempt_lifecycle")

            def append_summary_failure(
                self,
                attempt_id: str,
                *,
                failure_class: str,
                failure_message: str,
            ) -> None:
                del failure_class, failure_message
                self.summary_failures += 1
                self.event_types.append("summary_publication_failed")
                self.statuses[attempt_id] = runner.AttemptStatus.WRITE_AMBIGUOUS

        def execute_episode(
            _prepared: object,
            scheduled: SimpleNamespace,
            *,
            ledger: RecordingLedger,
            trace_writer: _FakeTraceWriter,
            completion_publisher: object,
            **_kwargs: object,
        ) -> dict[str, object]:
            ledger.append_started(scheduled.episode_attempt_id)
            references = trace_writer.references_for_attempt(
                scheduled.campaign_id,
                scheduled.episode_attempt_id,
            )
            completion_publisher(
                {
                    "task_completed": True,
                    "scientific_trace_references": [
                        reference.to_dict() for reference in references
                    ],
                },
                references,
            )
            return {"task_completed": True}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = SimpleNamespace(
                output_root=root,
                schedule=(row,),
                capabilities={
                    row.model_slot: SimpleNamespace(
                        model_tag=row.model_tag,
                        model_digest=row.model_digest,
                    )
                },
                snapshot=SimpleNamespace(sha256="a" * 64),
            )

            class CleanupFails:
                def __init__(self, **_kwargs: object) -> None:
                    self.path = root / "episode-temp"
                    self.path.mkdir()

                def __enter__(self) -> str:
                    return str(self.path)

                def __exit__(self, *_args: object) -> None:
                    raise OSError("cleanup failed after completion")

            with mock.patch.object(
                execution.tempfile,
                "TemporaryDirectory",
                CleanupFails,
            ):
                with self.assertRaisesRegex(OSError, "cleanup failed after completion"):
                    execution.execute_campaign(
                        prepared,
                        scheduled_rows=(row,),
                        denominator_rows=(row,),
                        resume=False,
                        stage="smoke",
                        ledger_type=RecordingLedger,
                        trace_type=_FakeTraceWriter,
                        client_builder=lambda *_args: {
                            row.model_slot: mock.sentinel.client
                        },
                        episode_runner=execute_episode,
                        pending_selector=runner._select_pending_rows,
                        summary_appender=runner._append_episode_summary,
                        failure_recorder=runner._record_infrastructure_failure,
                        completion_checker=runner._completion_errors,
                    )

            summaries = execution._load_summaries(root / "episodes.jsonl")

        self.assertEqual(len(summaries), 1)
        self.assertEqual(RecordingLedger.latest.summary_failures, 0)
        self.assertNotIn(
            "summary_publication_failed",
            RecordingLedger.latest.event_types,
        )
        self.assertIs(
            RecordingLedger.latest.attempt_status(row.episode_attempt_id),
            runner.AttemptStatus.COMPLETED,
        )


if __name__ == "__main__":
    unittest.main()
