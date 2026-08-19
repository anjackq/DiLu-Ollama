from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime import _minimal_factorial_runner_execution as execution
from dilu.runtime import minimal_factorial_runner as runner


class _FakeLedger:
    instances = 0

    def __init__(self, path: Path, *, campaign_id: str, resume: bool) -> None:
        type(self).instances += 1
        self.path = path
        self.campaign_id = campaign_id
        self.resume = resume
        self.statuses: dict[str, runner.AttemptStatus] = {}

    def attempt_status(self, attempt_id: str) -> runner.AttemptStatus | None:
        return self.statuses.get(attempt_id)

    def attempt_statuses(self) -> dict[str, runner.AttemptStatus]:
        return dict(self.statuses)

    def append_started(self, attempt_id: str) -> None:
        self.statuses[attempt_id] = runner.AttemptStatus.STARTED

    def append_terminal(
        self,
        attempt_id: str,
        *,
        status: runner.AttemptStatus,
        **_kwargs: object,
    ) -> None:
        self.statuses[attempt_id] = status

    def validate_trace_evidence(self, _writer: object) -> None:
        return None

    def can_resume(self, attempt_id: str) -> bool:
        return self.statuses.get(attempt_id) is runner.AttemptStatus.STARTED


class _FakeTraceWriter:
    instances = 0

    def __init__(
        self,
        path: Path,
        *,
        artifact_root: Path,
        resume: bool,
    ) -> None:
        type(self).instances += 1
        self.path = path
        self.artifact_root = artifact_root
        self.resume = resume

    def references_for_attempt(
        self,
        campaign_id: str,
        episode_attempt_id: str,
    ) -> tuple[SimpleNamespace, ...]:
        del campaign_id
        line_number = int(episode_attempt_id.rsplit("-", 1)[1]) + 1
        return (
            SimpleNamespace(
                to_dict=lambda: {
                    "relative_path": "traces/decision_traces.jsonl",
                    "line_number": line_number,
                    "record_sha256": "sha256:" + f"{line_number:064x}",
                    "schema_version": "iclr2027.scientific_trace.v1",
                    "schema_sha256": "sha256:" + "e" * 64,
                }
            ),
        )


def _row(index: int) -> SimpleNamespace:
    attempt_id = f"episode-{index}"
    return SimpleNamespace(
        stage="smoke",
        campaign_id="campaign-smoke",
        episode_attempt_id=attempt_id,
        model_slot=f"model-{index}",
        model_tag=f"tag-{index}",
        model_digest="sha256:" + f"{index + 1:064x}",
        condition_id="c000",
        condition=SimpleNamespace(retry_policy=mock.sentinel.retry),
        to_payload=lambda: {
            "episode_attempt_id": attempt_id,
            "stage": "smoke",
        },
    )


class MinimalFactorialExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeLedger.instances = 0
        _FakeTraceWriter.instances = 0

    def test_resources_and_model_clients_are_campaign_scoped_once(self) -> None:
        rows = (_row(0), _row(1))
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "smoke"
            prepared = SimpleNamespace(
                output_root=output_root,
                schedule=rows,
                capabilities={
                    row.model_slot: SimpleNamespace(
                        model_tag=row.model_tag,
                        model_digest=row.model_digest,
                    )
                    for row in rows
                },
                snapshot=SimpleNamespace(sha256="a" * 64),
            )
            clients = {
                "model-0": mock.sentinel.client_0,
                "model-1": mock.sentinel.client_1,
            }
            seen_clients: list[object] = []
            episode_temp_dirs: list[Path] = []

            def execute_episode(
                _prepared: object,
                row: SimpleNamespace,
                *,
                ledger: _FakeLedger,
                trace_writer: _FakeTraceWriter,
                client: object,
                episode_temp_dir: Path,
                completion_publisher: object,
            ) -> dict[str, object]:
                seen_clients.append(client)
                episode_temp_dirs.append(episode_temp_dir)
                ledger.append_started(row.episode_attempt_id)
                completion_publisher(
                    {"task_completed": True},
                    trace_writer.references_for_attempt(
                        row.campaign_id,
                        row.episode_attempt_id,
                    ),
                )
                return {"task_completed": True}

            with (
                mock.patch.object(
                    runner,
                    "ScientificAttemptLedger",
                    _FakeLedger,
                ),
                mock.patch.object(
                    runner,
                    "ScientificTraceWriter",
                    _FakeTraceWriter,
                ),
                mock.patch.object(
                    runner,
                    "build_model_clients",
                    return_value=clients,
                ) as build_clients,
                mock.patch.object(
                    runner,
                    "_run_scheduled_episode",
                    side_effect=execute_episode,
                ),
            ):
                summary = runner._execute_campaign(
                    prepared,
                    scheduled_rows=rows,
                    denominator_rows=rows,
                    resume=False,
                    stage="smoke",
                )
            self.assertTrue((output_root / "episodes.jsonl").is_file())
            self.assertFalse((output_root / "episode_summaries.jsonl").exists())
            summaries = execution._load_summaries(
                output_root / "episodes.jsonl",
                expected_campaign_provenance_sha256=(
                    summary.campaign_provenance_sha256
                ),
            )
            self.assertEqual(
                len(summaries[0]["scientific_trace_references"]),
                1,
            )
            self.assertTrue(episode_temp_dirs)
            self.assertTrue(
                all(not path.exists() for path in episode_temp_dirs),
                "episode temp directories must be cleaned after each call",
            )

        self.assertEqual(_FakeLedger.instances, 1)
        self.assertEqual(_FakeTraceWriter.instances, 1)
        build_clients.assert_called_once_with(
            prepared.capabilities,
            mock.sentinel.retry,
        )
        self.assertEqual(
            seen_clients,
            [mock.sentinel.client_0, mock.sentinel.client_1],
        )
        self.assertEqual(summary.completed, 2)
        self.assertEqual(summary.pending, 0)
        self.assertTrue(summary.promotion_allowed)

    def test_summary_is_published_before_completed_terminal(self) -> None:
        row = _row(0)
        events: list[str] = []

        class OrderedLedger(_FakeLedger):
            def append_terminal(
                self,
                attempt_id: str,
                *,
                status: runner.AttemptStatus,
                **kwargs: object,
            ) -> None:
                del kwargs
                events.append(status.value)
                super().append_terminal(attempt_id, status=status)

        def execute_episode(
            _prepared: object,
            scheduled: SimpleNamespace,
            *,
            ledger: OrderedLedger,
            trace_writer: _FakeTraceWriter,
            completion_publisher: object,
            **_kwargs: object,
        ) -> dict[str, object]:
            ledger.append_started(scheduled.episode_attempt_id)
            references = trace_writer.references_for_attempt(
                scheduled.campaign_id,
                scheduled.episode_attempt_id,
            )
            completion_publisher({"task_completed": True}, references)
            return {"task_completed": True}

        def append_summary(
            path: Path,
            summary: object,
            ledger: OrderedLedger,
        ) -> None:
            del path, summary
            self.assertIs(
                ledger.attempt_status(row.episode_attempt_id),
                runner.AttemptStatus.STARTED,
            )
            events.append("summary")

        with tempfile.TemporaryDirectory() as tmp:
            prepared = SimpleNamespace(
                output_root=Path(tmp),
                schedule=(row,),
                capabilities={
                    row.model_slot: SimpleNamespace(
                        model_tag=row.model_tag,
                        model_digest=row.model_digest,
                    )
                },
                snapshot=SimpleNamespace(sha256="a" * 64),
            )
            result = execution.execute_campaign(
                prepared,
                scheduled_rows=(row,),
                denominator_rows=(row,),
                resume=False,
                stage="smoke",
                ledger_type=OrderedLedger,
                trace_type=_FakeTraceWriter,
                client_builder=lambda *_args: {row.model_slot: mock.sentinel.client},
                episode_runner=execute_episode,
                pending_selector=runner._select_pending_rows,
                summary_appender=append_summary,
                failure_recorder=lambda *_args: None,
                completion_checker=lambda *_args: (),
            )

        self.assertEqual(events, ["summary", "completed"])
        self.assertEqual(result.completed, 1)

    def test_started_rows_require_ledger_resume_approval(self) -> None:
        started, unseen = _row(0), _row(1)
        ledger = mock.Mock()
        ledger.can_resume.return_value = False
        statuses = {
            started.episode_attempt_id: runner.AttemptStatus.STARTED,
        }

        approved = execution._ledger_approved_rows(
            (started, unseen),
            statuses,
            ledger,
        )

        self.assertEqual(approved, (unseen,))
        ledger.can_resume.assert_called_once_with(started.episode_attempt_id)

    def test_max_episodes_rejects_bool_zero_and_negative_values(self) -> None:
        for invalid in (True, False, 0, -1):
            with (
                self.subTest(max_episodes=invalid),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                execution._validate_max_episodes(invalid)
        for valid in (None, 1):
            with self.subTest(max_episodes=valid):
                execution._validate_max_episodes(valid)

    def test_limit_is_applied_after_exact_once_ledger_approval(self) -> None:
        rows = tuple(_row(index) for index in range(5))
        completed, request_owned, resumable, unseen_a, unseen_b = rows
        initial_statuses = {
            completed.episode_attempt_id: runner.AttemptStatus.COMPLETED,
            request_owned.episode_attempt_id: runner.AttemptStatus.STARTED,
            resumable.episode_attempt_id: runner.AttemptStatus.STARTED,
        }
        completion_checker = mock.Mock(return_value=())
        executed_ids: list[str] = []

        def execute_episode(
            _prepared: object,
            row: SimpleNamespace,
            *,
            ledger: _FakeLedger,
            **_kwargs: object,
        ) -> dict[str, object]:
            executed_ids.append(row.episode_attempt_id)
            ledger.statuses[row.episode_attempt_id] = runner.AttemptStatus.COMPLETED
            return {"task_completed": True}

        for max_episodes, expected in (
            (1, (resumable,)),
            (2, (resumable, unseen_a)),
            (None, (resumable, unseen_a, unseen_b)),
        ):
            executed_ids.clear()

            class BatchLedger(_FakeLedger):
                def __init__(
                    self, path: Path, *, campaign_id: str, resume: bool
                ) -> None:
                    super().__init__(path, campaign_id=campaign_id, resume=resume)
                    self.statuses.update(initial_statuses)

                def can_resume(self, attempt_id: str) -> bool:
                    return attempt_id != request_owned.episode_attempt_id

            with (
                self.subTest(max_episodes=max_episodes),
                tempfile.TemporaryDirectory() as tmp,
                mock.patch.object(execution, "reconcile_published_summaries"),
            ):
                prepared = self._prepared(Path(tmp), rows)
                result = execution.execute_campaign(
                    prepared,
                    scheduled_rows=rows,
                    denominator_rows=rows,
                    resume=True,
                    stage="stage2",
                    max_episodes=max_episodes,
                    ledger_type=BatchLedger,
                    trace_type=_FakeTraceWriter,
                    client_builder=lambda *_args: {
                        row.model_slot: mock.sentinel.client for row in rows
                    },
                    episode_runner=execute_episode,
                    pending_selector=runner._select_pending_rows,
                    summary_appender=mock.Mock(),
                    failure_recorder=mock.Mock(),
                    completion_checker=completion_checker,
                )

            self.assertEqual(
                executed_ids,
                [row.episode_attempt_id for row in expected],
            )
            self.assertNotIn(request_owned.episode_attempt_id, executed_ids)
            self.assertEqual(result.selected_this_invocation, len(expected))
            self.assertEqual(result.scheduled, len(rows))
            self.assertEqual(completion_checker.call_args.args[0], rows)

    @staticmethod
    def _prepared(output_root: Path, rows: tuple[SimpleNamespace, ...]) -> object:
        return SimpleNamespace(
            output_root=output_root,
            schedule=rows,
            capabilities={row.model_slot: row for row in rows},
            snapshot=SimpleNamespace(sha256="a" * 64),
        )


if __name__ == "__main__":
    unittest.main()
