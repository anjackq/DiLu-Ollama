from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime import minimal_factorial_runner as runner
from dilu.runtime import _minimal_factorial_runner_execution as execution


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
                trace_writer: object,
                client: object,
                episode_temp_dir: Path,
            ) -> dict[str, object]:
                del trace_writer
                seen_clients.append(client)
                episode_temp_dirs.append(episode_temp_dir)
                ledger.append_started(row.episode_attempt_id)
                ledger.append_terminal(
                    row.episode_attempt_id,
                    status=runner.AttemptStatus.COMPLETED,
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


if __name__ == "__main__":
    unittest.main()
