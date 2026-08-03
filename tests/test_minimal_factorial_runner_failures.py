from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime import _minimal_factorial_runner_status as runner_status
from dilu.runtime import minimal_factorial_runner as runner
from dilu.runtime.campaign_attempts import ScientificAttemptLedger
from dilu.runtime.scientific_trace import TraceReference
from tests.runtime_factorization_support import runtime


def _rows(stage: str, count: int) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(stage=stage, episode_attempt_id=f"{stage}-{index}")
        for index in range(count)
    )


class MinimalFactorialEvidenceTests(unittest.TestCase):
    def test_duplicate_scheduled_ids_block_before_execution(self) -> None:
        duplicate = SimpleNamespace(episode_attempt_id="episode-001")
        prepared = SimpleNamespace()

        with mock.patch.object(runner, "_execute_campaign_impl") as execute:
            with self.assertRaisesRegex(
                ValueError,
                "duplicate scheduled episode IDs",
            ):
                runner._execute_campaign(
                    prepared,
                    scheduled_rows=(duplicate, duplicate),
                    denominator_rows=(duplicate, duplicate),
                    resume=False,
                    stage="smoke",
                )

        execute.assert_not_called()

    def test_duplicate_scheduled_ids_block_completion(self) -> None:
        duplicate = SimpleNamespace(episode_attempt_id="episode-001")
        statuses = {
            duplicate.episode_attempt_id: runner.AttemptStatus.COMPLETED,
        }

        errors = runner._completion_errors(
            (duplicate, duplicate),
            ({"episode_attempt_id": "episode-001"},),
            statuses,
        )

        self.assertIn("duplicate scheduled episode IDs", errors)

    def test_ledger_exposes_read_only_attempt_status_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ScientificAttemptLedger(
                Path(tmp) / "attempts.jsonl",
                campaign_id="campaign-001",
            )
            self.assertIsNone(ledger.attempt_status("episode-001"))
            ledger.append_started("episode-001")

            self.assertIs(
                ledger.attempt_status("episode-001"),
                runner.AttemptStatus.STARTED,
            )
            self.assertEqual(
                ledger.attempt_statuses(),
                {"episode-001": runner.AttemptStatus.STARTED},
            )

    def test_scientific_runtime_resumes_ledger_approved_open_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bound = runtime(root)
            bound.begin_attempt()
            resumed_ledger = ScientificAttemptLedger(
                root / "campaign_attempts.jsonl",
                campaign_id=bound.identity.campaign_id,
                resume=True,
            )
            resumed = dataclasses.replace(
                bound,
                attempt_ledger=resumed_ledger,
            )

            resumed.begin_attempt()

            self.assertIs(
                resumed_ledger.attempt_status(bound.identity.episode_attempt_id),
                runner.AttemptStatus.STARTED,
            )

    def test_infrastructure_exception_becomes_typed_terminal_failure(self) -> None:
        ledger = mock.Mock()
        ledger.attempt_status.return_value = None
        row = SimpleNamespace(episode_attempt_id="episode-001")

        runner._record_infrastructure_failure(
            ledger,
            row,
            RuntimeError("simulator launch failed"),
        )

        ledger.append_started.assert_called_once_with("episode-001")
        ledger.append_terminal.assert_called_once_with(
            "episode-001",
            status=runner.AttemptStatus.FAILED,
            decision_count=0,
            failure_class="infrastructure_exception",
            failure_message="RuntimeError: simulator launch failed",
            trace_absence_reason="aborted_before_first_decision",
        )

    def test_post_terminal_summary_failure_appends_typed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "attempts.jsonl"
            ledger = ScientificAttemptLedger(
                ledger_path,
                campaign_id="campaign-001",
            )
            ledger.append_started("episode-001")
            ledger.append_terminal(
                "episode-001",
                status=runner.AttemptStatus.COMPLETED,
                decision_count=1,
                trace_references=(
                    TraceReference(
                        relative_path="traces/decision_traces.jsonl",
                        line_number=1,
                        record_sha256="sha256:" + "a" * 64,
                        schema_version="iclr2027.decision_trace.v1",
                        schema_sha256="sha256:" + "b" * 64,
                    ),
                ),
            )

            runner._record_infrastructure_failure(
                ledger,
                SimpleNamespace(episode_attempt_id="episode-001"),
                OSError("fake fsync failure after summary write"),
            )

            self.assertIs(
                ledger.attempt_status("episode-001"),
                runner.AttemptStatus.WRITE_AMBIGUOUS,
            )
            payloads = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                payloads[-1]["event_type"],
                "summary_publication_failed",
            )
            self.assertEqual(
                payloads[-1]["failure_class"],
                "summary_durability_failure",
            )
            self.assertEqual(
                payloads[-1]["prior_status"],
                runner.AttemptStatus.COMPLETED.value,
            )
            resumed = ScientificAttemptLedger(
                ledger_path,
                campaign_id="campaign-001",
                resume=True,
            )
            self.assertIs(
                resumed.attempt_status("episode-001"),
                runner.AttemptStatus.WRITE_AMBIGUOUS,
            )
            self.assertIs(
                runner_status._read_attempt_statuses(
                    ledger_path,
                    campaign_id="campaign-001",
                )["episode-001"],
                runner.AttemptStatus.WRITE_AMBIGUOUS,
            )

    def test_fsync_failure_after_summary_write_blocks_promotion(self) -> None:
        row = SimpleNamespace(
            stage="smoke",
            campaign_id="campaign-001",
            episode_attempt_id="episode-001",
            model_slot="qwen",
            model_tag="qwen:test",
            model_digest="sha256:" + "a" * 64,
            condition=SimpleNamespace(retry_policy=mock.sentinel.retry),
            to_payload=lambda: {"episode_attempt_id": "episode-001"},
        )

        class FakeLedger:
            def __init__(
                self,
                _path: Path,
                *,
                campaign_id: str,
                resume: bool,
            ) -> None:
                del campaign_id, resume
                self.statuses: dict[str, runner.AttemptStatus] = {}

            def attempt_status(
                self,
                attempt_id: str,
            ) -> runner.AttemptStatus | None:
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

            def append_summary_failure(
                self,
                attempt_id: str,
                *,
                failure_class: str,
                failure_message: str,
            ) -> None:
                del failure_class, failure_message
                self.statuses[attempt_id] = runner.AttemptStatus.WRITE_AMBIGUOUS

            def validate_trace_evidence(self, _writer: object) -> None:
                return None

        class FakeTrace:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

        def execute_episode(
            _prepared: object,
            scheduled: SimpleNamespace,
            *,
            ledger: FakeLedger,
            **_kwargs: object,
        ) -> dict[str, object]:
            ledger.append_started(scheduled.episode_attempt_id)
            ledger.append_terminal(
                scheduled.episode_attempt_id,
                status=runner.AttemptStatus.COMPLETED,
            )
            return {"task_completed": True}

        def fail_fsync_after_write(
            path: Path,
            summary: object,
            _ledger: object,
        ) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            raise OSError("fake fsync failure after summary write")

        with tempfile.TemporaryDirectory() as tmp:
            prepared = SimpleNamespace(
                output_root=Path(tmp),
                capabilities={
                    "qwen": SimpleNamespace(
                        model_tag=row.model_tag,
                        model_digest=row.model_digest,
                    )
                },
                snapshot=SimpleNamespace(sha256="b" * 64),
            )
            with (
                mock.patch.object(runner, "ScientificAttemptLedger", FakeLedger),
                mock.patch.object(runner, "ScientificTraceWriter", FakeTrace),
                mock.patch.object(
                    runner,
                    "build_model_clients",
                    return_value={"qwen": mock.sentinel.client},
                ),
                mock.patch.object(
                    runner,
                    "_run_scheduled_episode",
                    side_effect=execute_episode,
                ),
                mock.patch.object(
                    runner,
                    "_append_episode_summary",
                    side_effect=fail_fsync_after_write,
                ),
            ):
                result = runner._execute_campaign(
                    prepared,
                    scheduled_rows=(row,),
                    denominator_rows=(row,),
                    resume=False,
                    stage="smoke",
                )

        self.assertFalse(result.promotion_allowed)
        self.assertEqual(result.ambiguous, 1)

    def test_duplicate_and_denominator_mismatch_block_completion(self) -> None:
        rows = _rows("stage1", 2)
        statuses = {
            row.episode_attempt_id: runner.AttemptStatus.COMPLETED for row in rows
        }
        duplicate = [
            {"episode_attempt_id": rows[0].episode_attempt_id},
            {"episode_attempt_id": rows[0].episode_attempt_id},
        ]

        errors = runner._completion_errors(rows, duplicate, statuses)

        self.assertIn("duplicate episode summaries", errors)
        self.assertIn("episode summary denominator mismatch", errors)

    def test_summary_append_requires_terminal_attempt_evidence(self) -> None:
        ledger = mock.Mock()
        ledger.attempt_status.return_value = runner.AttemptStatus.STARTED
        summary = {
            "campaign_provenance_sha256": "sha256:" + "c" * 64,
            "episode_attempt_id": "episode-001",
        }
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            runner._append_episode_summary(
                Path("episodes.jsonl"),
                summary,
                ledger,
            )
        ledger.attempt_status.return_value = runner.AttemptStatus.COMPLETED
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            runner._append_episode_summary(
                path,
                summary,
                ledger,
            )
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
