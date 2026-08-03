from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime import _append_intent_io as append_io
from dilu.runtime import _minimal_factorial_runner_execution as execution
from dilu.runtime import minimal_factorial_runner as runner
from dilu.runtime._campaign_attempt_serialization import canonical_bytes, hash_payload


CAMPAIGN_PROVENANCE_SHA256 = "sha256:" + "c" * 64


def _summary(attempt_id: str, *, task_completed: bool = True) -> dict[str, object]:
    return {
        "campaign_provenance_sha256": CAMPAIGN_PROVENANCE_SHA256,
        "episode_attempt_id": attempt_id,
        "runtime_snapshot_sha256": "sha256:" + "d" * 64,
        "task_completed": task_completed,
    }


class _FakeLedger:
    def __init__(self, _path: Path, *, campaign_id: str, resume: bool) -> None:
        del campaign_id, resume
        self.statuses: dict[str, runner.AttemptStatus] = {}

    def attempt_statuses(self) -> dict[str, runner.AttemptStatus]:
        return dict(self.statuses)

    def attempt_status(
        self,
        attempt_id: str,
    ) -> runner.AttemptStatus | None:
        return self.statuses.get(attempt_id)

    def validate_trace_evidence(self, _writer: object) -> None:
        return None


class _FakeTrace:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass


def _row() -> SimpleNamespace:
    return SimpleNamespace(
        campaign_id="campaign-001",
        episode_attempt_id="episode-001",
        model_slot="qwen",
        model_tag="qwen:test",
        model_digest="sha256:" + "a" * 64,
        condition=SimpleNamespace(retry_policy=mock.sentinel.retry),
        to_payload=lambda: {"episode_attempt_id": "episode-001"},
    )


class MinimalFactorialSummaryEvidenceTests(unittest.TestCase):
    def test_summary_fsync_failure_leaves_ambiguity_marker(self) -> None:
        ledger = mock.Mock()
        ledger.attempt_status.return_value = runner.AttemptStatus.COMPLETED

        def write_then_fail(path: Path, line: bytes) -> None:
            with path.open("ab") as handle:
                handle.write(line)
                handle.flush()
            raise OSError("fake fsync failure after write")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_summaries.jsonl"
            with mock.patch.object(
                append_io,
                "_append_and_sync_data",
                side_effect=write_then_fail,
            ):
                with self.assertRaises(append_io.AppendCommitAmbiguousError):
                    runner._append_episode_summary(
                        path,
                        _summary("episode-001"),
                        ledger,
                    )

            self.assertTrue(append_io.append_intent_path_for(path).is_file())
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                execution._load_summaries(
                    path,
                    expected_campaign_provenance_sha256=(CAMPAIGN_PROVENANCE_SHA256),
                )

    def test_campaign_provenance_hash_binds_snapshot_and_denominator(self) -> None:
        rows = (
            SimpleNamespace(to_payload=lambda: {"episode_attempt_id": "episode-001"}),
            SimpleNamespace(to_payload=lambda: {"episode_attempt_id": "episode-002"}),
        )
        snapshot_sha256 = "f" * 64

        actual = execution._campaign_provenance_sha256(
            rows,
            snapshot_sha256,
        )

        self.assertEqual(
            actual,
            hash_payload(
                {
                    "schema_version": "iclr2027.campaign_provenance.v1",
                    "runtime_snapshot_sha256": "sha256:" + snapshot_sha256,
                    "scheduled_denominator": [
                        {"episode_attempt_id": "episode-001"},
                        {"episode_attempt_id": "episode-002"},
                    ],
                }
            ),
        )

    def test_summary_records_have_canonical_provenance_chain_and_root(self) -> None:
        ledger = mock.Mock()
        ledger.attempt_status.return_value = runner.AttemptStatus.COMPLETED
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_summaries.jsonl"
            first_summary = _summary("episode-001")
            second_summary = _summary("episode-002")

            runner._append_episode_summary(path, first_summary, ledger)
            runner._append_episode_summary(path, second_summary, ledger)

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            loaded = execution._load_summaries(
                path,
                expected_campaign_provenance_sha256=CAMPAIGN_PROVENANCE_SHA256,
            )

        self.assertEqual(records[0]["sequence"], 1)
        self.assertIsNone(records[0]["previous_record_sha256"])
        self.assertEqual(records[1]["sequence"], 2)
        self.assertEqual(
            records[1]["previous_record_sha256"],
            records[0]["record_sha256"],
        )
        self.assertEqual(
            records[0]["summary_provenance_sha256"],
            hash_payload(first_summary),
        )
        for record in records:
            unhashed = dict(record)
            claimed = unhashed.pop("record_sha256")
            self.assertEqual(claimed, hash_payload(unhashed))
        self.assertEqual(
            execution._summary_root_sha256(loaded),
            records[-1]["record_sha256"],
        )

    def test_tampered_summary_field_fails_resume_validation(self) -> None:
        self._assert_tamper_blocked("task_completed", False)

    def test_tampered_campaign_provenance_fails_resume_validation(self) -> None:
        self._assert_tamper_blocked(
            "campaign_provenance_sha256",
            "sha256:" + "e" * 64,
        )

    def test_execution_stamps_campaign_provenance_on_every_summary(self) -> None:
        row = _row()
        stage2_row = SimpleNamespace(
            to_payload=lambda: {"episode_attempt_id": "episode-002"},
        )
        captured: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp:
            prepared = SimpleNamespace(
                output_root=Path(tmp),
                schedule=(row, stage2_row),
                capabilities={
                    "qwen": SimpleNamespace(
                        model_tag=row.model_tag,
                        model_digest=row.model_digest,
                    )
                },
                snapshot=SimpleNamespace(sha256="f" * 64),
            )

            def run_episode(
                _prepared: object,
                scheduled: SimpleNamespace,
                *,
                ledger: _FakeLedger,
                **_kwargs: object,
            ) -> dict[str, object]:
                ledger.statuses[scheduled.episode_attempt_id] = (
                    runner.AttemptStatus.COMPLETED
                )
                return {"task_completed": True}

            result = execution.execute_campaign(
                prepared,
                scheduled_rows=(row,),
                denominator_rows=(row,),
                resume=False,
                stage="smoke",
                ledger_type=_FakeLedger,
                trace_type=_FakeTrace,
                client_builder=lambda *_args: {"qwen": mock.sentinel.client},
                episode_runner=run_episode,
                pending_selector=lambda rows, *_args, **_kwargs: tuple(rows),
                summary_appender=lambda _path, summary, _ledger: captured.append(
                    dict(summary)
                ),
                failure_recorder=lambda *_args: None,
                completion_checker=lambda *_args: (),
            )

        self.assertEqual(
            captured[0]["campaign_provenance_sha256"],
            execution._campaign_provenance_sha256(
                (row, stage2_row),
                "f" * 64,
            ),
        )
        self.assertEqual(
            result.campaign_provenance_sha256,
            execution._campaign_provenance_sha256(
                (row, stage2_row),
                "f" * 64,
            ),
        )

    def test_resume_validates_summary_hashes_before_execution(self) -> None:
        row = _row()
        snapshot_sha256 = "f" * 64
        campaign_hash = execution._campaign_provenance_sha256(
            (row,),
            snapshot_sha256,
        )
        ledger = mock.Mock()
        ledger.attempt_status.return_value = runner.AttemptStatus.COMPLETED
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "episode_summaries.jsonl"
            runner._append_episode_summary(
                path,
                {
                    **_summary("episode-001"),
                    "campaign_provenance_sha256": campaign_hash,
                },
                ledger,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["runtime_snapshot_sha256"] = "sha256:" + "0" * 64
            path.write_bytes(canonical_bytes(payload) + b"\n")
            prepared = SimpleNamespace(
                output_root=root,
                capabilities={},
                snapshot=SimpleNamespace(sha256=snapshot_sha256),
            )
            build_clients = mock.Mock(
                side_effect=AssertionError("execution began before validation")
            )

            with self.assertRaisesRegex(ValueError, "summary.*integrity"):
                execution.execute_campaign(
                    prepared,
                    scheduled_rows=(row,),
                    denominator_rows=(row,),
                    resume=True,
                    stage="smoke",
                    ledger_type=_FakeLedger,
                    trace_type=_FakeTrace,
                    client_builder=build_clients,
                    episode_runner=mock.Mock(),
                    pending_selector=mock.Mock(),
                    summary_appender=mock.Mock(),
                    failure_recorder=mock.Mock(),
                    completion_checker=mock.Mock(),
                )

        build_clients.assert_not_called()

    def _assert_tamper_blocked(self, field: str, value: object) -> None:
        ledger = mock.Mock()
        ledger.attempt_status.return_value = runner.AttemptStatus.COMPLETED
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_summaries.jsonl"
            runner._append_episode_summary(
                path,
                _summary("episode-001"),
                ledger,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload[field] = value
            path.write_bytes(canonical_bytes(payload) + b"\n")

            with self.assertRaisesRegex(ValueError, "summary.*integrity|provenance"):
                execution._load_summaries(
                    path,
                    expected_campaign_provenance_sha256=(CAMPAIGN_PROVENANCE_SHA256),
                )


if __name__ == "__main__":
    unittest.main()
