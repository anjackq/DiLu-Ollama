from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from dilu.runtime import _append_intent_io as intent_io
from dilu.runtime.campaign_attempts import (
    ScientificAttemptLedger,
    ScientificAttemptWriteError,
)
from dilu.runtime.scientific_trace import (
    ScientificTraceWriteError,
    ScientificTraceWriter,
    append_trace_before_step,
)
from tests.test_scientific_trace import _record


CAMPAIGN_ID = "campaign-001"
ATTEMPT_ID = "episode-attempt-001"


class AppendIntentDurabilityTests(unittest.TestCase):
    def test_intent_fsync_failure_does_not_touch_jsonl(self) -> None:
        for artifact_kind in ("attempt_ledger", "scientific_trace"):
            with (
                self.subTest(artifact_kind=artifact_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                target, path = self._target(root, artifact_kind)

                with mock.patch.object(
                    intent_io.os,
                    "fsync",
                    side_effect=OSError("intent fsync failed"),
                ):
                    with self.assertRaises(self._write_error(artifact_kind)):
                        self._append(target, artifact_kind)

                self.assertEqual(path.read_bytes() if path.exists() else b"", b"")
                self.assertTrue(intent_io.append_intent_path_for(path).exists())
                self._assert_fresh_resume_rejects(root, path, artifact_kind)

    def test_data_fsync_failure_keeps_pending_and_blocks_fresh_resume(self) -> None:
        for artifact_kind in ("attempt_ledger", "scientific_trace"):
            with (
                self.subTest(artifact_kind=artifact_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                target, path = self._target(root, artifact_kind)

                with mock.patch.object(
                    intent_io.os,
                    "fsync",
                    side_effect=(None, OSError("data fsync failed")),
                ):
                    with self.assertRaises(self._write_error(artifact_kind)):
                        self._append(target, artifact_kind)

                self.assertGreater(path.stat().st_size, 0)
                self.assertTrue(intent_io.append_intent_path_for(path).exists())
                self._assert_fresh_resume_rejects(root, path, artifact_kind)

    def test_data_close_failure_keeps_pending_and_blocks_fresh_resume(self) -> None:
        for artifact_kind in ("attempt_ledger", "scientific_trace"):
            with (
                self.subTest(artifact_kind=artifact_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                target, path = self._target(root, artifact_kind)
                original_append = intent_io._append_and_sync_data

                def append_then_fail(*args: Any, **kwargs: Any) -> None:
                    original_append(*args, **kwargs)
                    raise OSError("data close outcome is ambiguous")

                with mock.patch.object(
                    intent_io,
                    "_append_and_sync_data",
                    side_effect=append_then_fail,
                ):
                    with self.assertRaises(self._write_error(artifact_kind)):
                        self._append(target, artifact_kind)

                self.assertGreater(path.stat().st_size, 0)
                self.assertTrue(intent_io.append_intent_path_for(path).exists())
                with self.assertRaises(self._write_error(artifact_kind)):
                    self._append(target, artifact_kind)
                self._assert_fresh_resume_rejects(root, path, artifact_kind)

    def test_real_partial_tail_keeps_pending_and_blocks_fresh_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            writer = ScientificTraceWriter(path, artifact_root=root)

            def write_one_byte(target: Path, data: bytes) -> None:
                target.write_bytes(data[:1])
                raise OSError("partial data append")

            with mock.patch.object(
                intent_io,
                "_append_and_sync_data",
                side_effect=write_one_byte,
            ):
                with self.assertRaises(ScientificTraceWriteError):
                    writer.append(_record())

            self.assertEqual(path.read_bytes(), b"{")
            self.assertTrue(intent_io.append_intent_path_for(path).exists())
            self._assert_fresh_resume_rejects(root, path, "scientific_trace")

    def test_both_intent_and_data_fsync_precede_environment_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ScientificTraceWriter(
                Path(tmp) / "decision_traces.jsonl",
                artifact_root=Path(tmp),
            )
            events: list[str] = []

            def step(action_id: int) -> str:
                events.append(f"step:{action_id}")
                return "stepped"

            with mock.patch.object(
                os,
                "fsync",
                side_effect=lambda descriptor: events.append("fsync"),
            ):
                append_trace_before_step(writer, _record(), step)

            self.assertEqual(events, ["fsync", "fsync", "step:3"])

    def test_cleanup_failure_and_crash_point_keep_pending(self) -> None:
        failures = (OSError("cleanup failed"), KeyboardInterrupt("crash point"))
        for artifact_kind in ("attempt_ledger", "scientific_trace"):
            for failure in failures:
                with (
                    self.subTest(
                        artifact_kind=artifact_kind,
                        failure=type(failure).__name__,
                    ),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    target, path = self._target(root, artifact_kind)
                    expected_error: type[BaseException] = (
                        KeyboardInterrupt
                        if isinstance(failure, KeyboardInterrupt)
                        else self._write_error(artifact_kind)
                    )

                    with mock.patch.object(
                        intent_io,
                        "clear_append_intent",
                        side_effect=failure,
                    ):
                        with self.assertRaises(expected_error):
                            self._append(target, artifact_kind)

                    pending = intent_io.append_intent_path_for(path)
                    self.assertTrue(pending.exists())
                    payload = json.loads(pending.read_text(encoding="utf-8"))
                    self.assertEqual(payload["artifact_kind"], artifact_kind)
                    self.assertEqual(payload["episode_attempt_id"], ATTEMPT_ID)
                    self.assertEqual(payload["expected_offset"], 0)
                    self.assertEqual(payload["byte_length"], path.stat().st_size)
                    self.assertRegex(payload["record_sha256"], r"^sha256:[0-9a-f]{64}$")
                    self.assertRegex(payload["line_sha256"], r"^sha256:[0-9a-f]{64}$")
                    self._assert_fresh_resume_rejects(root, path, artifact_kind)

    def test_clean_success_removes_pending_and_fresh_resume_accepts_once(self) -> None:
        for artifact_kind in ("attempt_ledger", "scientific_trace"):
            with (
                self.subTest(artifact_kind=artifact_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                target, path = self._target(root, artifact_kind)

                self._append(target, artifact_kind)

                self.assertFalse(intent_io.append_intent_path_for(path).exists())
                lines = path.read_bytes().splitlines()
                self.assertEqual(len(lines), 1)
                if artifact_kind == "attempt_ledger":
                    resumed = ScientificAttemptLedger(
                        path,
                        campaign_id=CAMPAIGN_ID,
                        resume=True,
                    )
                    self.assertTrue(resumed.can_resume(ATTEMPT_ID))
                else:
                    resumed = ScientificTraceWriter(
                        path,
                        artifact_root=root,
                        resume=True,
                    )
                    self.assertEqual(resumed.next_line_number, 2)

    @staticmethod
    def _target(root: Path, artifact_kind: str) -> tuple[object, Path]:
        if artifact_kind == "attempt_ledger":
            path = root / "campaign_attempts.jsonl"
            return ScientificAttemptLedger(path, campaign_id=CAMPAIGN_ID), path
        path = root / "decision_traces.jsonl"
        return ScientificTraceWriter(path, artifact_root=root), path

    @staticmethod
    def _append(target: object, artifact_kind: str) -> None:
        if artifact_kind == "attempt_ledger":
            assert isinstance(target, ScientificAttemptLedger)
            target.append_started(ATTEMPT_ID)
            return
        assert isinstance(target, ScientificTraceWriter)
        target.append(_record())

    @staticmethod
    def _write_error(artifact_kind: str) -> type[Exception]:
        if artifact_kind == "attempt_ledger":
            return ScientificAttemptWriteError
        return ScientificTraceWriteError

    def _assert_fresh_resume_rejects(
        self,
        root: Path,
        path: Path,
        artifact_kind: str,
    ) -> None:
        if artifact_kind == "attempt_ledger":
            with self.assertRaises(ScientificAttemptWriteError):
                ScientificAttemptLedger(path, campaign_id=CAMPAIGN_ID, resume=True)
            return
        with self.assertRaises(ScientificTraceWriteError):
            ScientificTraceWriter(path, artifact_root=root, resume=True)


if __name__ == "__main__":
    unittest.main()
