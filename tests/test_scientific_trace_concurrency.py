from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime.scientific_trace import (
    ScientificTraceCommitAmbiguousError,
    ScientificTraceValidationError,
    ScientificTraceWriter,
)
from tests.test_scientific_trace import _record


def _retag_record(*, attempt_id: str, request_id: str):
    record = _record()
    key = dataclasses.replace(
        record.context.key,
        episode_attempt_id=attempt_id,
    )
    request = dataclasses.replace(record.generation.request, request_id=request_id)
    attempts = tuple(
        dataclasses.replace(
            attempt,
            request_id=request_id,
            attempt_id=f"{request_id}:a{attempt.attempt_index}",
        )
        for attempt in record.generation.attempts
    )
    return dataclasses.replace(
        record,
        context=dataclasses.replace(record.context, key=key),
        generation=dataclasses.replace(
            record.generation,
            request=request,
            attempts=attempts,
        ),
    )


class ScientificTraceConcurrencyTests(unittest.TestCase):
    def test_stale_writers_refresh_under_exclusive_append_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            first = ScientificTraceWriter(path, artifact_root=root)
            second = ScientificTraceWriter(path, artifact_root=root)

            first_reference = first.append(
                _retag_record(attempt_id="attempt-001", request_id="request-001")
            )
            second_reference = second.append(
                _retag_record(attempt_id="attempt-002", request_id="request-002")
            )

            self.assertEqual(first_reference.line_number, 1)
            self.assertEqual(second_reference.line_number, 2)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_stale_writer_cannot_reuse_campaign_request_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            first = ScientificTraceWriter(path, artifact_root=root)
            second = ScientificTraceWriter(path, artifact_root=root)
            first.append(
                _retag_record(attempt_id="attempt-001", request_id="request-shared")
            )

            with self.assertRaises(ScientificTraceValidationError):
                second.append(
                    _retag_record(
                        attempt_id="attempt-002",
                        request_id="request-shared",
                    )
                )

    def test_commit_ambiguity_poison_survives_fresh_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            writer = ScientificTraceWriter(path, artifact_root=root)
            with mock.patch.object(
                os,
                "fsync",
                side_effect=(None, OSError("fsync failed")),
            ):
                with self.assertRaises(ScientificTraceCommitAmbiguousError):
                    writer.append(_record())

            with self.assertRaises(ScientificTraceCommitAmbiguousError):
                ScientificTraceWriter(path, artifact_root=root, resume=True)

    def test_prewrite_validation_is_not_commit_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )

            with self.assertRaises(ScientificTraceValidationError):
                writer.append(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
