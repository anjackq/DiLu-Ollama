from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime import _campaign_attempt_store as attempt_store
from dilu.runtime.campaign_attempts import (
    TRACE_COMMIT_AMBIGUOUS,
    AttemptStatus,
    ScientificAttemptLedger,
    ScientificAttemptRecord,
    ScientificAttemptWriteError,
)
from dilu.runtime.scientific_trace import TraceReference


CAMPAIGN_ID = "campaign-001"
ATTEMPT_ID = "episode-attempt-001"


def _trace_reference(line_number: int = 1) -> TraceReference:
    return TraceReference(
        relative_path="decision_traces.jsonl",
        line_number=line_number,
        record_sha256="sha256:" + f"{line_number:064x}",
        schema_version="iclr2027.decision_trace.v1",
        schema_sha256="sha256:" + "a" * 64,
    )


def _ledger(root: Path, *, resume: bool = False) -> ScientificAttemptLedger:
    return ScientificAttemptLedger(
        root / "campaign_attempts.jsonl",
        campaign_id=CAMPAIGN_ID,
        resume=resume,
    )


class ScientificAttemptRecordTests(unittest.TestCase):
    def test_record_is_frozen_and_requires_typed_immutable_evidence(self) -> None:
        record = ScientificAttemptRecord(
            campaign_id=CAMPAIGN_ID,
            episode_attempt_id=ATTEMPT_ID,
            status=AttemptStatus.STARTED,
            decision_count=0,
        )

        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.status = AttemptStatus.FAILED  # type: ignore[misc]
        with self.assertRaises(ValueError):
            ScientificAttemptRecord(
                campaign_id=CAMPAIGN_ID,
                episode_attempt_id=ATTEMPT_ID,
                status="started",  # type: ignore[arg-type]
                decision_count=0,
            )
        with self.assertRaises(ValueError):
            ScientificAttemptRecord(
                campaign_id=CAMPAIGN_ID,
                episode_attempt_id=ATTEMPT_ID,
                status=AttemptStatus.COMPLETED,
                decision_count=1,
                trace_references=[_trace_reference()],  # type: ignore[arg-type]
            )

    def test_zero_decision_failure_requires_explicit_trace_absence_reason(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            ScientificAttemptRecord(
                campaign_id=CAMPAIGN_ID,
                episode_attempt_id=ATTEMPT_ID,
                status=AttemptStatus.FAILED,
                decision_count=0,
                failure_class="runtime_protocol_error",
                failure_message="availability was unresolved",
            )

        record = ScientificAttemptRecord(
            campaign_id=CAMPAIGN_ID,
            episode_attempt_id=ATTEMPT_ID,
            status=AttemptStatus.FAILED,
            decision_count=0,
            failure_class="runtime_protocol_error",
            failure_message="availability was unresolved",
            trace_absence_reason="aborted_before_first_decision",
        )
        self.assertEqual(record.trace_references, ())

    def test_decided_abort_requires_all_committed_trace_references(self) -> None:
        with self.assertRaises(ValueError):
            ScientificAttemptRecord(
                campaign_id=CAMPAIGN_ID,
                episode_attempt_id=ATTEMPT_ID,
                status=AttemptStatus.FAILED,
                decision_count=1,
                failure_class="simulator_failure",
                failure_message="env.step failed",
            )

        reference = _trace_reference()
        record = ScientificAttemptRecord(
            campaign_id=CAMPAIGN_ID,
            episode_attempt_id=ATTEMPT_ID,
            status=AttemptStatus.FAILED,
            decision_count=1,
            trace_references=(reference,),
            failure_class="simulator_failure",
            failure_message="env.step failed",
        )
        self.assertEqual(record.trace_references, (reference,))
        self.assertIsNone(record.trace_absence_reason)

    def test_write_ambiguity_uses_reserved_reason(self) -> None:
        with self.assertRaises(ValueError):
            ScientificAttemptRecord(
                campaign_id=CAMPAIGN_ID,
                episode_attempt_id=ATTEMPT_ID,
                status=AttemptStatus.WRITE_AMBIGUOUS,
                decision_count=0,
                failure_class="trace_write_failure",
                failure_message="close failed",
                trace_absence_reason=TRACE_COMMIT_AMBIGUOUS,
            )

        record = ScientificAttemptRecord(
            campaign_id=CAMPAIGN_ID,
            episode_attempt_id=ATTEMPT_ID,
            status=AttemptStatus.WRITE_AMBIGUOUS,
            decision_count=0,
            failure_class=TRACE_COMMIT_AMBIGUOUS,
            failure_message="trace fsync outcome is unknown",
            trace_absence_reason=TRACE_COMMIT_AMBIGUOUS,
        )
        self.assertEqual(record.failure_class, TRACE_COMMIT_AMBIGUOUS)


class ScientificAttemptLedgerTests(unittest.TestCase):
    def test_append_is_canonical_durable_and_hash_chained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = _ledger(root)
            first = ledger.append_started(ATTEMPT_ID)
            second = ledger.register_request_id("request-001", ATTEMPT_ID)
            third = ledger.append_terminal(
                ATTEMPT_ID,
                status=AttemptStatus.COMPLETED,
                decision_count=1,
                trace_references=(_trace_reference(),),
            )

            with (root / "campaign_attempts.jsonl").open("rb") as handle:
                raw_lines = handle.readlines()
            payloads = [json.loads(line) for line in raw_lines]
            for raw_line, payload in zip(raw_lines, payloads, strict=True):
                canonical = json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
                self.assertEqual(raw_line, canonical + b"\n")
                claimed_hash = payload.pop("record_sha256")
                unhashed = json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
                self.assertEqual(
                    claimed_hash,
                    "sha256:" + hashlib.sha256(unhashed).hexdigest(),
                )

            self.assertIsNone(payloads[0]["previous_record_sha256"])
            first_previous = payloads[1]["previous_record_sha256"]
            self.assertEqual(first_previous, first.record_sha256)
            self.assertEqual(
                payloads[2]["previous_record_sha256"], second.record_sha256
            )
            self.assertEqual(third.line_number, 3)

    def test_each_attempt_has_one_start_and_one_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = _ledger(Path(tmp))
            ledger.append_started(ATTEMPT_ID)
            self.assertTrue(ledger.can_resume(ATTEMPT_ID))
            with self.assertRaises(ScientificAttemptWriteError):
                ledger.append_started(ATTEMPT_ID)

            ledger.append_terminal(
                ATTEMPT_ID,
                status=AttemptStatus.BLOCKED,
                decision_count=0,
                failure_class="runtime_protocol_error",
                failure_message="precondition failed",
                trace_absence_reason="aborted_before_first_decision",
            )
            self.assertFalse(ledger.can_resume(ATTEMPT_ID))
            with self.assertRaises(ScientificAttemptWriteError):
                ledger.append_terminal(
                    ATTEMPT_ID,
                    status=AttemptStatus.FAILED,
                    decision_count=0,
                    failure_class="runtime_protocol_error",
                    failure_message="duplicate terminal",
                    trace_absence_reason="aborted_before_first_decision",
                )
            with self.assertRaises(ScientificAttemptWriteError):
                ledger.register_request_id("request-after-terminal", ATTEMPT_ID)

    def test_request_registry_is_campaign_wide_and_owned_by_started_attempt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = _ledger(Path(tmp))
            ledger.append_started(ATTEMPT_ID)
            ledger.register_request_id("request-001", ATTEMPT_ID)
            ledger.append_started("episode-attempt-002")

            with self.assertRaises(ScientificAttemptWriteError):
                ledger.register_request_id("request-001", "episode-attempt-002")
            with self.assertRaises(ScientificAttemptWriteError):
                ledger.register_request_id("request-001", ATTEMPT_ID)
            with self.assertRaises(ScientificAttemptWriteError):
                ledger.register_request_id("request-orphan", "episode-attempt-missing")

    def test_stale_writers_refresh_under_one_append_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _ledger(root)
            second = _ledger(root)

            first.append_started(ATTEMPT_ID)
            second.append_started("episode-attempt-002")
            first.register_request_id("request-001", ATTEMPT_ID)
            with self.assertRaises(ScientificAttemptWriteError):
                second.register_request_id("request-001", "episode-attempt-002")
            resumed = _ledger(root, resume=True)
            self.assertFalse(resumed.can_resume(ATTEMPT_ID))
            self.assertTrue(resumed.can_resume("episode-attempt-002"))

    def test_can_resume_refreshes_after_another_writer_terminates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _ledger(root)
            second = _ledger(root)
            first.append_started(ATTEMPT_ID)

            self.assertTrue(second.can_resume(ATTEMPT_ID))
            first.append_terminal(
                ATTEMPT_ID,
                status=AttemptStatus.COMPLETED,
                decision_count=1,
                trace_references=(_trace_reference(),),
            )

            self.assertFalse(second.can_resume(ATTEMPT_ID))

    def test_single_owner_appends_do_not_rescan_committed_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = _ledger(Path(tmp))
            with mock.patch.object(
                attempt_store.json,
                "loads",
                wraps=json.loads,
            ) as loads:
                ledger.append_started(ATTEMPT_ID)
                ledger.register_request_id("request-001", ATTEMPT_ID)
                ledger.register_request_id("request-002", ATTEMPT_ID)

            self.assertEqual(loads.call_count, 0)

    def test_terminal_and_trace_ambiguous_attempts_cannot_resume(self) -> None:
        terminal_cases = (
            (
                AttemptStatus.COMPLETED,
                1,
                (_trace_reference(),),
                None,
                None,
                None,
            ),
            (
                AttemptStatus.WRITE_AMBIGUOUS,
                0,
                (),
                TRACE_COMMIT_AMBIGUOUS,
                "trace close failed",
                TRACE_COMMIT_AMBIGUOUS,
            ),
        )
        for (
            status,
            count,
            references,
            failure_class,
            message,
            absence,
        ) in terminal_cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ledger = _ledger(root)
                ledger.append_started(ATTEMPT_ID)
                ledger.append_terminal(
                    ATTEMPT_ID,
                    status=status,
                    decision_count=count,
                    trace_references=references,
                    failure_class=failure_class,
                    failure_message=message,
                    trace_absence_reason=absence,
                )
                self.assertFalse(ledger.can_resume(ATTEMPT_ID))
                self.assertFalse(_ledger(root, resume=True).can_resume(ATTEMPT_ID))

    def test_resume_rejects_integrity_chain_and_tail_tampering(self) -> None:
        mutations = ("content", "chain", "tail")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "campaign_attempts.jsonl"
                ledger = _ledger(root)
                ledger.append_started(ATTEMPT_ID)
                ledger.register_request_id("request-001", ATTEMPT_ID)
                with path.open("rb") as handle:
                    lines = handle.readlines()
                if mutation == "tail":
                    lines[-1] = lines[-1][:-1]
                else:
                    payload = json.loads(lines[1])
                    if mutation == "content":
                        payload["request_id"] = "request-tampered"
                    else:
                        payload["previous_record_sha256"] = "sha256:" + "0" * 64
                    lines[1] = (
                        json.dumps(
                            payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
                with path.open("wb") as handle:
                    handle.writelines(lines)

                with self.assertRaises(ScientificAttemptWriteError):
                    _ledger(root, resume=True)

    def test_storage_ambiguity_poisoning_prevents_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = _ledger(root)
            with mock.patch.object(os, "fsync", side_effect=OSError("fsync failed")):
                with self.assertRaises(ScientificAttemptWriteError):
                    ledger.append_started(ATTEMPT_ID)

            self.assertFalse(ledger.can_resume(ATTEMPT_ID))
            with self.assertRaises(ScientificAttemptWriteError):
                ledger.append_started("episode-attempt-002")
            with self.assertRaises(ScientificAttemptWriteError):
                _ledger(root, resume=True)


if __name__ == "__main__":
    unittest.main()
