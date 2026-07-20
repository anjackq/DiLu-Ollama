from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dilu.runtime._campaign_attempt_io import lock_path_for
from dilu.runtime.campaign_attempts import (
    AttemptStatus,
    ScientificAttemptLedger,
    ScientificAttemptWriteError,
)
from dilu.runtime.scientific_trace import ScientificTraceWriter
from tests.test_scientific_trace import _record


class CampaignTraceEvidenceTests(unittest.TestCase):
    def test_busy_ledger_is_not_reported_as_nonresumable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "campaign_attempts.jsonl"
            ledger = ScientificAttemptLedger(
                ledger_path,
                campaign_id="campaign-001",
            )
            ledger.append_started("episode-attempt-001")
            lock_path_for(ledger_path).write_text("busy", encoding="utf-8")

            with self.assertRaises(ScientificAttemptWriteError):
                ledger.can_resume("episode-attempt-001")

    def test_terminal_references_join_to_validated_trace_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            ledger = ScientificAttemptLedger(
                root / "campaign_attempts.jsonl",
                campaign_id="campaign-001",
            )
            first_reference = writer.append(_record())
            second_reference = writer.append(_record(1))
            ledger.append_started("episode-attempt-001")
            ledger.append_terminal(
                "episode-attempt-001",
                status=AttemptStatus.COMPLETED,
                decision_count=2,
                trace_references=(first_reference, second_reference),
            )

            ledger.validate_trace_evidence(writer)

    def test_terminal_cannot_reference_trace_owned_by_another_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            ledger = ScientificAttemptLedger(
                root / "campaign_attempts.jsonl",
                campaign_id="campaign-001",
            )
            other_attempt_reference = writer.append(_record())
            ledger.append_started("episode-attempt-002")
            ledger.append_terminal(
                "episode-attempt-002",
                status=AttemptStatus.COMPLETED,
                decision_count=1,
                trace_references=(other_attempt_reference,),
            )

            with self.assertRaisesRegex(
                ScientificAttemptWriteError,
                "ordered trace evidence",
            ):
                ledger.validate_trace_evidence(writer)

    def test_terminal_trace_references_must_preserve_decision_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            ledger = ScientificAttemptLedger(
                root / "campaign_attempts.jsonl",
                campaign_id="campaign-001",
            )
            first_reference = writer.append(_record())
            second_reference = writer.append(_record(1))
            ledger.append_started("episode-attempt-001")
            ledger.append_terminal(
                "episode-attempt-001",
                status=AttemptStatus.COMPLETED,
                decision_count=2,
                trace_references=(second_reference, first_reference),
            )

            with self.assertRaisesRegex(
                ScientificAttemptWriteError,
                "ordered trace evidence",
            ):
                ledger.validate_trace_evidence(writer)

    def test_missing_trace_artifact_blocks_campaign_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_path = root / "decision_traces.jsonl"
            writer = ScientificTraceWriter(trace_path, artifact_root=root)
            ledger = ScientificAttemptLedger(
                root / "campaign_attempts.jsonl",
                campaign_id="campaign-001",
            )
            reference = writer.append(_record())
            ledger.append_started("episode-attempt-001")
            ledger.append_terminal(
                "episode-attempt-001",
                status=AttemptStatus.COMPLETED,
                decision_count=1,
                trace_references=(reference,),
            )
            trace_path.unlink()

            with self.assertRaises(ScientificAttemptWriteError):
                ledger.validate_trace_evidence(writer)


if __name__ == "__main__":
    unittest.main()
