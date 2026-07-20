from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime.campaign_attempts import AttemptStatus
from dilu.runtime.scientific_trace import (
    ScientificTraceValidationError,
    ScientificTraceWriteError,
)
from tests.runtime_factorization_support import (
    FakeEnvironment,
    SecondAppendAmbiguousTraceWriter,
    run_episode,
    runtime,
    terminal_attempt_payload,
)


class AttemptLifecycleAmbiguityTest(unittest.TestCase):
    def test_second_ambiguous_append_retains_durable_trace_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = SecondAppendAmbiguousTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            bound_runtime = runtime(root, writer=writer)

            with self.assertRaises(ScientificTraceWriteError):
                run_episode(
                    root,
                    bound_runtime,
                    environment=FakeEnvironment(terminate_after=2),
                    max_steps_override=2,
                )

            lifecycle = terminal_attempt_payload(root)
            self.assertEqual(
                lifecycle["status"],
                AttemptStatus.WRITE_AMBIGUOUS.value,
            )
            self.assertEqual(lifecycle["decision_count"], 1)
            self.assertEqual(len(lifecycle["trace_references"]), 1)
            self.assertEqual(lifecycle["trace_references"][0]["line_number"], 1)
            self.assertFalse(
                bound_runtime.attempt_ledger.can_resume(
                    bound_runtime.identity.episode_attempt_id
                )
            )

    def test_trace_finalization_failure_writes_terminal_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bound_runtime = runtime(root)

            with mock.patch.object(
                type(bound_runtime),
                "current_trace_references",
                side_effect=ScientificTraceValidationError("trace finalization failed"),
            ):
                with self.assertRaises(ScientificTraceValidationError):
                    run_episode(root, bound_runtime)

            lifecycle = terminal_attempt_payload(root)
            self.assertEqual(lifecycle["status"], AttemptStatus.FAILED.value)
            self.assertEqual(lifecycle["decision_count"], 1)
            self.assertEqual(len(lifecycle["trace_references"]), 1)
            self.assertFalse(
                bound_runtime.attempt_ledger.can_resume(
                    bound_runtime.identity.episode_attempt_id
                )
            )

    def test_completion_validation_failure_writes_terminal_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bound_runtime = runtime(root)

            with mock.patch.object(
                type(bound_runtime),
                "complete_attempt",
                side_effect=ValueError("completion rejected"),
            ):
                with self.assertRaises(ValueError):
                    run_episode(root, bound_runtime)

            lifecycle = terminal_attempt_payload(root)
            self.assertEqual(lifecycle["status"], AttemptStatus.FAILED.value)
            self.assertEqual(lifecycle["decision_count"], 1)
            self.assertEqual(len(lifecycle["trace_references"]), 1)
            self.assertFalse(
                bound_runtime.attempt_ledger.can_resume(
                    bound_runtime.identity.episode_attempt_id
                )
            )


if __name__ == "__main__":
    unittest.main()
