from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dilu.runtime as runtime_api
from tests.runtime_factorization_support import run_episode, runtime


class ScientificRuntimeCompletionPublicationTests(unittest.TestCase):
    def test_completion_publisher_rejects_result_trace_reference_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bound_runtime = runtime(Path(tmp))
            publisher = mock.Mock()
            bound_runtime = dataclasses.replace(
                bound_runtime,
                completion_publisher=publisher,
            )
            bound_runtime.begin_attempt()

            with self.assertRaisesRegex(ValueError, "trace references"):
                bound_runtime.complete_attempt(
                    (),
                    result={
                        "scientific_trace_references": [{"relative_path": "wrong"}]
                    },
                )

        publisher.assert_not_called()

    def test_completion_publisher_observes_started_attempt_and_exact_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bound_runtime = runtime(root)
            published: list[tuple[dict[str, object], tuple[object, ...]]] = []

            def publish_completion(
                result: dict[str, object],
                references: tuple[object, ...],
            ) -> None:
                self.assertIs(
                    bound_runtime.attempt_ledger.attempt_status(
                        bound_runtime.identity.episode_attempt_id
                    ),
                    runtime_api.AttemptStatus.STARTED,
                )
                published.append((dict(result), references))
                bound_runtime.attempt_ledger.append_terminal(
                    bound_runtime.identity.episode_attempt_id,
                    status=runtime_api.AttemptStatus.COMPLETED,
                    decision_count=len(references),
                    trace_references=references,
                )

            bound_runtime = dataclasses.replace(
                bound_runtime,
                completion_publisher=publish_completion,
            )

            result = run_episode(root, bound_runtime)

            final_status = bound_runtime.attempt_ledger.attempt_status(
                bound_runtime.identity.episode_attempt_id
            )

        self.assertEqual(len(published), 1)
        self.assertEqual(
            published[0][0]["scientific_trace_references"],
            [reference.to_dict() for reference in published[0][1]],
        )
        self.assertEqual(
            result["scientific_trace_references"],
            published[0][0]["scientific_trace_references"],
        )
        self.assertIs(
            final_status,
            runtime_api.AttemptStatus.COMPLETED,
        )


if __name__ == "__main__":
    unittest.main()
