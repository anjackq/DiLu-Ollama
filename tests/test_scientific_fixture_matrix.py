from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import requests

from dilu.runtime.harness_config import OutputEnforcement, resolve_main_conditions
from dilu.runtime.ollama_scientific_client import (
    OllamaScientificClient,
    PreAcceptTransportUnavailable,
    ScientificGenerationAbort,
)
from tests.runtime_factorization_support import (
    run_episode,
    runtime,
    terminal_attempt_payload,
)
from tests.scientific_transport_support import (
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_retry_policy,
    success_payload,
)
from tests.test_scientific_driver_action_resolution import _scientific_config


def _post_for(outcome: str) -> Callable[..., FakeResponse]:
    if outcome == "strict":
        return lambda *args, **kwargs: FakeResponse(success_payload())
    if outcome == "recoverable":
        return lambda *args, **kwargs: FakeResponse(success_payload("3"))
    if outcome == "invalid":
        return lambda *args, **kwargs: FakeResponse(
            success_payload("choose the left lane")
        )
    if outcome == "empty":
        return lambda *args, **kwargs: FakeResponse(success_payload("", eval_count=0))
    if outcome == "timeout":
        return lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.Timeout("timeout")
        )
    if outcome == "unavailable":
        return lambda *args, **kwargs: (_ for _ in ()).throw(
            PreAcceptTransportUnavailable("connection refused")
        )
    if outcome == "schema_rejected":
        return lambda *args, **kwargs: FakeResponse(
            {"error": "invalid format schema"},
            status_code=400,
        )
    raise ValueError(f"Unknown fixture outcome: {outcome}")


def _client(outcome: str) -> OllamaScientificClient:
    now = [0.0]

    def post(*args: Any, **kwargs: Any) -> FakeResponse:
        return _post_for(outcome)(*args, **kwargs)

    def sleep(seconds: float) -> None:
        now[0] += seconds

    return OllamaScientificClient(
        capabilities=make_capabilities(),
        retry_policy=make_retry_policy(),
        identity_inspector=identity_inspector_for(),
        post=post,
        sleep=sleep,
        clock=lambda: now[0],
    )


class ScientificFixtureMatrixTests(unittest.TestCase):
    def test_fixture_matrix_preserves_success_fallback_and_abort_classes(self) -> None:
        outcomes = (
            "strict",
            "recoverable",
            "invalid",
            "empty",
            "timeout",
            "unavailable",
            "schema_rejected",
        )
        for index, outcome in enumerate(outcomes):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                enforcement = (
                    OutputEnforcement.BACKEND_SCHEMA
                    if outcome == "schema_rejected"
                    else OutputEnforcement.PROMPT_ONLY
                )
                config = next(
                    item
                    for item in resolve_main_conditions(_scientific_config())
                    if item.condition.output_enforcement is enforcement
                )
                bound_runtime = runtime(
                    root,
                    attempt=index,
                    config=config,
                    transport_client=_client(outcome),
                )
                if outcome not in {"unavailable", "schema_rejected"}:
                    result = run_episode(root, bound_runtime)
                    expected_fallbacks = 0 if outcome == "strict" else 1
                    self.assertEqual(
                        result["fallback_action_count"],
                        expected_fallbacks,
                    )
                else:
                    with self.assertRaises(ScientificGenerationAbort):
                        run_episode(root, bound_runtime)
                    lifecycle = terminal_attempt_payload(root)
                    self.assertEqual(lifecycle["status"], "blocked")
                trace = json.loads(
                    (root / "decision_traces.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()[0]
                )
                expected_failure = {
                    "strict": None,
                    "recoverable": "syntax_invalid",
                    "invalid": "syntax_invalid",
                    "empty": "model_empty_output",
                    "timeout": "generation_timeout",
                    "unavailable": "transport_unavailable_before_accept",
                    "schema_rejected": "schema_rejection",
                }[outcome]
                actual_failure = (
                    trace["failure"]["failure_class"]
                    if trace["failure"] is not None
                    else None
                )
                self.assertEqual(actual_failure, expected_failure)


if __name__ == "__main__":
    unittest.main()
