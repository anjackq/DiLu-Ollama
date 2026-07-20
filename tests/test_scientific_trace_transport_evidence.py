from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime._scientific_trace_hashing import capability_snapshot_sha256
from dilu.runtime.ollama_scientific_client import (
    OllamaScientificClient,
    PreAcceptTransportUnavailable,
)
from dilu.runtime.runtime_failures import RuntimeFailureClass
from dilu.runtime.scientific_trace import (
    DecisionTraceRecord,
    ScientificTraceWriteError,
    ScientificTraceWriter,
    TraceDisposition,
)
from dilu.runtime.scientific_transport_records import GenerationResult
from tests.scientific_transport_support import (
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_retry_policy,
    success_payload,
)
from tests.test_scientific_trace import _record


def _blocked_record(generation: GenerationResult) -> DecisionTraceRecord:
    base = _record()
    return DecisionTraceRecord(
        context=base.context,
        harness_config=base.harness_config,
        prompt_artifact=base.prompt_artifact,
        generation=generation,
        resolution=None,
        shield_stack=None,
        disposition=TraceDisposition.BLOCKED_BEFORE_EXECUTION,
        decision_latency_ms=generation.latency_ms,
    )


class ScientificTraceTransportEvidenceTests(unittest.TestCase):
    def test_capability_preflight_block_rejects_forged_verified_snapshot(
        self,
    ) -> None:
        base = _record()
        generation = OllamaScientificClient(
            capabilities=make_capabilities(seed_verified=False),
            retry_policy=make_retry_policy(),
            identity_inspector=identity_inspector_for(),
            post=mock.Mock(side_effect=AssertionError("HTTP must not be called")),
            sleep=lambda _: None,
        ).generate(base.generation.request)
        record = _blocked_record(generation)

        evidence = record.to_dict()["generation"]["transport_evidence"]
        self.assertIsNone(evidence["effective_profile"])
        self.assertIsNone(evidence["effective_think_mode"])
        self.assertFalse(evidence["seed_verified"])
        with self.assertRaises(ValueError):
            dataclasses.replace(generation, capabilities=make_capabilities())
        with tempfile.TemporaryDirectory() as tmp:
            ScientificTraceWriter(
                Path(tmp) / "decision_traces.jsonl",
                artifact_root=Path(tmp),
            ).append(record)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            tampered = record.to_dict()
            evidence = tampered["generation"]["transport_evidence"]
            evidence["seed_verified"] = True
            evidence["capability_snapshot_sha256"] = capability_snapshot_sha256(
                evidence
            )
            path.write_text(
                json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ScientificTraceWriteError):
                ScientificTraceWriter(path, artifact_root=root, resume=True)

    def test_identity_preflight_block_remains_valid(self) -> None:
        base = _record()
        generation = OllamaScientificClient(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            identity_inspector=identity_inspector_for("sha256:" + "b" * 64),
            post=mock.Mock(side_effect=AssertionError("HTTP must not be called")),
            sleep=lambda _: None,
        ).generate(base.generation.request)
        record = _blocked_record(generation)

        self.assertEqual(len(generation.identity_checks), 1)
        self.assertFalse(generation.identity_checks[0].succeeded)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            ).append(record)

    def test_no_think_leakage_does_not_claim_effective_think_mode(self) -> None:
        base = _record()
        payload = success_payload()
        payload["message"]["thinking"] = "forged reasoning"
        generation = OllamaScientificClient(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            identity_inspector=identity_inspector_for(),
            post=lambda *args, **kwargs: FakeResponse(payload),
            sleep=lambda _: None,
        ).generate(base.generation.request)
        evidence = _blocked_record(generation).to_dict()["generation"][
            "transport_evidence"
        ]

        self.assertEqual(generation.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)
        self.assertEqual(evidence["effective_profile"], "ollama_native_chat")
        self.assertIsNone(evidence["effective_think_mode"])

    def test_unaccepted_retry_has_no_synthetic_effective_transport(self) -> None:
        base = _record()
        now = [0.0]

        def clock() -> float:
            return now[0]

        def sleep(seconds: float) -> None:
            now[0] += seconds

        def unavailable(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise PreAcceptTransportUnavailable("not accepted")

        generation = OllamaScientificClient(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            identity_inspector=identity_inspector_for(),
            post=unavailable,
            sleep=sleep,
            clock=clock,
        ).generate(base.generation.request)
        record = _blocked_record(generation)
        evidence = record.to_dict()["generation"]["transport_evidence"]
        self.assertIsNone(evidence["effective_profile"])
        self.assertIsNone(evidence["effective_think_mode"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            payload = record.to_dict()
            generation_payload = payload["generation"]
            generation_payload["retry_cooldown_ms"] = 1.0
            generation_payload["latency_ms"] -= 9999.0
            payload["decision_latency_ms"] -= 9999.0
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ScientificTraceWriteError):
                ScientificTraceWriter(path, artifact_root=root, resume=True)

    def test_timeout_with_unknown_acceptance_has_no_effective_transport(self) -> None:
        payload = _record(timeout=True).to_dict()
        attempt = payload["generation"]["attempts"][0]
        evidence = payload["generation"]["transport_evidence"]

        self.assertIsNone(attempt["accepted_by_server"])
        self.assertIsNone(evidence["effective_profile"])
        self.assertIsNone(evidence["effective_think_mode"])


if __name__ == "__main__":
    unittest.main()
