import unittest
from collections.abc import Callable
from typing import Any

import requests

from dilu.runtime.harness_config import OutputEnforcement
from dilu.runtime.ollama_transport import OllamaModelIdentity
from dilu.runtime.ollama_scientific_client import (
    OllamaScientificClient,
    PreAcceptTransportUnavailable,
    is_retryable_failure,
)
from dilu.runtime.runtime_failures import RuntimeFailureClass
from tests.scientific_transport_support import (
    MODEL_DIGEST,
    FakeResponse as _FakeResponse,
    identity_inspector_for as _identity_inspector_for,
    make_capabilities as _capabilities,
    make_request as _request,
    make_retry_policy as _retry_policy,
    success_payload as _success_payload,
)


def _scientific_client(**kwargs: Any) -> OllamaScientificClient:
    identity_inspector = kwargs.pop(
        "identity_inspector",
        _identity_inspector_for(),
    )
    return OllamaScientificClient(
        identity_inspector=identity_inspector,
        **kwargs,
    )


class RetryAndFailureTests(unittest.TestCase):
    def test_one_retry_only_for_transport_unavailable_before_accept(self) -> None:
        calls: list[int] = []
        sleeps: list[float] = []
        now = [0.0]

        def clock() -> float:
            return now[0]

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        def post(*args: Any, **kwargs: Any) -> _FakeResponse:
            del args, kwargs
            calls.append(1)
            if len(calls) == 1:
                raise PreAcceptTransportUnavailable("connection refused")
            return _FakeResponse(_success_payload())

        client = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=post,
            sleep=sleep,
            clock=clock,
        )

        result = client.generate(_request())

        self.assertTrue(result.succeeded)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [10.0])
        self.assertEqual(result.retry_cooldown_ms, 10000.0)
        self.assertEqual(result.retry_cooldown_policy_ms, 10000.0)
        self.assertGreaterEqual(result.latency_ms, result.retry_cooldown_ms)
        self.assertEqual(
            tuple(attempt.error_class for attempt in result.attempts),
            (RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT, None),
        )
        self.assertEqual(
            result.attempt_ids,
            ("req-case-001-step-000:a1", "req-case-001-step-000:a2"),
        )

    def test_timeout_empty_and_schema_rejection_are_not_retried(self) -> None:
        cases: tuple[
            tuple[str, Callable[..., _FakeResponse], RuntimeFailureClass], ...
        ] = (
            (
                "timeout",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    requests.Timeout("timeout")
                ),
                RuntimeFailureClass.GENERATION_TIMEOUT,
            ),
            (
                "empty",
                lambda *args, **kwargs: _FakeResponse(
                    _success_payload(" \t", eval_count=0)
                ),
                RuntimeFailureClass.MODEL_EMPTY_OUTPUT,
            ),
            (
                "schema",
                lambda *args, **kwargs: _FakeResponse(
                    {"error": "invalid format schema"}, status_code=400
                ),
                RuntimeFailureClass.SCHEMA_REJECTION,
            ),
        )
        for name, post, expected_failure in cases:
            with self.subTest(name=name):
                client = _scientific_client(
                    capabilities=_capabilities(),
                    retry_policy=_retry_policy(),
                    post=post,
                    sleep=lambda _: self.fail("Non-retryable failure slept."),
                )
                request = _request(
                    OutputEnforcement.BACKEND_SCHEMA
                    if name == "schema"
                    else OutputEnforcement.PROMPT_ONLY
                )
                result = client.generate(request)
                self.assertEqual(len(result.attempts), 1)
                self.assertEqual(result.error_class, expected_failure)
                if name == "timeout":
                    self.assertIsNone(result.raw_response)
                    self.assertTrue(result.operational_fallback_allowed)
                if name == "empty":
                    self.assertTrue(result.transport_succeeded)
                    self.assertTrue(result.operational_fallback_allowed)
                if name == "schema":
                    self.assertFalse(result.operational_fallback_allowed)
                    self.assertIn("format schema", result.transport_error_body)

    def test_unverified_http_400_is_transport_drift_with_error_body(self) -> None:
        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: _FakeResponse(
                {"error": "model is not available"}, status_code=400
            ),
        ).generate(_request(OutputEnforcement.BACKEND_SCHEMA))

        self.assertEqual(result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)
        self.assertIn("model is not available", result.transport_error_body)

    def test_schema_keyword_incidental_to_model_error_is_transport_drift(self) -> None:
        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: _FakeResponse(
                {
                    "error": (
                        "model unavailable; requested output format was not evaluated"
                    )
                },
                status_code=400,
            ),
        ).generate(_request(OutputEnforcement.BACKEND_SCHEMA))

        self.assertEqual(result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)

    def test_generation_response_with_redirect_history_is_transport_drift(self) -> None:
        response = _FakeResponse(_success_payload())
        response.history = [object()]
        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: response,
        ).generate(_request())

        self.assertEqual(result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)
        self.assertIsNone(result.raw_response)

    def test_malformed_2xx_content_preserves_transport_body(self) -> None:
        payload = _success_payload()
        payload["message"]["content"] = {"unexpected": "object"}
        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: _FakeResponse(payload),
        ).generate(_request())

        self.assertEqual(result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)
        self.assertIsNone(result.raw_response)
        self.assertIn("unexpected", result.transport_error_body)

    def test_post_generation_digest_drift_preserves_raw_and_aborts(self) -> None:
        inspections = [
            OllamaModelIdentity("qwen3:0.6b", MODEL_DIGEST),
            OllamaModelIdentity("qwen3:0.6b", "sha256:" + "b" * 64),
        ]

        def inspect_identity(*args: Any, **kwargs: Any) -> OllamaModelIdentity:
            del args, kwargs
            return inspections.pop(0)

        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            identity_inspector=inspect_identity,
            post=lambda *args, **kwargs: _FakeResponse(_success_payload()),
        ).generate(_request())

        self.assertEqual(result.raw_response, "Response to user:#### 3")
        self.assertIsNone(result.contract_text)
        self.assertEqual(result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)
        self.assertIn("post_generation", result.error_message)
        self.assertTrue(result.requires_cell_abort)

    def test_generic_connection_error_is_not_assumed_pre_accept(self) -> None:
        client = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: (_ for _ in ()).throw(
                requests.ConnectionError("accept boundary unknown")
            ),
            sleep=lambda _: self.fail("Ambiguous connection errors cannot retry."),
        )

        result = client.generate(_request())

        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)

    def test_payload_validity_drift_is_fail_closed_without_retry(self) -> None:
        invalid_payloads = (
            {**_success_payload(), "done": False},
            {**_success_payload(), "prompt_eval_count": None},
            {**_success_payload(), "prompt_eval_count": 0},
            {**_success_payload(), "eval_count": 0},
            {
                **_success_payload(),
                "message": {
                    "role": "user",
                    "content": "Response to user:#### 3",
                    "thinking": "",
                },
            },
            {
                **_success_payload(),
                "message": {
                    "content": "Response to user:#### 3",
                    "thinking": "hidden reasoning",
                },
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                result = _scientific_client(
                    capabilities=_capabilities(),
                    retry_policy=_retry_policy(),
                    post=lambda *args, payload=payload, **kwargs: _FakeResponse(
                        payload
                    ),
                ).generate(_request())
                self.assertEqual(len(result.attempts), 1)
                self.assertEqual(
                    result.error_class,
                    RuntimeFailureClass.TRANSPORT_DRIFT,
                )
                self.assertFalse(result.operational_fallback_allowed)
                if isinstance(payload.get("message", {}).get("content"), str):
                    self.assertEqual(
                        result.raw_response,
                        payload["message"]["content"],
                    )

    def test_prompt_only_invalid_syntax_remains_transport_success(self) -> None:
        raw = "Action id: 3"
        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: _FakeResponse(_success_payload(raw)),
        ).generate(_request())

        self.assertTrue(result.transport_succeeded)
        self.assertTrue(result.succeeded)
        self.assertEqual((result.raw_response, result.contract_text), (raw, raw))

    def test_schema_decoding_rejects_noncanonical_generated_string(self) -> None:
        raw = '"Response to user:#### 3\\n"'
        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: _FakeResponse(_success_payload(raw)),
        ).generate(_request(OutputEnforcement.BACKEND_SCHEMA))

        self.assertEqual(result.raw_response, raw)
        self.assertIsNone(result.contract_text)
        self.assertEqual(result.error_class, RuntimeFailureClass.SCHEMA_REJECTION)

    def test_only_pre_accept_transport_unavailability_is_retryable(self) -> None:
        non_retryable = (
            RuntimeFailureClass.GENERATION_TIMEOUT,
            RuntimeFailureClass.MODEL_EMPTY_OUTPUT,
            RuntimeFailureClass.SCHEMA_REJECTION,
            RuntimeFailureClass.SYNTAX_INVALID,
            RuntimeFailureClass.ACTION_UNAVAILABLE,
            RuntimeFailureClass.TRANSPORT_DRIFT,
        )

        self.assertTrue(
            is_retryable_failure(
                RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT
            )
        )
        for failure in non_retryable:
            self.assertFalse(is_retryable_failure(failure))

    def test_capability_or_identity_drift_fails_before_http(self) -> None:
        calls: list[int] = []

        def post(*args: Any, **kwargs: Any) -> _FakeResponse:
            del args, kwargs
            calls.append(1)
            return _FakeResponse(_success_payload())

        mismatches = (
            (_request(digest="sha256:" + "b" * 64), _capabilities()),
            (_request(), _capabilities(seed_verified=False)),
            (
                _request(OutputEnforcement.BACKEND_SCHEMA),
                _capabilities(schema_verified=False),
            ),
        )
        for request, capabilities in mismatches:
            with self.subTest(request=request, capabilities=capabilities):
                result = _scientific_client(
                    capabilities=capabilities,
                    retry_policy=_retry_policy(),
                    post=post,
                ).generate(request)
                self.assertEqual(
                    result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT
                )
                self.assertEqual(result.attempts, ())
        self.assertEqual(calls, [])

    def test_current_model_digest_drift_fails_before_generation_http(self) -> None:
        calls: list[int] = []

        def post(*args: Any, **kwargs: Any) -> _FakeResponse:
            del args, kwargs
            calls.append(1)
            return _FakeResponse(_success_payload())

        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=post,
            identity_inspector=lambda *_, **__: OllamaModelIdentity(
                model_tag="qwen3:0.6b",
                model_digest="sha256:" + "b" * 64,
            ),
        ).generate(_request())

        self.assertEqual(result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)
        self.assertEqual(result.error_message, "current_model_digest_drift")
        self.assertEqual(result.attempts, ())
        self.assertEqual(calls, [])

    def test_response_model_tag_drift_is_not_retried(self) -> None:
        payload = _success_payload()
        payload["model"] = "qwen3:latest"
        client = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: _FakeResponse(payload),
        )

        result = client.generate(_request())

        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)


if __name__ == "__main__":
    unittest.main()
