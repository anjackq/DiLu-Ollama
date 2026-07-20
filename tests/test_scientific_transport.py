import dataclasses
import unittest
from typing import Any

from dilu.runtime.harness_config import OutputEnforcement, ThinkMode
from dilu.runtime.ollama_transport import OllamaModelIdentity
from dilu.runtime.ollama_scientific_client import (
    OllamaScientificClient,
    build_native_chat_payload,
)
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


class RequestAndPayloadTests(unittest.TestCase):
    def test_request_and_result_are_immutable_and_capture_transport_fields(
        self,
    ) -> None:
        client = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: _FakeResponse(_success_payload()),
        )

        result = client.generate(_request())

        self.assertTrue(result.succeeded)
        self.assertTrue(result.transport_succeeded)
        self.assertEqual(result.model_tag, "qwen3:0.6b")
        self.assertEqual(result.model_digest, MODEL_DIGEST)
        self.assertEqual(result.request_id, "req-case-001-step-000")
        self.assertEqual(result.attempt_ids, ("req-case-001-step-000:a1",))
        self.assertEqual(result.native_endpoint, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(result.options.seed, 1058710636)
        self.assertEqual(result.output_enforcement, OutputEnforcement.PROMPT_ONLY)
        self.assertEqual(result.think_mode, ThinkMode.NO_THINK)
        self.assertEqual(result.raw_response, "Response to user:#### 3")
        self.assertEqual(result.contract_text, "Response to user:#### 3")
        self.assertEqual(result.stop_reason, "stop")
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (19, 7))
        self.assertEqual(result.backend_timing.total_duration_ns, 120_000_000)
        self.assertEqual(result.backend_timing.load_duration_ns, 10_000_000)
        self.assertGreaterEqual(result.latency_ms, 0.0)
        self.assertIsNone(result.error_class)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.raw_response = "changed"

    def test_total_latency_includes_pre_and_post_identity_checks(self) -> None:
        now = [0.0]

        def clock() -> float:
            return now[0]

        def inspect_identity(*args: Any, **kwargs: Any) -> OllamaModelIdentity:
            del args, kwargs
            now[0] += 0.05
            return OllamaModelIdentity("qwen3:0.6b", MODEL_DIGEST)

        def post(*args: Any, **kwargs: Any) -> _FakeResponse:
            del args, kwargs
            now[0] += 0.02
            return _FakeResponse(_success_payload())

        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            identity_inspector=inspect_identity,
            post=post,
            clock=clock,
        ).generate(_request())

        self.assertAlmostEqual(result.identity_latency_ms, 100.0)
        self.assertAlmostEqual(result.generation_latency_ms, 20.0)
        self.assertAlmostEqual(result.latency_ms, 120.0)
        self.assertEqual(
            tuple(check.phase for check in result.identity_checks),
            ("pre", "post"),
        )

    def test_prompt_only_and_schema_payloads_differ_only_by_native_format(self) -> None:
        prompt_payload = build_native_chat_payload(_request())
        schema_payload = build_native_chat_payload(
            _request(OutputEnforcement.BACKEND_SCHEMA)
        )

        schema_format = schema_payload.pop("format")
        self.assertEqual(prompt_payload, schema_payload)
        self.assertEqual(
            schema_format,
            {
                "type": "string",
                "enum": [
                    f"Response to user:#### {action_id}" for action_id in range(5)
                ],
            },
        )

    def test_native_post_disables_redirect_following(self) -> None:
        observed: dict[str, Any] = {}

        def post(*args: Any, **kwargs: Any) -> _FakeResponse:
            del args
            observed.update(kwargs)
            return _FakeResponse(_success_payload())

        result = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=post,
        ).generate(_request())

        self.assertTrue(result.succeeded)
        self.assertIs(observed["allow_redirects"], False)

    def test_schema_response_is_preserved_verbatim_without_synthetic_rewrite(
        self,
    ) -> None:
        raw_json_string = '"Response to user:#### 3"'
        client = _scientific_client(
            capabilities=_capabilities(),
            retry_policy=_retry_policy(),
            post=lambda *args, **kwargs: _FakeResponse(
                _success_payload(raw_json_string)
            ),
        )

        result = client.generate(_request(OutputEnforcement.BACKEND_SCHEMA))

        self.assertEqual(result.raw_response, raw_json_string)
        self.assertEqual(result.contract_text, "Response to user:#### 3")

    def test_scientific_request_rejects_compat_endpoint_or_auto_think(self) -> None:
        with self.assertRaises(ValueError):
            _request(endpoint="http://127.0.0.1:11434/v1/chat/completions")
        with self.assertRaises(ValueError):
            _request(think_mode=ThinkMode.AUTO)


if __name__ == "__main__":
    unittest.main()
