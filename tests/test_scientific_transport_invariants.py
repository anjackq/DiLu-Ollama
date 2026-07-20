import dataclasses
import unittest
from unittest.mock import patch

import requests

from dilu.runtime.harness_config import ThinkMode
from dilu.runtime.harness_config import OutputEnforcement
from dilu.runtime.ollama_scientific_client import (
    GenerationAttempt,
    NativeGenerationOptions,
    OllamaScientificClient,
    PreAcceptTransportUnavailable,
    ScientificTransportCapabilities,
)
from dilu.runtime.runtime_failures import RuntimeFailureClass
from tests.scientific_transport_support import (
    MODEL_DIGEST,
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_request,
    make_retry_policy,
    success_payload,
)


def _scientific_client(**kwargs: object) -> OllamaScientificClient:
    return OllamaScientificClient(
        identity_inspector=identity_inspector_for(),
        **kwargs,
    )


class RetryInvariantTests(unittest.TestCase):
    def test_two_preaccept_failures_stop_after_exactly_one_retry(self) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        sleeps: list[float] = []
        now = [0.0]

        def clock() -> float:
            return now[0]

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        def unavailable(*args: object, **kwargs: object) -> FakeResponse:
            calls.append((args, kwargs))
            raise PreAcceptTransportUnavailable("not accepted")

        result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=unavailable,
            sleep=sleep,
            clock=clock,
        ).generate(make_request())

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(sleeps, [10.0])
        self.assertEqual(result.retry_cooldown_ms, 10000.0)
        self.assertEqual(result.retry_cooldown_policy_ms, 10000.0)
        self.assertEqual(
            tuple(attempt.accepted_by_server for attempt in result.attempts),
            (False, False),
        )
        self.assertEqual(
            result.error_class,
            RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT,
        )
        self.assertTrue(result.requires_cell_abort)

        with self.assertRaises(ValueError):
            dataclasses.replace(
                result,
                retry_cooldown_ms=1.0,
                latency_ms=result.latency_ms - 9999.0,
            )

    def test_default_adapter_classifies_nested_new_connection_error(self) -> None:
        class NewConnectionError(Exception):
            pass

        connection_error = requests.ConnectionError("connection failed")
        connection_error.__cause__ = NewConnectionError("socket never connected")
        now = [0.0]

        def clock() -> float:
            return now[0]

        def sleep(seconds: float) -> None:
            now[0] += seconds

        with patch(
            "dilu.runtime.ollama_scientific_client.requests.post",
            side_effect=connection_error,
        ):
            result = _scientific_client(
                capabilities=make_capabilities(),
                retry_policy=make_retry_policy(),
                sleep=sleep,
                clock=clock,
            ).generate(make_request())

        self.assertEqual(len(result.attempts), 2)
        self.assertTrue(
            all(
                attempt.error_class
                is RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT
                for attempt in result.attempts
            )
        )

    def test_non_success_redirect_status_is_transport_drift(self) -> None:
        result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=lambda *args, **kwargs: FakeResponse({}, status_code=302),
        ).generate(make_request())

        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.error_class, RuntimeFailureClass.TRANSPORT_DRIFT)


class ImmutableTypeInvariantTests(unittest.TestCase):
    def test_nonfinite_generation_controls_are_rejected(self) -> None:
        for temperature in (float("nan"), float("inf")):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    NativeGenerationOptions(
                        seed=1,
                        temperature=temperature,
                        num_ctx=4096,
                        num_predict=64,
                    )

        request = make_request()
        with self.assertRaises(ValueError):
            dataclasses.replace(request, timeout_sec=float("inf"))
        with self.assertRaises(ValueError):
            dataclasses.replace(request.options, seed=1 << 32)

    def test_attempt_rejects_success_without_raw_response(self) -> None:
        with self.assertRaises(ValueError):
            GenerationAttempt(
                request_id="req-1",
                attempt_id="req-1:a1",
                attempt_index=1,
                accepted_by_server=True,
                http_status=200,
                response_body='{"message":{"content":"x"}}',
                raw_response=None,
                contract_text=None,
                transport_error_body=None,
                thinking_response="",
                stop_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                backend_timing=None,
                latency_ms=1.0,
                error_class=None,
                error_message=None,
            )

    def test_capability_schema_mechanism_is_frozen(self) -> None:
        with self.assertRaises(ValueError):
            ScientificTransportCapabilities(
                model_tag="qwen3:0.6b",
                model_digest=MODEL_DIGEST,
                native_endpoint="http://127.0.0.1:11434/api/chat",
                supported_think_modes=(ThinkMode.NO_THINK,),
                seed_verified=True,
                schema_verified=True,
                capability_probe_id="s1-probe",
                capability_artifact_hash="sha256:" + "c" * 64,
                schema_mechanism="unregistered_schema",
            )

    def test_capability_think_modes_must_be_immutable(self) -> None:
        with self.assertRaises(ValueError):
            dataclasses.replace(
                make_capabilities(),
                supported_think_modes=[ThinkMode.NO_THINK],
            )

    def test_result_rejects_attempt_chain_from_another_request(self) -> None:
        result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=lambda *args, **kwargs: FakeResponse(success_payload()),
        ).generate(make_request())
        foreign_attempt = dataclasses.replace(
            result.attempts[0],
            request_id="foreign-request",
        )

        with self.assertRaises(ValueError):
            dataclasses.replace(result, attempts=(foreign_attempt,))

    def test_result_rejects_fabricated_summary_fields(self) -> None:
        result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=lambda *args, **kwargs: FakeResponse(success_payload()),
        ).generate(make_request())

        invalid_replacements = (
            {"error_message": "fabricated"},
            {"latency_ms": result.latency_ms + 1.0},
            {"retry_cooldown_ms": 10000.0},
            {"retry_cooldown_policy_ms": 10000.0},
        )
        for replacement in invalid_replacements:
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    dataclasses.replace(result, **replacement)

    def test_result_rejects_retry_after_success(self) -> None:
        result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=lambda *args, **kwargs: FakeResponse(success_payload()),
        ).generate(make_request())
        second = dataclasses.replace(
            result.attempts[0],
            attempt_id=f"{result.request_id}:a2",
            attempt_index=2,
        )

        with self.assertRaises(ValueError):
            dataclasses.replace(result, attempts=(result.attempts[0], second))

    def test_result_binds_request_contract_and_raw_response_evidence(self) -> None:
        prompt_result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=lambda *args, **kwargs: FakeResponse(success_payload()),
        ).generate(make_request())
        prompt_attempt = prompt_result.attempts[0]
        invalid_prompt_replacements = (
            {
                "attempts": (
                    dataclasses.replace(
                        prompt_attempt,
                        thinking_response="forged reasoning",
                    ),
                ),
                "thinking_response": "forged reasoning",
            },
            {
                "attempts": (
                    dataclasses.replace(prompt_attempt, raw_response="forged"),
                ),
                "raw_response": "forged",
            },
            {
                "attempts": (
                    dataclasses.replace(prompt_attempt, response_body="forged"),
                ),
                "response_body": "forged",
            },
            {
                "attempts": (
                    dataclasses.replace(
                        prompt_attempt,
                        contract_text=None,
                        error_class=RuntimeFailureClass.SCHEMA_REJECTION,
                        error_message="forged schema rejection",
                    ),
                ),
                "contract_text": None,
                "error_class": RuntimeFailureClass.SCHEMA_REJECTION,
                "error_message": "forged schema rejection",
            },
        )
        for replacement in invalid_prompt_replacements:
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    dataclasses.replace(prompt_result, **replacement)

        schema_result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=lambda *args, **kwargs: FakeResponse(
                success_payload('"Response to user:#### 3"')
            ),
        ).generate(make_request(OutputEnforcement.BACKEND_SCHEMA))
        with self.assertRaises(ValueError):
            dataclasses.replace(
                schema_result,
                attempts=(
                    dataclasses.replace(
                        schema_result.attempts[0],
                        raw_response="not-json",
                    ),
                ),
                raw_response="not-json",
            )

    def test_model_output_and_transport_error_body_are_mutually_exclusive(self) -> None:
        result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=lambda *args, **kwargs: FakeResponse(
                success_payload('"Response to user:#### 3\\n"')
            ),
        ).generate(make_request(OutputEnforcement.BACKEND_SCHEMA))

        self.assertEqual(result.error_class, RuntimeFailureClass.SCHEMA_REJECTION)

        with self.assertRaises(ValueError):
            dataclasses.replace(
                result.attempts[0],
                transport_error_body='{"error":"transport"}',
            )

    def test_success_stop_reason_must_be_canonical_text(self) -> None:
        result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=lambda *args, **kwargs: FakeResponse(success_payload()),
        ).generate(make_request())

        for invalid in ("   ", 7):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    dataclasses.replace(result.attempts[0], stop_reason=invalid)

    def test_success_result_cannot_drop_identity_check_chain(self) -> None:
        result = _scientific_client(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            post=lambda *args, **kwargs: FakeResponse(success_payload()),
        ).generate(make_request())

        with self.assertRaises(ValueError):
            dataclasses.replace(
                result,
                identity_checks=(),
                identity_latency_ms=0.0,
                latency_ms=result.generation_latency_ms,
            )

    def test_malformed_accepted_payload_preserves_complete_response_body(self) -> None:
        payloads = (
            {
                **success_payload(),
                "message": {
                    "role": "user",
                    "content": "Response to user:#### 3",
                    "thinking": "",
                },
            },
            {**success_payload(), "eval_count": 0},
            {**success_payload(), "total_duration": "invalid"},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                response = FakeResponse(payload)
                result = _scientific_client(
                    capabilities=make_capabilities(),
                    retry_policy=make_retry_policy(),
                    post=lambda *args, **kwargs: response,
                ).generate(make_request())

                self.assertEqual(
                    result.error_class,
                    RuntimeFailureClass.TRANSPORT_DRIFT,
                )
                self.assertEqual(result.response_body, response.text)
                self.assertEqual(result.attempts[0].response_body, response.text)


if __name__ == "__main__":
    unittest.main()
