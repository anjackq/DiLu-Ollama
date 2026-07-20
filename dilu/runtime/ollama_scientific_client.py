from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import requests

from ._scientific_transport_checks import (
    PreAcceptTransportUnavailable,
    is_verified_schema_rejection as _is_verified_schema_rejection,
    requests_post_with_preaccept_classification as _requests_post_with_preaccept_classification,
    response_text as _response_text,
    transport_preflight_error as _transport_preflight_error,
)
from ._scientific_transport_response import (
    blocked_result,
    failed_attempt,
    parse_native_response_attempt,
    result_from_attempts,
)
from .harness_config import OutputEnforcement, RetryPolicy
from .ollama_transport import OllamaModelIdentity
from .runtime_failures import RuntimeFailureClass
from .scientific_transport_records import GenerationAttempt, GenerationResult
from .scientific_transport_evidence import ModelIdentityCheck
from .scientific_transport_types import (
    GenerationRequest,
    NativeGenerationOptions,
    ScientificGenerationContext,
    ScientificTransportCapabilities,
    build_native_chat_payload,
    canonical_action_text_schema,
)


class ScientificGenerationAbort(RuntimeError):
    def __init__(
        self,
        result: GenerationResult,
        *,
        trace_reference: Any | None = None,
    ) -> None:
        if not isinstance(result, GenerationResult) or not result.requires_cell_abort:
            raise ValueError(
                "ScientificGenerationAbort requires a cell-aborting result."
            )
        if trace_reference is not None:
            from ._scientific_trace_store import TraceReference

            if not isinstance(trace_reference, TraceReference):
                raise ValueError("trace_reference must be a TraceReference or None.")
        super().__init__(f"Scientific generation aborted: {result.error_class.value}")
        self.result = result
        self.trace_reference = trace_reference


class ScientificGenerationTimeout(TimeoutError):
    def __init__(self, result: GenerationResult) -> None:
        if (
            not isinstance(result, GenerationResult)
            or result.error_class is not RuntimeFailureClass.GENERATION_TIMEOUT
            or not result.operational_fallback_allowed
        ):
            raise ValueError(
                "ScientificGenerationTimeout requires a typed timeout result."
            )
        super().__init__(result.error_message or "scientific generation timeout")
        self.result = result


class OllamaScientificClient:
    def __init__(
        self,
        *,
        capabilities: ScientificTransportCapabilities,
        retry_policy: RetryPolicy,
        identity_inspector: Callable[..., OllamaModelIdentity],
        post: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(capabilities, ScientificTransportCapabilities):
            raise ValueError("capabilities must be ScientificTransportCapabilities.")
        if not isinstance(retry_policy, RetryPolicy):
            raise ValueError("retry_policy must be RetryPolicy.")
        retry_policy.validate_scientific()
        if not callable(identity_inspector):
            raise ValueError("identity_inspector must be callable.")
        self._capabilities = capabilities
        self._retry_policy = retry_policy
        self._identity_inspector = identity_inspector
        self._post = post or _requests_post_with_preaccept_classification
        self._sleep = sleep
        self._clock = clock
        self._headers = tuple(sorted((headers or {}).items()))

    @property
    def capabilities(self) -> ScientificTransportCapabilities:
        return self._capabilities

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry_policy

    def generate(self, request: GenerationRequest) -> GenerationResult:
        preflight_error = _transport_preflight_error(request, self._capabilities)
        if preflight_error is not None:
            return blocked_result(request, self._capabilities, preflight_error)

        identity_checks: list[ModelIdentityCheck] = []
        attempts: list[GenerationAttempt] = []
        cooldown_observed_ms = 0.0
        cooldown_policy_ms = 0.0
        maximum_attempts = 1 + self._retry_policy.max_transport_unavailable_retries
        for attempt_index in range(1, maximum_attempts + 1):
            pre_identity = self._current_identity_check(
                request,
                attempt_index,
                "pre",
            )
            identity_checks.append(pre_identity)
            if not pre_identity.succeeded:
                if not attempts:
                    return blocked_result(
                        request,
                        self._capabilities,
                        pre_identity.error_message or "current_model_identity_invalid",
                        identity_checks,
                    )
                attempts.append(
                    failed_attempt(
                        request,
                        f"{request.request_id}:a{attempt_index}",
                        attempt_index,
                        accepted_by_server=False,
                        status_code=None,
                        started=self._clock(),
                        clock=self._clock,
                        failure=RuntimeFailureClass.TRANSPORT_DRIFT,
                        error=ValueError(pre_identity.error_message),
                    )
                )
                break
            attempt = self._run_attempt(request, attempt_index)
            if attempt.accepted_by_server is not False:
                post_identity = self._current_identity_check(
                    request,
                    attempt_index,
                    "post",
                )
                identity_checks.append(post_identity)
                if not post_identity.succeeded:
                    attempt = replace(
                        attempt,
                        contract_text=None,
                        error_class=RuntimeFailureClass.TRANSPORT_DRIFT,
                        error_message=(
                            "post_generation:"
                            f"{post_identity.error_message or 'identity_invalid'}"
                        ),
                    )
            attempts.append(attempt)
            if attempt.error_class is None:
                break
            if not is_retryable_failure(attempt.error_class):
                break
            if attempt_index >= maximum_attempts:
                break
            cooldown = self._retry_policy.retry_cooldown_sec
            cooldown_started = self._clock()
            self._sleep(cooldown)
            cooldown_observed_ms += max(
                0.0,
                (self._clock() - cooldown_started) * 1000.0,
            )
            cooldown_policy_ms += cooldown * 1000.0
        return result_from_attempts(
            request,
            self._capabilities,
            identity_checks,
            attempts,
            cooldown_observed_ms,
            cooldown_policy_ms,
        )

    def _current_identity_check(
        self,
        request: GenerationRequest,
        attempt_index: int,
        phase: str,
    ) -> ModelIdentityCheck:
        started = self._clock()
        identity: OllamaModelIdentity | None = None
        error_message: str | None = None
        try:
            identity = self._identity_inspector(
                request.native_endpoint,
                request.model_tag,
                timeout_sec=min(float(request.timeout_sec), 10.0),
            )
        except Exception as exc:
            error_message = f"current_model_identity_unavailable:{type(exc).__name__}"
        if identity is not None and not isinstance(identity, OllamaModelIdentity):
            identity = None
            error_message = "current_model_identity_invalid"
        if identity is not None and identity.model_tag != request.model_tag:
            error_message = "current_model_tag_drift"
        if identity is not None and identity.model_digest != request.model_digest:
            error_message = "current_model_digest_drift"
        return ModelIdentityCheck(
            attempt_index=attempt_index,
            phase=phase,
            requested_model_tag=request.model_tag,
            requested_model_digest=request.model_digest,
            observed_model_tag=(identity.model_tag if identity is not None else None),
            observed_model_digest=(
                identity.model_digest if identity is not None else None
            ),
            latency_ms=max(0.0, (self._clock() - started) * 1000.0),
            error_message=error_message,
        )

    def _run_attempt(
        self,
        request: GenerationRequest,
        attempt_index: int,
    ) -> GenerationAttempt:
        attempt_id = f"{request.request_id}:a{attempt_index}"
        started = self._clock()
        try:
            response = self._post(
                request.native_endpoint,
                json=build_native_chat_payload(request),
                headers=dict(self._headers),
                timeout=float(request.timeout_sec),
                allow_redirects=False,
            )
        except PreAcceptTransportUnavailable as exc:
            return self._failure_attempt(
                request,
                attempt_id,
                attempt_index,
                started,
                False,
                RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT,
                exc,
            )
        except requests.Timeout as exc:
            return self._failure_attempt(
                request,
                attempt_id,
                attempt_index,
                started,
                None,
                RuntimeFailureClass.GENERATION_TIMEOUT,
                exc,
            )
        except requests.ConnectionError as exc:
            return self._failure_attempt(
                request,
                attempt_id,
                attempt_index,
                started,
                None,
                RuntimeFailureClass.TRANSPORT_DRIFT,
                exc,
            )
        except Exception as exc:
            return self._failure_attempt(
                request,
                attempt_id,
                attempt_index,
                started,
                None,
                RuntimeFailureClass.TRANSPORT_DRIFT,
                exc,
            )

        status_code = getattr(response, "status_code", None)
        if getattr(response, "history", ()):
            return failed_attempt(
                request,
                attempt_id,
                attempt_index,
                accepted_by_server=None,
                status_code=status_code if isinstance(status_code, int) else None,
                started=started,
                clock=self._clock,
                failure=RuntimeFailureClass.TRANSPORT_DRIFT,
                error=ValueError("generation_redirect_history_detected"),
                response_body=_response_text(response),
                transport_error_body=_response_text(response),
            )
        if not isinstance(status_code, int) or not 200 <= status_code < 300:
            transport_error_body = _response_text(response)
            try:
                response.raise_for_status()
            except Exception as exc:
                error = exc
            else:
                error = ValueError(f"non_success_http_status:{status_code}")
            failure = RuntimeFailureClass.TRANSPORT_DRIFT
            if (
                request.output_enforcement is OutputEnforcement.BACKEND_SCHEMA
                and status_code
                in {
                    400,
                    422,
                }
                and _is_verified_schema_rejection(transport_error_body)
            ):
                failure = RuntimeFailureClass.SCHEMA_REJECTION
            return failed_attempt(
                request,
                attempt_id,
                attempt_index,
                accepted_by_server=True,
                status_code=status_code,
                started=started,
                clock=self._clock,
                failure=failure,
                error=error,
                response_body=transport_error_body,
                transport_error_body=transport_error_body,
            )

        try:
            data = response.json()
        except Exception as exc:
            return self._failure_attempt(
                request,
                attempt_id,
                attempt_index,
                started,
                True,
                RuntimeFailureClass.TRANSPORT_DRIFT,
                exc,
                status_code=status_code,
                response_body=_response_text(response),
                transport_error_body=_response_text(response),
            )
        return parse_native_response_attempt(
            request,
            attempt_id,
            attempt_index,
            status_code,
            data,
            _response_text(response),
            started,
            self._clock,
        )

    def _failure_attempt(
        self,
        request: GenerationRequest,
        attempt_id: str,
        attempt_index: int,
        started: float,
        accepted_by_server: bool | None,
        failure: RuntimeFailureClass,
        error: Exception,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
        transport_error_body: str | None = None,
    ) -> GenerationAttempt:
        return failed_attempt(
            request,
            attempt_id,
            attempt_index,
            accepted_by_server=accepted_by_server,
            status_code=status_code,
            started=started,
            clock=self._clock,
            failure=failure,
            error=error,
            response_body=response_body,
            transport_error_body=transport_error_body,
        )


def is_retryable_failure(failure: RuntimeFailureClass) -> bool:
    return failure is RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT


__all__ = [
    "NativeGenerationOptions",
    "OllamaScientificClient",
    "PreAcceptTransportUnavailable",
    "ScientificGenerationAbort",
    "ScientificGenerationTimeout",
    "ScientificGenerationContext",
    "ScientificTransportCapabilities",
    "build_native_chat_payload",
    "canonical_action_text_schema",
    "is_retryable_failure",
]
