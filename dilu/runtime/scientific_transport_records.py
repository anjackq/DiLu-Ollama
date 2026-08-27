from __future__ import annotations

from dataclasses import dataclass

from ._scientific_contract_validation import (
    validate_output_contract_semantics,
    validate_success_response_body,
)
from ._scientific_transport_attempt import GenerationAttempt
from ._scientific_transport_validation import (
    require_canonical_text as _require_canonical_text,
    require_nonnegative_number as _require_nonnegative_number,
    require_optional_nonnegative_int as _require_optional_nonnegative_int,
)
from .harness_config import OutputEnforcement, ThinkMode
from .runtime_failures import RuntimeFailureClass
from .scientific_transport_evidence import BackendTiming, ModelIdentityCheck
from .scientific_transport_types import (
    GenerationRequest,
    NativeGenerationOptions,
    ScientificTransportCapabilities,
)


SCIENTIFIC_RETRY_COOLDOWN_MS = 10000.0
SCHEMA_OUTPUT_ENFORCEMENTS = (
    OutputEnforcement.BACKEND_SCHEMA,
    OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
)


@dataclass(frozen=True)
class GenerationResult:
    request: GenerationRequest
    capabilities: ScientificTransportCapabilities
    identity_checks: tuple[ModelIdentityCheck, ...]
    attempts: tuple[GenerationAttempt, ...]
    response_body: str | None
    raw_response: str | None
    contract_text: str | None
    transport_error_body: str | None
    thinking_response: str
    stop_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    backend_timing: BackendTiming | None
    latency_ms: float
    identity_latency_ms: float
    generation_latency_ms: float
    retry_cooldown_ms: float
    retry_cooldown_policy_ms: float
    error_class: RuntimeFailureClass | None
    error_message: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationRequest):
            raise ValueError("request must be GenerationRequest.")
        if not isinstance(self.capabilities, ScientificTransportCapabilities):
            raise ValueError("capabilities must be ScientificTransportCapabilities.")
        capabilities_bind_request = not (
            self.capabilities.model_tag != self.request.model_tag
            or self.capabilities.model_digest != self.request.model_digest
            or self.capabilities.native_endpoint != self.request.native_endpoint
            or self.request.think_mode not in self.capabilities.supported_think_modes
        )
        preflight_block = (
            not self.attempts
            and self.error_class is RuntimeFailureClass.TRANSPORT_DRIFT
        )
        capability_preflight_failed = (
            not capabilities_bind_request
            or not self.capabilities.seed_verified
            or (
                self.request.output_enforcement in SCHEMA_OUTPUT_ENFORCEMENTS
                and not self.capabilities.schema_verified
            )
        )
        if not capabilities_bind_request and not preflight_block:
            raise ValueError("Transport capabilities do not bind the request.")
        if self.attempts and (
            not self.capabilities.seed_verified
            or (
                self.request.output_enforcement in SCHEMA_OUTPUT_ENFORCEMENTS
                and not self.capabilities.schema_verified
            )
        ):
            raise ValueError("Attempted generation lacks verified capabilities.")
        if not isinstance(self.identity_checks, tuple) or not all(
            isinstance(check, ModelIdentityCheck) for check in self.identity_checks
        ):
            raise ValueError("identity_checks must be an immutable typed tuple.")
        if (
            preflight_block
            and not self.identity_checks
            and not capability_preflight_failed
        ):
            raise ValueError(
                "Preflight block requires capability or identity failure evidence."
            )
        if preflight_block and capability_preflight_failed and self.identity_checks:
            raise ValueError(
                "Capability-preflight failure cannot contain identity checks."
            )
        if not isinstance(self.attempts, tuple):
            raise ValueError("attempts must be an immutable tuple.")
        if len(self.attempts) > 2:
            raise ValueError("Scientific transport permits at most two attempts.")
        if self.raw_response is None and self.contract_text is not None:
            raise ValueError("contract_text requires a raw response.")
        _require_optional_text("response_body", self.response_body)
        _require_optional_text("transport_error_body", self.transport_error_body)
        _require_optional_nonnegative_int("prompt_tokens", self.prompt_tokens)
        _require_optional_nonnegative_int("completion_tokens", self.completion_tokens)
        if self.backend_timing is not None and not isinstance(
            self.backend_timing, BackendTiming
        ):
            raise ValueError("backend_timing must be BackendTiming or None.")
        self._validate_timing_fields()
        self._validate_identity_chain()
        if self.attempts:
            self._validate_attempt_chain()
        else:
            self._validate_preflight_block()
        if self.error_class is None and self.contract_text is None:
            raise ValueError("Successful results require contract text.")

    def _validate_timing_fields(self) -> None:
        for name in (
            "latency_ms",
            "identity_latency_ms",
            "generation_latency_ms",
            "retry_cooldown_ms",
            "retry_cooldown_policy_ms",
        ):
            _require_nonnegative_number(name, getattr(self, name))
        expected_identity = sum(check.latency_ms for check in self.identity_checks)
        if self.identity_latency_ms != expected_identity:
            raise ValueError("Identity latency must match identity checks.")
        expected_total = (
            self.identity_latency_ms
            + self.generation_latency_ms
            + self.retry_cooldown_ms
        )
        if self.latency_ms != expected_total:
            raise ValueError("Total latency must match observed timing components.")

    def _validate_identity_chain(self) -> None:
        for check in self.identity_checks:
            if (
                check.requested_model_tag != self.request.model_tag
                or check.requested_model_digest != self.request.model_digest
            ):
                raise ValueError("Identity checks must match the generation request.")

        if not self.attempts:
            if not self.identity_checks:
                return
            only_check = self.identity_checks[0]
            if (
                len(self.identity_checks) != 1
                or only_check.attempt_index != 1
                or only_check.phase != "pre"
                or only_check.succeeded
            ):
                raise ValueError(
                    "A blocked live-identity check must be one failed pre check."
                )
            return

        expected_chain: list[tuple[int, str]] = []
        for attempt in self.attempts:
            expected_chain.append((attempt.attempt_index, "pre"))
            if attempt.accepted_by_server is not False:
                expected_chain.append((attempt.attempt_index, "post"))
        observed_chain = [
            (check.attempt_index, check.phase) for check in self.identity_checks
        ]
        if observed_chain != expected_chain:
            raise ValueError("Identity checks must exactly bind the attempt chain.")

        if self.error_class is None or self.operational_fallback_allowed:
            if not all(check.succeeded for check in self.identity_checks):
                raise ValueError(
                    "Accepted evidence requires successful model identity checks."
                )
        elif any(not check.succeeded for check in self.identity_checks):
            if self.error_class is not RuntimeFailureClass.TRANSPORT_DRIFT:
                raise ValueError("Identity-check failure must be transport drift.")

    def _validate_preflight_block(self) -> None:
        if self.error_class is not RuntimeFailureClass.TRANSPORT_DRIFT:
            raise ValueError("Preflight-blocked results must be transport drift.")
        response_values = (
            self.response_body,
            self.raw_response,
            self.contract_text,
            self.transport_error_body,
            self.stop_reason,
            self.prompt_tokens,
            self.completion_tokens,
            self.backend_timing,
        )
        if (
            any(value is not None for value in response_values)
            or self.thinking_response
        ):
            raise ValueError("Blocked results cannot contain generation evidence.")
        _require_canonical_text("error_message", self.error_message)
        if (
            self.generation_latency_ms != 0.0
            or self.retry_cooldown_ms != 0.0
            or self.retry_cooldown_policy_ms != 0.0
        ):
            raise ValueError("Blocked results cannot contain generation timing.")

    def _validate_attempt_chain(self) -> None:
        for expected_index, attempt in enumerate(self.attempts, start=1):
            if not isinstance(attempt, GenerationAttempt):
                raise ValueError("attempts must contain GenerationAttempt values.")
            if (
                attempt.request_id != self.request.request_id
                or attempt.attempt_index != expected_index
                or attempt.attempt_id != f"{self.request.request_id}:a{expected_index}"
            ):
                raise ValueError("Generation attempts must form one ordered chain.")
            self._validate_attempt_contract(attempt)
        self._validate_retry_semantics()
        self._validate_final_attempt()
        expected_generation = sum(item.latency_ms for item in self.attempts)
        if self.generation_latency_ms != expected_generation:
            raise ValueError("Generation latency must match its attempt chain.")

    def _validate_attempt_contract(self, attempt: GenerationAttempt) -> None:
        validate_output_contract_semantics(
            output_enforcement=self.request.output_enforcement.value,
            think_mode=self.request.think_mode.value,
            error_class=(
                None if attempt.error_class is None else attempt.error_class.value
            ),
            raw_output=attempt.raw_response,
            contract_text=attempt.contract_text,
            thinking_output=attempt.thinking_response,
        )
        if attempt.error_class is None:
            timing = attempt.backend_timing
            validate_success_response_body(
                model_tag=self.request.model_tag,
                response_body=attempt.response_body,
                raw_output=attempt.raw_response,
                thinking_output=attempt.thinking_response,
                stop_reason=attempt.stop_reason,
                prompt_tokens=attempt.prompt_tokens,
                completion_tokens=attempt.completion_tokens,
                backend_timing=(
                    None
                    if timing is None
                    else (
                        timing.total_duration_ns,
                        timing.load_duration_ns,
                        timing.prompt_eval_duration_ns,
                        timing.eval_duration_ns,
                    )
                ),
            )

    def _validate_retry_semantics(self) -> None:
        retry_failure = RuntimeFailureClass.TRANSPORT_UNAVAILABLE_BEFORE_ACCEPT
        if len(self.attempts) == 1:
            if self.attempts[0].error_class is retry_failure:
                raise ValueError("Pre-accept unavailability requires the fixed retry.")
            if self.retry_cooldown_ms or self.retry_cooldown_policy_ms:
                raise ValueError(
                    "Single-attempt results cannot contain retry cooldown."
                )
            return
        first = self.attempts[0]
        if (
            first.error_class is not retry_failure
            or first.accepted_by_server is not False
        ):
            raise ValueError("Only proved pre-accept unavailability permits retry.")
        if self.retry_cooldown_policy_ms != SCIENTIFIC_RETRY_COOLDOWN_MS:
            raise ValueError("Scientific retry policy must be exactly 10 seconds.")
        if self.retry_cooldown_ms < SCIENTIFIC_RETRY_COOLDOWN_MS:
            raise ValueError("Scientific retry did not observe the fixed cooldown.")

    def _validate_final_attempt(self) -> None:
        final = self.attempts[-1]
        expected = (
            final.response_body,
            final.raw_response,
            final.contract_text,
            final.transport_error_body,
            final.thinking_response,
            final.stop_reason,
            final.prompt_tokens,
            final.completion_tokens,
            final.backend_timing,
            final.error_class,
            final.error_message,
        )
        observed = (
            self.response_body,
            self.raw_response,
            self.contract_text,
            self.transport_error_body,
            self.thinking_response,
            self.stop_reason,
            self.prompt_tokens,
            self.completion_tokens,
            self.backend_timing,
            self.error_class,
            self.error_message,
        )
        if observed != expected:
            raise ValueError("GenerationResult must preserve its final attempt.")

    @property
    def succeeded(self) -> bool:
        return self.error_class is None

    @property
    def transport_succeeded(self) -> bool:
        if not self.attempts:
            return False
        final = self.attempts[-1]
        return (
            final.accepted_by_server is True
            and final.http_status is not None
            and 200 <= final.http_status < 300
            and final.error_class is not RuntimeFailureClass.TRANSPORT_DRIFT
        )

    @property
    def operational_fallback_allowed(self) -> bool:
        return self.error_class in {
            RuntimeFailureClass.GENERATION_TIMEOUT,
            RuntimeFailureClass.MODEL_EMPTY_OUTPUT,
        }

    @property
    def requires_cell_abort(self) -> bool:
        return self.error_class is not None and not self.operational_fallback_allowed

    @property
    def model_tag(self) -> str:
        return self.request.model_tag

    @property
    def model_digest(self) -> str:
        return self.request.model_digest

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def attempt_ids(self) -> tuple[str, ...]:
        return tuple(attempt.attempt_id for attempt in self.attempts)

    @property
    def native_endpoint(self) -> str:
        return self.request.native_endpoint

    @property
    def options(self) -> NativeGenerationOptions:
        return self.request.options

    @property
    def output_enforcement(self) -> OutputEnforcement:
        return self.request.output_enforcement

    @property
    def think_mode(self) -> ThinkMode:
        return self.request.think_mode

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


def _require_optional_text(name: str, value: str | None) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None.")


__all__ = [
    "BackendTiming",
    "GenerationAttempt",
    "GenerationResult",
    "ModelIdentityCheck",
    "SCHEMA_OUTPUT_ENFORCEMENTS",
    "SCIENTIFIC_RETRY_COOLDOWN_MS",
]
