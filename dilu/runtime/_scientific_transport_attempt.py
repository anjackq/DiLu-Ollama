from __future__ import annotations

from dataclasses import dataclass

from ._scientific_transport_validation import (
    require_bool as _require_bool,
    require_canonical_text as _require_canonical_text,
    require_nonnegative_number as _require_nonnegative_number,
    require_optional_nonnegative_int as _require_optional_nonnegative_int,
    require_positive_int as _require_positive_int,
)
from .runtime_failures import RuntimeFailureClass
from .scientific_transport_evidence import BackendTiming


@dataclass(frozen=True)
class GenerationAttempt:
    request_id: str
    attempt_id: str
    attempt_index: int
    accepted_by_server: bool | None
    http_status: int | None
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
    error_class: RuntimeFailureClass | None
    error_message: str | None

    def __post_init__(self) -> None:
        _require_canonical_text("request_id", self.request_id)
        _require_canonical_text("attempt_id", self.attempt_id)
        _require_positive_int("attempt_index", self.attempt_index)
        if self.accepted_by_server is not None:
            _require_bool("accepted_by_server", self.accepted_by_server)
        if self.http_status is not None:
            _require_positive_int("http_status", self.http_status)
        _require_optional_text("response_body", self.response_body)
        _require_optional_text("raw_response", self.raw_response)
        _require_optional_text("contract_text", self.contract_text)
        _require_optional_text("transport_error_body", self.transport_error_body)
        if self.raw_response is None and self.contract_text is not None:
            raise ValueError("contract_text requires a raw response.")
        if self.raw_response is not None and self.transport_error_body is not None:
            raise ValueError(
                "Model output and transport error body are mutually exclusive."
            )
        if not isinstance(self.thinking_response, str):
            raise ValueError("thinking_response must be a string.")
        if self.stop_reason is not None:
            _require_canonical_text("stop_reason", self.stop_reason)
        _require_optional_nonnegative_int("prompt_tokens", self.prompt_tokens)
        _require_optional_nonnegative_int("completion_tokens", self.completion_tokens)
        if self.backend_timing is not None and not isinstance(
            self.backend_timing, BackendTiming
        ):
            raise ValueError("backend_timing must be BackendTiming or None.")
        _require_nonnegative_number("latency_ms", self.latency_ms)
        if self.error_class is not None and not isinstance(
            self.error_class, RuntimeFailureClass
        ):
            raise ValueError("error_class must be RuntimeFailureClass or None.")
        self._validate_success_or_failure()

    def _validate_success_or_failure(self) -> None:
        if self.error_class is None:
            if self.error_message is not None or self.transport_error_body is not None:
                raise ValueError("Successful attempts cannot contain error evidence.")
            if self.raw_response is None or self.contract_text is None:
                raise ValueError("Successful attempts require response evidence.")
            if self.response_body is None:
                raise ValueError("Successful attempts require the HTTP response body.")
            if self.prompt_tokens is None or self.completion_tokens is None:
                raise ValueError("Successful attempts require token counts.")
            if self.backend_timing is None:
                raise ValueError("Successful attempts require backend timing.")
            if (
                self.accepted_by_server is not True
                or self.http_status is None
                or not 200 <= self.http_status < 300
                or not _is_canonical_text(self.stop_reason)
            ):
                raise ValueError(
                    "Successful attempts require accepted completion metadata."
                )
            return
        _require_canonical_text("error_message", self.error_message)
        if self.accepted_by_server is False and (
            self.response_body is not None
            or self.raw_response is not None
            or self.contract_text is not None
            or self.transport_error_body is not None
            or self.prompt_tokens is not None
            or self.completion_tokens is not None
            or self.backend_timing is not None
            or self.thinking_response
        ):
            raise ValueError("Pre-accept failures cannot contain response evidence.")


def _require_optional_text(name: str, value: str | None) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None.")


def _is_canonical_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


__all__ = ["GenerationAttempt"]
