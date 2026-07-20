from __future__ import annotations

from dataclasses import dataclass

from ._scientific_transport_validation import (
    require_canonical_text,
    require_model_digest,
    require_nonnegative_number,
    require_positive_int,
)


@dataclass(frozen=True)
class BackendTiming:
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int

    def __post_init__(self) -> None:
        for field_name in (
            "total_duration_ns",
            "load_duration_ns",
            "prompt_eval_duration_ns",
            "eval_duration_ns",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")


@dataclass(frozen=True)
class ModelIdentityCheck:
    attempt_index: int
    phase: str
    requested_model_tag: str
    requested_model_digest: str
    observed_model_tag: str | None
    observed_model_digest: str | None
    latency_ms: float
    error_message: str | None

    def __post_init__(self) -> None:
        require_positive_int("attempt_index", self.attempt_index)
        if self.phase not in {"pre", "post"}:
            raise ValueError("identity-check phase must be pre or post.")
        require_canonical_text("requested_model_tag", self.requested_model_tag)
        require_model_digest("requested_model_digest", self.requested_model_digest)
        _require_optional_text("observed_model_tag", self.observed_model_tag)
        if self.observed_model_digest is not None:
            require_model_digest("observed_model_digest", self.observed_model_digest)
        require_nonnegative_number("latency_ms", self.latency_ms)
        if self.error_message is None:
            if (
                self.observed_model_tag != self.requested_model_tag
                or self.observed_model_digest != self.requested_model_digest
            ):
                raise ValueError("Successful identity checks must match the request.")
        else:
            require_canonical_text("error_message", self.error_message)

    @property
    def succeeded(self) -> bool:
        return self.error_message is None


def _require_optional_text(name: str, value: str | None) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None.")


__all__ = ["BackendTiming", "ModelIdentityCheck"]
