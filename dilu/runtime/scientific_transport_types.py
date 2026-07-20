from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._scientific_transport_validation import (
    require_bool as _require_bool,
    require_canonical_text as _require_canonical_text,
    require_model_digest as _require_model_digest,
    require_native_endpoint as _require_native_endpoint,
    require_positive_int as _require_positive_int,
    require_uint32 as _require_uint32,
)
from .harness_config import OutputEnforcement, ThinkMode


CANONICAL_ACTION_TEXT_VALUES = tuple(
    f"Response to user:#### {action_id}" for action_id in range(5)
)
SCHEMA_MECHANISM = "ollama_api_chat_json_string_enum_v1"


@dataclass(frozen=True)
class NativeGenerationOptions:
    seed: int
    temperature: float
    num_ctx: int
    num_predict: int

    def __post_init__(self) -> None:
        _require_uint32("seed", self.seed)
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise ValueError("temperature must be numeric.")
        if not math.isfinite(float(self.temperature)) or float(self.temperature) < 0.0:
            raise ValueError("temperature must be non-negative.")
        _require_positive_int("num_ctx", self.num_ctx)
        _require_positive_int("num_predict", self.num_predict)

    def to_payload(self) -> dict[str, int | float]:
        return {
            "seed": self.seed,
            "temperature": float(self.temperature),
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }


@dataclass(frozen=True)
class ScientificTransportCapabilities:
    model_tag: str
    model_digest: str
    native_endpoint: str
    supported_think_modes: tuple[ThinkMode, ...]
    seed_verified: bool
    schema_verified: bool
    capability_probe_id: str
    capability_artifact_hash: str
    schema_mechanism: str

    def __post_init__(self) -> None:
        _require_canonical_text("model_tag", self.model_tag)
        _require_model_digest("model_digest", self.model_digest)
        _require_native_endpoint(self.native_endpoint)
        if (
            not isinstance(self.supported_think_modes, tuple)
            or not self.supported_think_modes
        ):
            raise ValueError("At least one explicit think mode must be supported.")
        for mode in self.supported_think_modes:
            if not isinstance(mode, ThinkMode) or mode is ThinkMode.AUTO:
                raise ValueError(
                    "Supported think modes must be explicit ThinkMode values."
                )
        _require_bool("seed_verified", self.seed_verified)
        _require_bool("schema_verified", self.schema_verified)
        _require_canonical_text("capability_probe_id", self.capability_probe_id)
        _require_model_digest("capability_artifact_hash", self.capability_artifact_hash)
        if self.schema_mechanism != SCHEMA_MECHANISM:
            raise ValueError("Unknown backend schema mechanism.")


@dataclass(frozen=True)
class GenerationRequest:
    model_tag: str
    model_digest: str
    request_id: str
    messages: tuple[tuple[str, str], ...]
    native_endpoint: str
    options: NativeGenerationOptions
    output_enforcement: OutputEnforcement
    think_mode: ThinkMode
    timeout_sec: float

    def __post_init__(self) -> None:
        _require_canonical_text("model_tag", self.model_tag)
        _require_model_digest("model_digest", self.model_digest)
        _require_canonical_text("request_id", self.request_id)
        _require_native_endpoint(self.native_endpoint)
        if not isinstance(self.options, NativeGenerationOptions):
            raise ValueError("options must be NativeGenerationOptions.")
        if not isinstance(self.output_enforcement, OutputEnforcement):
            raise ValueError("output_enforcement must be OutputEnforcement.")
        if (
            not isinstance(self.think_mode, ThinkMode)
            or self.think_mode is ThinkMode.AUTO
        ):
            raise ValueError("Scientific think mode must be explicit.")
        if isinstance(self.timeout_sec, bool) or not isinstance(
            self.timeout_sec, (int, float)
        ):
            raise ValueError("timeout_sec must be numeric.")
        if not math.isfinite(float(self.timeout_sec)) or float(self.timeout_sec) <= 0.0:
            raise ValueError("timeout_sec must be positive.")
        if not isinstance(self.messages, tuple) or not self.messages:
            raise ValueError("messages must be a non-empty immutable tuple.")
        for message in self.messages:
            if (
                not isinstance(message, tuple)
                or len(message) != 2
                or message[0] not in {"system", "user", "assistant"}
                or not isinstance(message[1], str)
            ):
                raise ValueError(
                    "Each message must be a canonical (role, content) tuple."
                )


@dataclass(frozen=True)
class ScientificGenerationContext:
    request_id: str
    model_digest: str
    generation_seed: int

    def __post_init__(self) -> None:
        _require_canonical_text("request_id", self.request_id)
        _require_model_digest("model_digest", self.model_digest)
        _require_uint32("generation_seed", self.generation_seed)


def canonical_action_text_schema() -> dict[str, Any]:
    return {"type": "string", "enum": list(CANONICAL_ACTION_TEXT_VALUES)}


def build_native_chat_payload(request: GenerationRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model_tag,
        "messages": [
            {"role": role, "content": content} for role, content in request.messages
        ],
        "stream": False,
        "options": request.options.to_payload(),
        "think": request.think_mode is ThinkMode.THINK,
    }
    if request.output_enforcement is OutputEnforcement.BACKEND_SCHEMA:
        payload["format"] = canonical_action_text_schema()
    return payload


__all__ = [
    "CANONICAL_ACTION_TEXT_VALUES",
    "GenerationRequest",
    "NativeGenerationOptions",
    "ScientificGenerationContext",
    "ScientificTransportCapabilities",
    "SCHEMA_MECHANISM",
    "build_native_chat_payload",
    "canonical_action_text_schema",
]
