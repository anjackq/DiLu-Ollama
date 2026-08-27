from __future__ import annotations

import json
from typing import Any

import requests

from dilu.runtime.harness_config import OutputEnforcement, RetryPolicy, ThinkMode
from dilu.runtime.ollama_transport import OllamaModelIdentity
from dilu.runtime.ollama_scientific_client import (
    GenerationRequest,
    NativeGenerationOptions,
    ScientificTransportCapabilities,
)


MODEL_DIGEST = "sha256:" + "a" * 64


def identity_inspector_for(model_digest: str = MODEL_DIGEST):
    def inspect(
        api_base: str,
        model_tag: str,
        *,
        timeout_sec: float,
    ) -> OllamaModelIdentity:
        del api_base, timeout_sec
        return OllamaModelIdentity(model_tag=model_tag, model_digest=model_digest)

    return inspect


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        status_code: int = 200,
        *,
        text: str | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if text is None else text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"status={self.status_code}")
            error.response = self
            raise error

    def json(self) -> dict[str, Any]:
        return self._payload


def make_request(
    output_enforcement: OutputEnforcement = OutputEnforcement.PROMPT_ONLY,
    *,
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    think_mode: ThinkMode = ThinkMode.NO_THINK,
    digest: str = MODEL_DIGEST,
    available_action_ids: tuple[int, ...] | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        model_tag="qwen3:0.6b",
        model_digest=digest,
        request_id="req-case-001-step-000",
        messages=(("system", "contract"), ("user", "scenario")),
        native_endpoint=endpoint,
        options=NativeGenerationOptions(
            seed=1058710636,
            temperature=0.0,
            num_ctx=4096,
            num_predict=64,
        ),
        output_enforcement=output_enforcement,
        think_mode=think_mode,
        timeout_sec=60.0,
        available_action_ids=available_action_ids,
    )


def make_capabilities(
    *,
    digest: str = MODEL_DIGEST,
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    seed_verified: bool = True,
    schema_verified: bool = True,
) -> ScientificTransportCapabilities:
    return ScientificTransportCapabilities(
        model_tag="qwen3:0.6b",
        model_digest=digest,
        native_endpoint=endpoint,
        supported_think_modes=(ThinkMode.NO_THINK,),
        seed_verified=seed_verified,
        schema_verified=schema_verified,
        capability_probe_id="s1-transport-probe-placeholder",
        capability_artifact_hash="sha256:" + "c" * 64,
        schema_mechanism="ollama_api_chat_json_string_enum_v1",
    )


def make_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_transport_unavailable_retries=1,
        retry_cooldown_sec=10.0,
        retry_on_timeout=False,
        retry_on_empty_output=False,
        retry_on_schema_rejection=False,
    )


def success_payload(
    content: str = "Response to user:#### 3",
    *,
    eval_count: int = 7,
) -> dict[str, Any]:
    return {
        "model": "qwen3:0.6b",
        "message": {"role": "assistant", "content": content, "thinking": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 19,
        "eval_count": eval_count,
        "total_duration": 120_000_000,
        "load_duration": 10_000_000,
        "prompt_eval_duration": 40_000_000,
        "eval_duration": 60_000_000,
    }
