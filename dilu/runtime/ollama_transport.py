from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaNativeChatResolution:
    configured_mode: str
    effective_native_chat: bool
    reason: str
    think_mode: str
    thinking_family: bool


def normalize_ollama_native_chat_mode(raw: Any) -> str:
    if isinstance(raw, bool):
        return "true" if raw else "false"
    mode = str(raw if raw is not None else "auto").strip().lower()
    if mode in {"1", "true", "yes", "on", "native", "native_chat"}:
        return "true"
    if mode in {"0", "false", "no", "off", "openai", "openai_compat", "v1"}:
        return "false"
    return "auto"


def normalize_ollama_think_mode(raw: Any) -> str:
    mode = str(raw if raw is not None else "auto").strip().lower()
    if mode in {"think", "true", "on", "1"}:
        return "think"
    if mode in {"no_think", "nothink", "no-think", "false", "off", "0"}:
        return "no_think"
    return "auto"


def ollama_model_maybe_supports_thinking(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    if not name:
        return False
    thinking_markers = (
        "qwen",
        "qwq",
        "deepseek-r1",
        "deepseek-v3.1",
        "deepseek",
    )
    return any(marker in name for marker in thinking_markers)


def resolve_ollama_native_chat_mode(
    model_name: str,
    configured_mode: Any,
    think_mode: Any,
) -> OllamaNativeChatResolution:
    normalized_mode = normalize_ollama_native_chat_mode(configured_mode)
    normalized_think_mode = normalize_ollama_think_mode(think_mode)
    thinking_family = ollama_model_maybe_supports_thinking(model_name)

    if normalized_mode == "true":
        return OllamaNativeChatResolution(
            configured_mode="true",
            effective_native_chat=True,
            reason="manual_true",
            think_mode=normalized_think_mode,
            thinking_family=thinking_family,
        )
    if normalized_mode == "false":
        return OllamaNativeChatResolution(
            configured_mode="false",
            effective_native_chat=False,
            reason="manual_false",
            think_mode=normalized_think_mode,
            thinking_family=thinking_family,
        )
    if thinking_family and normalized_think_mode in {"think", "no_think"}:
        return OllamaNativeChatResolution(
            configured_mode="auto",
            effective_native_chat=True,
            reason=f"thinking_family_{normalized_think_mode}",
            think_mode=normalized_think_mode,
            thinking_family=True,
        )
    return OllamaNativeChatResolution(
        configured_mode="auto",
        effective_native_chat=False,
        reason="non_thinking_model" if not thinking_family else "think_mode_auto",
        think_mode=normalized_think_mode,
        thinking_family=thinking_family,
    )
