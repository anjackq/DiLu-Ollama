from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

import requests


OLLAMA_DIGEST_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z", re.IGNORECASE)
OLLAMA_BARE_DIGEST_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z", re.IGNORECASE)


@dataclass(frozen=True)
class OllamaModelIdentity:
    model_tag: str
    model_digest: str

    def __post_init__(self) -> None:
        _require_canonical_model_tag(self.model_tag)
        normalized_digest = _normalize_model_digest(self.model_digest)
        object.__setattr__(self, "model_digest", normalized_digest)


@dataclass(frozen=True)
class OllamaNativeChatResolution:
    configured_mode: str
    effective_native_chat: bool
    reason: str
    think_mode: str
    thinking_family: bool


def ollama_tags_url(api_base: str) -> str:
    raw = str(api_base or "").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Ollama API base must be an absolute HTTP(S) URL.")
    path = parsed.path.rstrip("/")
    suffixes = ("/v1/chat/completions", "/api/chat", "/api/tags", "/v1")
    for suffix in suffixes:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if path:
        raise ValueError("Ollama API base contains an unsupported path prefix.")
    return urlunparse((parsed.scheme, parsed.netloc, "/api/tags", "", "", ""))


def parse_ollama_model_identity(
    payload: Any,
    requested_model_tag: str,
) -> OllamaModelIdentity:
    _require_canonical_model_tag(requested_model_tag)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Ollama /api/tags payload must contain a models list.")
    exact_matches = []
    for entry in payload["models"]:
        if not isinstance(entry, dict):
            continue
        present_tags = [entry[key] for key in ("name", "model") if key in entry]
        if requested_model_tag not in present_tags:
            continue
        if any(tag != requested_model_tag for tag in present_tags):
            raise ValueError(
                "Ollama model identity fields conflict with requested tag."
            )
        exact_matches.append(entry)
    if len(exact_matches) != 1:
        raise ValueError(
            "Ollama model identity requires exactly one exact tag match; "
            f"found {len(exact_matches)}."
        )
    digest = exact_matches[0].get("digest")
    return OllamaModelIdentity(
        model_tag=requested_model_tag,
        model_digest=_normalize_ollama_tags_digest(digest),
    )


def inspect_ollama_model_identity(
    api_base: str,
    model_tag: str,
    *,
    get: Callable[..., Any] = requests.get,
    timeout_sec: float = 10.0,
) -> OllamaModelIdentity:
    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
        raise ValueError("timeout_sec must be numeric.")
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive.")
    response = get(
        ollama_tags_url(api_base),
        timeout=float(timeout_sec),
        allow_redirects=False,
    )
    if getattr(response, "history", ()):
        raise ValueError("Ollama identity lookup cannot follow redirects.")
    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or not 200 <= status_code < 300:
        raise ValueError("Ollama identity lookup requires a direct 2xx response.")
    response.raise_for_status()
    return parse_ollama_model_identity(response.json(), model_tag)


def _require_canonical_model_tag(value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("model_tag must be a non-empty canonical string.")


def _normalize_model_digest(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("model_digest must be a string.")
    normalized = value.strip().lower()
    if not OLLAMA_DIGEST_PATTERN.fullmatch(normalized):
        raise ValueError("model_digest must be a full sha256:<64 hex> digest.")
    return normalized


def _normalize_ollama_tags_digest(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if OLLAMA_BARE_DIGEST_PATTERN.fullmatch(normalized):
            return f"sha256:{normalized}"
    return _normalize_model_digest(value)


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
