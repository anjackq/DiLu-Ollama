import os
from pathlib import Path
from typing import Any, Dict, Optional

from .ollama_transport import (
    normalize_ollama_think_mode,
    resolve_ollama_native_chat_mode,
)


def _pick_model(config: Dict[str, Any], api_type: str, chat_model_override: Optional[str]) -> Optional[str]:
    if chat_model_override:
        return str(chat_model_override)
    if api_type == "openai":
        model = config.get("OPENAI_CHAT_MODEL")
    elif api_type == "ollama":
        model = config.get("OLLAMA_CHAT_MODEL")
    elif api_type == "gemini":
        model = config.get("GEMINI_CHAT_MODEL")
    else:
        return None
    if model is None:
        return None
    model_str = str(model).strip()
    if not model_str or model_str.lower() == "none":
        return None
    return model_str


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _positive_int_or_none(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _first_nonempty_config_value(config: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_nonempty_env_value(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _unquote_dotenv_value(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def load_project_dotenv_if_present(dotenv_path: Optional[str] = None) -> bool:
    """
    Load a repo-local .env file without overriding explicit process env vars.
    """
    path = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    loaded_any = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _unquote_dotenv_value(value)
        loaded_any = True
    return loaded_any


def _resolve_openai_api_key(config: Dict[str, Any]) -> str:
    return (
        _first_nonempty_config_value(config, "OPENAI_KEY", "OPENAI_API_KEY")
        or _first_nonempty_env_value("OPENAI_API_KEY", "OPENAI_KEY", "OPENROUTER_API_KEY")
        or ""
    )


def _resolve_openai_api_base(config: Dict[str, Any]) -> Optional[str]:
    return _first_nonempty_config_value(config, "OPENAI_BASE_URL", "OPENAI_API_BASE") or _first_nonempty_env_value(
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    )


def _set_env_if_present(name: str, value: Optional[str]) -> None:
    if value is not None and str(value).strip():
        os.environ[name] = str(value).strip()


def _apply_ollama_resource_controls(config: Dict[str, Any]) -> None:
    """
    Export optional Ollama runtime controls for the current evaluation process.

    These keys are intentionally opt-in so the standard config does not silently
    change local Ollama behavior. `DILU_OLLAMA_NUM_CTX` is consumed by the
    DriverAgent native /api/chat payload, while OLLAMA_* variables are useful
    when Ollama is started from the same process environment.
    """
    mappings = {
        "ollama_runtime_num_ctx": "DILU_OLLAMA_NUM_CTX",
        "ollama_runtime_keep_alive": "DILU_OLLAMA_KEEP_ALIVE",
        "ollama_runtime_max_loaded_models": "OLLAMA_MAX_LOADED_MODELS",
        "ollama_runtime_num_parallel": "OLLAMA_NUM_PARALLEL",
        "ollama_runtime_max_queue": "OLLAMA_MAX_QUEUE",
    }
    for config_key, env_key in mappings.items():
        if config.get(config_key) is not None:
            _set_env_if_present(env_key, str(config.get(config_key)))


def openai_compatible_default_headers_from_env() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_APP_TITLE") or os.environ.get("OPENROUTER_SITE_NAME")
    if referer:
        headers["HTTP-Referer"] = str(referer).strip()
    if title:
        headers["X-OpenRouter-Title"] = str(title).strip()
    return {key: value for key, value in headers.items() if value}


def openai_compatible_default_headers_from_config(config: Dict[str, Any]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    referer = _first_nonempty_config_value(config, "OPENROUTER_HTTP_REFERER")
    title = _first_nonempty_config_value(config, "OPENROUTER_APP_TITLE", "OPENROUTER_SITE_NAME")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def _apply_output_runtime_controls(config: Dict[str, Any]) -> None:
    """
    Runtime output controls are configured here (outside policy resolution).
    """
    disable_streaming = _as_bool(config.get("eval_disable_streaming", True), default=True)
    disable_checker = _as_bool(config.get("eval_disable_checker_llm", True), default=True)
    base_tokens = (
        _positive_int_or_none(config.get("max_output_tokens"))
        or _positive_int_or_none(config.get("eval_decision_max_output_tokens"))
        or 512
    )
    runtime_tokens = _positive_int_or_none(config.get("runtime_max_output_tokens")) or base_tokens

    os.environ["DILU_USE_STREAMING"] = "0" if disable_streaming else "1"
    os.environ["DILU_ENABLE_CHECKER_LLM"] = "0" if disable_checker else "1"
    os.environ["DILU_MAX_OUTPUT_TOKENS"] = str(max(1, int(base_tokens)))
    os.environ["DILU_RUNTIME_MAX_OUTPUT_TOKENS"] = str(max(1, int(runtime_tokens)))
    os.environ["DILU_ENABLE_INTENT_RESOLVER"] = "1" if _as_bool(
        config.get("eval_enable_intent_resolver", False),
        default=False,
    ) else "0"
    os.environ["DILU_INTENT_RESOLVER_API_TYPE"] = str(
        config.get("intent_resolver_api_type", "ollama") or "ollama"
    ).strip()
    os.environ["DILU_INTENT_RESOLVER_MODEL"] = str(
        config.get("intent_resolver_model", "") or ""
    ).strip()
    os.environ["DILU_INTENT_RESOLVER_TIMEOUT_SEC"] = str(
        config.get("intent_resolver_timeout_sec", 5) or 5
    )
    os.environ["DILU_INTENT_RESOLVER_MAX_OUTPUT_TOKENS"] = str(
        _positive_int_or_none(config.get("intent_resolver_max_output_tokens")) or 32
    )
    os.environ["DILU_INTENT_RESOLVER_ABSTAIN_ON_AMBIGUOUS"] = "1" if _as_bool(
        config.get("intent_resolver_abstain_on_ambiguous", True),
        default=True,
    ) else "0"


def _resolve_quiet_mode(config: Dict[str, Any], mode: str, quiet_override: Optional[bool]) -> bool:
    if quiet_override is not None:
        return bool(quiet_override)
    mode_normalized = str(mode or "runtime").strip().lower()
    global_default = _as_bool(config.get("quiet_mode", False), default=False)
    mode_key = "eval_quiet_mode" if mode_normalized == "eval" else "runtime_quiet_mode"
    mode_value = config.get(mode_key)
    if mode_value is None:
        return global_default
    return _as_bool(mode_value, default=global_default)


def _resolve_progress_bar(config: Dict[str, Any], mode: str, progress_override: Optional[bool]) -> bool:
    if progress_override is not None:
        return bool(progress_override)
    mode_normalized = str(mode or "runtime").strip().lower()
    global_default = _as_bool(config.get("progress_bar", True), default=True)
    mode_key = "eval_progress_bar" if mode_normalized == "eval" else "runtime_progress_bar"
    mode_value = config.get(mode_key)
    if mode_value is None:
        return global_default
    return _as_bool(mode_value, default=global_default)


def configure_runtime_env(
    config: Dict[str, Any],
    chat_model_override: Optional[str] = None,
    mode: str = "runtime",
    quiet_override: Optional[bool] = None,
    progress_override: Optional[bool] = None,
) -> Optional[str]:
    """
    Configure provider-specific env vars used by DiLu runtime scripts.

    Returns the selected chat model for providers that require one.
    """
    load_project_dotenv_if_present()
    api_type = str(config["OPENAI_API_TYPE"]).strip().lower()
    selected_model = _pick_model(config, api_type, chat_model_override)
    _apply_output_runtime_controls(config)
    os.environ["DILU_QUIET_MODE"] = "1" if _resolve_quiet_mode(config, mode, quiet_override) else "0"
    os.environ["DILU_PROGRESS_BAR"] = "1" if _resolve_progress_bar(config, mode, progress_override) else "0"

    if api_type == "azure":
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = str(config["AZURE_API_VERSION"])
        os.environ["OPENAI_API_BASE"] = str(config["AZURE_API_BASE"])
        os.environ["OPENAI_API_KEY"] = str(config["AZURE_API_KEY"])
        os.environ["AZURE_CHAT_DEPLOY_NAME"] = str(config["AZURE_CHAT_DEPLOY_NAME"])
        os.environ["AZURE_EMBED_DEPLOY_NAME"] = str(config["AZURE_EMBED_DEPLOY_NAME"])
        return selected_model

    if api_type == "openai":
        if not selected_model:
            raise ValueError("OPENAI_CHAT_MODEL must be set when OPENAI_API_TYPE is 'openai'.")
        openai_api_key = _resolve_openai_api_key(config)
        if not openai_api_key:
            raise ValueError(
                "OPENAI_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY must be set when OPENAI_API_TYPE is 'openai'."
            )
        openai_api_base = _resolve_openai_api_base(config)
        os.environ["OPENAI_API_TYPE"] = "openai"
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENAI_CHAT_MODEL"] = selected_model
        if openai_api_base:
            os.environ["OPENAI_BASE_URL"] = openai_api_base
            os.environ["OPENAI_API_BASE"] = openai_api_base
        _set_env_if_present(
            "OPENROUTER_HTTP_REFERER",
            _first_nonempty_config_value(config, "OPENROUTER_HTTP_REFERER"),
        )
        _set_env_if_present(
            "OPENROUTER_APP_TITLE",
            _first_nonempty_config_value(config, "OPENROUTER_APP_TITLE", "OPENROUTER_SITE_NAME"),
        )
        if config.get("OPENAI_REFLECTION_MODEL"):
            os.environ["OPENAI_REFLECTION_MODEL"] = str(config["OPENAI_REFLECTION_MODEL"])
        return selected_model

    if api_type == "ollama":
        if not selected_model:
            raise ValueError("OLLAMA_CHAT_MODEL must be set when OPENAI_API_TYPE is 'ollama'.")
        _apply_ollama_resource_controls(config)
        ollama_api_base = str(config.get("OLLAMA_API_BASE", "http://localhost:11434/v1"))
        ollama_api_key = str(config.get("OLLAMA_API_KEY", "ollama"))
        ollama_think_mode = normalize_ollama_think_mode(config.get("OLLAMA_THINK_MODE", "auto"))
        native_resolution = resolve_ollama_native_chat_mode(
            selected_model,
            config.get("OLLAMA_USE_NATIVE_CHAT", "auto"),
            ollama_think_mode,
        )
        ollama_native_chat_timeout_sec = str(config.get("OLLAMA_NATIVE_CHAT_TIMEOUT_SEC", 60))
        os.environ["OPENAI_API_TYPE"] = "ollama"
        os.environ["OLLAMA_API_BASE"] = ollama_api_base
        os.environ["OPENAI_BASE_URL"] = ollama_api_base
        # Keep legacy var for compatibility with some clients/tools.
        os.environ["OPENAI_API_BASE"] = ollama_api_base
        os.environ["OLLAMA_API_KEY"] = ollama_api_key
        os.environ["OLLAMA_CHAT_MODEL"] = selected_model
        os.environ["OLLAMA_EMBED_MODEL"] = str(config["OLLAMA_EMBED_MODEL"])
        os.environ["OLLAMA_THINK_MODE"] = ollama_think_mode
        os.environ["OLLAMA_USE_NATIVE_CHAT"] = "1" if native_resolution.effective_native_chat else "0"
        os.environ["OLLAMA_USE_NATIVE_CHAT_CONFIGURED"] = native_resolution.configured_mode
        os.environ["OLLAMA_USE_NATIVE_CHAT_EFFECTIVE"] = (
            "1" if native_resolution.effective_native_chat else "0"
        )
        os.environ["OLLAMA_NATIVE_CHAT_RESOLUTION_REASON"] = native_resolution.reason
        os.environ["OLLAMA_MODEL_THINKING_FAMILY"] = "1" if native_resolution.thinking_family else "0"
        os.environ["OLLAMA_NATIVE_CHAT_TIMEOUT_SEC"] = ollama_native_chat_timeout_sec
        if config.get("OLLAMA_REFLECTION_MODEL"):
            os.environ["OLLAMA_REFLECTION_MODEL"] = str(config["OLLAMA_REFLECTION_MODEL"])
        return selected_model

    if api_type == "gemini":
        if not selected_model:
            raise ValueError("GEMINI_CHAT_MODEL must be set when OPENAI_API_TYPE is 'gemini'.")
        gemini_api_key = str(config.get("GEMINI_API_KEY", "")).strip()
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set when OPENAI_API_TYPE is 'gemini'.")
        os.environ["OPENAI_API_TYPE"] = "gemini"
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        os.environ["GEMINI_CHAT_MODEL"] = selected_model
        if config.get("GEMINI_REFLECTION_MODEL"):
            os.environ["GEMINI_REFLECTION_MODEL"] = str(config["GEMINI_REFLECTION_MODEL"])
        # Gemini mode reuses existing embedding backends (no native Gemini embeddings in this phase).
        # If Ollama embedding settings exist in config, expose them as env vars for DrivingMemory.
        if config.get("OLLAMA_API_BASE"):
            os.environ["OLLAMA_API_BASE"] = str(config["OLLAMA_API_BASE"])
        if config.get("OLLAMA_API_KEY"):
            os.environ["OLLAMA_API_KEY"] = str(config["OLLAMA_API_KEY"])
        if config.get("OLLAMA_EMBED_MODEL"):
            os.environ["OLLAMA_EMBED_MODEL"] = str(config["OLLAMA_EMBED_MODEL"])
        return selected_model

    raise ValueError(f"Unsupported OPENAI_API_TYPE: {config['OPENAI_API_TYPE']}")
