import os
import json
import re
import textwrap
import time
from contextlib import contextmanager
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from rich import print
from typing import Any, Dict, List
import requests

# UPDATED IMPORTS
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.callbacks import OpenAICallbackHandler

from dilu.scenario.envScenario import EnvScenario
from dilu.driver_agent.prompt_modules import build_prompt_artifact
from dilu.runtime.action_resolution import (
    ActionResolutionResult,
    ActionSyntaxStatus,
    require_fixed_idle_available,
    resolve_action,
)
from dilu.runtime.energy_monitor import estimate_generated_tokens
from dilu.runtime.harness_config import HarnessConfig, OutputEnforcement
from dilu.runtime.runtime_failures import (
    ProtocolInvariantCode,
    ProtocolInvariantViolation,
    RuntimeFailureClass,
    RuntimeProtocolError,
)
from dilu.runtime.token_usage import (
    combine_token_usage_records,
    build_token_usage_record_from_langchain_message,
    build_token_usage_record_from_ollama_native_payload,
    build_whitespace_estimate_token_usage,
)
from dilu.runtime.llm_env import openai_compatible_default_headers_from_env
from dilu.runtime.ollama_transport import (
    normalize_ollama_think_mode,
    ollama_model_maybe_supports_thinking,
    resolve_ollama_native_chat_mode,
)
from dilu.runtime.ollama_scientific_client import (
    GenerationRequest,
    GenerationResult,
    NativeGenerationOptions,
    OllamaScientificClient,
    ScientificGenerationAbort,
    ScientificGenerationContext,
    ScientificGenerationTimeout,
)
from dilu.runtime.scientific_runtime import ScientificEpisodeRuntime

delimiter = "####"
ChatGoogleGenerativeAI = None
_GEMINI_IMPORT_ERROR = None
ACTION_RECOVERY_PATTERN = re.compile(
    r"Response to user:\s*\#{4}\s*(?:<[^>]+>\s*)?(-?\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ACTION_ANYWHERE_PATTERN = re.compile(r"\b-?\d+\b")
RUNTIME_ACTION_LINE_PATTERN = re.compile(
    r"^Response to user:\s*\#{4}\s*(?:<[^>]+>\s*)?(-?\d+)\s*$",
    re.IGNORECASE,
)
RUNTIME_MISSING_DELIMITER_ACTION_LINE_PATTERN = re.compile(
    r"^Response to user:\s*(?:<[^>]+>\s*)?(-?\d+)\s*$",
    re.IGNORECASE,
)
RUNTIME_LABELED_ACTION_PATTERN = re.compile(r"^Action id:\s*(-?\d+)\s*$", re.IGNORECASE)
RUNTIME_REASON_LINE_PATTERN = re.compile(r"^Reason:\s*(.+?)\s*$", re.IGNORECASE)
RUNTIME_RESPONSE_PREFIX_PATTERN = re.compile(r"^Response to user:", re.IGNORECASE)
RECOVERED_RUNTIME_PARSE_PATHS = {
    "missing_delimiter_recovered",
    "labeled_backup",
    "delimiter_tail",
    "regex_recovered",
    "loose_recovered",
    "semantic_label_recovered",
    "intent_resolver_direct",
    "checker_direct",
    "checker_regex_recovered",
    "checker_loose_recovered",
}
UNPARSEABLE_RUNTIME_PARSE_PATHS = {
    "parse_fallback",
    "empty_response_fallback",
    "max_tokens_empty_response_fallback",
    "incomplete_output_fallback",
    "checker_fallback",
}
SUPPORTED_PROMPT_PROFILES = {"harness_v2", "legacy_dilu_like"}


def normalize_prompt_profile(value: str | None) -> str:
    profile = str(value or "harness_v2").strip().lower().replace("-", "_")
    return profile if profile in SUPPORTED_PROMPT_PROFILES else "harness_v2"


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text is not None:
                    chunks.append(str(text))
                else:
                    chunks.append(str(part))
            else:
                text = getattr(part, "text", None)
                chunks.append(str(text) if text is not None else str(part))
        return "".join(chunks)
    return str(content)


def _load_chat_google_generative_ai():
    global ChatGoogleGenerativeAI, _GEMINI_IMPORT_ERROR
    if ChatGoogleGenerativeAI is not None:
        return ChatGoogleGenerativeAI
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI as GeminiChat
    except Exception as exc:
        _GEMINI_IMPORT_ERROR = exc
        return None
    ChatGoogleGenerativeAI = GeminiChat
    _GEMINI_IMPORT_ERROR = None
    return ChatGoogleGenerativeAI


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except Exception:
        return default


def _env_positive_int_or_none(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _is_timeout_exception(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, requests.Timeout))


def _normalize_ollama_think_mode(raw: str) -> str:
    return normalize_ollama_think_mode(raw)


def _ollama_native_chat_url(api_base: str) -> str:
    base = (api_base or "http://localhost:11434/v1").strip()
    if base.endswith("/"):
        base = base[:-1]
    parsed = urlparse(base)
    normalized_path = parsed.path.rstrip("/")
    if normalized_path.endswith("/v1"):
        root_path = normalized_path[:-3]
    elif normalized_path == "/v1":
        root_path = ""
    else:
        root_path = normalized_path
    if not root_path.endswith("/"):
        root_path += "/"
    return f"{parsed.scheme}://{parsed.netloc}{root_path}api/chat"


def _ollama_model_maybe_supports_think(model_name: str) -> bool:
    return ollama_model_maybe_supports_thinking(model_name)


def _ollama_role_from_message(msg) -> str:
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, AIMessage):
        return "assistant"
    return "user"


# ... (Keep example_message and example_answer variables as they are in original) ...
example_message = textwrap.dedent(
    f"""\
        {delimiter} Driving scenario description:
        You are driving on a road with 4 lanes, and you are currently driving in the second lane from the left. Your speed is 25.00 m/s, acceleration is 0.00 m/s^2, and lane position is 363.14 m. 
        There are other vehicles driving around you, and below is their basic information:
        - Vehicle `912` is driving on the same lane of you and is ahead of you. The speed of it is 23.30 m/s, acceleration is 0.00 m/s^2, and lane position is 382.33 m.
        - Vehicle `864` is driving on the lane to your right and is ahead of you. The speed of it is 21.30 m/s, acceleration is 0.00 m/s^2, and lane position is 373.74 m.
        - Vehicle `488` is driving on the lane to your left and is ahead of you. The speed of it is 23.61 $m/s$, acceleration is 0.00 $m/s^2$, and lane position is 368.75 $m$.

        {delimiter} Your available actions:
        IDLE - remain in the current lane with current speed Action_id: 1
        Turn-left - change lane to the left of the current lane Action_id: 0
        Turn-right - change lane to the right of the current lane Action_id: 2
        Acceleration - accelerate the vehicle Action_id: 3
        Deceleration - decelerate the vehicle Action_id: 4
        """
)
example_answer = textwrap.dedent(
    f"""\
        Response to user:{delimiter} 4
        Reason: The lead car is critically close and slower, while adjacent lanes are blocked, so deceleration is required.
        """
)


class DriverAgent:
    def __init__(
        self,
        sce: EnvScenario,
        temperature: float = 0,
        verbose: bool = False,
        scientific_runtime: ScientificEpisodeRuntime | None = None,
    ) -> None:
        self.sce = sce
        self.verbose = bool(verbose)
        self.quiet_mode = (
            True
            if scientific_runtime is not None
            else _env_bool("DILU_QUIET_MODE", False)
        )
        self.temperature = float(temperature)
        if scientific_runtime is not None:
            self._initialize_scientific_runtime(scientific_runtime)
            return
        self.oai_api_type = os.getenv("OPENAI_API_TYPE")
        self.decision_timeout_sec = _env_float("DILU_DECISION_TIMEOUT_SEC", 60.0)
        # For local Ollama models, invoke mode avoids long stream stalls on small models.
        default_streaming = self.oai_api_type != "ollama"
        self.use_streaming = _env_bool("DILU_USE_STREAMING", default_streaming)
        self.enable_checker_llm = _env_bool("DILU_ENABLE_CHECKER_LLM", True)
        self.prompt_profile = normalize_prompt_profile(
            os.getenv("DILU_PROMPT_PROFILE", "harness_v2")
        )
        self.enable_intent_resolver = _env_bool("DILU_ENABLE_INTENT_RESOLVER", False)
        self.intent_resolver_api_type = (
            os.getenv("DILU_INTENT_RESOLVER_API_TYPE", "ollama").strip().lower()
        )
        self.intent_resolver_model = os.getenv("DILU_INTENT_RESOLVER_MODEL", "").strip()
        self.intent_resolver_timeout_sec = _env_float(
            "DILU_INTENT_RESOLVER_TIMEOUT_SEC", 5.0
        )
        self.intent_resolver_max_output_tokens = max(
            1,
            int(os.getenv("DILU_INTENT_RESOLVER_MAX_OUTPUT_TOKENS", "32")),
        )
        self.intent_resolver_abstain_on_ambiguous = _env_bool(
            "DILU_INTENT_RESOLVER_ABSTAIN_ON_AMBIGUOUS",
            True,
        )
        self.last_intent_resolver_prompt = None
        max_tokens_default = 2000
        self.max_tokens = int(
            os.getenv("DILU_MAX_OUTPUT_TOKENS", str(max_tokens_default))
        )
        self.runtime_max_output_tokens = max(
            1,
            int(os.getenv("DILU_RUNTIME_MAX_OUTPUT_TOKENS", str(self.max_tokens))),
        )
        self.ollama_think_mode = _normalize_ollama_think_mode(
            os.getenv("OLLAMA_THINK_MODE", "auto")
        )
        self.ollama_native_chat_timeout_sec = _env_float(
            "OLLAMA_NATIVE_CHAT_TIMEOUT_SEC", self.decision_timeout_sec
        )
        self.ollama_native_num_ctx = _env_positive_int_or_none("DILU_OLLAMA_NUM_CTX")
        self.ollama_native_keep_alive = (
            str(os.getenv("DILU_OLLAMA_KEEP_ALIVE", "") or "").strip() or None
        )
        self.ollama_chat_url = _ollama_native_chat_url(
            os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
        )
        self.ollama_model_name = os.getenv("OLLAMA_CHAT_MODEL")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        self.ollama_native_chat_resolution = resolve_ollama_native_chat_mode(
            self.ollama_model_name,
            os.getenv(
                "OLLAMA_USE_NATIVE_CHAT_CONFIGURED",
                os.getenv("OLLAMA_USE_NATIVE_CHAT", "auto"),
            ),
            self.ollama_think_mode,
        )
        self.ollama_use_native_chat = bool(
            self.ollama_native_chat_resolution.effective_native_chat
        )
        self.ollama_use_native_chat_configured = (
            self.ollama_native_chat_resolution.configured_mode
        )
        self.ollama_native_chat_resolution_reason = (
            self.ollama_native_chat_resolution.reason
        )
        self.ollama_native_think_supported = None
        self.ollama_native_timed_out = False
        self.ollama_model_think_heuristic = _ollama_model_maybe_supports_think(
            self.ollama_model_name
        )
        self.ollama_think_downgrade_noted = False
        self.last_ollama_transport = "provider_default"
        self.last_ollama_effective_think_mode = self.ollama_think_mode
        self.last_ollama_native_retry_used = False
        self.last_ollama_native_timeout = False
        self.last_ollama_native_timeout_short_circuit = False
        self.last_ollama_native_num_predict = None
        self.last_ollama_native_num_ctx = None
        self.last_ollama_native_keep_alive = None
        self.last_decision_meta = {
            "timed_out": False,
            "used_fallback": False,
            "fallback_reason": None,
            "parse_mode": "unknown",
            "checker_used": False,
            "selected_action": None,
            "decision_elapsed_sec": 0.0,
            "ollama_transport": None,
            "ollama_native_chat_configured": (
                self.ollama_use_native_chat_configured
                if self.oai_api_type == "ollama"
                else None
            ),
            "ollama_native_chat_effective": (
                self.ollama_use_native_chat if self.oai_api_type == "ollama" else None
            ),
            "ollama_native_chat_resolution_reason": (
                self.ollama_native_chat_resolution_reason
                if self.oai_api_type == "ollama"
                else None
            ),
            "ollama_requested_think_mode": None,
            "ollama_effective_think_mode": None,
            "ollama_native_retry_used": False,
            "ollama_native_timeout": False,
            "ollama_native_timeout_short_circuit": False,
            "ollama_native_num_predict": None,
            "ollama_native_num_ctx": None,
            "ollama_native_keep_alive": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "token_count_method": None,
            "token_usage_source": None,
            "response_contract_satisfied": False,
            "response_contract_recovered": False,
            "response_recovery_reason": None,
            "response_unparseable": False,
            "response_action_line_present": False,
            "response_action_line_count": 0,
            "response_reason_line_present": False,
            "runtime_parse_path": "unknown",
            "response_first_nonempty_line": None,
            "response_truncated_before_contract": False,
            "semantic_recovery_used": False,
            "semantic_recovery_label": None,
            "semantic_recovery_reason": None,
            "intent_resolver_used": False,
            "intent_resolver_model": None,
            "intent_resolver_action_id": None,
            "intent_resolver_abstained": False,
            "intent_resolver_reason": None,
            "final_action_source": "unknown",
        }
        if self.oai_api_type == "azure":
            self._log_info("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                callbacks=[OpenAICallbackHandler()],
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=self.max_tokens,
                request_timeout=self.decision_timeout_sec,
                streaming=self.use_streaming,
            )
        elif self.oai_api_type == "openai":
            openai_api_base = os.getenv("OPENAI_BASE_URL") or os.getenv(
                "OPENAI_API_BASE"
            )
            openai_api_key = os.getenv("OPENAI_API_KEY")
            default_headers = openai_compatible_default_headers_from_env()
            if openai_api_base:
                self._log_info(f"Use OpenAI-compatible API at {openai_api_base}")
            else:
                self._log_info("Use OpenAI API")
            chat_kwargs = {}
            if default_headers:
                chat_kwargs["default_headers"] = default_headers
            self.llm = ChatOpenAI(
                temperature=temperature,
                callbacks=[OpenAICallbackHandler()],
                model_name=os.getenv("OPENAI_CHAT_MODEL"),
                openai_api_base=openai_api_base,
                openai_api_key=openai_api_key,
                max_tokens=self.max_tokens,
                request_timeout=self.decision_timeout_sec,
                streaming=self.use_streaming,
                **chat_kwargs,
            )
        # [ADD] Added support for local Ollama models
        elif self.oai_api_type == "ollama":
            model_name = self.ollama_model_name
            api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
            self.ollama_api_base = api_base
            api_key = self.ollama_api_key

            if not model_name:
                raise ValueError("OLLAMA_CHAT_MODEL is not configured.")

            self._log_info(f"Using Local Ollama API: {model_name} at {api_base}")
            if self.ollama_use_native_chat:
                effective_mode = self._get_ollama_effective_think_mode()
                self._log_info(
                    f"[yellow]DriverAgent Ollama mode: native /api/chat | "
                    f"configured={self.ollama_use_native_chat_configured} "
                    f"reason={self.ollama_native_chat_resolution_reason} | "
                    f"think_mode={self.ollama_think_mode} | effective={effective_mode}[/yellow]"
                )
            else:
                self._log_info(
                    f"[yellow]DriverAgent Ollama mode: OpenAI-compatible /v1 | "
                    f"configured={self.ollama_use_native_chat_configured} "
                    f"reason={self.ollama_native_chat_resolution_reason} | "
                    f"think_mode={self.ollama_think_mode}[/yellow]"
                )

            # Keep OpenAI-compatible client as fallback for native failures.
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                openai_api_base=api_base,  # or base_url for newer langchain versions
                openai_api_key=api_key,
                max_tokens=self.max_tokens,
                request_timeout=self.decision_timeout_sec,
                streaming=self.use_streaming,
            )
        elif self.oai_api_type == "gemini":
            gemini_chat = _load_chat_google_generative_ai()
            if gemini_chat is None:
                raise ImportError(
                    "Gemini support requires 'langchain-google-genai'. Install with: pip install langchain-google-genai"
                ) from _GEMINI_IMPORT_ERROR
            model_name = os.getenv("GEMINI_CHAT_MODEL")
            api_key = os.getenv("GEMINI_API_KEY")
            if not model_name:
                raise ValueError("GEMINI_CHAT_MODEL is not configured.")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not configured.")
            self._log_info(f"Using Gemini API: {model_name}")
            self.llm = gemini_chat(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=self.max_tokens,
                timeout=self.decision_timeout_sec,
            )
        else:
            raise ValueError(f"Unknown OPENAI_API_TYPE: {self.oai_api_type}")

    def _initialize_scientific_runtime(
        self,
        runtime: ScientificEpisodeRuntime,
    ) -> None:
        if not isinstance(runtime, ScientificEpisodeRuntime):
            raise ValueError("scientific_runtime must be ScientificEpisodeRuntime.")
        runtime.validate_binding()
        config = runtime.harness_config
        self.scientific_runtime = runtime
        self.scientific_harness_config = config
        self.scientific_transport_client = runtime.transport_client
        self.scientific_generation_context = None
        self.oai_api_type = "ollama"
        self.decision_timeout_sec = config.transport.timeout_sec
        self.use_streaming = False
        self.enable_checker_llm = False
        self.prompt_profile = config.condition.policy_content.value
        self.enable_intent_resolver = False
        self.intent_resolver_api_type = "disabled"
        self.intent_resolver_model = ""
        self.intent_resolver_timeout_sec = 0.0
        self.intent_resolver_max_output_tokens = 1
        self.intent_resolver_abstain_on_ambiguous = True
        self.last_intent_resolver_prompt = None
        self.max_tokens = config.transport.max_output_tokens
        self.runtime_max_output_tokens = config.transport.max_output_tokens
        self.ollama_think_mode = config.transport.think_mode.value
        self.ollama_native_chat_timeout_sec = config.transport.timeout_sec
        self.ollama_native_num_ctx = config.transport.context_tokens
        self.ollama_native_keep_alive = None
        self.ollama_chat_url = runtime.runtime_lock.native_endpoint
        self.ollama_model_name = runtime.model_tag
        self.ollama_api_key = ""
        self.ollama_api_base = runtime.runtime_lock.native_endpoint
        self.ollama_native_chat_resolution = resolve_ollama_native_chat_mode(
            self.ollama_model_name,
            "true",
            self.ollama_think_mode,
        )
        self.ollama_use_native_chat = True
        self.ollama_use_native_chat_configured = "true"
        self.ollama_native_chat_resolution_reason = "scientific_runtime_lock"
        self.ollama_native_think_supported = True
        self.ollama_native_timed_out = False
        self.ollama_model_think_heuristic = False
        self.ollama_think_downgrade_noted = False
        self.last_ollama_transport = "ollama_native_chat"
        self.last_ollama_effective_think_mode = self.ollama_think_mode
        self.last_ollama_native_retry_used = False
        self.last_ollama_native_timeout = False
        self.last_ollama_native_timeout_short_circuit = False
        self.last_ollama_native_num_predict = None
        self.last_ollama_native_num_ctx = None
        self.last_ollama_native_keep_alive = None
        self.last_decision_meta = {}
        self.last_prompt_artifact = None
        self.last_generation_result = None
        self.last_action_resolution = None
        self.last_scientific_available_action_ids = None
        self.llm = None

    @property
    def _step_logs_enabled(self) -> bool:
        return self.verbose and (not self.quiet_mode)

    def _log_info(self, message: str) -> None:
        if not self.quiet_mode:
            print(message)

    def _log_step(self, message: str, *, end: str = "\n", flush: bool = False) -> None:
        if self._step_logs_enabled:
            print(message, end=end, flush=flush)

    def _log_warn(self, message: str) -> None:
        print(message)

    def _log_error(self, message: str) -> None:
        print(message)

    def set_ollama_native_chat_timeout_sec(self, timeout_sec: float) -> float:
        timeout_sec = max(1.0, float(timeout_sec))
        self.ollama_native_chat_timeout_sec = timeout_sec
        os.environ["OLLAMA_NATIVE_CHAT_TIMEOUT_SEC"] = str(timeout_sec)
        # Allow the next decision to re-attempt native chat using the updated timeout.
        self.ollama_native_timed_out = False
        return timeout_sec

    def set_decision_timeout_sec(self, timeout_sec: float) -> float:
        timeout_sec = max(1.0, float(timeout_sec))
        self.decision_timeout_sec = timeout_sec
        os.environ["DILU_DECISION_TIMEOUT_SEC"] = str(timeout_sec)
        try:
            if hasattr(self.llm, "request_timeout"):
                setattr(self.llm, "request_timeout", timeout_sec)
            if hasattr(self.llm, "timeout"):
                setattr(self.llm, "timeout", timeout_sec)
        except Exception:
            pass
        return timeout_sec

    def _valid_action_ids(self) -> List[int]:
        try:
            return sorted(
                int(action_id) for action_id in self.sce.available_action_ids()
            )
        except Exception:
            return [0, 1, 2, 3, 4]

    def _fallback_action_id(self) -> int:
        try:
            return int(self.sce.preferred_fallback_action_id())
        except Exception:
            valid = self._valid_action_ids()
            return 4 if 4 in valid else valid[0]

    def _decision_fallback_action_id(self) -> int:
        scientific_config = getattr(self, "scientific_harness_config", None)
        if scientific_config is None:
            return self._fallback_action_id()
        if not isinstance(scientific_config, HarnessConfig):
            raise ValueError("Scientific harness config must be a HarnessConfig.")
        scientific_config.validate_scientific()
        return require_fixed_idle_available(self._scientific_available_action_ids())

    def _scientific_available_action_ids(self) -> List[int]:
        try:
            available_action_ids = list(self.sce.available_action_ids())
        except Exception as exc:
            violation = ProtocolInvariantViolation.from_mapping(
                ProtocolInvariantCode.ACTION_AVAILABILITY_UNRESOLVED,
                "Scientific action availability could not be read from the environment.",
                {"error_type": type(exc).__name__},
            )
            raise RuntimeProtocolError(violation) from exc
        try:
            idle_action_id = int(self.sce.action_id_for_token("IDLE"))
        except Exception as exc:
            violation = ProtocolInvariantViolation.from_mapping(
                ProtocolInvariantCode.ACTION_TOKEN_MAPPING_MISMATCH,
                "Scientific IDLE action token could not be resolved.",
                {"error_type": type(exc).__name__},
            )
            raise RuntimeProtocolError(violation) from exc
        if idle_action_id != 1:
            violation = ProtocolInvariantViolation.from_mapping(
                ProtocolInvariantCode.ACTION_TOKEN_MAPPING_MISMATCH,
                "Scientific IDLE action token must map exactly to action ID 1.",
                {"observed_action_id": idle_action_id},
            )
            raise RuntimeProtocolError(violation)
        try:
            require_fixed_idle_available(available_action_ids)
        except ValueError as exc:
            violation = ProtocolInvariantViolation.from_mapping(
                ProtocolInvariantCode.ACTION_AVAILABILITY_UNRESOLVED,
                "Scientific action availability is outside the canonical domain.",
                {"error_type": type(exc).__name__},
            )
            raise RuntimeProtocolError(violation) from exc
        return sorted({int(action_id) for action_id in available_action_ids})

    def _resolve_scientific_action(
        self,
        raw_response: str,
        available_action_ids: List[int],
        *,
        timed_out: bool,
    ) -> ActionResolutionResult:
        scientific_config = getattr(self, "scientific_harness_config", None)
        if not isinstance(scientific_config, HarnessConfig):
            raise ValueError("Scientific harness config must be a HarnessConfig.")
        scientific_config.validate_scientific()
        resolution = resolve_action(
            raw_response,
            available_action_ids=available_action_ids,
            timed_out=timed_out,
            parser_mode=scientific_config.parser_mode,
            resolver_mode=scientific_config.resolver_mode,
            fallback_policy=scientific_config.fallback_policy,
        )
        self.last_action_resolution = resolution
        return resolution

    def _finalize_scientific_decision(
        self,
        *,
        response_content: str,
        human_message: str,
        response_diagnostics: Dict[str, Any],
        decision_meta: Dict[str, Any],
        token_usage: Dict[str, Any] | None,
        start_time: float,
        valid_action_ids: List[int],
        timed_out: bool,
    ) -> tuple[int, str, str, str]:
        resolution = self._resolve_scientific_action(
            response_content,
            valid_action_ids,
            timed_out=timed_out,
        )
        strict_valid = resolution.syntax_status is ActionSyntaxStatus.STRICT_VALID
        first_nonempty_line = next(
            (line.strip() for line in response_content.splitlines() if line.strip()),
            None,
        )
        parse_path = f"typed_{resolution.syntax_status.value}"
        if resolution.used_fallback:
            parse_path += "_fixed_idle_fallback"
        decision_meta.update(response_diagnostics)
        decision_meta.update(
            {
                "timed_out": timed_out,
                "used_fallback": resolution.used_fallback,
                "fallback_reason": (
                    resolution.violation.value
                    if resolution.used_fallback and resolution.violation is not None
                    else None
                ),
                "parse_mode": parse_path,
                "runtime_parse_path": parse_path,
                "original_selected_action": (
                    resolution.strict_action
                    if resolution.strict_action is not None
                    else resolution.recovered_action
                ),
                "selected_action": resolution.final_resolved_action,
                "final_action_source": (
                    "fixed_idle_fallback"
                    if resolution.used_fallback
                    else "strict_action"
                ),
                "response_contract_satisfied": strict_valid,
                "response_contract_recovered": False,
                "response_recovery_reason": None,
                "response_unparseable": not strict_valid,
                "response_action_line_present": resolution.strict_action is not None,
                "response_action_line_count": (
                    1 if resolution.strict_action is not None else 0
                ),
                "response_reason_line_present": False,
                "response_first_nonempty_line": (
                    None if first_nonempty_line is None else first_nonempty_line[:160]
                ),
                "decision_elapsed_sec": round(time.time() - start_time, 3),
            }
        )
        decision_meta.update(self._scientific_generation_metadata())
        if token_usage is None:
            token_usage = build_whitespace_estimate_token_usage(
                estimate_generated_tokens(response_content)
            )
        decision_meta.update(token_usage)
        self.last_decision_meta = decision_meta
        self._log_step(f"Result: {resolution.final_resolved_action}")
        return (
            resolution.final_resolved_action,
            response_content,
            human_message,
            "",
        )

    def _action_descriptions(self) -> dict[int, dict[str, str]]:
        try:
            return self.sce.action_catalog_with_descriptions()
        except Exception:
            return {}

    def _allowed_action_text(self) -> str:
        return ", ".join(str(action_id) for action_id in self._valid_action_ids())

    def _action_table_markdown(self) -> str:
        rows = [
            "| Action_id | Action Description |",
            "|-----------|--------------------|",
        ]
        action_descriptions = self._action_descriptions()
        for action_id in self._valid_action_ids():
            description = action_descriptions.get(int(action_id), {}).get(
                "description", f"Action {action_id}"
            )
            rows.append(f"| {action_id} | {description} |")
        return "\n".join(rows)

    def _intent_resolver_action_table(self) -> str:
        rows = [
            "| action_id | token | description |",
            "|-----------|-------|-------------|",
        ]
        action_descriptions = self._action_descriptions()
        for action_id in self._valid_action_ids():
            entry = action_descriptions.get(int(action_id), {})
            token = str(entry.get("token") or f"ACTION_{action_id}")
            description = str(entry.get("description") or f"Action {action_id}")
            rows.append(f"| {action_id} | {token} | {description} |")
        return "\n".join(rows)

    def _is_valid_action_id(self, value: int) -> bool:
        return int(value) in set(self._valid_action_ids())

    def _extract_valid_action_from_text(self, text: str) -> int:
        try:
            value = int(str(text).strip())
            if self._is_valid_action_id(value):
                return value
        except Exception:
            pass

        recovered = ACTION_RECOVERY_PATTERN.findall(text or "")
        for token in reversed(recovered):
            value = int(token)
            if self._is_valid_action_id(value):
                return value

        matches = ACTION_ANYWHERE_PATTERN.findall(text or "")
        for token in reversed(matches):
            value = int(token)
            if self._is_valid_action_id(value):
                return value

        raise ValueError("No valid action id found in text")

    def _default_action_catalog_entry(self, action_id: int) -> Dict[str, str]:
        defaults = {
            0: {"token": "LANE_LEFT", "description": "Turn-left"},
            1: {"token": "IDLE", "description": "IDLE"},
            2: {"token": "LANE_RIGHT", "description": "Turn-right"},
            3: {"token": "FASTER", "description": "Acceleration"},
            4: {"token": "SLOWER", "description": "Deceleration"},
        }
        return dict(
            defaults.get(
                int(action_id),
                {"token": f"ACTION_{action_id}", "description": f"Action {action_id}"},
            )
        )

    def _semantic_aliases_for_action(
        self, action_id: int, entry: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        action_id = int(action_id)
        default_entry = self._default_action_catalog_entry(action_id)
        token = str(
            entry.get("token") or default_entry.get("token") or f"ACTION_{action_id}"
        ).strip()
        description = str(
            entry.get("description")
            or default_entry.get("description")
            or f"Action {action_id}"
        ).strip()
        aliases = {token, description}
        if action_id == 0:
            aliases.update({"LANE_LEFT", "Turn-left", "turn left", "left lane"})
        elif action_id == 1:
            aliases.update({"IDLE", "keep lane", "maintain speed"})
        elif action_id == 2:
            aliases.update({"LANE_RIGHT", "Turn-right", "turn right", "right lane"})
        elif action_id == 3:
            aliases.update({"FASTER", "Acceleration", "accelerate"})
        elif action_id == 4:
            aliases.update({"SLOWER", "Deceleration", "decelerate"})
        cleaned_aliases = sorted(
            {alias.strip() for alias in aliases if str(alias or "").strip()},
            key=lambda value: (-len(value), value.lower()),
        )
        return token, cleaned_aliases

    def _semantic_action_segment(self, text: str) -> str:
        nonempty_lines = self._response_nonempty_lines(text)
        for line in nonempty_lines:
            if not RUNTIME_RESPONSE_PREFIX_PATTERN.match(line):
                continue
            segment = (
                line.split(delimiter, 1)[-1]
                if delimiter in line
                else line.split(":", 1)[-1]
            )
            return segment.strip()
        if delimiter in str(text or ""):
            return str(text or "").split(delimiter, 1)[-1].splitlines()[0].strip()
        return str(text or "").strip()

    @staticmethod
    def _semantic_normalize(text: str) -> str:
        lowered = str(text or "").lower()
        lowered = re.sub(r"[_\-]+", " ", lowered)
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
        return f" {lowered.strip()} "

    def _extract_semantic_action_from_text(self, text: str) -> tuple[int, str]:
        segment = self._semantic_action_segment(text)
        if not segment:
            raise ValueError("semantic label unavailable")
        normalized_segment = self._semantic_normalize(segment)
        action_descriptions = self._action_descriptions()
        all_action_ids = sorted(
            {
                int(action_id)
                for action_id in set(action_descriptions.keys())
                | {0, 1, 2, 3, 4}
                | set(self._valid_action_ids())
            }
        )
        matched: Dict[int, str] = {}
        for action_id in all_action_ids:
            entry = dict(self._default_action_catalog_entry(action_id))
            entry.update(action_descriptions.get(int(action_id), {}) or {})
            token, aliases = self._semantic_aliases_for_action(action_id, entry)
            for alias in aliases:
                normalized_alias = self._semantic_normalize(alias).strip()
                if not normalized_alias:
                    continue
                if re.search(
                    rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])",
                    normalized_segment,
                ):
                    matched[int(action_id)] = token
                    break
        if not matched:
            raise ValueError("semantic label unavailable")
        if len(matched) > 1:
            labels = ", ".join(matched[action_id] for action_id in sorted(matched))
            raise ValueError(f"semantic label ambiguous: {labels}")
        action_id, token = next(iter(matched.items()))
        if not self._is_valid_action_id(action_id):
            raise ValueError(
                f"semantic label unavailable for current action set: {token}"
            )
        return int(action_id), str(token)

    def _intent_resolver_enabled(self) -> bool:
        return bool(
            getattr(self, "enable_intent_resolver", False)
            and str(getattr(self, "intent_resolver_api_type", "ollama") or "")
            .strip()
            .lower()
            == "ollama"
            and str(getattr(self, "intent_resolver_model", "") or "").strip()
        )

    def _intent_resolver_prompt(self, raw_output: str) -> str:
        action_table = self._intent_resolver_action_table()
        allowed_actions = self._allowed_action_text()
        return textwrap.dedent(
            f"""\
        You are an action-intent decoder, not a driving policy.
        Decode the driver's intended action from the raw output only.
        Use only the valid action table below. Do not infer from any traffic scenario.
        Abstain with null when the intent is ambiguous, unsafe to infer, or not clearly mapped to one valid action.

        Valid action table:
        {action_table}

        Raw driver output:
        {raw_output}

        Return JSON only, with this exact schema:
        {{"action_id": <one of [{allowed_actions}] or null>, "confidence": "high"|"low", "reason": "<short reason>"}}
        """
        ).strip()

    def _invoke_intent_resolver_model(self, prompt: str) -> str:
        model_name = str(getattr(self, "intent_resolver_model", "") or "").strip()
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract one action id from malformed driver output. "
                        "Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": int(
                    getattr(self, "intent_resolver_max_output_tokens", 32) or 32
                ),
            },
        }
        payload = self._apply_ollama_think_mode(payload, "no_think")
        response = requests.post(
            self.ollama_chat_url,
            headers=self._ollama_request_headers(),
            json=payload,
            timeout=float(getattr(self, "intent_resolver_timeout_sec", 5.0) or 5.0),
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            message = data.get("message")
            if isinstance(message, dict):
                return str(message.get("content") or "")
            return str(data.get("response") or "")
        return str(data or "")

    @staticmethod
    def _parse_intent_resolver_json(text: str) -> Dict[str, Any]:
        content = str(text or "").strip()
        if not content:
            raise ValueError("empty intent resolver response")
        try:
            parsed = json.loads(content)
        except Exception:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match is None:
                raise ValueError("invalid intent resolver json")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("intent resolver response is not an object")
        return parsed

    def _resolve_action_with_intent_resolver(
        self, raw_output: str
    ) -> tuple[int | None, Dict[str, Any]]:
        metadata = {
            "intent_resolver_used": False,
            "intent_resolver_model": str(
                getattr(self, "intent_resolver_model", "") or ""
            )
            or None,
            "intent_resolver_action_id": None,
            "intent_resolver_abstained": False,
            "intent_resolver_reason": None,
        }
        if not self._intent_resolver_enabled():
            metadata["intent_resolver_reason"] = "disabled"
            return None, metadata
        metadata["intent_resolver_used"] = True
        prompt = self._intent_resolver_prompt(raw_output)
        self.last_intent_resolver_prompt = prompt
        try:
            resolver_text = self._invoke_intent_resolver_model(prompt)
            parsed = self._parse_intent_resolver_json(resolver_text)
        except TimeoutError:
            metadata["intent_resolver_reason"] = "timeout"
            return None, metadata
        except Exception as exc:
            metadata["intent_resolver_reason"] = (
                f"invalid_response: {type(exc).__name__}"
            )
            return None, metadata

        reason = str(parsed.get("reason") or "").strip()
        metadata["intent_resolver_reason"] = reason[:160] if reason else None
        confidence = str(parsed.get("confidence") or "").strip().lower()
        raw_action_id = parsed.get("action_id")
        if raw_action_id is None:
            metadata["intent_resolver_abstained"] = True
            return None, metadata
        if confidence != "high" and bool(
            getattr(self, "intent_resolver_abstain_on_ambiguous", True)
        ):
            metadata["intent_resolver_abstained"] = True
            return None, metadata
        try:
            action_id = int(raw_action_id)
        except Exception:
            metadata["intent_resolver_reason"] = (
                metadata["intent_resolver_reason"] or "invalid_action_id"
            )
            metadata["intent_resolver_abstained"] = True
            return None, metadata
        metadata["intent_resolver_action_id"] = int(action_id)
        if not self._is_valid_action_id(action_id):
            metadata["intent_resolver_reason"] = (
                metadata["intent_resolver_reason"] or "invalid_action_id"
            )
            metadata["intent_resolver_abstained"] = True
            return None, metadata
        metadata["intent_resolver_abstained"] = False
        return int(action_id), metadata

    def _response_nonempty_lines(self, text: str) -> List[str]:
        return [line.strip() for line in str(text or "").splitlines() if line.strip()]

    def _runtime_response_contract_diagnostics(
        self,
        text: str,
        response_diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        nonempty_lines = self._response_nonempty_lines(text)
        first_nonempty_line = nonempty_lines[0] if nonempty_lines else None
        response_action_line_present = any(
            line.lower().startswith("response to user:") for line in nonempty_lines
        )
        response_action_line_count = sum(
            1 for line in nonempty_lines if RUNTIME_RESPONSE_PREFIX_PATTERN.match(line)
        )
        response_reason_line_present = any(
            RUNTIME_REASON_LINE_PATTERN.fullmatch(line) is not None
            for line in nonempty_lines
        )
        response_contract_satisfied = False
        if first_nonempty_line is not None:
            match = RUNTIME_ACTION_LINE_PATTERN.fullmatch(first_nonempty_line)
            if match is not None:
                try:
                    response_contract_satisfied = self._is_valid_action_id(
                        int(match.group(1))
                    )
                except Exception:
                    response_contract_satisfied = False
        finish_reason = (
            str(response_diagnostics.get("response_finish_reason") or "")
            .strip()
            .upper()
        )
        response_truncated_before_contract = bool(
            finish_reason == "MAX_TOKENS" and not response_action_line_present
        )
        return {
            "response_contract_satisfied": bool(response_contract_satisfied),
            "response_contract_recovered": False,
            "response_recovery_reason": None,
            "response_unparseable": False,
            "response_action_line_present": bool(response_action_line_present),
            "response_action_line_count": int(response_action_line_count),
            "response_reason_line_present": bool(response_reason_line_present),
            "response_first_nonempty_line": (
                None if first_nonempty_line is None else first_nonempty_line[:160]
            ),
            "response_truncated_before_contract": bool(
                response_truncated_before_contract
            ),
        }

    def _mark_runtime_parse_result(
        self, decision_meta: Dict[str, Any], parse_path: str
    ) -> None:
        recovered = parse_path in RECOVERED_RUNTIME_PARSE_PATHS
        decision_meta["response_contract_recovered"] = bool(recovered)
        if parse_path == "missing_delimiter_recovered":
            decision_meta["response_recovery_reason"] = "missing_delimiter_action_line"
        elif recovered:
            decision_meta["response_recovery_reason"] = "non_strict_runtime_parse"
        else:
            decision_meta["response_recovery_reason"] = None
        decision_meta["response_unparseable"] = False

    def _mark_runtime_parse_failure(
        self, decision_meta: Dict[str, Any], parse_path: str
    ) -> None:
        decision_meta["response_contract_recovered"] = False
        decision_meta["response_recovery_reason"] = None
        decision_meta["response_unparseable"] = (
            parse_path in UNPARSEABLE_RUNTIME_PARSE_PATHS
        )

    def _extract_runtime_action_from_text(self, text: str) -> tuple[int, str]:
        nonempty_lines = self._response_nonempty_lines(text)
        if nonempty_lines:
            first_nonempty_line = nonempty_lines[0]
            match = RUNTIME_ACTION_LINE_PATTERN.fullmatch(first_nonempty_line)
            if match is not None:
                value = int(match.group(1))
                if self._is_valid_action_id(value):
                    return value, "strict_action_line"
            missing_delimiter_match = (
                RUNTIME_MISSING_DELIMITER_ACTION_LINE_PATTERN.fullmatch(
                    first_nonempty_line
                )
            )
            if missing_delimiter_match is not None:
                response_lines = [
                    line
                    for line in nonempty_lines
                    if RUNTIME_RESPONSE_PREFIX_PATTERN.match(line)
                ]
                if len(response_lines) != 1:
                    raise ValueError("Ambiguous runtime response action lines")
                value = int(missing_delimiter_match.group(1))
                if self._is_valid_action_id(value):
                    return value, "missing_delimiter_recovered"
        for line in nonempty_lines:
            match = RUNTIME_LABELED_ACTION_PATTERN.fullmatch(line)
            if match is None:
                continue
            value = _safe_int(match.group(1), default=-999)
            if self._is_valid_action_id(value):
                return value, "labeled_backup"
        raise ValueError("No valid runtime action line found in text")

    def _response_diagnostics_from_message(
        self, response: Any, response_content: str
    ) -> Dict[str, Any]:
        response_metadata = getattr(response, "response_metadata", None) or {}
        usage_metadata = getattr(response, "usage_metadata", None) or {}
        output_token_details = {}
        if isinstance(usage_metadata, dict):
            output_token_details = usage_metadata.get("output_token_details") or {}
            if not isinstance(output_token_details, dict):
                output_token_details = {}
        visible_content = str(response_content or "").strip()
        return {
            "response_finish_reason": (
                str(response_metadata.get("finish_reason"))
                if isinstance(response_metadata, dict)
                and response_metadata.get("finish_reason") is not None
                else None
            ),
            "response_model_provider": (
                str(response_metadata.get("model_provider"))
                if isinstance(response_metadata, dict)
                and response_metadata.get("model_provider") is not None
                else None
            ),
            "response_visible_chars": len(visible_content),
            "response_empty": len(visible_content) == 0,
            "reasoning_tokens": _safe_int(
                output_token_details.get("reasoning"), default=0
            ),
            "output_tokens": (
                _safe_int(usage_metadata.get("output_tokens"), default=0)
                if isinstance(usage_metadata, dict)
                else 0
            ),
        }

    def _empty_selector_fallback_reason(
        self, response_content: str, response_diagnostics: Dict[str, Any]
    ) -> str:
        if str(response_content or "").strip():
            return "parse_fallback"
        if (
            str(response_diagnostics.get("response_finish_reason") or "")
            .strip()
            .upper()
            == "MAX_TOKENS"
        ):
            return "max_tokens_empty_response_fallback"
        return "empty_response_fallback"

    def _highway_flow_safety_rules(self) -> str:
        scenario_family = getattr(self.sce, "scenario_family", lambda: "highway")()
        if str(scenario_family or "highway").strip().lower() != "highway":
            return ""
        return textwrap.dedent(
            """\
        HIGHWAY TRAFFIC-FLOW SAFETY:
        - Unnecessary stopping or near-stopping in a live highway lane is unsafe because it creates rear-end collision risk.
        - Do not repeatedly choose Deceleration or IDLE until the ego vehicle stops unless an immediate front-collision risk makes it unavoidable.
        - If ego speed is already very low and the front gap is safe, prefer Acceleration or steady lane keeping over further Deceleration.
        - After Deceleration resolves a front-risk situation, recover toward the current lane's traffic-flow band instead of camping at the lowest available highway speed.
        - In the rightmost cruising lane, 60 km/h is a slow-flow floor, not the default cruising target when the road ahead is clear.
        - If ego is in the rightmost lane, below 80 km/h, and the front gap/TTC is safe or not closing, prefer Acceleration over repeated IDLE.
        - Do not accelerate behind a slower same-lane lead car unless the projected front gap and projected TTC remain safe after acceleration.
        - If same-lane front TTC is collapsing, choose Deceleration early rather than waiting until the final unsafe gap.
        - Treat safe progress and maintaining traffic flow as part of safety, not merely efficiency.
        - Lane ranks use right-hand traffic: lane_rank=0 is the leftmost/overtaking lane, and the maximum lane_rank is the rightmost cruising lane.
        - Highway speed discipline: right/slow flow is 60-100 km/h, middle/normal flow is 100-120 km/h, and left/overtaking flow is 120-130 km/h.
        - Low-speed left-lane camping is a traffic-flow safety issue; if ego is in a faster lane, below that lane's flow band, and not actively overtaking, prefer a safe LANE_RIGHT when target-lane gaps are clear.
        - Hard lane-change safety overrides lane discipline: never change lanes if front or rear target-lane gaps are unsafe or uncertain.
        """
        )

    def _build_system_message(self, fallback_action_id: int) -> str:
        scientific_config = getattr(self, "scientific_harness_config", None)
        if scientific_config is not None:
            if not isinstance(scientific_config, HarnessConfig):
                raise ValueError("Scientific harness config must be a HarnessConfig.")
            scientific_config.validate_scientific()
            if fallback_action_id != 1:
                raise ValueError("Scientific fixed-IDLE fallback requires action ID 1.")
            artifact = build_prompt_artifact(
                scientific_config.condition.policy_content,
                output_enforcement=scientific_config.condition.output_enforcement,
                few_shot_num=0,
            )
            self.last_prompt_artifact = artifact
            return artifact.system_prompt()

        if (
            normalize_prompt_profile(getattr(self, "prompt_profile", "harness_v2"))
            == "legacy_dilu_like"
        ):
            return self._build_legacy_dilu_like_system_message()

        scenario_family = getattr(self.sce, "scenario_family", lambda: "highway")()
        allowed_actions = self._allowed_action_text()
        if scenario_family == "intersection":
            feasibility_check = "Confirm that the action is compatible with an intersection-only longitudinal decision."
            drive_logic = textwrap.dedent(
                f"""\
            ### DRIVE LOGIC:
            1. SAFETY: Avoid entering a conflict zone unless the gap is clearly safe. If uncertain, use the safer slower action when available.
            2. PROGRESS: Once a safe gap exists, continue through the intersection without unnecessary hesitation.
            3. NO LANE CHANGES: Treat this as a longitudinal-only decision problem. Do not invent lane changes.
            4. EFFICIENCY: Prefer smooth progress, but only after safety is satisfied.
            """
            )
        elif scenario_family == "merge":
            feasibility_check = "Confirm that the action preserves a safe merge gap and does not force nearby traffic."
            drive_logic = textwrap.dedent(
                f"""\
            ### DRIVE LOGIC:
            1. SAFETY: Maintain a safe gap to nearby traffic and avoid forcing a merge.
            2. MERGE PROGRESS: If a safe mainline gap exists, prioritize merging smoothly into the target lane.
            3. NO-UNNECESSARY-BRAKING: Avoid full hesitation when a safe merge opportunity is already present.
            4. EFFICIENCY: Preserve stable speed and lane control while completing the merge.
            """
            )
        else:
            feasibility_check = "Confirm lane-change feasibility: target-lane front and rear vehicles must both leave clear safety gaps."
            drive_logic = textwrap.dedent(
                f"""\
            ### DRIVE LOGIC:
            1. SAFETY: If a lead car is closer than 25m and your speed is higher, you should prefer the safer slower action when available.
            2. NO-UNNECESSARY-LANE-CHANGE: If the lead car in your current lane is not slower than you (or not close enough to block you), do not change lane just for preference.
            3. EFFICIENCY: Once front-collision risk is resolved, recover toward the current lane's flow band instead of camping at the lowest speed.
            4. TRAFFIC RULE (RIGHT-HAND TRAFFIC): Prefer overtaking from the LEFT lane when overtaking is necessary and safe.
            """
            )

        return textwrap.dedent(
            f"""\
        You are an autonomous driving decision module.
        Think privately about safety, traffic flow, and action validity, but do not output your private reasoning.

        {drive_logic}
        {self._highway_flow_safety_rules()}

        PRIVATE DECISION CHECKLIST - DO NOT OUTPUT:
        1. Safety check: reject collision-prone actions and unavailable action ids.
        2. Check front collision risk, target-lane front/rear gaps, and projected TTC.
        3. Efficiency check: keep safe traffic flow and recover speed after immediate risk clears.
        4. {feasibility_check}
        5. Choose the safest valid action id from the allowed set.

        HARD LANE-CHANGE SAFETY RULES:
        - Never choose a lane-change action if a target-lane vehicle is within 15 m ahead or behind the ego vehicle.
        - Never choose a lane-change action if target-lane front/rear distance is unknown or ambiguous.
        - If a lane-change action is unsafe or unavailable, choose a longitudinal action that preserves safety and traffic flow.

        RESPONSE CONTRACT:
        - Output at most two non-empty lines.
        - First non-empty line must be exactly:
        Response to user:{delimiter} <action_id>
        - Optional second line may be:
        Reason: <one short sentence>
        - Use exactly one real integer action id from: {allowed_actions}
        - Do not output angle brackets, markdown, bullet points, code fences, JSON, or step-by-step reasoning.
        - Do not omit "{delimiter}"; a missing delimiter can be recovered but is counted as a runtime contract failure.
        """
        )

    def _build_legacy_dilu_like_system_message(self) -> str:
        allowed_actions = self._allowed_action_text()
        return textwrap.dedent(
            f"""\
        You are an autonomous driving decision module.
        Select one high-level discrete action for the ego vehicle from the available action set.

        Available action ids: {allowed_actions}
        - 0: change lane left
        - 1: keep lane / idle
        - 2: change lane right
        - 3: accelerate
        - 4: decelerate

        Prefer safe behavior and avoid collisions. Consider surrounding vehicles, current speed,
        and lane availability before choosing an action.

        RESPONSE FORMAT:
        Response to user:{delimiter} <action_id>
        """
        )

    def _run_with_timeout(self, fn, *args):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn, *args)
        try:
            return future.result(timeout=self.decision_timeout_sec)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Decision timeout after {self.decision_timeout_sec:.1f}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @contextmanager
    def _temporary_max_output_tokens(self, max_output_tokens_override: int | None):
        if max_output_tokens_override is None:
            yield
            return
        attr_names = ("max_tokens", "max_output_tokens")
        original_values = {}
        for attr_name in attr_names:
            if hasattr(self.llm, attr_name):
                try:
                    original_values[attr_name] = getattr(self.llm, attr_name)
                    setattr(self.llm, attr_name, int(max_output_tokens_override))
                except Exception:
                    pass
        try:
            yield
        finally:
            for attr_name, value in original_values.items():
                try:
                    setattr(self.llm, attr_name, value)
                except Exception:
                    pass

    def _runtime_response_diagnostics_for_text(
        self,
        response_content: str,
        token_usage: Dict[str, Any] | None,
        *,
        finish_reason: str | None = None,
    ) -> Dict[str, Any]:
        token_usage = token_usage or {}
        visible_content = str(response_content or "").strip()
        return {
            "response_finish_reason": finish_reason,
            "response_model_provider": None,
            "response_visible_chars": len(visible_content),
            "response_empty": len(visible_content) == 0,
            "reasoning_tokens": 0,
            "output_tokens": _safe_int(token_usage.get("completion_tokens"), default=0),
        }

    def _ollama_native_runtime_diagnostics(self) -> Dict[str, Any]:
        return {
            "ollama_native_num_predict": self.last_ollama_native_num_predict,
            "ollama_native_num_ctx": self.last_ollama_native_num_ctx,
            "ollama_native_keep_alive": self.last_ollama_native_keep_alive,
        }

    def _scientific_generation_request(
        self,
        messages,
        max_output_tokens_override: int | None,
    ) -> GenerationRequest:
        config = getattr(self, "scientific_harness_config", None)
        context = getattr(self, "scientific_generation_context", None)
        if not isinstance(config, HarnessConfig):
            raise ValueError("Scientific generation requires HarnessConfig.")
        if not isinstance(context, ScientificGenerationContext):
            raise ValueError("Scientific generation context is not configured.")
        config.validate_scientific()
        if self.oai_api_type != "ollama":
            raise ValueError("Scientific generation requires native Ollama.")
        if max_output_tokens_override not in {
            None,
            config.transport.max_output_tokens,
        }:
            raise ValueError("Scientific max-output override drifted from the config.")
        native_messages = tuple(
            (item["role"], item["content"])
            for item in self._to_ollama_messages(messages)
        )
        available_action_ids = self._scientific_grounded_available_action_ids(config)
        return GenerationRequest(
            model_tag=self.ollama_model_name,
            model_digest=context.model_digest,
            request_id=context.request_id,
            messages=native_messages,
            native_endpoint=self.ollama_chat_url,
            options=NativeGenerationOptions(
                seed=context.generation_seed,
                temperature=config.transport.temperature,
                num_ctx=config.transport.context_tokens,
                num_predict=config.transport.max_output_tokens,
            ),
            output_enforcement=config.condition.output_enforcement,
            think_mode=config.transport.think_mode,
            timeout_sec=config.transport.timeout_sec,
            available_action_ids=available_action_ids,
        )

    def _scientific_grounded_available_action_ids(
        self,
        config: HarnessConfig,
    ) -> tuple[int, ...] | None:
        """Thread the current decision's availability into O2 requests only.

        Every other condition (O0/O1) must receive `None` here so the
        canonical, static enum keeps building exactly as before.
        """
        grounded = OutputEnforcement.BACKEND_SCHEMA_GROUNDED
        if config.condition.output_enforcement is not grounded:
            return None
        available = getattr(self, "last_scientific_available_action_ids", None)
        if not isinstance(available, tuple) or not available:
            violation = ProtocolInvariantViolation.from_mapping(
                ProtocolInvariantCode.ACTION_AVAILABILITY_UNRESOLVED,
                "Grounded generation requires resolved action availability.",
            )
            raise RuntimeProtocolError(violation)
        return tuple(int(action_id) for action_id in available)

    def _invoke_scientific_response(
        self,
        messages,
        max_output_tokens_override: int | None,
    ) -> tuple[str, Dict[str, Any] | None, Dict[str, Any]]:
        client = getattr(self, "scientific_transport_client", None)
        if not isinstance(client, OllamaScientificClient):
            raise ValueError("Scientific transport client is not configured.")
        request = self._scientific_generation_request(
            messages,
            max_output_tokens_override,
        )
        result = client.generate(request)
        self.last_generation_result = result
        if result.requires_cell_abort:
            raise ScientificGenerationAbort(result)
        if result.error_class is RuntimeFailureClass.GENERATION_TIMEOUT:
            raise ScientificGenerationTimeout(result)
        if result.contract_text is None:
            raise ValueError("Scientific generation produced no contract text.")
        token_usage = None
        if result.prompt_tokens is not None and result.completion_tokens is not None:
            token_usage = {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "token_count_method": "ollama_native_payload",
                "token_usage_source": "ollama_native_scientific",
            }
        diagnostics = self._runtime_response_diagnostics_for_text(
            result.contract_text,
            token_usage,
            finish_reason=result.stop_reason,
        )
        diagnostics["response_model_provider"] = "ollama_native_scientific"
        return result.contract_text, token_usage, diagnostics

    def _scientific_generation_metadata(self) -> Dict[str, Any]:
        result = getattr(self, "last_generation_result", None)
        if not isinstance(result, GenerationResult):
            return {}
        return {
            "scientific_request_id": result.request_id,
            "scientific_attempt_ids": list(result.attempt_ids),
            "model_digest": result.model_digest,
            "generation_seed": result.options.seed,
            "scientific_native_endpoint": result.native_endpoint,
            "scientific_output_enforcement": result.output_enforcement.value,
            "scientific_think_mode": result.think_mode.value,
            "scientific_response_body": result.response_body,
            "scientific_raw_response": result.raw_response,
            "scientific_contract_text": result.contract_text,
            "scientific_transport_error_body": result.transport_error_body,
            "scientific_stop_reason": result.stop_reason,
            "scientific_latency_ms": result.latency_ms,
            "scientific_identity_latency_ms": result.identity_latency_ms,
            "scientific_generation_latency_ms": result.generation_latency_ms,
            "scientific_retry_cooldown_ms": result.retry_cooldown_ms,
            "scientific_retry_cooldown_policy_ms": (result.retry_cooldown_policy_ms),
            "scientific_identity_checks": [
                {
                    "attempt_index": check.attempt_index,
                    "phase": check.phase,
                    "observed_model_tag": check.observed_model_tag,
                    "observed_model_digest": check.observed_model_digest,
                    "latency_ms": check.latency_ms,
                    "error_message": check.error_message,
                }
                for check in result.identity_checks
            ],
            "scientific_backend_total_duration_ns": (
                result.backend_timing.total_duration_ns
                if result.backend_timing is not None
                else None
            ),
            "scientific_backend_load_duration_ns": (
                result.backend_timing.load_duration_ns
                if result.backend_timing is not None
                else None
            ),
            "scientific_backend_prompt_eval_duration_ns": (
                result.backend_timing.prompt_eval_duration_ns
                if result.backend_timing is not None
                else None
            ),
            "scientific_backend_eval_duration_ns": (
                result.backend_timing.eval_duration_ns
                if result.backend_timing is not None
                else None
            ),
            "scientific_transport_succeeded": result.transport_succeeded,
            "scientific_generation_error": (
                result.error_class.value if result.error_class is not None else None
            ),
        }

    def _invoke_response_with_diagnostics(
        self,
        messages,
        max_output_tokens_override: int | None = None,
    ) -> tuple[str, Dict[str, Any] | None, Dict[str, Any]]:
        if getattr(self, "scientific_harness_config", None) is not None:
            return self._invoke_scientific_response(
                messages,
                max_output_tokens_override,
            )
        if self.oai_api_type == "ollama" and self.ollama_use_native_chat:
            content, _thinking, usage = self._ollama_native_invoke(messages)
            diagnostics = self._runtime_response_diagnostics_for_text(content, usage)
            diagnostics.update(self._ollama_native_runtime_diagnostics())
            return content, usage, diagnostics

        if self.oai_api_type == "ollama":
            content, usage, finish_reason = self._ollama_openai_compat_invoke(
                messages,
                max_output_tokens=max_output_tokens_override,
            )
            diagnostics = self._runtime_response_diagnostics_for_text(
                content,
                usage,
                finish_reason=finish_reason,
            )
            return content, usage, diagnostics

        with self._temporary_max_output_tokens(max_output_tokens_override):
            response = self._run_with_timeout(self.llm.invoke, messages)
        content = _content_to_text(getattr(response, "content", ""))
        token_usage = build_token_usage_record_from_langchain_message(response)
        diagnostics = self._response_diagnostics_from_message(response, content)
        return content, token_usage, diagnostics

    def _invoke_response(self, messages) -> str:
        content, token_usage, _diagnostics = self._invoke_response_with_diagnostics(
            messages
        )
        return content, token_usage

    def _stream_response(self, messages) -> str:
        if self.oai_api_type == "ollama" and self.ollama_use_native_chat:
            content, _thinking, usage = self._ollama_native_stream(messages)
            return content, usage

        def _collect_stream(msgs):
            chunks = []
            token_usage = None
            for chunk in self.llm.stream(msgs):
                chunk_text = _content_to_text(getattr(chunk, "content", ""))
                if chunk_text:
                    chunks.append(chunk_text)
                    self._log_step(chunk_text, end="", flush=True)
                chunk_usage = build_token_usage_record_from_langchain_message(chunk)
                if chunk_usage is not None:
                    token_usage = chunk_usage
            return "".join(chunks), token_usage

        return self._run_with_timeout(_collect_stream, messages)

    def _to_ollama_messages(self, messages) -> list:
        payload_messages = []
        for msg in messages:
            payload_messages.append(
                {
                    "role": _ollama_role_from_message(msg),
                    "content": _content_to_text(getattr(msg, "content", "")),
                }
            )
        return payload_messages

    def _get_ollama_effective_think_mode(self) -> str:
        mode = self.ollama_think_mode
        if mode != "think":
            return mode
        if self.ollama_native_think_supported is False:
            return "auto"
        if self.ollama_native_think_supported is True:
            return "think"
        if not self.ollama_model_think_heuristic:
            self.ollama_native_think_supported = False
            if not self.ollama_think_downgrade_noted:
                self._log_warn(
                    f"[yellow]Native Ollama think flag is likely unsupported for {self.ollama_model_name}. "
                    "Using effective think_mode=auto.[/yellow]"
                )
                self.ollama_think_downgrade_noted = True
            return "auto"
        return "think"

    def _apply_ollama_think_mode(self, payload: dict, mode: str | None = None) -> dict:
        mode = _normalize_ollama_think_mode(
            mode or self._get_ollama_effective_think_mode()
        )
        if mode == "think":
            payload["think"] = True
        elif mode == "no_think":
            payload["think"] = False
        return payload

    def _ollama_native_options(self) -> Dict[str, Any]:
        output_cap = max(
            1,
            int(
                getattr(self, "runtime_max_output_tokens", None)
                or getattr(self, "max_tokens", 512)
                or 512
            ),
        )
        options: Dict[str, Any] = {
            "num_predict": output_cap,
            "temperature": float(getattr(self, "temperature", 0.0) or 0.0),
        }
        self.last_ollama_native_num_predict = output_cap
        num_ctx = getattr(self, "ollama_native_num_ctx", None)
        if num_ctx is not None:
            options["num_ctx"] = int(num_ctx)
            self.last_ollama_native_num_ctx = int(num_ctx)
        else:
            self.last_ollama_native_num_ctx = None
        return options

    def _ollama_native_keep_alive(self) -> str | None:
        keep_alive = getattr(self, "ollama_native_keep_alive", None)
        self.last_ollama_native_keep_alive = keep_alive
        return keep_alive

    def _ollama_request_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.ollama_api_key}"}

    def _ollama_openai_compat_url(self) -> str:
        raw_base = (
            str(getattr(self, "ollama_api_base", "") or "").strip()
            or str(os.getenv("OLLAMA_API_BASE", "") or "").strip()
            or str(getattr(self, "ollama_chat_url", "") or "").strip()
            or "http://localhost:11434/v1"
        )
        base = raw_base.rstrip("/")
        if base.endswith("/v1/chat/completions"):
            return base
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/api/chat"):
            return base[: -len("/api/chat")] + "/v1/chat/completions"
        if base.endswith("/api"):
            return base[: -len("/api")] + "/v1/chat/completions"
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _ollama_openai_compat_usage_record(self, usage: Any) -> Dict[str, Any] | None:
        if not isinstance(usage, dict):
            return None
        prompt_tokens = _safe_int(usage.get("prompt_tokens"), default=0)
        completion_tokens = _safe_int(usage.get("completion_tokens"), default=0)
        total_tokens = _safe_int(
            usage.get("total_tokens"), default=prompt_tokens + completion_tokens
        )
        return {
            "prompt_tokens": max(0, int(prompt_tokens)),
            "completion_tokens": max(0, int(completion_tokens)),
            "total_tokens": max(0, int(total_tokens)),
            "token_count_method": "ollama_openai_usage",
            "token_usage_source": "openai_compat",
        }

    def _ollama_openai_compat_invoke_once(
        self, messages, max_output_tokens: int | None = None
    ):
        output_cap = max(
            1,
            int(
                max_output_tokens
                or getattr(self, "runtime_max_output_tokens", None)
                or getattr(self, "max_tokens", 512)
                or 512
            ),
        )
        payload = {
            "model": self.ollama_model_name,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
            "temperature": float(getattr(self, "temperature", 0.0) or 0.0),
            "max_tokens": output_cap,
        }
        response = requests.post(
            self._ollama_openai_compat_url(),
            json=payload,
            headers=self._ollama_request_headers(),
            timeout=float(getattr(self, "decision_timeout_sec", 60.0) or 60.0),
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        content = _content_to_text((message or {}).get("content", ""))
        finish_reason = (
            choice.get("finish_reason") if isinstance(choice, dict) else None
        )
        usage = self._ollama_openai_compat_usage_record(
            data.get("usage") if isinstance(data, dict) else None
        )
        return content, usage, finish_reason

    def _ollama_openai_compat_invoke(
        self, messages, max_output_tokens: int | None = None
    ):
        self.last_ollama_transport = "openai_compat_direct"
        self.last_ollama_effective_think_mode = self._get_ollama_effective_think_mode()
        self.last_ollama_native_retry_used = False
        self.last_ollama_native_timeout = False
        self.last_ollama_native_timeout_short_circuit = False
        return self._run_with_timeout(
            self._ollama_openai_compat_invoke_once,
            messages,
            max_output_tokens,
        )

    def _effective_ollama_native_timeout_sec(self) -> float:
        # Decision timeout is the hard cap in timeout-only policy mode.
        return max(
            1.0,
            min(
                float(self.decision_timeout_sec),
                float(self.ollama_native_chat_timeout_sec),
            ),
        )

    def _ollama_native_invoke_once(self, messages, think_mode: str):
        payload = {
            "model": self.ollama_model_name,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
            "options": self._ollama_native_options(),
        }
        keep_alive = self._ollama_native_keep_alive()
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        payload = self._apply_ollama_think_mode(payload, mode=think_mode)
        response = requests.post(
            self.ollama_chat_url,
            json=payload,
            headers=self._ollama_request_headers(),
            timeout=self._effective_ollama_native_timeout_sec(),
        )
        response.raise_for_status()
        data = response.json()
        msg = data.get("message", {}) or {}
        return (
            _content_to_text(msg.get("content", "")),
            _content_to_text(msg.get("thinking", "")),
            build_token_usage_record_from_ollama_native_payload(data),
        )

    def _ollama_native_stream_once(self, messages, think_mode: str):
        payload = {
            "model": self.ollama_model_name,
            "messages": self._to_ollama_messages(messages),
            "stream": True,
            "options": self._ollama_native_options(),
        }
        keep_alive = self._ollama_native_keep_alive()
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        payload = self._apply_ollama_think_mode(payload, mode=think_mode)
        content_chunks = []
        thinking_chunks = []
        final_usage = None
        with requests.post(
            self.ollama_chat_url,
            json=payload,
            headers=self._ollama_request_headers(),
            timeout=self._effective_ollama_native_timeout_sec(),
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    line = raw_line.decode("utf-8")
                    data = json.loads(line)
                except Exception:
                    continue
                msg = data.get("message", {}) or {}
                chunk_text = _content_to_text(msg.get("content", ""))
                chunk_thinking = _content_to_text(msg.get("thinking", ""))
                if chunk_text:
                    content_chunks.append(chunk_text)
                    self._log_step(chunk_text, end="", flush=True)
                if chunk_thinking:
                    thinking_chunks.append(chunk_thinking)
                if data.get("done"):
                    final_usage = build_token_usage_record_from_ollama_native_payload(
                        data
                    )
                    break
        return "".join(content_chunks), "".join(thinking_chunks), final_usage

    def _ollama_native_invoke(self, messages):
        requested_mode = self.ollama_think_mode
        effective_mode = self._get_ollama_effective_think_mode()
        if self.ollama_native_timed_out:
            self.last_ollama_transport = "native_timeout_short_circuit"
            self.last_ollama_effective_think_mode = effective_mode
            self.last_ollama_native_retry_used = False
            self.last_ollama_native_timeout = True
            self.last_ollama_native_timeout_short_circuit = True
            raise TimeoutError(
                f"Native Ollama timeout short-circuit active for {self.ollama_model_name}"
            )
        self.last_ollama_transport = "native"
        self.last_ollama_effective_think_mode = effective_mode
        self.last_ollama_native_retry_used = False
        self.last_ollama_native_timeout = False
        self.last_ollama_native_timeout_short_circuit = False
        try:
            result = self._run_with_timeout(
                self._ollama_native_invoke_once, messages, effective_mode
            )
            if effective_mode == "think":
                self.ollama_native_think_supported = True
            return result
        except Exception as exc:
            if _is_timeout_exception(exc):
                self.ollama_native_timed_out = True
                self.last_ollama_transport = "native_timeout"
                self.last_ollama_native_timeout = True
                self._log_warn(
                    f"[yellow]Native Ollama chat timed out for {self.ollama_model_name}. "
                    "Skipping /v1 retry and using timeout safety fallback.[/yellow]"
                )
                raise TimeoutError(str(exc))
            if (
                requested_mode == "think"
                and effective_mode == "think"
                and isinstance(exc, requests.HTTPError)
            ):
                self.ollama_native_think_supported = False
                self.last_ollama_native_retry_used = True
                self.last_ollama_effective_think_mode = "auto"
                self._log_warn(
                    f"[yellow]Native Ollama rejected think=true for {self.ollama_model_name}. "
                    "Retrying without think flag.[/yellow]"
                )
                try:
                    result = self._run_with_timeout(
                        self._ollama_native_invoke_once, messages, "auto"
                    )
                    if not self.ollama_think_downgrade_noted:
                        self._log_warn(
                            f"[yellow]Native Ollama continuing with effective think_mode=auto for "
                            f"{self.ollama_model_name}.[/yellow]"
                        )
                        self.ollama_think_downgrade_noted = True
                    return result
                except Exception as retry_exc:
                    if _is_timeout_exception(retry_exc):
                        self.ollama_native_timed_out = True
                        self.last_ollama_transport = "native_timeout"
                        self.last_ollama_native_timeout = True
                        self._log_warn(
                            f"[yellow]Native Ollama retry without think timed out for {self.ollama_model_name}. "
                            "Skipping /v1 retry and using timeout safety fallback.[/yellow]"
                        )
                        raise TimeoutError(str(retry_exc))
                    self.last_ollama_transport = "openai_compat_fallback"
                    self._log_warn(
                        f"[yellow]Native Ollama retry without think failed ({type(retry_exc).__name__}). "
                        "Falling back to OpenAI-compatible path.[/yellow]"
                    )
                    response = self._run_with_timeout(self.llm.invoke, messages)
                    return (
                        _content_to_text(getattr(response, "content", "")),
                        "",
                        build_token_usage_record_from_langchain_message(response),
                    )
            self.last_ollama_transport = "openai_compat_fallback"
            self._log_warn(
                f"[yellow]Native Ollama chat failed ({type(exc).__name__}). Falling back to OpenAI-compatible path.[/yellow]"
            )
            response = self._run_with_timeout(self.llm.invoke, messages)
            return (
                _content_to_text(getattr(response, "content", "")),
                "",
                build_token_usage_record_from_langchain_message(response),
            )

    def _ollama_native_stream(self, messages):
        requested_mode = self.ollama_think_mode
        effective_mode = self._get_ollama_effective_think_mode()
        if self.ollama_native_timed_out:
            self.last_ollama_transport = "native_timeout_short_circuit"
            self.last_ollama_effective_think_mode = effective_mode
            self.last_ollama_native_retry_used = False
            self.last_ollama_native_timeout = True
            self.last_ollama_native_timeout_short_circuit = True
            raise TimeoutError(
                f"Native Ollama timeout short-circuit active for {self.ollama_model_name}"
            )
        self.last_ollama_transport = "native"
        self.last_ollama_effective_think_mode = effective_mode
        self.last_ollama_native_retry_used = False
        self.last_ollama_native_timeout = False
        self.last_ollama_native_timeout_short_circuit = False
        try:
            result = self._run_with_timeout(
                self._ollama_native_stream_once, messages, effective_mode
            )
            if effective_mode == "think":
                self.ollama_native_think_supported = True
            return result
        except Exception as exc:
            if _is_timeout_exception(exc):
                self.ollama_native_timed_out = True
                self.last_ollama_transport = "native_timeout"
                self.last_ollama_native_timeout = True
                self._log_warn(
                    f"[yellow]Native Ollama stream timed out for {self.ollama_model_name}. "
                    "Skipping /v1 retry and using timeout safety fallback.[/yellow]"
                )
                raise TimeoutError(str(exc))
            if (
                requested_mode == "think"
                and effective_mode == "think"
                and isinstance(exc, requests.HTTPError)
            ):
                self.ollama_native_think_supported = False
                self.last_ollama_native_retry_used = True
                self.last_ollama_effective_think_mode = "auto"
                self._log_warn(
                    f"[yellow]Native Ollama rejected think=true for {self.ollama_model_name}. "
                    "Retrying without think flag.[/yellow]"
                )
                try:
                    result = self._run_with_timeout(
                        self._ollama_native_stream_once, messages, "auto"
                    )
                    if not self.ollama_think_downgrade_noted:
                        self._log_warn(
                            f"[yellow]Native Ollama continuing with effective think_mode=auto for "
                            f"{self.ollama_model_name}.[/yellow]"
                        )
                        self.ollama_think_downgrade_noted = True
                    return result
                except Exception as retry_exc:
                    if _is_timeout_exception(retry_exc):
                        self.ollama_native_timed_out = True
                        self.last_ollama_transport = "native_timeout"
                        self.last_ollama_native_timeout = True
                        self._log_warn(
                            f"[yellow]Native Ollama stream retry without think timed out for {self.ollama_model_name}. "
                            "Skipping /v1 retry and using timeout safety fallback.[/yellow]"
                        )
                        raise TimeoutError(str(retry_exc))
                    self.last_ollama_transport = "openai_compat_fallback"
                    self._log_warn(
                        f"[yellow]Native Ollama retry without think failed ({type(retry_exc).__name__}). "
                        "Falling back to OpenAI-compatible stream.[/yellow]"
                    )

                    def _collect_stream(msgs):
                        chunks = []
                        token_usage = None
                        for chunk in self.llm.stream(msgs):
                            chunk_text = _content_to_text(getattr(chunk, "content", ""))
                            if chunk_text:
                                chunks.append(chunk_text)
                                self._log_step(chunk_text, end="", flush=True)
                            chunk_usage = (
                                build_token_usage_record_from_langchain_message(chunk)
                            )
                            if chunk_usage is not None:
                                token_usage = chunk_usage
                        return "".join(chunks), token_usage

                    content, usage = self._run_with_timeout(_collect_stream, messages)
                    return content, "", usage
            self.last_ollama_transport = "openai_compat_fallback"
            self._log_warn(
                f"[yellow]Native Ollama stream failed ({type(exc).__name__}). Falling back to OpenAI-compatible stream.[/yellow]"
            )

            def _collect_stream(msgs):
                chunks = []
                token_usage = None
                for chunk in self.llm.stream(msgs):
                    chunk_text = _content_to_text(getattr(chunk, "content", ""))
                    if chunk_text:
                        chunks.append(chunk_text)
                        self._log_step(chunk_text, end="", flush=True)
                    chunk_usage = build_token_usage_record_from_langchain_message(chunk)
                    if chunk_usage is not None:
                        token_usage = chunk_usage
                return "".join(chunks), token_usage

            content, usage = self._run_with_timeout(_collect_stream, messages)
            return content, "", usage

    def few_shot_decision(
        self,
        scenario_description: str = "Not available",
        previous_decisions: str = "Not available",
        available_actions: str = "Not available",
        driving_intensions: str = "Not available",
        fewshot_messages: List[str] = None,
        fewshot_answers: List[str] = None,
    ):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/
        if getattr(self, "scientific_harness_config", None) is not None and (
            fewshot_messages or fewshot_answers
        ):
            raise ValueError(
                "Scientific confirmatory decisions require zero few-shot examples."
            )
        scientific_mode = getattr(self, "scientific_harness_config", None) is not None
        if scientific_mode:
            self.last_action_resolution = None
            self.last_generation_result = None
            self.last_decision_meta = None
            valid_action_ids = self._scientific_available_action_ids()
            self.last_scientific_available_action_ids = tuple(valid_action_ids)
            fallback_action_id = require_fixed_idle_available(valid_action_ids)
        else:
            fallback_action_id = self._decision_fallback_action_id()
            valid_action_ids = self._valid_action_ids()
        system_message = self._build_system_message(fallback_action_id)

        fewshot_intro = (
            "Above messages are some examples of how you make a decision successfully in the past. "
            "Those scenarios are similar to the current scenario. You should refer to those examples "
            "to make a decision for the current scenario."
            if fewshot_messages
            else "Use only the current scenario."
        )
        human_message = f"""\
        {fewshot_intro}

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}

        Think privately; output only the response contract from the system message.
        """
        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(HumanMessage(content=fewshot_messages[i]))
            messages.append(AIMessage(content=fewshot_answers[i]))
        messages.append(HumanMessage(content=human_message))
        # print("fewshot number:", (len(messages) - 2)/2)
        start_time = time.time()
        decision_meta = {
            "timed_out": False,
            "used_fallback": False,
            "fallback_reason": None,
            "parse_mode": "direct",
            "checker_used": False,
            "selected_action": None,
            "decision_elapsed_sec": 0.0,
            "ollama_transport": None,
            "ollama_native_chat_configured": (
                self.ollama_use_native_chat_configured
                if self.oai_api_type == "ollama"
                else None
            ),
            "ollama_native_chat_effective": (
                self.ollama_use_native_chat if self.oai_api_type == "ollama" else None
            ),
            "ollama_native_chat_resolution_reason": (
                self.ollama_native_chat_resolution_reason
                if self.oai_api_type == "ollama"
                else None
            ),
            "ollama_requested_think_mode": (
                self.ollama_think_mode if self.oai_api_type == "ollama" else None
            ),
            "ollama_effective_think_mode": None,
            "ollama_native_retry_used": False,
            "ollama_native_timeout": False,
            "ollama_native_timeout_short_circuit": False,
            "ollama_native_num_predict": None,
            "ollama_native_num_ctx": None,
            "ollama_native_keep_alive": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "token_count_method": None,
            "token_usage_source": None,
            "response_contract_satisfied": False,
            "response_contract_recovered": False,
            "response_recovery_reason": None,
            "response_unparseable": False,
            "response_action_line_present": False,
            "response_action_line_count": 0,
            "response_reason_line_present": False,
            "runtime_parse_path": "unknown",
            "response_first_nonempty_line": None,
            "response_truncated_before_contract": False,
            "response_finish_reason": None,
            "response_model_provider": None,
            "response_visible_chars": 0,
            "response_empty": False,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "semantic_recovery_used": False,
            "semantic_recovery_label": None,
            "semantic_recovery_reason": None,
            "intent_resolver_used": False,
            "intent_resolver_model": None,
            "intent_resolver_action_id": None,
            "intent_resolver_abstained": False,
            "intent_resolver_reason": None,
            "final_action_source": "unknown",
        }

        # NOTE: get_openai_callback might return 0 for Ollama
        # with get_openai_callback() as cb:
        # response = self.llm.invoke(messages) # invoke instead of __call__

        self._log_step("[cyan]Agent answer:[/cyan]")
        response_content = ""
        token_usage = None
        response_diagnostics: Dict[str, Any] = {}
        try:
            if self.use_streaming and not scientific_mode:
                response_content, token_usage = self._stream_response(messages)
                response_diagnostics = self._runtime_response_diagnostics_for_text(
                    response_content, token_usage
                )
            else:
                response_content, token_usage, response_diagnostics = (
                    self._invoke_response_with_diagnostics(
                        messages,
                        max_output_tokens_override=self.runtime_max_output_tokens,
                    )
                )
                self._log_step(response_content, end="", flush=True)
            self._log_step("\n")
        except TimeoutError as exc:
            if scientific_mode:
                result = getattr(self, "last_generation_result", None)
                if (
                    not isinstance(exc, ScientificGenerationTimeout)
                    or result is not exc.result
                ):
                    raise
                response_diagnostics = self._runtime_response_diagnostics_for_text(
                    response_content,
                    token_usage,
                )
                self._log_error(
                    f"\n[red]Decision timeout after {self.decision_timeout_sec:.1f}s. "
                    f"Fixed IDLE fallback action: {fallback_action_id}[/red]"
                )
                return self._finalize_scientific_decision(
                    response_content=response_content,
                    human_message=human_message,
                    response_diagnostics=response_diagnostics,
                    decision_meta=decision_meta,
                    token_usage=token_usage,
                    start_time=start_time,
                    valid_action_ids=valid_action_ids,
                    timed_out=True,
                )
            response_content = f"Response to user:{delimiter} {fallback_action_id}"
            response_diagnostics = self._runtime_response_diagnostics_for_text(
                response_content, token_usage
            )
            self._log_error(
                f"\n[red]Decision timeout after {self.decision_timeout_sec:.1f}s. "
                f"Fallback action: {fallback_action_id}[/red]"
            )
            decision_meta.update(response_diagnostics)
            decision_meta.update(
                self._runtime_response_contract_diagnostics(
                    response_content, response_diagnostics
                )
            )
            decision_meta["timed_out"] = True
            decision_meta["used_fallback"] = True
            decision_meta["fallback_reason"] = "decision_timeout"
            decision_meta["parse_mode"] = "timeout_fallback"
            decision_meta["runtime_parse_path"] = "timeout_fallback"
            decision_meta["response_unparseable"] = False
            decision_meta["selected_action"] = fallback_action_id
            decision_meta["final_action_source"] = "safe_fallback"
            decision_meta["decision_elapsed_sec"] = round(time.time() - start_time, 3)
            if self.oai_api_type == "ollama":
                decision_meta["ollama_transport"] = self.last_ollama_transport
                decision_meta["ollama_effective_think_mode"] = (
                    self.last_ollama_effective_think_mode
                )
                decision_meta["ollama_native_retry_used"] = bool(
                    self.last_ollama_native_retry_used
                )
                decision_meta["ollama_native_timeout"] = bool(
                    self.last_ollama_native_timeout
                )
                decision_meta["ollama_native_timeout_short_circuit"] = bool(
                    self.last_ollama_native_timeout_short_circuit
                )
                decision_meta.update(self._ollama_native_runtime_diagnostics())
            token_usage = build_whitespace_estimate_token_usage(0)
            decision_meta.update(token_usage)
            self.last_decision_meta = decision_meta
            few_shot_answers_store = ""
            for i in range(len(fewshot_messages)):
                few_shot_answers_store += fewshot_answers[i] + "\n---------------\n"
            self._log_step(f"Result: {fallback_action_id}")
            return (
                fallback_action_id,
                response_content,
                human_message,
                few_shot_answers_store,
            )

        if scientific_mode:
            return self._finalize_scientific_decision(
                response_content=response_content,
                human_message=human_message,
                response_diagnostics=response_diagnostics,
                decision_meta=decision_meta,
                token_usage=token_usage,
                start_time=start_time,
                valid_action_ids=valid_action_ids,
                timed_out=False,
            )

        decision_meta.update(response_diagnostics)
        decision_meta.update(
            self._runtime_response_contract_diagnostics(
                response_content, response_diagnostics
            )
        )
        result = fallback_action_id
        parse_path = "unknown"
        response_text = str(response_content or "")

        def _accept_parse(action_id: int, path: str) -> None:
            nonlocal result, parse_path
            result = int(action_id)
            parse_path = str(path)
            decision_meta["parse_mode"] = parse_path
            decision_meta["runtime_parse_path"] = parse_path
            decision_meta["final_action_source"] = parse_path
            self._mark_runtime_parse_result(decision_meta, parse_path)

        def _use_fallback(path: str) -> None:
            nonlocal result, parse_path
            result = fallback_action_id
            parse_path = str(path)
            decision_meta["used_fallback"] = True
            decision_meta["fallback_reason"] = parse_path
            decision_meta["parse_mode"] = parse_path
            decision_meta["runtime_parse_path"] = parse_path
            decision_meta["final_action_source"] = "safe_fallback"
            self._mark_runtime_parse_failure(decision_meta, parse_path)

        if not response_text.strip():
            _use_fallback(
                self._empty_selector_fallback_reason(
                    response_text, response_diagnostics
                )
            )
            self._log_step(
                f"[red]LLM returned an empty action response. Fallback to action {fallback_action_id}.[/red]"
            )
        else:
            try:
                action_id, path = self._extract_runtime_action_from_text(response_text)
                _accept_parse(action_id, path)
            except ValueError:
                tail = (
                    response_text.split(delimiter)[-1].strip()
                    if delimiter in response_text
                    else ""
                )
                try:
                    if tail:
                        _accept_parse(
                            self._extract_valid_action_from_text(tail), "delimiter_tail"
                        )
                    else:
                        raise ValueError("No delimiter tail available")
                except ValueError:
                    try:
                        _accept_parse(
                            self._extract_valid_action_from_text(response_text),
                            "regex_recovered",
                        )
                        self._log_step(
                            f"[yellow]Recovered action via regex parse:[/yellow] {result}"
                        )
                    except ValueError:
                        try:
                            semantic_action_id, semantic_label = (
                                self._extract_semantic_action_from_text(response_text)
                            )
                            decision_meta["semantic_recovery_used"] = True
                            decision_meta["semantic_recovery_label"] = semantic_label
                            decision_meta["semantic_recovery_reason"] = None
                            _accept_parse(
                                semantic_action_id, "semantic_label_recovered"
                            )
                            self._log_step(
                                f"[yellow]Recovered action via semantic label:[/yellow] {semantic_label} -> {result}"
                            )
                        except ValueError as semantic_exc:
                            decision_meta["semantic_recovery_used"] = False
                            decision_meta["semantic_recovery_label"] = None
                            decision_meta["semantic_recovery_reason"] = str(
                                semantic_exc
                            )[:160]
                            resolver_action_id, resolver_meta = (
                                self._resolve_action_with_intent_resolver(response_text)
                            )
                            decision_meta.update(resolver_meta)
                            if resolver_action_id is not None:
                                _accept_parse(
                                    resolver_action_id, "intent_resolver_direct"
                                )
                                self._log_step(
                                    f"[yellow]Recovered action via intent resolver:[/yellow] {result}"
                                )
                            elif not self.enable_checker_llm:
                                fallback_path = (
                                    "incomplete_output_fallback"
                                    if decision_meta.get(
                                        "response_truncated_before_contract"
                                    )
                                    else "parse_fallback"
                                )
                                _use_fallback(fallback_path)
                                self._log_step(
                                    f"[red]Output parse failed. Checker disabled; fallback to action {fallback_action_id}.[/red]"
                                )
                            else:
                                decision_meta["checker_used"] = True
                                self._log_step(
                                    "Output is not a valid action contract, checking the output..."
                                )
                                action_table = self._action_table_markdown()
                                allowed_actions = self._allowed_action_text()
                                check_message = f"""
                                You are an output checking assistant.

                                The driving agent output was:
                                {response_text}

                                Valid actions are:
                                {action_table}

                                Return exactly one valid action id from: {allowed_actions}

                                Answer format:
                                {delimiter} <correct action_id>
                                """
                                checker_messages = [HumanMessage(content=check_message)]
                                checker_token_usage = None
                                try:
                                    check_response = self._run_with_timeout(
                                        self.llm.invoke, checker_messages
                                    )
                                    checker_token_usage = (
                                        build_token_usage_record_from_langchain_message(
                                            check_response
                                        )
                                    )
                                    check_text = _content_to_text(
                                        getattr(check_response, "content", "")
                                    ).strip()
                                    if checker_token_usage is None and check_text:
                                        checker_token_usage = (
                                            build_whitespace_estimate_token_usage(
                                                estimate_generated_tokens(check_text)
                                            )
                                        )
                                except TimeoutError:
                                    check_text = ""
                                    decision_meta["timed_out"] = True
                                    self._log_step(
                                        "[yellow]Checker timed out. Applying safe fallback parse.[/yellow]"
                                    )
                                token_usage = combine_token_usage_records(
                                    token_usage, checker_token_usage
                                )

                                checker_tail = (
                                    check_text.split(delimiter)[-1].strip()
                                    if delimiter in check_text
                                    else check_text
                                )
                                try:
                                    _accept_parse(
                                        self._extract_valid_action_from_text(
                                            checker_tail
                                        ),
                                        "checker_direct",
                                    )
                                except ValueError:
                                    try:
                                        _accept_parse(
                                            self._extract_valid_action_from_text(
                                                check_text
                                            ),
                                            "checker_regex_recovered",
                                        )
                                        self._log_step(
                                            f"[yellow]Recovered action from checker output:[/yellow] {result}"
                                        )
                                    except ValueError:
                                        _use_fallback("checker_fallback")
                                        self._log_step(
                                            f"[red]Checker output parse failed. Falling back to safe action {fallback_action_id}.[/red]"
                                        )

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + "\n---------------\n"
        decision_meta["selected_action"] = int(result)
        decision_meta["decision_elapsed_sec"] = round(time.time() - start_time, 3)
        if self.oai_api_type == "ollama":
            decision_meta["ollama_transport"] = self.last_ollama_transport
            decision_meta["ollama_effective_think_mode"] = (
                self.last_ollama_effective_think_mode
            )
            decision_meta["ollama_native_retry_used"] = bool(
                self.last_ollama_native_retry_used
            )
            decision_meta["ollama_native_timeout"] = bool(
                self.last_ollama_native_timeout
            )
            decision_meta["ollama_native_timeout_short_circuit"] = bool(
                self.last_ollama_native_timeout_short_circuit
            )
            decision_meta.update(self._ollama_native_runtime_diagnostics())
        if token_usage is None:
            token_usage = build_whitespace_estimate_token_usage(
                estimate_generated_tokens(response_content)
            )
        decision_meta.update(token_usage)
        self.last_decision_meta = decision_meta
        self._log_step(f"Result: {result}")
        return result, response_content, human_message, few_shot_answers_store
