from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .prompts import render_system_prompt, render_user_prompt
from ..llm_env import openai_compatible_default_headers_from_config

ChatGoogleGenerativeAI = None
_GEMINI_IMPORT_ERROR = None


@dataclass(frozen=True)
class CodegenResponse:
    content: str


def _load_chat_google_generative_ai():
    global ChatGoogleGenerativeAI, _GEMINI_IMPORT_ERROR
    if ChatGoogleGenerativeAI is not None:
        return ChatGoogleGenerativeAI
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI as GeminiChat
    except Exception as exc:  # pragma: no cover - optional dependency
        _GEMINI_IMPORT_ERROR = exc
        return None
    ChatGoogleGenerativeAI = GeminiChat
    _GEMINI_IMPORT_ERROR = None
    return ChatGoogleGenerativeAI


class HighwayCodegenAgent:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        model_name: str,
        request_timeout: float = 60.0,
        few_shot: bool = False,
        temperature: float = 0.0,
    ) -> None:
        self.config = dict(config)
        self.model_name = model_name
        self.request_timeout = float(request_timeout)
        self.few_shot = bool(few_shot)
        self.temperature = float(temperature)
        self.client = self._build_client()

    def _build_client(self):
        api_type = str(self.config["OPENAI_API_TYPE"]).strip().lower()
        if api_type == "ollama":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=str(self.config.get("OLLAMA_API_KEY", "ollama")),
                base_url=str(self.config.get("OLLAMA_API_BASE", "http://localhost:11434/v1")),
            )
        if api_type == "openai":
            api_key = (
                str(self.config.get("OPENAI_KEY") or self.config.get("OPENAI_API_KEY") or "").strip()
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("OPENAI_KEY")
                or os.environ.get("OPENROUTER_API_KEY")
            )
            base_url = str(self.config.get("OPENAI_BASE_URL") or self.config.get("OPENAI_API_BASE") or "").strip()
            default_headers = openai_compatible_default_headers_from_config(self.config)
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=api_key,
                base_url=base_url or None,
                **({"default_headers": default_headers} if default_headers else {}),
            )
        if api_type == "azure":
            return AzureChatOpenAI(
                azure_deployment=str(self.config["AZURE_CHAT_DEPLOY_NAME"]),
                api_version=str(self.config["AZURE_API_VERSION"]),
                azure_endpoint=str(self.config["AZURE_API_BASE"]),
                api_key=str(self.config["AZURE_API_KEY"]),
                temperature=self.temperature,
                timeout=self.request_timeout,
            )
        if api_type == "gemini":
            gemini_chat = _load_chat_google_generative_ai()
            if gemini_chat is None:
                raise ImportError("langchain_google_genai is required for Gemini code generation.") from _GEMINI_IMPORT_ERROR
            return gemini_chat(
                model=self.model_name,
                google_api_key=str(self.config["GEMINI_API_KEY"]),
                temperature=self.temperature,
                timeout=self.request_timeout,
            )
        raise ValueError(f"Unsupported OPENAI_API_TYPE for highway codegen: {api_type}")

    def generate(self, *, command: str, context_info: str) -> CodegenResponse:
        messages = [
            SystemMessage(content=render_system_prompt(few_shot=self.few_shot)),
            HumanMessage(content=render_user_prompt(command, context_info)),
        ]
        response = self.client.invoke(messages)
        return CodegenResponse(content=str(getattr(response, "content", response)))
