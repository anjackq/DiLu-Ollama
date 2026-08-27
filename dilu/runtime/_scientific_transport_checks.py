from __future__ import annotations

import json
import re
from typing import Any

import requests

from .harness_config import OutputEnforcement
from .scientific_transport_types import (
    SCHEMA_MECHANISM,
    SCHEMA_MECHANISM_GROUNDED,
    GenerationRequest,
    ScientificTransportCapabilities,
)


class PreAcceptTransportUnavailable(RuntimeError):
    """Raised only when the adapter proves Ollama accepted no request."""


def response_text(response: Any) -> str | None:
    value = getattr(response, "text", None)
    return value if isinstance(value, str) else None


def is_verified_schema_rejection(body: str | None) -> bool:
    if not isinstance(body, str):
        return False
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), str):
        return False
    error = payload["error"].strip().lower()
    patterns = (
        r"\binvalid\s+(?:json\s+)?(?:schema|format)\b",
        r"\b(?:schema|format)\s+(?:is\s+)?(?:invalid|unsupported)\b",
        r"\bdoes\s+not\s+support\s+(?:json\s+)?(?:schema|format)\b",
        r"\bfailed\s+to\s+(?:parse|validate)\s+(?:json\s+)?(?:schema|format)\b",
    )
    return any(re.search(pattern, error) for pattern in patterns)


def transport_preflight_error(
    request: GenerationRequest,
    capabilities: ScientificTransportCapabilities,
) -> str | None:
    if request.model_tag != capabilities.model_tag:
        return "model_tag_drift"
    if request.model_digest != capabilities.model_digest:
        return "model_digest_drift"
    if request.native_endpoint != capabilities.native_endpoint:
        return "native_endpoint_drift"
    if request.think_mode not in capabilities.supported_think_modes:
        return "think_mode_not_verified"
    if not capabilities.seed_verified:
        return "generation_seed_not_verified"
    if capabilities.schema_mechanism not in (SCHEMA_MECHANISM, SCHEMA_MECHANISM_GROUNDED):
        return "schema_mechanism_drift"
    if (
        request.output_enforcement
        in (OutputEnforcement.BACKEND_SCHEMA, OutputEnforcement.BACKEND_SCHEMA_GROUNDED)
        and not capabilities.schema_verified
    ):
        return "backend_schema_not_verified"
    return None


def requests_post_with_preaccept_classification(*args: Any, **kwargs: Any) -> Any:
    try:
        return requests.post(*args, **kwargs)
    except requests.ConnectionError as exc:
        if contains_new_connection_error(exc):
            raise PreAcceptTransportUnavailable(str(exc)) from exc
        raise


def contains_new_connection_error(error: BaseException) -> bool:
    pending: list[Any] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if type(current).__name__ == "NewConnectionError":
            return True
        for candidate in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
            getattr(current, "reason", None),
        ):
            if isinstance(candidate, BaseException):
                pending.append(candidate)
        pending.extend(
            item
            for item in getattr(current, "args", ())
            if isinstance(item, BaseException)
        )
    return False


__all__ = [
    "PreAcceptTransportUnavailable",
    "is_verified_schema_rejection",
    "requests_post_with_preaccept_classification",
    "response_text",
    "transport_preflight_error",
]
