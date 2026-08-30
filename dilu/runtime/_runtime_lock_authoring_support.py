"""Native capability probing and canonical publication for S1 authoring."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ._runtime_lock_response_evidence import derive_response_evidence
from .action_resolution import (
    FIXED_IDLE_ACTION_ID,
    ActionResolutionResult,
)
from .harness_config import (
    OutputEnforcement,
    ThinkMode,
)
from .ollama_transport import (
    OllamaModelIdentity,
    inspect_ollama_model_identity,
    ollama_tags_url,
)
from .scientific_transport_types import (
    GenerationRequest,
    NativeGenerationOptions,
    build_native_chat_payload,
)

GetCallable = Callable[..., Any]
PostCallable = Callable[..., Any]
S1_COLD_START_OBSERVATION_TIMEOUT_SEC = 120.0
OLLAMA_NATIVE_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE = (
    "ollama_native_capability_preflight_v2"
)
_FIXED_IDLE_ACTION_TEXT = f"Response to user:#### {FIXED_IDLE_ACTION_ID}"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def bytes_sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def publish_once(path: Path, content: bytes) -> None:
    """Atomically publish canonical bytes or verify an identical prior write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(
                    f"Frozen artifact already exists with drift: {path}."
                ) from None
        finally:
            temporary.unlink(missing_ok=True)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def probe_model(
    *,
    model_slot: str,
    model_tag: str,
    native_endpoint: str,
    seed: int,
    temperature: float,
    context_tokens: int,
    max_output_tokens: int,
    timeout_sec: float,
    think_mode: ThinkMode,
    canonical_schema_bytes: bytes,
    get: GetCallable,
    post: PostCallable,
    extra_requests: Callable[[OllamaModelIdentity], tuple[GenerationRequest, ...]]
    | None = None,
) -> tuple[OllamaModelIdentity, tuple[dict[str, Any], ...]]:
    """Send the trusted prompt/repeat/schema probe sequence and return evidence.

    ``extra_requests``, when given, is called with the resolved pre-probe
    identity and its returned requests are appended after the trusted
    three (e.g. the ICLR 2027 grounded-decoding (O2) probe); ``None``
    (the default) reproduces the exact V5/V7 three-call sequence.
    """
    before = _inspect_direct_identity(
        native_endpoint,
        model_tag,
        get=get,
        timeout_sec=timeout_sec,
    )
    records: list[dict[str, Any]] = []
    payload_bodies: list[bytes] = []
    resolutions: list[ActionResolutionResult] = []
    requests = build_probe_requests(
        model_slot=model_slot,
        identity=before,
        native_endpoint=native_endpoint,
        seed=seed,
        temperature=temperature,
        context_tokens=context_tokens,
        max_output_tokens=max_output_tokens,
        timeout_sec=timeout_sec,
        think_mode=think_mode,
    )
    if extra_requests is not None:
        requests = requests + extra_requests(before)
    for request in requests:
        payload = build_native_chat_payload(request)
        if (
            request.output_enforcement is OutputEnforcement.BACKEND_SCHEMA
            and canonical_bytes(payload.get("format")) != canonical_schema_bytes
        ):
            raise ValueError("Native capability schema drift before direct POST.")
        payload_body = canonical_bytes(payload)
        payload_bodies.append(payload_body)
        record, resolution = _direct_call(
            request=request,
            payload=payload,
            payload_body=payload_body,
            post=post,
        )
        records.append(record)
        resolutions.append(resolution)
    if (
        payload_bodies[0] != payload_bodies[1]
        or resolutions[0].final_resolved_action != resolutions[1].final_resolved_action
    ):
        raise ValueError("Prompt-only repeat evidence mismatch.")
    after = _inspect_direct_identity(
        native_endpoint,
        model_tag,
        get=get,
        timeout_sec=timeout_sec,
    )
    if after != before:
        raise ValueError("Ollama model identity drift after native probe.")
    identity_before = asdict(before)
    identity_after = asdict(after)
    return before, tuple(
        {
            **record,
            "identity_before": identity_before,
            "identity_after": identity_after,
        }
        for record in records
    )


def _inspect_direct_identity(
    native_endpoint: str,
    model_tag: str,
    *,
    get: GetCallable,
    timeout_sec: float,
) -> OllamaModelIdentity:
    expected_url = ollama_tags_url(native_endpoint)

    def direct_get(
        url: str,
        *,
        timeout: float,
        allow_redirects: bool,
    ) -> object:
        if url != expected_url:
            raise ValueError("Ollama identity endpoint construction drift.")
        response = get(
            url,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )
        if getattr(response, "url", None) != expected_url:
            raise ValueError("Ollama identity endpoint drift or fallback.")
        return response

    return inspect_ollama_model_identity(
        native_endpoint,
        model_tag,
        get=direct_get,
        timeout_sec=timeout_sec,
    )


def build_probe_requests(
    *,
    model_slot: str,
    identity: OllamaModelIdentity,
    native_endpoint: str,
    seed: int,
    temperature: float,
    context_tokens: int,
    max_output_tokens: int,
    timeout_sec: float,
    think_mode: ThinkMode,
) -> tuple[GenerationRequest, ...]:
    """Build the exact trusted prompt, repeat, and schema request sequence."""
    labels = ("prompt", "prompt-repeat", "schema")
    enforcements = (
        OutputEnforcement.PROMPT_ONLY,
        OutputEnforcement.PROMPT_ONLY,
        OutputEnforcement.BACKEND_SCHEMA,
    )
    timeouts = (
        S1_COLD_START_OBSERVATION_TIMEOUT_SEC,
        timeout_sec,
        timeout_sec,
    )
    messages = (
        (
            "system",
            f"Return exactly this text and nothing else: {_FIXED_IDLE_ACTION_TEXT}",
        ),
        ("user", "Perform the response-format capability check."),
    )
    options = NativeGenerationOptions(
        seed=seed,
        temperature=temperature,
        num_ctx=context_tokens,
        num_predict=max_output_tokens,
    )
    return tuple(
        GenerationRequest(
            model_tag=identity.model_tag,
            model_digest=identity.model_digest,
            request_id=f"s1-{model_slot}-{label}",
            messages=messages,
            native_endpoint=native_endpoint,
            options=options,
            output_enforcement=enforcement,
            think_mode=think_mode,
            timeout_sec=request_timeout_sec,
        )
        for label, enforcement, request_timeout_sec in zip(
            labels,
            enforcements,
            timeouts,
            strict=True,
        )
    )


def build_request_evidence(request: GenerationRequest) -> dict[str, object]:
    """Serialize every frozen request field recorded by direct probing."""
    return {
        "model_tag": request.model_tag,
        "model_digest": request.model_digest,
        "request_id": request.request_id,
        "messages": [
            {"role": role, "content": content} for role, content in request.messages
        ],
        "native_endpoint": request.native_endpoint,
        "options": request.options.to_payload(),
        "output_enforcement": request.output_enforcement.value,
        "think_mode": request.think_mode.value,
        "timeout_sec": float(request.timeout_sec),
    }


def _direct_call(
    *,
    request: GenerationRequest,
    payload: Mapping[str, Any],
    payload_body: bytes,
    post: PostCallable,
) -> tuple[dict[str, Any], ActionResolutionResult]:
    response = post(
        request.native_endpoint,
        data=payload_body,
        headers={"Content-Type": "application/json"},
        timeout=float(request.timeout_sec),
        allow_redirects=False,
    )
    if getattr(response, "history", ()):
        raise ValueError("Native capability probe cannot follow a redirect.")
    if getattr(response, "url", None) != request.native_endpoint:
        raise ValueError("Native capability probe endpoint drift or fallback.")
    status = getattr(response, "status_code", None)
    if not isinstance(status, int) or not 200 <= status < 300:
        raise ValueError("Native capability probe requires a direct 2xx response.")
    response_body = getattr(response, "text", None)
    if not isinstance(response_body, str):
        raise ValueError("Native capability probe response body is malformed.")
    try:
        response_payload = response.json()
    except (TypeError, ValueError) as exc:
        raise ValueError("Native capability probe returned malformed JSON.") from exc
    response_evidence, resolution = derive_response_evidence(
        request,
        status,
        response_payload,
        response_body,
    )
    return (
        {
            "request": build_request_evidence(request),
            "payload": dict(payload),
            "payload_sha256": bytes_sha256(payload_body),
            "request_body": payload_body.decode("utf-8"),
            **response_evidence,
        },
        resolution,
    )
