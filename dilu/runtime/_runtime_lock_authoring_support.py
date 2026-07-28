"""Native capability probing and canonical publication for S1 authoring."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ._scientific_transport_response import parse_native_response_attempt
from .action_resolution import parse_canonical_action
from .harness_config import OutputEnforcement, ThinkMode
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


def validate_exact_lock_tree(
    root: Path,
    expected_relative_files: Sequence[Path],
) -> None:
    """Require one unredirected, case-exact filesystem tree."""
    paths = tuple(expected_relative_files)
    if any(
        (
            not isinstance(relative, Path)
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        )
        for relative in paths
    ):
        raise ValueError("Expected runtime-lock path is not canonical.")
    expected_files = {relative.as_posix() for relative in paths}
    if len(expected_files) != len(paths):
        raise ValueError("Expected runtime-lock paths contain duplicates.")
    expected_directories = {
        parent.as_posix()
        for relative in paths
        for parent in relative.parents
        if parent != Path(".")
    }
    expected_casefold: dict[str, str] = {}
    for expected in expected_files | expected_directories:
        _record_case_exact(expected_casefold, expected)
    if not root.is_dir() or _path_is_redirect(root):
        raise ValueError(
            "Runtime-lock root is missing, redirected, or not a directory."
        )

    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    observed_casefold: dict[str, str] = {}
    _collect_tree_entries(
        root,
        Path(),
        observed_files,
        observed_directories,
        observed_casefold,
    )
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("Runtime-lock filesystem tree is not exact.")


def _collect_tree_entries(
    directory: Path,
    relative_root: Path,
    files: set[str],
    directories: set[str],
    casefold_index: dict[str, str],
) -> None:
    with os.scandir(directory) as entries:
        for entry in entries:
            relative = relative_root / entry.name
            serialized = relative.as_posix()
            _record_case_exact(casefold_index, serialized)
            if entry.is_symlink() or _stat_is_reparse(
                entry.stat(follow_symlinks=False)
            ):
                raise ValueError("Runtime-lock tree contains a redirected entry.")
            if entry.is_dir(follow_symlinks=False):
                directories.add(serialized)
                _collect_tree_entries(
                    Path(entry.path),
                    relative,
                    files,
                    directories,
                    casefold_index,
                )
            elif entry.is_file(follow_symlinks=False):
                files.add(serialized)
            else:
                raise ValueError("Runtime-lock tree contains an invalid entry type.")


def _record_case_exact(index: dict[str, str], value: str) -> None:
    folded = value.casefold()
    previous = index.setdefault(folded, value)
    if previous != value:
        raise ValueError("Runtime-lock tree contains a case-colliding entry.")


def _path_is_redirect(path: Path) -> bool:
    return path.is_symlink() or _stat_is_reparse(path.stat(follow_symlinks=False))


def _stat_is_reparse(value: os.stat_result) -> bool:
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(value, "st_file_attributes", 0) & marker)


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
) -> tuple[OllamaModelIdentity, tuple[dict[str, Any], ...]]:
    before = _inspect_direct_identity(
        native_endpoint,
        model_tag,
        get=get,
        timeout_sec=timeout_sec,
    )
    records: list[dict[str, Any]] = []
    payload_bodies: list[bytes] = []
    actions: list[int] = []
    for index, enforcement in enumerate(
        (
            OutputEnforcement.PROMPT_ONLY,
            OutputEnforcement.PROMPT_ONLY,
            OutputEnforcement.BACKEND_SCHEMA,
        )
    ):
        request = _probe_request(
            model_slot=model_slot,
            identity=before,
            native_endpoint=native_endpoint,
            seed=seed,
            temperature=temperature,
            context_tokens=context_tokens,
            max_output_tokens=max_output_tokens,
            timeout_sec=timeout_sec,
            think_mode=think_mode,
            enforcement=enforcement,
            index=index,
        )
        payload = build_native_chat_payload(request)
        if (
            enforcement is OutputEnforcement.BACKEND_SCHEMA
            and canonical_bytes(payload.get("format")) != canonical_schema_bytes
        ):
            raise ValueError("Native capability schema drift before direct POST.")
        payload_body = canonical_bytes(payload)
        payload_bodies.append(payload_body)
        record, action = _direct_call(
            request=request,
            payload=payload,
            payload_body=payload_body,
            post=post,
        )
        records.append(record)
        actions.append(action)
    if payload_bodies[0] != payload_bodies[1] or actions[0] != actions[1]:
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


def _probe_request(
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
    enforcement: OutputEnforcement,
    index: int,
) -> GenerationRequest:
    label = ("prompt", "prompt-repeat", "schema")[index]
    return GenerationRequest(
        model_tag=identity.model_tag,
        model_digest=identity.model_digest,
        request_id=f"s1-{model_slot}-{label}",
        messages=(
            (
                "system",
                "Return exactly one canonical driving action: "
                "Response to user:#### N, where N is 0 through 4.",
            ),
            ("user", "Choose the canonical IDLE action for this capability probe."),
        ),
        native_endpoint=native_endpoint,
        options=NativeGenerationOptions(
            seed=seed,
            temperature=temperature,
            num_ctx=context_tokens,
            num_predict=max_output_tokens,
        ),
        output_enforcement=enforcement,
        think_mode=think_mode,
        timeout_sec=timeout_sec,
    )


def _direct_call(
    *,
    request: GenerationRequest,
    payload: Mapping[str, Any],
    payload_body: bytes,
    post: PostCallable,
) -> tuple[dict[str, Any], int]:
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
    attempt = parse_native_response_attempt(
        request,
        f"{request.request_id}:a1",
        1,
        status,
        response_payload,
        response_body,
        0.0,
        lambda: 0.0,
    )
    if attempt.error_class is not None:
        label = (
            "schema rejection"
            if request.output_enforcement is OutputEnforcement.BACKEND_SCHEMA
            else "malformed native response"
        )
        raise ValueError(f"Native capability probe {label}: {attempt.error_message}.")
    if attempt.contract_text is None or attempt.backend_timing is None:
        raise ValueError("Native capability probe omitted required evidence.")
    action = parse_canonical_action(attempt.contract_text)
    request_evidence = {
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
    return (
        {
            "request": request_evidence,
            "payload": dict(payload),
            "payload_sha256": bytes_sha256(payload_body),
            "request_body": payload_body.decode("utf-8"),
            "http_status": status,
            "response_body": response_body,
            "raw_response": attempt.raw_response,
            "canonical_action": action,
            "stop_reason": attempt.stop_reason,
            "prompt_tokens": attempt.prompt_tokens,
            "completion_tokens": attempt.completion_tokens,
            "total_tokens": attempt.prompt_tokens + attempt.completion_tokens,
            "backend_timing": asdict(attempt.backend_timing),
        },
        action,
    )
