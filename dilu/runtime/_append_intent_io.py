from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


APPEND_INTENT_SCHEMA_VERSION = "iclr2027.append_intent.v1"
_SHA256_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z")


class AppendIntentWriteError(RuntimeError):
    pass


class AppendCommitAmbiguousError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppendIntent:
    artifact_kind: str
    episode_attempt_id: str
    expected_offset: int
    byte_length: int
    record_sha256: str
    line_sha256: str
    schema_version: str = APPEND_INTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("artifact_kind", "episode_attempt_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty canonical text.")
        for name in ("expected_offset", "byte_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.byte_length == 0:
            raise ValueError("byte_length must be positive.")
        for name in ("record_sha256", "line_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a full sha256 digest.")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "episode_attempt_id": self.episode_attempt_id,
            "expected_offset": self.expected_offset,
            "byte_length": self.byte_length,
            "record_sha256": self.record_sha256,
            "line_sha256": self.line_sha256,
            "schema_version": self.schema_version,
        }


def append_intent_path_for(path: Path) -> Path:
    return path.with_name(f"{path.name}.append_pending")


def durable_append_with_intent(
    path: Path,
    line: bytes,
    *,
    artifact_kind: str,
    episode_attempt_id: str,
    expected_offset: int,
    record_sha256: str,
) -> None:
    resolved = Path(path).resolve()
    if not isinstance(line, bytes) or not line or not line.endswith(b"\n"):
        raise ValueError("line must be non-empty newline-terminated bytes.")
    intent = AppendIntent(
        artifact_kind=artifact_kind,
        episode_attempt_id=episode_attempt_id,
        expected_offset=expected_offset,
        byte_length=len(line),
        record_sha256=record_sha256,
        line_sha256=_sha256(line),
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    actual_offset = resolved.stat().st_size if resolved.exists() else 0
    if actual_offset != expected_offset:
        raise AppendIntentWriteError("Artifact offset drifted before intent commit.")
    pending_path = append_intent_path_for(resolved)
    persist_append_intent(pending_path, intent)
    try:
        _append_and_sync_data(resolved, line)
        _verify_committed_line(resolved, line, expected_offset)
    except Exception as exc:
        raise AppendCommitAmbiguousError(
            "Artifact append outcome is ambiguous; append intent remains pending."
        ) from exc
    try:
        clear_append_intent(pending_path)
    except Exception as exc:
        raise AppendCommitAmbiguousError(
            "Append-intent cleanup failed after durable data commit."
        ) from exc


def persist_append_intent(path: Path, intent: AppendIntent) -> None:
    encoded = _canonical_bytes(intent.to_dict()) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor: int | None = None
    failure: Exception | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(
                f"Partial append-intent write: {written}/{len(encoded)} bytes."
            )
        os.fsync(descriptor)
    except Exception as exc:
        failure = exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except Exception as exc:
                failure = failure or exc
    if failure is not None:
        raise AppendIntentWriteError(
            "Append intent was not durably committed."
        ) from failure


def clear_append_intent(path: Path) -> None:
    path.unlink()


def _append_and_sync_data(path: Path, line: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor: int | None = None
    failure: Exception | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError(f"Partial artifact append: {written}/{len(line)} bytes.")
        os.fsync(descriptor)
    except Exception as exc:
        failure = exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except Exception as exc:
                failure = failure or exc
    if failure is not None:
        raise failure


def _verify_committed_line(path: Path, line: bytes, expected_offset: int) -> None:
    if path.stat().st_size != expected_offset + len(line):
        raise OSError("Artifact size drifted after append.")
    with path.open("rb") as handle:
        handle.seek(expected_offset)
        if handle.read(len(line)) != line or handle.read(1):
            raise OSError("Artifact tail does not match the prepared append intent.")


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


__all__ = [
    "APPEND_INTENT_SCHEMA_VERSION",
    "AppendCommitAmbiguousError",
    "AppendIntent",
    "AppendIntentWriteError",
    "append_intent_path_for",
    "clear_append_intent",
    "durable_append_with_intent",
    "persist_append_intent",
]
