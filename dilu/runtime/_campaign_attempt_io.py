from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def lock_path_for(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def poison_path_for(path: Path) -> Path:
    return path.with_name(f"{path.name}.poisoned")


@contextmanager
def exclusive_append_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def persist_poison_marker(path: Path, episode_attempt_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return
    try:
        payload = f"trace_commit_ambiguous:{episode_attempt_id}\n".encode("utf-8")
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("Partial campaign poison-marker write.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "exclusive_append_lock",
    "lock_path_for",
    "persist_poison_marker",
    "poison_path_for",
]
