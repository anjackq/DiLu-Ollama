"""Sibling staging and atomic no-replace installation for S1 artifacts."""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ._runtime_lock_tree_validation import validate_unredirected_artifact_paths


@dataclass(frozen=True)
class ParentIdentity:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class TransactionState:
    destination: Path
    stage: Path
    parent_identity: ParentIdentity
    stage_device: int
    stage_inode: int


@contextmanager
def staged_destination(destination: Path) -> Iterator[TransactionState]:
    """Yield a private sibling stage and remove it unless atomically installed."""
    parent = destination.parent
    validate_unredirected_artifact_paths((destination / ".install-probe",))
    if os.path.lexists(destination):
        raise ValueError("Fresh authoring destination already exists.")
    created_parents = _create_missing_parents(parent)
    identity = _parent_identity(parent)
    parent_handle = _lock_parent_directory(parent)
    try:
        _require_parent_identity(identity)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-",
                dir=parent,
            )
        )
        stage_stat = stage.stat(follow_symlinks=False)
        state = TransactionState(
            destination,
            stage,
            identity,
            stage_stat.st_dev,
            stage_stat.st_ino,
        )
        try:
            yield state
        finally:
            if _parent_matches(identity) and os.path.lexists(stage):
                _require_stage_identity(state)
                _cleanup_stage(stage, parent)
    finally:
        _unlock_parent_directory(parent_handle)
        if not os.path.lexists(destination):
            _cleanup_created_parents(created_parents)


def guard_staged_destination(state: TransactionState) -> None:
    """Require the locked parent and private stage to retain physical identity."""
    _require_parent_identity(state.parent_identity)
    _require_stage_identity(state)
    validate_unredirected_artifact_paths((state.stage / ".publication-probe",))
    _require_parent_identity(state.parent_identity)


def install_staged_destination(
    *,
    state: TransactionState,
    final_artifact_paths: tuple[Path, ...],
) -> None:
    """Atomically install a complete stage without replacing any destination."""
    stage = state.stage
    destination = state.destination
    guard_staged_destination(state)
    validate_unredirected_artifact_paths(final_artifact_paths)
    if os.path.lexists(destination):
        raise ValueError("Final authoring destination appeared before install.")
    _require_parent_identity(state.parent_identity)
    _rename_noreplace(stage, destination)
    _require_parent_identity(state.parent_identity)
    validate_unredirected_artifact_paths(final_artifact_paths)


def _parent_identity(parent: Path) -> ParentIdentity:
    if not parent.is_dir():
        raise ValueError("Authoring destination parent must already exist.")
    stat_result = parent.stat(follow_symlinks=False)
    return ParentIdentity(parent, stat_result.st_dev, stat_result.st_ino)


def _require_parent_identity(expected: ParentIdentity) -> None:
    observed = _parent_identity(expected.path)
    if observed != expected:
        raise ValueError("Authoring destination parent identity drifted.")


def _parent_matches(expected: ParentIdentity) -> bool:
    try:
        _require_parent_identity(expected)
    except (OSError, ValueError):
        return False
    return True


def _require_stage_identity(state: TransactionState) -> None:
    if not state.stage.is_dir() or state.stage.is_symlink():
        raise ValueError("Authoring staging directory identity drifted.")
    observed = state.stage.stat(follow_symlinks=False)
    if (observed.st_dev, observed.st_ino) != (
        state.stage_device,
        state.stage_inode,
    ):
        raise ValueError("Authoring staging directory identity drifted.")


def _create_missing_parents(parent: Path) -> tuple[ParentIdentity, ...]:
    missing: list[Path] = []
    candidate = parent
    while not os.path.lexists(candidate):
        missing.append(candidate)
        candidate = candidate.parent
    validate_unredirected_artifact_paths((candidate / ".parent-probe",))
    created: list[ParentIdentity] = []
    for path in reversed(missing):
        path.mkdir()
        created.append(_parent_identity(path))
    return tuple(created)


def _cleanup_created_parents(created: tuple[ParentIdentity, ...]) -> None:
    for identity in reversed(created):
        if not _parent_matches(identity):
            return
        try:
            identity.path.rmdir()
        except OSError:
            return


def _lock_parent_directory(parent: Path) -> int | None:
    if os.name != "nt":
        return None
    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(parent),
        0,
        0x00000001 | 0x00000002,
        None,
        3,
        0x02000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError()
    return int(handle)


def _unlock_parent_directory(handle: int | None) -> None:
    if handle is None:
        return
    if not ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle)):
        raise ctypes.WinError()


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), destination)
        return
    raise OSError(errno.ENOTSUP, "Atomic no-replace directory install unavailable")


def _cleanup_stage(stage: Path, expected_parent: Path) -> None:
    if stage.parent != expected_parent or not stage.name.startswith("."):
        raise RuntimeError("Refusing to clean an unexpected staging path.")
    is_junction = getattr(stage, "is_junction", lambda: False)
    if stage.is_symlink() or is_junction():
        os.rmdir(stage)
    else:
        shutil.rmtree(stage)
