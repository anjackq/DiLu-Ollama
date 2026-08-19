"""Shared initialization and read-only snapshots for scientific traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._append_intent_io import append_intent_path_for
from ._campaign_attempt_io import lock_path_for, poison_path_for
from ._scientific_trace_artifacts import ScientificTraceWriteError, TraceReference


@dataclass(frozen=True)
class ScientificTraceSnapshot:
    references_by_attempt: Mapping[
        tuple[str, str],
        tuple[TraceReference, ...],
    ]


def initialize_scientific_trace_state(
    writer: Any,
    path: Path,
    artifact_root: Path,
) -> None:
    writer.path = Path(path).resolve()
    writer.artifact_root = Path(artifact_root).resolve()
    try:
        relative = writer.path.relative_to(writer.artifact_root)
    except ValueError as exc:
        raise ValueError(
            "Scientific trace path must be inside artifact_root."
        ) from exc
    if not relative.parts:
        raise ValueError("Scientific trace path must name a file.")
    writer.relative_path = relative.as_posix()
    writer._lock_path = lock_path_for(writer.path)
    writer._poison_path = poison_path_for(writer.path)
    writer._pending_path = append_intent_path_for(writer.path)
    writer._keys = set()
    writer._last_by_episode = {}
    writer._signature_by_episode = {}
    writer._terminal_episodes = set()
    writer._request_owners = {}
    writer._references_by_episode = {}
    writer._reference_index = set()
    writer._line_count = 0
    writer._byte_offset = 0
    writer._poisoned = False


def read_validated_trace_snapshot(
    writer_type: type[Any],
    path: Path,
    *,
    artifact_root: Path,
) -> ScientificTraceSnapshot:
    writer = object.__new__(writer_type)
    initialize_scientific_trace_state(writer, path, artifact_root)
    _require_quiescent(writer)
    before = _file_state(writer.path)
    if writer.path.exists() and writer.path.stat().st_size:
        writer._scan_existing()
    after = _file_state(writer.path)
    _require_quiescent(writer)
    final = _file_state(writer.path)
    if before != after or after != final:
        raise ScientificTraceWriteError(
            "Scientific trace changed during read-only validation."
        )
    _require_quiescent(writer)
    return ScientificTraceSnapshot(
        MappingProxyType(
            {
                key: tuple(references)
                for key, references in writer._references_by_episode.items()
            }
        )
    )


def _require_quiescent(writer: Any) -> None:
    if any(
        marker.exists()
        for marker in (writer._lock_path, writer._pending_path, writer._poison_path)
    ):
        raise ScientificTraceWriteError(
            "Scientific trace is busy or has ambiguous durable-append evidence."
        )


def _file_state(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


__all__ = [
    "ScientificTraceSnapshot",
    "initialize_scientific_trace_state",
    "read_validated_trace_snapshot",
]
