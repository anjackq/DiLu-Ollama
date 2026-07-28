"""Immutable values and atomic publication for minimal factorial scheduling."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, init=False)
class FrozenSpec:
    values: Mapping[str, Any]

    def __init__(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("FrozenSpec values must be a mapping.")
        object.__setattr__(self, "values", freeze(values))

    def __getattr__(self, name: str) -> Any:
        try:
            value = self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return FrozenSpec(value) if isinstance(value, Mapping) else value

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def to_dict(self) -> dict[str, Any]:
        return plain(self.values)


@dataclass(frozen=True, init=False)
class TransportSpec(FrozenSpec):
    native_endpoint: str
    think_mode: str
    temperature: float
    context_tokens: int
    max_output_tokens: int
    timeout_sec: float
    generation_seed_master: int


@dataclass(frozen=True, init=False)
class RuntimeSources(FrozenSpec):
    runtime_config: str
    base_runtime_config: str
    protocol_constants: str
    require_clean_git: bool


@dataclass(frozen=True, init=False)
class FixedHarnessSpec(FrozenSpec):
    pass


@dataclass(frozen=True, init=False)
class SimulationSpec(FrozenSpec):
    pass


@dataclass(frozen=True, init=False)
class ScoringSpec(FrozenSpec):
    pass


@dataclass(frozen=True, init=False)
class SelectionSpec(FrozenSpec):
    pass


@dataclass(frozen=True, init=False)
class BootstrapSpec(FrozenSpec):
    draws: int
    version: str


@dataclass(frozen=True, init=False)
class OutputSpec(FrozenSpec):
    root: str
    s1: str
    smoke: str
    llm_campaign: str
    baselines: str
    analysis: str


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def publish_once(path: Path, content: bytes) -> None:
    """Atomically publish content once; an existing artifact must byte-match."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_synced_temp(path.parent, content)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Frozen campaign manifest already exists: {path}.")
        finally:
            temporary.unlink(missing_ok=True)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _write_synced_temp(directory: Path, content: bytes) -> Path:
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)
