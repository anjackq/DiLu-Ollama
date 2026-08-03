"""Trusted runtime-lock verification for registered analysis inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ._minimal_factorial_schedule_support import canonical_sha256
from ._scientific_runtime_binding import load_verified_runtime_lock_binding

_LOCK_TO_EPISODE = {
    "source_artifact_sha256": "runtime_lock_source_artifact_sha256",
    "authorization_artifact_sha256": ("runtime_lock_authorization_artifact_sha256"),
    "binding_sha256": "runtime_lock_binding_sha256",
    "prompt_sha256": "prompt_sha256",
    "capability_artifact_sha256": "capability_artifact_sha256",
    "capability_snapshot_sha256": "capability_snapshot_sha256",
    "trace_schema_sha256": "trace_schema_sha256",
}


def validate_authorized_runtime_locks(
    lock_root: Path,
    claim: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    *,
    lock_loader: Callable[..., Any] = load_verified_runtime_lock_binding,
) -> tuple[str, ...]:
    errors: set[str] = set()
    schedule = _rows(claim.get("schedule"), "claim schedule")
    manifest = _object(claim.get("manifest"), "registered manifest")
    snapshot = _object(claim.get("runtime_snapshot"), "runtime snapshot")
    transport = _object(manifest.get("transport"), "registered transport")
    scheduled_cells = _cells(schedule)
    episode_cells = _cells(episodes)
    if len(scheduled_cells) != 16 or set(episode_cells) != set(scheduled_cells):
        return ("authorized runtime-lock gate requires the exact 16 cells",)
    for cell, scheduled_rows in sorted(scheduled_cells.items()):
        model_slot, condition_id = cell
        lock_dir = Path(lock_root) / model_slot / condition_id
        try:
            binding = lock_loader(
                runtime_lock_path=lock_dir / "RUNTIME_PROTOCOL_LOCK.json",
                authorization_path=lock_dir / "PROTOCOL_FROZEN.json",
            )
            _validate_binding(
                binding,
                scheduled_rows[0],
                episode_cells[cell],
                transport,
                snapshot,
                errors,
            )
        except Exception as exc:
            errors.add(
                f"authorized runtime-lock {model_slot}/{condition_id} failed: {exc}"
            )
    return tuple(sorted(errors))


def _validate_binding(
    binding: Any,
    scheduled: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    transport: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    errors: set[str],
) -> None:
    condition = _object(scheduled.get("condition"), "scheduled condition")
    condition_transport = _object(condition.get("transport"), "condition transport")
    expected = {
        "condition_id": scheduled.get("condition_id"),
        "config_sha256": "sha256:" + canonical_sha256(condition),
        "model_tag": scheduled.get("model_tag"),
        "model_digest": scheduled.get("model_digest"),
        "native_endpoint": transport.get("native_endpoint"),
        "think_mode": condition_transport.get("think_mode"),
        "benchmark_fingerprint": scheduled.get("benchmark_fingerprint"),
        "code_revision": scheduled.get("code_revision"),
    }
    for field, value in expected.items():
        observed = getattr(binding, field)
        if field == "think_mode":
            observed = getattr(observed, "value", observed)
        if observed != value:
            errors.add(f"authorized runtime-lock {field} drifted")
    expected_trace = _sha(snapshot.get("trace_schema_sha256"))
    if binding.trace_schema_sha256 != expected_trace:
        errors.add("authorized runtime-lock trace schema drifted")
    for row in episodes:
        if row.get("model_tag") != binding.model_tag:
            errors.add("episode model tag drifted from authorized runtime lock")
        if row.get("model_digest") != binding.model_digest:
            errors.add("episode model digest drifted from authorized runtime lock")
        if row.get("config_sha256") != binding.config_sha256:
            errors.add("episode config drifted from authorized runtime lock")
        for lock_field, episode_field in _LOCK_TO_EPISODE.items():
            if row.get(episode_field) != getattr(binding, lock_field):
                errors.add(
                    f"episode {episode_field} drifted from authorized runtime lock"
                )


def _cells(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    output: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (_text(row, "model_slot"), _text(row, "condition_id"))
        output.setdefault(key, []).append(row)
    return output


def _rows(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError(f"{name} must be a list of objects")
    return tuple(value)


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _sha(value: Any) -> str:
    text = str(value or "")
    return text if text.startswith("sha256:") else "sha256:" + text


__all__ = ["validate_authorized_runtime_locks"]
