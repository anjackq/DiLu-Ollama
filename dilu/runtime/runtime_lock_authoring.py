"""S1 native capability probing and transactional runtime-lock authoring."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import requests

from ._minimal_factorial_manifest import REVISION_RE, RuntimeSnapshot, case_fingerprint
from ._minimal_factorial_schedule_support import canonical_sha256
from ._runtime_lock_authoring_support import (
    GetCallable,
    PostCallable,
    canonical_bytes,
)
from ._runtime_lock_authoring_workflow import (
    BoundaryHook,
    RuntimeLockArtifact,
    artifact_paths,
    build_fresh_campaign_plan,
    publish_staged_campaign,
)
from ._runtime_lock_existing import ExistingCampaign, load_existing_campaign
from ._runtime_lock_transaction import (
    guard_staged_destination,
    install_staged_destination,
    staged_destination,
)
from ._runtime_lock_tree_validation import (
    validate_exact_lock_tree,
    validate_unredirected_artifact_paths,
)
from .minimal_factorial_schedule import (
    ExperimentManifest,
    build_runtime_snapshot,
    load_experiment_manifest,
)
from .ollama_transport import OllamaModelIdentity
from .scientific_transport_types import (
    ScientificTransportCapabilities,
    canonical_action_text_schema,
)


@dataclass(frozen=True)
class S1AuthoringResult:
    preflight_path: Path
    preflight_sha256: str
    model_bindings: Mapping[str, OllamaModelIdentity]
    capabilities: Mapping[str, ScientificTransportCapabilities]
    lock_artifacts: tuple[RuntimeLockArtifact, ...]


def author_verified_runtime_locks(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    get: GetCallable | None = None,
    post: PostCallable | None = None,
    publication_hook: Callable[[Path], None] | None = None,
    publication_boundary_hook: BoundaryHook | None = None,
) -> S1AuthoringResult:
    """Load an exact frozen destination or transactionally author a fresh one."""
    root = _require_repo_root(repo_root)
    manifest = load_experiment_manifest(
        root / "configs" / "iclr2027" / "minimal_factorial.yaml"
    )
    case_set = _load_checked_case_set(root, manifest)
    destination = (
        root / manifest.outputs.root if output_root is None else Path(output_root)
    )
    schema_bytes = canonical_bytes(canonical_action_text_schema())
    validate_unredirected_artifact_paths((destination / ".authoring-probe",))

    if os.path.lexists(destination):
        snapshot = build_runtime_snapshot(manifest, case_set)
        _validate_snapshot(snapshot)
        existing = load_existing_campaign(
            destination=destination,
            manifest=manifest,
            case_set=case_set,
            snapshot=snapshot,
            canonical_schema_bytes=schema_bytes,
        )
        return _result(destination, existing)

    snapshot = build_runtime_snapshot(manifest, case_set)
    _validate_snapshot(snapshot)
    resolved_get = requests.get if get is None else get
    resolved_post = requests.post if post is None else post
    with staged_destination(destination) as transaction:
        stage = transaction.stage
        plan = build_fresh_campaign_plan(
            destination=stage,
            manifest=manifest,
            case_set=case_set,
            snapshot=snapshot,
            canonical_schema_bytes=schema_bytes,
            get=resolved_get,
            post=resolved_post,
        )
        _validate_runtime_stability(
            manifest,
            case_set,
            snapshot,
            schema_bytes,
        )
        publish_staged_campaign(
            destination=stage,
            manifest=manifest,
            case_set=case_set,
            snapshot=snapshot,
            plan=plan,
            boundary_hook=publication_boundary_hook,
            boundary_guard=lambda: guard_staged_destination(transaction),
        )
        staged_paths = artifact_paths(stage, plan.locks)
        staged_relative_paths = tuple(path.relative_to(stage) for path in staged_paths)
        validate_exact_lock_tree(stage, staged_relative_paths)
        if publication_hook is not None:
            guard_staged_destination(transaction)
            publication_hook(stage)
            guard_staged_destination(transaction)
        validate_exact_lock_tree(stage, staged_relative_paths)
        final_paths = tuple(destination / path for path in staged_relative_paths)
        install_staged_destination(
            state=transaction,
            final_artifact_paths=final_paths,
        )

    existing = load_existing_campaign(
        destination=destination,
        manifest=manifest,
        case_set=case_set,
        snapshot=snapshot,
        canonical_schema_bytes=schema_bytes,
    )
    return _result(destination, existing)


def _result(destination: Path, existing: ExistingCampaign) -> S1AuthoringResult:
    return S1AuthoringResult(
        destination / "s1" / "model_preflight.json",
        existing.preflight_sha256,
        MappingProxyType(dict(existing.bindings)),
        MappingProxyType(dict(existing.capabilities)),
        existing.lock_artifacts,
    )


def _validate_runtime_stability(
    manifest: ExperimentManifest,
    case_set: Mapping[str, Any],
    snapshot: RuntimeSnapshot,
    schema_bytes: bytes,
) -> None:
    final_snapshot = build_runtime_snapshot(manifest, case_set)
    if final_snapshot.sha256 != snapshot.sha256 or dict(final_snapshot.payload) != dict(
        snapshot.payload
    ):
        raise ValueError("Runtime snapshot drift after capability probe.")
    if canonical_bytes(canonical_action_text_schema()) != schema_bytes:
        raise ValueError("Canonical action schema drift after capability probe.")


def _load_checked_case_set(
    root: Path,
    manifest: ExperimentManifest,
) -> dict[str, Any]:
    case_path = (root / manifest.case_path).resolve(strict=True)
    try:
        case_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Frozen case set must be inside the repository.") from exc
    case_set = json.loads(case_path.read_text(encoding="utf-8"))
    case_fingerprint(case_set)
    return case_set


def _require_repo_root(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError("repo_root must be a pathlib.Path.")
    root = value.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repo_root must identify a directory.")
    return root


def _validate_snapshot(snapshot: RuntimeSnapshot) -> None:
    if not isinstance(snapshot, RuntimeSnapshot):
        raise TypeError("Snapshot builder must return RuntimeSnapshot.")
    revision = snapshot.payload.get("code_revision")
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise ValueError("Runtime snapshot revision is not an exact commit SHA.")
    if snapshot.sha256 != canonical_sha256(snapshot.payload):
        raise ValueError("Runtime snapshot hash drift.")


__all__ = [
    "RuntimeLockArtifact",
    "S1AuthoringResult",
    "author_verified_runtime_locks",
]
