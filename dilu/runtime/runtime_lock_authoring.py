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

from ._grounded_decoding_lock_authoring import (
    GROUNDED_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE,
    build_v8_capabilities,
    probe_v8_models,
    resolve_comparator_digest_matches,
    v8_artifact_paths,
)
from ._grounded_decoding_manifest_support import (
    GroundedDecodingManifest,
    load_grounded_decoding_manifest,
)
from ._minimal_factorial_manifest import REVISION_RE, RuntimeSnapshot, case_fingerprint
from ._minimal_factorial_schedule_support import canonical_sha256
from ._runtime_lock_authoring_support import (
    GetCallable,
    PostCallable,
    bytes_sha256,
    canonical_bytes,
    publish_once,
)
from ._runtime_lock_authoring_workflow import (
    BoundaryHook,
    RuntimeLockArtifact,
    artifact_paths,
    build_fresh_campaign_plan,
    build_lock_plans,
    publish_staged_campaign,
    verify_lock_plan,
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
from .grounded_decoding_schedule import build_v8_smoke_schedule
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


@dataclass(frozen=True)
class GroundedDecodingS1Result:
    """S1 result for the ICLR 2027 grounded-decoding (V8) campaign.

    Scoped to exactly what Task 5 authors: the capability preflight (with
    the extra grounded probe and the ``comparator_digest_match`` record)
    and the 10 lock/authorization pairs (5 models x {c120, c121}). Unlike
    ``S1AuthoringResult``, no ``smoke``/``llm_campaign`` frozen schedule
    files are written here -- V8's episode schedule is built on demand from
    ``grounded_decoding_schedule.build_v8_schedule`` by the campaign runner.
    """

    preflight_path: Path
    preflight_sha256: str
    model_bindings: Mapping[str, OllamaModelIdentity]
    capabilities: Mapping[str, ScientificTransportCapabilities]
    comparator_digest_match: Mapping[str, bool]
    lock_artifacts: tuple[RuntimeLockArtifact, ...]


def author_verified_runtime_locks(
    repo_root: Path,
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
    get: GetCallable | None = None,
    post: PostCallable | None = None,
    publication_hook: Callable[[Path], None] | None = None,
    publication_boundary_hook: BoundaryHook | None = None,
) -> S1AuthoringResult:
    """Load an exact frozen destination or transactionally author a fresh one."""
    root = _require_repo_root(repo_root)
    source = (
        root / "configs" / "iclr2027" / "minimal_factorial.yaml"
        if manifest_path is None
        else Path(manifest_path).resolve(strict=True)
    )
    manifest = load_experiment_manifest(source)
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


def author_verified_grounded_decoding_locks(
    repo_root: Path,
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
    v5_manifest_path: Path | None = None,
    v7_manifest_path: Path | None = None,
    get: GetCallable | None = None,
    post: PostCallable | None = None,
    publication_hook: Callable[[Path], None] | None = None,
) -> GroundedDecodingS1Result:
    """Transactionally author the V8 S1 capability preflight and locks.

    Fresh-authoring only (no idempotent existing-destination reload, unlike
    ``author_verified_runtime_locks``): Task 5 scopes V8's S1 step to the
    capability probe (3 trusted calls plus 1 grounded-schema call per
    model), the comparator digest comparison, and the 10 lock/authorization
    pairs (5 models x {c120, c121}), all reusing the existing
    ``build_lock_plans``/``verify_lock_plan`` machinery unmodified.
    """
    root = _require_repo_root(repo_root)
    source = (
        root / "configs" / "iclr2027" / "grounded_decoding_v8.yaml"
        if manifest_path is None
        else Path(manifest_path).resolve(strict=True)
    )
    manifest = load_grounded_decoding_manifest(source)
    case_set = _load_checked_case_set(root, manifest)
    destination = (
        root / manifest.outputs.root if output_root is None else Path(output_root)
    )
    validate_unredirected_artifact_paths((destination / ".authoring-probe",))

    v5_manifest = load_experiment_manifest(
        root / "configs" / "iclr2027" / "minimal_factorial.yaml"
        if v5_manifest_path is None
        else Path(v5_manifest_path).resolve(strict=True)
    )
    v7_manifest = load_experiment_manifest(
        root / "configs" / "iclr2027" / "model_breadth_factorial_v7.yaml"
        if v7_manifest_path is None
        else Path(v7_manifest_path).resolve(strict=True)
    )

    snapshot = build_runtime_snapshot(manifest, case_set)
    _validate_snapshot(snapshot)
    resolved_get = requests.get if get is None else get
    resolved_post = requests.post if post is None else post

    with staged_destination(destination) as transaction:
        stage = transaction.stage
        bindings, records = probe_v8_models(manifest, get=resolved_get, post=resolved_post)
        _validate_v8_runtime_stability(manifest, case_set, snapshot)
        comparator_digest_match = resolve_comparator_digest_matches(
            manifest, bindings, v5_manifest, v7_manifest
        )
        preflight_bytes = canonical_bytes(
            {
                "artifact_type": GROUNDED_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE,
                "runtime_snapshot_sha256": "sha256:" + snapshot.sha256,
                "comparator_digest_match": dict(comparator_digest_match),
                "records": records,
            }
        )
        preflight_sha256 = bytes_sha256(preflight_bytes)
        capabilities = build_v8_capabilities(manifest, bindings, preflight_sha256)
        smoke = build_v8_smoke_schedule(manifest, case_set, bindings, runtime_snapshot=snapshot)
        locks = build_lock_plans(stage, manifest, smoke, capabilities)

        guard_staged_destination(transaction)
        preflight_path = stage / "s1" / "model_preflight.json"
        publish_once(preflight_path, preflight_bytes)
        for lock in locks:
            guard_staged_destination(transaction)
            publish_once(lock.runtime_path, lock.runtime_bytes)
            publish_once(lock.authorization_path, lock.authorization_bytes)
            verify_lock_plan(lock)

        staged_paths = v8_artifact_paths(stage, locks)
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

    # Re-derive lock plans rooted at the final (post-install) destination --
    # the staged-phase plans above point at the staging directory, which no
    # longer exists once install_staged_destination renames it away. This
    # mirrors how author_verified_runtime_locks re-derives its result via a
    # fresh load_existing_campaign() call after installation.
    final_locks = build_lock_plans(destination, manifest, smoke, capabilities)
    artifacts = tuple(verify_lock_plan(lock) for lock in final_locks)

    return GroundedDecodingS1Result(
        destination / "s1" / "model_preflight.json",
        preflight_sha256,
        MappingProxyType(dict(bindings)),
        MappingProxyType(dict(capabilities)),
        MappingProxyType(dict(comparator_digest_match)),
        artifacts,
    )


def _validate_v8_runtime_stability(
    manifest: GroundedDecodingManifest,
    case_set: Mapping[str, Any],
    snapshot: RuntimeSnapshot,
) -> None:
    final_snapshot = build_runtime_snapshot(manifest, case_set)
    if final_snapshot.sha256 != snapshot.sha256 or dict(final_snapshot.payload) != dict(
        snapshot.payload
    ):
        raise ValueError("Runtime snapshot drift after capability probe.")


__all__ = [
    "GroundedDecodingS1Result",
    "RuntimeLockArtifact",
    "S1AuthoringResult",
    "author_verified_grounded_decoding_locks",
    "author_verified_runtime_locks",
]
