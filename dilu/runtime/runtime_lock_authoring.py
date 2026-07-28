"""S1 native capability probing and verified runtime-lock authoring."""

from __future__ import annotations

import json
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
    bytes_sha256,
    canonical_bytes,
    probe_model,
    publish_once,
)
from ._scientific_runtime_binding import (
    RuntimeLockBinding,
    VerifiedRuntimeLockBinding,
    load_verified_runtime_lock_binding,
)
from .harness_config import ThinkMode
from .minimal_factorial_schedule import (
    ExperimentManifest,
    ScheduledEpisode,
    build_harness_config,
    build_runtime_snapshot,
    build_smoke_schedule,
    build_union_schedule,
    load_experiment_manifest,
    write_frozen_campaign_manifest,
)
from .ollama_transport import OllamaModelIdentity
from .scientific_transport_types import (
    SCHEMA_MECHANISM,
    ScientificTransportCapabilities,
    canonical_action_text_schema,
)


@dataclass(frozen=True)
class RuntimeLockArtifact:
    model_slot: str
    condition_id: str
    runtime_lock_path: Path
    authorization_path: Path
    verified_binding: VerifiedRuntimeLockBinding


@dataclass(frozen=True)
class S1AuthoringResult:
    preflight_path: Path
    preflight_sha256: str
    model_bindings: Mapping[str, OllamaModelIdentity]
    capabilities: Mapping[str, ScientificTransportCapabilities]
    lock_artifacts: tuple[RuntimeLockArtifact, ...]


@dataclass(frozen=True)
class _LockPlan:
    model_slot: str
    condition_id: str
    mapping: Mapping[str, str]
    runtime_path: Path
    authorization_path: Path
    runtime_bytes: bytes
    authorization_bytes: bytes


def author_verified_runtime_locks(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    get: GetCallable | None = None,
    post: PostCallable | None = None,
    publication_hook: Callable[[Path], None] | None = None,
) -> S1AuthoringResult:
    """Probe two frozen Ollama models and author all 16 S1 lock pairs."""
    root = _require_repo_root(repo_root)
    manifest = load_experiment_manifest(
        root / "configs" / "iclr2027" / "minimal_factorial.yaml"
    )
    case_set = _load_checked_case_set(root, manifest)
    schema_before = canonical_bytes(canonical_action_text_schema())
    snapshot = build_runtime_snapshot(manifest, case_set)
    _validate_snapshot(snapshot)

    resolved_get = requests.get if get is None else get
    resolved_post = requests.post if post is None else post
    bindings, records = _probe_models(
        manifest,
        canonical_schema_bytes=schema_before,
        get=resolved_get,
        post=resolved_post,
    )
    preflight_payload = {
        "artifact_type": "ollama_native_capability_preflight_v1",
        "runtime_snapshot_sha256": "sha256:" + snapshot.sha256,
        "records": records,
    }
    preflight_bytes = canonical_bytes(preflight_payload)
    preflight_sha256 = bytes_sha256(preflight_bytes)
    capabilities = _capabilities(manifest, bindings, preflight_sha256)
    digests = {slot: identity.model_digest for slot, identity in bindings.items()}
    smoke = build_smoke_schedule(
        manifest,
        case_set,
        digests,
        runtime_snapshot=snapshot,
    )
    union = build_union_schedule(
        manifest,
        case_set,
        digests,
        runtime_snapshot=snapshot,
    )
    final_snapshot = build_runtime_snapshot(manifest, case_set)
    if final_snapshot.sha256 != snapshot.sha256 or dict(final_snapshot.payload) != dict(
        snapshot.payload
    ):
        raise ValueError("Runtime snapshot drift after capability probe.")
    if canonical_bytes(canonical_action_text_schema()) != schema_before:
        raise ValueError("Canonical action schema drift after capability probe.")

    destination = (
        root / manifest.outputs.root if output_root is None else Path(output_root)
    )
    plans = _build_lock_plans(
        destination,
        manifest,
        smoke,
        capabilities,
    )
    _publish_campaign_artifacts(
        destination=destination,
        manifest=manifest,
        case_set=case_set,
        snapshot=snapshot,
        bindings=bindings,
        smoke=smoke,
        union=union,
        preflight_bytes=preflight_bytes,
    )
    lock_artifacts = _publish_and_verify_locks(plans)
    if publication_hook is not None:
        publication_hook(destination)
    return S1AuthoringResult(
        preflight_path=destination / "s1" / "model_preflight.json",
        preflight_sha256=preflight_sha256,
        model_bindings=MappingProxyType(dict(bindings)),
        capabilities=MappingProxyType(dict(capabilities)),
        lock_artifacts=lock_artifacts,
    )


def _probe_models(
    manifest: ExperimentManifest,
    *,
    canonical_schema_bytes: bytes,
    get: GetCallable,
    post: PostCallable,
) -> tuple[dict[str, OllamaModelIdentity], list[dict[str, Any]]]:
    bindings: dict[str, OllamaModelIdentity] = {}
    records: list[dict[str, Any]] = []
    think_mode = ThinkMode(manifest.transport.think_mode)
    for index, model in enumerate(manifest.models):
        identity, model_records = probe_model(
            model_slot=model.slot,
            model_tag=model.tag,
            native_endpoint=manifest.transport.native_endpoint,
            seed=manifest.transport.generation_seed_master + index,
            temperature=manifest.transport.temperature,
            context_tokens=manifest.transport.context_tokens,
            max_output_tokens=manifest.transport.max_output_tokens,
            timeout_sec=manifest.transport.timeout_sec,
            think_mode=think_mode,
            canonical_schema_bytes=canonical_schema_bytes,
            get=get,
            post=post,
        )
        bindings[model.slot] = identity
        records.extend({"model_slot": model.slot, **record} for record in model_records)
    if len(records) != 6:
        raise ValueError("Capability preflight must contain exactly six direct calls.")
    return bindings, records


def _capabilities(
    manifest: ExperimentManifest,
    bindings: Mapping[str, OllamaModelIdentity],
    artifact_hash: str,
) -> dict[str, ScientificTransportCapabilities]:
    think_mode = ThinkMode(manifest.transport.think_mode)
    return {
        slot: ScientificTransportCapabilities(
            model_tag=identity.model_tag,
            model_digest=identity.model_digest,
            native_endpoint=manifest.transport.native_endpoint,
            supported_think_modes=(think_mode,),
            seed_verified=True,
            schema_verified=True,
            capability_probe_id=f"s1-native-probe-{slot}",
            capability_artifact_hash=artifact_hash,
            schema_mechanism=SCHEMA_MECHANISM,
        )
        for slot, identity in bindings.items()
    }


def _build_lock_plans(
    destination: Path,
    manifest: ExperimentManifest,
    smoke: tuple[ScheduledEpisode, ...],
    capabilities: Mapping[str, ScientificTransportCapabilities],
) -> tuple[_LockPlan, ...]:
    rows = {(row.model_slot, row.condition_id): row for row in smoke}
    expected = {
        (model.slot, f"c{index:03b}") for model in manifest.models for index in range(8)
    }
    if set(rows) != expected or len(smoke) != 16:
        raise ValueError("Smoke schedule does not provide exactly 16 lock identities.")
    plans: list[_LockPlan] = []
    for model in manifest.models:
        for index in range(8):
            condition = build_harness_config(manifest, index)
            condition_id = condition.condition_id()
            row = rows[(model.slot, condition_id)]
            if condition.to_canonical_dict() != row.condition.to_canonical_dict():
                raise ValueError(
                    "Scheduled condition drifted from exact HarnessConfig."
                )
            binding = RuntimeLockBinding.from_runtime(
                harness_config=condition,
                identity=row.identity(),
                capabilities=capabilities[model.slot],
            )
            mapping = binding.to_dict()
            runtime_bytes = canonical_bytes(mapping)
            authorization_bytes = canonical_bytes(
                {
                    "artifact_type": "runtime_lock_authorization_v1",
                    "runtime_lock_sha256": bytes_sha256(runtime_bytes),
                }
            )
            lock_root = destination / "s1" / "locks" / model.slot / condition_id
            plans.append(
                _LockPlan(
                    model.slot,
                    condition_id,
                    MappingProxyType(mapping),
                    lock_root / "RUNTIME_PROTOCOL_LOCK.json",
                    lock_root / "PROTOCOL_FROZEN.json",
                    runtime_bytes,
                    authorization_bytes,
                )
            )
    if len(plans) != 16:
        raise ValueError("Runtime-lock authoring requires exactly 16 unique pairs.")
    return tuple(plans)


def _publish_campaign_artifacts(
    *,
    destination: Path,
    manifest: ExperimentManifest,
    case_set: Mapping[str, Any],
    snapshot: RuntimeSnapshot,
    bindings: Mapping[str, OllamaModelIdentity],
    smoke: tuple[ScheduledEpisode, ...],
    union: tuple[ScheduledEpisode, ...],
    preflight_bytes: bytes,
) -> None:
    publish_once(destination / "s1" / "model_preflight.json", preflight_bytes)
    write_frozen_campaign_manifest(
        destination / "smoke" / "campaign_manifest.json",
        manifest,
        snapshot,
        smoke,
        case_set=case_set,
        model_bindings=bindings,
    )
    write_frozen_campaign_manifest(
        destination / "llm_campaign" / "campaign_manifest.json",
        manifest,
        snapshot,
        union,
        case_set=case_set,
        model_bindings=bindings,
    )
    publish_once(
        destination / "llm_campaign" / "union_schedule.json",
        canonical_bytes([row.to_payload() for row in union]),
    )


def _publish_and_verify_locks(
    plans: tuple[_LockPlan, ...],
) -> tuple[RuntimeLockArtifact, ...]:
    artifacts: list[RuntimeLockArtifact] = []
    for plan in plans:
        publish_once(plan.runtime_path, plan.runtime_bytes)
        publish_once(plan.authorization_path, plan.authorization_bytes)
        loaded = load_verified_runtime_lock_binding(
            runtime_lock_path=plan.runtime_path,
            authorization_path=plan.authorization_path,
        )
        if loaded.to_dict() != dict(plan.mapping):
            raise ValueError("Authored runtime-lock loader round trip drifted.")
        artifacts.append(
            RuntimeLockArtifact(
                plan.model_slot,
                plan.condition_id,
                plan.runtime_path,
                plan.authorization_path,
                loaded,
            )
        )
    return tuple(artifacts)


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
