"""Pure campaign planning and staged artifact publication."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

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
    RuntimeSnapshot,
    ScheduledEpisode,
    build_harness_config,
    build_smoke_schedule,
    build_union_schedule,
    write_frozen_campaign_manifest,
)
from .ollama_transport import OllamaModelIdentity
from .scientific_transport_types import (
    SCHEMA_MECHANISM,
    ScientificTransportCapabilities,
)

BoundaryHook = Callable[[int, Path], None]
BoundaryGuard = Callable[[], None]


@dataclass(frozen=True)
class RuntimeLockArtifact:
    model_slot: str
    condition_id: str
    runtime_lock_path: Path
    authorization_path: Path
    verified_binding: VerifiedRuntimeLockBinding


@dataclass(frozen=True)
class LockPlan:
    model_slot: str
    condition_id: str
    mapping: Mapping[str, str]
    runtime_path: Path
    authorization_path: Path
    runtime_bytes: bytes
    authorization_bytes: bytes


@dataclass(frozen=True)
class CampaignPlan:
    preflight_bytes: bytes
    preflight_sha256: str
    bindings: Mapping[str, OllamaModelIdentity]
    capabilities: Mapping[str, ScientificTransportCapabilities]
    smoke: tuple[ScheduledEpisode, ...]
    union: tuple[ScheduledEpisode, ...]
    locks: tuple[LockPlan, ...]


def build_fresh_campaign_plan(
    *,
    destination: Path,
    manifest: ExperimentManifest,
    case_set: Mapping[str, Any],
    snapshot: RuntimeSnapshot,
    canonical_schema_bytes: bytes,
    get: GetCallable,
    post: PostCallable,
) -> CampaignPlan:
    bindings, records = probe_models(
        manifest,
        canonical_schema_bytes=canonical_schema_bytes,
        get=get,
        post=post,
    )
    preflight_bytes = canonical_bytes(
        {
            "artifact_type": "ollama_native_capability_preflight_v1",
            "runtime_snapshot_sha256": "sha256:" + snapshot.sha256,
            "records": records,
        }
    )
    preflight_sha256 = bytes_sha256(preflight_bytes)
    capabilities = build_capabilities(manifest, bindings, preflight_sha256)
    smoke, union = build_schedules(manifest, case_set, snapshot, bindings)
    locks = build_lock_plans(destination, manifest, smoke, capabilities)
    return CampaignPlan(
        preflight_bytes,
        preflight_sha256,
        MappingProxyType(bindings),
        MappingProxyType(capabilities),
        smoke,
        union,
        locks,
    )


def probe_models(
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


def build_capabilities(
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


def build_schedules(
    manifest: ExperimentManifest,
    case_set: Mapping[str, Any],
    snapshot: RuntimeSnapshot,
    bindings: Mapping[str, OllamaModelIdentity],
) -> tuple[tuple[ScheduledEpisode, ...], tuple[ScheduledEpisode, ...]]:
    digests = {slot: identity.model_digest for slot, identity in bindings.items()}
    smoke = build_smoke_schedule(manifest, case_set, digests, runtime_snapshot=snapshot)
    union = build_union_schedule(manifest, case_set, digests, runtime_snapshot=snapshot)
    return smoke, union


def build_lock_plans(
    destination: Path,
    manifest: ExperimentManifest,
    smoke: tuple[ScheduledEpisode, ...],
    capabilities: Mapping[str, ScientificTransportCapabilities],
) -> tuple[LockPlan, ...]:
    rows = {(row.model_slot, row.condition_id): row for row in smoke}
    expected = {
        (model.slot, f"c{index:03b}") for model in manifest.models for index in range(8)
    }
    if set(rows) != expected or len(smoke) != 16:
        raise ValueError("Smoke schedule does not provide exactly 16 lock identities.")
    plans: list[LockPlan] = []
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
                LockPlan(
                    model.slot,
                    condition_id,
                    MappingProxyType(mapping),
                    lock_root / "RUNTIME_PROTOCOL_LOCK.json",
                    lock_root / "PROTOCOL_FROZEN.json",
                    runtime_bytes,
                    authorization_bytes,
                )
            )
    return tuple(plans)


def artifact_paths(destination: Path, locks: tuple[LockPlan, ...]) -> tuple[Path, ...]:
    return (
        destination / "s1" / "model_preflight.json",
        destination / "smoke" / "campaign_manifest.json",
        destination / "llm_campaign" / "campaign_manifest.json",
        destination / "llm_campaign" / "union_schedule.json",
        *(
            path
            for plan in locks
            for path in (plan.runtime_path, plan.authorization_path)
        ),
    )


def publish_staged_campaign(
    *,
    destination: Path,
    manifest: ExperimentManifest,
    case_set: Mapping[str, Any],
    snapshot: RuntimeSnapshot,
    plan: CampaignPlan,
    boundary_hook: BoundaryHook | None,
    boundary_guard: BoundaryGuard,
) -> tuple[RuntimeLockArtifact, ...]:
    boundary_index = 0

    def boundary(path: Path) -> None:
        nonlocal boundary_index
        boundary_guard()
        if boundary_hook is not None:
            boundary_hook(boundary_index, path)
        boundary_guard()
        boundary_index += 1

    preflight_path = destination / "s1" / "model_preflight.json"
    boundary(preflight_path)
    publish_once(preflight_path, plan.preflight_bytes)
    smoke_path = destination / "smoke" / "campaign_manifest.json"
    boundary(smoke_path)
    write_frozen_campaign_manifest(
        smoke_path,
        manifest,
        snapshot,
        plan.smoke,
        case_set=case_set,
        model_bindings=plan.bindings,
    )
    claim_path = destination / "llm_campaign" / "campaign_manifest.json"
    boundary(claim_path)
    write_frozen_campaign_manifest(
        claim_path,
        manifest,
        snapshot,
        plan.union,
        case_set=case_set,
        model_bindings=plan.bindings,
    )
    union_path = destination / "llm_campaign" / "union_schedule.json"
    boundary(union_path)
    publish_once(
        union_path,
        canonical_bytes([row.to_payload() for row in plan.union]),
    )
    artifacts: list[RuntimeLockArtifact] = []
    for lock in plan.locks:
        boundary(lock.runtime_path)
        publish_once(lock.runtime_path, lock.runtime_bytes)
        boundary(lock.authorization_path)
        publish_once(lock.authorization_path, lock.authorization_bytes)
        artifacts.append(verify_lock_plan(lock))
    if boundary_index != 36:
        raise ValueError("Staged campaign publication boundary count drifted.")
    return tuple(artifacts)


def verify_lock_plan(plan: LockPlan) -> RuntimeLockArtifact:
    loaded = load_verified_runtime_lock_binding(
        runtime_lock_path=plan.runtime_path,
        authorization_path=plan.authorization_path,
    )
    if loaded.to_dict() != dict(plan.mapping):
        raise ValueError("Authored runtime-lock loader round trip drifted.")
    return RuntimeLockArtifact(
        plan.model_slot,
        plan.condition_id,
        plan.runtime_path,
        plan.authorization_path,
        loaded,
    )
