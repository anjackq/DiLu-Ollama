"""Frozen campaign loading helpers for the minimal-factorial runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ._minimal_factorial_manifest import serialize_frozen_campaign
from ._minimal_factorial_schedule_support import plain
from ._runtime_lock_authoring_support import (
    OLLAMA_NATIVE_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE,
    canonical_bytes,
)
from ._runtime_lock_authoring_workflow import build_capabilities
from ._runtime_lock_existing import _load_bindings, _load_canonical_object
from .minimal_factorial_schedule import (
    ExperimentManifest,
    RuntimeSnapshot,
    ScheduledEpisode,
)
from .ollama_transport import OllamaModelIdentity
from .scientific_transport_types import (
    ScientificTransportCapabilities,
    canonical_action_text_schema,
)


@dataclass(frozen=True)
class FrozenS1:
    model_bindings: Mapping[str, OllamaModelIdentity]
    capabilities: Mapping[str, ScientificTransportCapabilities]


@dataclass(frozen=True)
class FrozenThresholds:
    ttc_threshold_sec: float
    headway_threshold_m: float
    rear_ttc_threshold_sec: float
    rear_headway_threshold_m: float
    low_speed_blocking_threshold_mps: float
    blocking_front_gap_safe_m: float
    blocking_front_ttc_safe_sec: float
    stop_threshold_mps: float
    near_stop_threshold_mps: float
    slow_decision_threshold_sec: float


@dataclass(frozen=True)
class ValidatedCampaign:
    repo_root: Path
    manifest: ExperimentManifest
    case_set: Mapping[str, Any]
    snapshot: RuntimeSnapshot


@dataclass(frozen=True)
class PreparedCampaign:
    repo_root: Path
    manifest: ExperimentManifest
    snapshot: RuntimeSnapshot
    case_set: Mapping[str, Any]
    case_by_id: Mapping[str, Mapping[str, Any]]
    schedule: tuple[ScheduledEpisode, ...]
    output_root: Path
    lock_root: Path
    runtime_config: Mapping[str, Any]
    environment_config: Mapping[str, Mapping[str, Any]]
    target_env_id: str
    default_max_steps: int
    thresholds: Any
    capabilities: Mapping[str, ScientificTransportCapabilities]


def validate_live_snapshot(
    manifest_path: Path,
    *,
    repo_root: Any,
    load_manifest: Any,
    load_cases: Any,
    build_snapshot: Any,
) -> ValidatedCampaign:
    root = repo_root(manifest_path)
    manifest = load_manifest(manifest_path.resolve())
    case_set = load_cases(root, manifest)
    snapshot = build_snapshot(manifest, case_set)
    return ValidatedCampaign(root, manifest, case_set, snapshot)


def open_frozen_campaign(
    validated: ValidatedCampaign,
    campaign: str,
    *,
    load_s1: Any,
    build_smoke: Any,
    build_union: Any,
    verify: Any,
) -> PreparedCampaign:
    if campaign not in {"smoke", "claim"}:
        raise ValueError("campaign must be 'smoke' or 'claim'.")
    manifest = validated.manifest
    s1 = load_s1(validated.repo_root, manifest, validated.snapshot)
    digests = {
        slot: identity.model_digest for slot, identity in s1.model_bindings.items()
    }
    builder = build_smoke if campaign == "smoke" else build_union
    schedule = builder(
        manifest,
        validated.case_set,
        digests,
        runtime_snapshot=validated.snapshot,
    )
    directory = (
        manifest.outputs.smoke if campaign == "smoke" else manifest.outputs.llm_campaign
    )
    base = validated.repo_root / manifest.outputs.root
    output_root = base / directory
    verify(
        output_root / "campaign_manifest.json",
        manifest,
        validated.snapshot,
        schedule,
        validated.case_set,
        union_path=(
            output_root / "union_schedule.json" if campaign == "claim" else None
        ),
    )
    runtime = plain(validated.snapshot.payload["runtime_config"])
    environment = plain(validated.snapshot.payload["environment_config"])
    target = manifest.simulation.target_env_id
    return PreparedCampaign(
        validated.repo_root,
        manifest,
        validated.snapshot,
        validated.case_set,
        {case["case_id"]: case for case in validated.case_set["cases"]},
        tuple(schedule),
        output_root,
        base / "s1" / "locks",
        runtime,
        {target: environment},
        target,
        int(runtime.get("simulation_duration", environment.get("duration", 30))),
        thresholds(runtime),
        s1.capabilities,
    )


def load_checked_case_set(
    repo_root: Path,
    manifest: ExperimentManifest,
) -> dict[str, Any]:
    import json

    path = (repo_root / manifest.case_path).resolve(strict=True)
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Frozen case set must be inside the repository.") from exc
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Frozen case set must be a JSON object.")
    return value


def thresholds(config: Mapping[str, Any]) -> FrozenThresholds:
    return FrozenThresholds(
        float(config.get("metrics_ttc_threshold_sec", 2.0)),
        float(config.get("metrics_headway_threshold_m", 15.0)),
        float(config.get("metrics_rear_ttc_threshold_sec", 2.5)),
        float(config.get("metrics_rear_headway_threshold_m", 12.0)),
        float(config.get("metrics_low_speed_blocking_threshold_mps", 8.5)),
        float(config.get("metrics_blocking_front_gap_safe_m", 25.0)),
        float(config.get("metrics_blocking_front_ttc_safe_sec", 4.0)),
        float(config.get("metrics_stop_threshold_mps", 0.5)),
        float(config.get("metrics_near_stop_threshold_mps", 2.0)),
        float(config.get("eval_slow_decision_threshold_sec", 5.0)),
    )


def load_frozen_s1(
    root: Path,
    manifest: ExperimentManifest,
    snapshot: RuntimeSnapshot,
) -> FrozenS1:
    preflight_path = root / manifest.outputs.root / "s1" / "model_preflight.json"
    content, preflight = _load_canonical_object(preflight_path)
    if (
        set(preflight) != {"artifact_type", "runtime_snapshot_sha256", "records"}
        or preflight["artifact_type"]
        != OLLAMA_NATIVE_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE
        or preflight["runtime_snapshot_sha256"] != "sha256:" + snapshot.sha256
    ):
        raise ValueError("Frozen S1 preflight drifted from runtime snapshot.")
    records = preflight["records"]
    if not isinstance(records, list):
        raise ValueError("Frozen S1 preflight records are malformed.")
    bindings = _load_bindings(
        manifest,
        records,
        canonical_schema_bytes=canonical_bytes(canonical_action_text_schema()),
    )
    artifact_hash = "sha256:" + __import__("hashlib").sha256(content).hexdigest()
    capabilities = build_capabilities(manifest, bindings, artifact_hash)
    return FrozenS1(
        MappingProxyType(dict(bindings)),
        MappingProxyType(dict(capabilities)),
    )


def verify_frozen_campaign(
    path: Path,
    manifest: ExperimentManifest,
    snapshot: RuntimeSnapshot,
    schedule: Sequence[ScheduledEpisode],
    case_set: Mapping[str, Any],
    *,
    union_path: Path | None = None,
) -> None:
    expected = serialize_frozen_campaign(manifest, snapshot, schedule, case_set)
    if path.read_bytes() != expected:
        raise ValueError("Frozen campaign manifest bytes drifted.")
    if union_path is not None:
        union_bytes = canonical_bytes([row.to_payload() for row in schedule])
        if union_path.read_bytes() != union_bytes:
            raise ValueError("Frozen union schedule bytes drifted.")


__all__ = [
    "FrozenS1",
    "FrozenThresholds",
    "PreparedCampaign",
    "ValidatedCampaign",
    "load_checked_case_set",
    "load_frozen_s1",
    "open_frozen_campaign",
    "thresholds",
    "validate_live_snapshot",
    "verify_frozen_campaign",
]
