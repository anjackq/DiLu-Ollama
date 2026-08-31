"""Thin runner that reuses the locked minimal-factorial execution path for V8."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from ._grounded_decoding_lock_authoring import (
    GROUNDED_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE,
    build_v8_capabilities,
)
from ._minimal_factorial_runner_campaign import (
    FrozenS1,
    PreparedCampaign,
    load_checked_case_set,
    thresholds,
    verify_frozen_campaign,
)
from ._minimal_factorial_schedule_support import plain
from ._runtime_lock_authoring_support import canonical_bytes
from ._runtime_lock_existing import _load_bindings, _load_canonical_object
from .grounded_decoding_campaign import freeze_v8_campaign_artifacts
from .grounded_decoding_schedule import (
    GroundedDecodingManifest,
    build_runtime_snapshot,
    build_v8_schedule,
    build_v8_smoke_schedule,
    load_grounded_decoding_manifest,
)
from .minimal_factorial_runner import (
    RunSummary,
    _campaign_status_impl,
    _execute_campaign,
)
from .runtime_lock_authoring import author_verified_grounded_decoding_locks
from .scientific_transport_types import canonical_action_text_schema


def run_probe_lock(manifest_path: Path) -> Path:
    source = manifest_path.resolve()
    authored = author_verified_grounded_decoding_locks(
        _repo_root(source), manifest_path=source
    )
    if all(authored.comparator_digest_match.values()):
        _freeze_from_authored_s1(source, authored)
    return authored.preflight_path


def run_smoke(manifest_path: Path, *, resume: bool) -> RunSummary:
    prepared = _prepare_campaign(manifest_path, campaign="smoke")
    return _execute_campaign(
        prepared,
        scheduled_rows=prepared.schedule,
        denominator_rows=prepared.schedule,
        resume=resume,
        stage="smoke",
    )


def run_claim_stage(
    manifest_path: Path,
    *,
    stage: Literal["stage1", "stage2"],
    resume: bool,
    max_episodes: int | None = None,
) -> RunSummary:
    if stage not in {"stage1", "stage2"}:
        raise ValueError("stage must be 'stage1' or 'stage2'.")
    prepared = _prepare_campaign(manifest_path, campaign="claim")
    stage_name = "stage1" if stage == "stage1" else "stage2_additional"
    scheduled = tuple(row for row in prepared.schedule if row.stage == stage_name)
    denominator = scheduled if stage == "stage1" else tuple(prepared.schedule)
    return _execute_campaign(
        prepared,
        scheduled_rows=scheduled,
        denominator_rows=denominator,
        resume=resume,
        stage=stage,
        max_episodes=max_episodes,
    )


def campaign_status(manifest_path: Path) -> dict[str, Any]:
    manifest = load_grounded_decoding_manifest(manifest_path.resolve())
    base = manifest.repo_root() / manifest.outputs.root
    campaigns = []
    for campaign, directory in (
        ("smoke", manifest.outputs.smoke),
        ("claim", manifest.outputs.llm_campaign),
    ):
        if (base / directory / "campaign_manifest.json").is_file():
            campaigns.append(_prepare_campaign(manifest_path, campaign=campaign))
    return _campaign_status_impl(tuple(campaigns))


def _freeze_from_authored_s1(manifest_path: Path, authored: Any) -> None:
    manifest = load_grounded_decoding_manifest(manifest_path)
    root = manifest.repo_root()
    case_set = load_checked_case_set(root, manifest)
    snapshot = build_runtime_snapshot(manifest, case_set)
    freeze_v8_campaign_artifacts(
        repo_root=root,
        manifest=manifest,
        case_set=case_set,
        snapshot=snapshot,
        model_bindings=authored.model_bindings,
    )


def _prepare_campaign(manifest_path: Path, *, campaign: str) -> PreparedCampaign:
    if campaign not in {"smoke", "claim"}:
        raise ValueError("campaign must be 'smoke' or 'claim'.")
    manifest = load_grounded_decoding_manifest(manifest_path.resolve())
    root = manifest.repo_root()
    case_set = load_checked_case_set(root, manifest)
    snapshot = build_runtime_snapshot(manifest, case_set)
    s1 = _load_v8_s1(root, manifest, snapshot)
    if campaign == "smoke":
        schedule = build_v8_smoke_schedule(
            manifest, case_set, s1.model_bindings, runtime_snapshot=snapshot
        )
        directory = manifest.outputs.smoke
    else:
        schedule = build_v8_schedule(
            manifest, case_set, s1.model_bindings, runtime_snapshot=snapshot
        ).all_claim_bearing
        directory = manifest.outputs.llm_campaign
    base = root / manifest.outputs.root
    output_root = base / directory
    verify_frozen_campaign(
        output_root / "campaign_manifest.json",
        manifest,
        snapshot,
        schedule,
        case_set,
        union_path=(
            output_root / "union_schedule.json" if campaign == "claim" else None
        ),
    )
    runtime = plain(snapshot.payload["runtime_config"])
    environment = plain(snapshot.payload["environment_config"])
    target = manifest.simulation.target_env_id
    return PreparedCampaign(
        root,
        manifest,
        snapshot,
        case_set,
        {case["case_id"]: case for case in case_set["cases"]},
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


def _load_v8_s1(
    root: Path,
    manifest: GroundedDecodingManifest,
    snapshot: Any,
) -> FrozenS1:
    path = root / manifest.outputs.root / "s1" / "model_preflight.json"
    content, preflight = _load_canonical_object(path)
    expected_fields = {
        "artifact_type",
        "runtime_snapshot_sha256",
        "comparator_digest_match",
        "records",
    }
    if (
        set(preflight) != expected_fields
        or preflight["artifact_type"] != GROUNDED_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE
        or preflight["runtime_snapshot_sha256"] != "sha256:" + snapshot.sha256
    ):
        raise ValueError("Frozen V8 S1 preflight drifted from runtime snapshot.")
    records = preflight["records"]
    expected_slots = [model.slot for model in manifest.models for _ in range(4)]
    if (
        not isinstance(records, list)
        or [
            record.get("model_slot") if isinstance(record, Mapping) else None
            for record in records
        ]
        != expected_slots
    ):
        raise ValueError("Frozen V8 S1 record order drifted.")
    trusted_three = [
        record
        for model_index in range(len(manifest.models))
        for record in records[model_index * 4 : model_index * 4 + 3]
    ]
    bindings = _load_bindings(
        manifest,
        trusted_three,
        canonical_schema_bytes=canonical_bytes(canonical_action_text_schema()),
    )
    artifact_hash = "sha256:" + hashlib.sha256(content).hexdigest()
    capabilities = build_v8_capabilities(manifest, bindings, artifact_hash)
    return FrozenS1(
        MappingProxyType(dict(bindings)), MappingProxyType(dict(capabilities))
    )


def _repo_root(manifest_path: Path) -> Path:
    return manifest_path.resolve().parents[2]


__all__ = [
    "campaign_status",
    "run_claim_stage",
    "run_probe_lock",
    "run_smoke",
]
