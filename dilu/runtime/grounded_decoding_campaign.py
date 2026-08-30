"""Freeze the exact smoke and claim schedules for grounded-decoding V8."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._minimal_factorial_manifest import write_frozen_campaign_manifest
from ._minimal_factorial_schedule_support import publish_once
from ._runtime_lock_authoring_support import canonical_bytes
from .grounded_decoding_schedule import (
    GroundedDecodingManifest,
    RuntimeSnapshot,
    build_comparator_contract,
    build_v8_schedule,
    build_v8_smoke_schedule,
    validate_comparator_pairing,
)
from .minimal_factorial_schedule import load_experiment_manifest
from .ollama_transport import OllamaModelIdentity


@dataclass(frozen=True)
class V8CampaignArtifacts:
    smoke_manifest: Path
    claim_manifest: Path
    union_schedule: Path


def freeze_v8_campaign_artifacts(
    *,
    repo_root: Path,
    manifest: GroundedDecodingManifest,
    case_set: Mapping[str, Any],
    snapshot: RuntimeSnapshot,
    model_bindings: Mapping[str, OllamaModelIdentity],
    output_root: Path | None = None,
) -> V8CampaignArtifacts:
    """Publish byte-stable 10-row smoke and 480-row claim manifests."""
    smoke = build_v8_smoke_schedule(
        manifest,
        case_set,
        model_bindings,
        runtime_snapshot=snapshot,
    )
    schedule = build_v8_schedule(
        manifest,
        case_set,
        model_bindings,
        runtime_snapshot=snapshot,
    )
    v5_manifest = load_experiment_manifest(
        repo_root / "configs" / "iclr2027" / "minimal_factorial.yaml"
    )
    v7_manifest = load_experiment_manifest(
        repo_root / "configs" / "iclr2027" / "model_breadth_factorial_v7.yaml"
    )
    contract = build_comparator_contract(
        manifest,
        case_set,
        v5_manifest,
        v7_manifest,
        runtime_snapshot=snapshot,
    )
    validate_comparator_pairing(schedule, contract)

    destination = output_root or repo_root / manifest.outputs.root
    smoke_path = destination / manifest.outputs.smoke / "campaign_manifest.json"
    claim_path = destination / manifest.outputs.llm_campaign / "campaign_manifest.json"
    union_path = destination / manifest.outputs.llm_campaign / "union_schedule.json"
    write_frozen_campaign_manifest(
        smoke_path,
        manifest,
        snapshot,
        smoke,
        case_set=case_set,
    )
    write_frozen_campaign_manifest(
        claim_path,
        manifest,
        snapshot,
        schedule.all_claim_bearing,
        case_set=case_set,
    )
    publish_once(
        union_path,
        canonical_bytes([row.to_payload() for row in schedule.all_claim_bearing]),
    )
    return V8CampaignArtifacts(smoke_path, claim_path, union_path)


__all__ = ["V8CampaignArtifacts", "freeze_v8_campaign_artifacts"]
