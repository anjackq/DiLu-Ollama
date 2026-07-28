"""Public deterministic schedule API for the ICLR 2027 minimal factorial."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from ._minimal_factorial_manifest import (
    ExperimentManifest,
    ModelSpec,
    RuntimeSnapshot,
    build_harness_config,
    build_runtime_snapshot,
    case_fingerprint,
    load_experiment_manifest,
    write_frozen_campaign_manifest,
)
from ._scientific_runtime_binding import ScientificEpisodeIdentity
from ._scientific_transport_validation import require_model_digest


@dataclass(frozen=True)
class ScheduledEpisode:
    stage: str
    campaign_id: str
    model_slot: str
    model_tag: str
    model_digest: str
    condition: Any
    condition_id: str
    case_id: str
    simulator_seed: int
    episode_attempt_id: str
    pair_id: str
    template_id: str
    replicate_id: int
    primary_snapshot_id: str
    benchmark_fingerprint: str
    code_revision: str

    def identity(self) -> ScientificEpisodeIdentity:
        return ScientificEpisodeIdentity(
            self.campaign_id,
            self.episode_attempt_id,
            self.case_id,
            self.pair_id,
            self.template_id,
            self.replicate_id,
            self.simulator_seed,
            self.primary_snapshot_id,
            self.benchmark_fingerprint,
            self.code_revision,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["condition"] = self.condition.to_canonical_dict()
        return payload


def select_smoke_case(
    case_set: Mapping[str, Any], campaign_id: str
) -> Mapping[str, Any]:
    case_fingerprint(case_set)
    return min(
        case_set["cases"],
        key=lambda case: _digest(f"{campaign_id}|smoke|{case['case_id']}"),
    )


def select_stage1_cases(
    case_set: Mapping[str, Any], campaign_id: str
) -> tuple[Mapping[str, Any], ...]:
    case_fingerprint(case_set)
    selected: list[Mapping[str, Any]] = []
    for category in sorted({case["category"] for case in case_set["cases"]}):
        cases = [case for case in case_set["cases"] if case["category"] == category]
        selected.extend(
            sorted(cases, key=lambda case: _digest(f"{campaign_id}|{case['case_id']}"))[
                :3
            ]
        )
    return tuple(selected)


def build_smoke_schedule(
    manifest: ExperimentManifest,
    case_set: Mapping[str, Any],
    model_digests: Mapping[str, str],
    *,
    runtime_snapshot: RuntimeSnapshot,
) -> tuple[ScheduledEpisode, ...]:
    fingerprint, revision = _binding(runtime_snapshot, case_set)
    return _episodes(
        "smoke",
        manifest.smoke_campaign_id,
        manifest,
        (select_smoke_case(case_set, manifest.campaign_id),),
        range(8),
        model_digests,
        revision,
        fingerprint,
    )


def build_union_schedule(
    manifest: ExperimentManifest,
    case_set: Mapping[str, Any],
    model_digests: Mapping[str, str],
    *,
    runtime_snapshot: RuntimeSnapshot,
) -> tuple[ScheduledEpisode, ...]:
    fingerprint, revision = _binding(runtime_snapshot, case_set)
    stage1 = select_stage1_cases(case_set, manifest.campaign_id)
    selected = {case["case_id"] for case in stage1}
    remaining = tuple(
        case for case in case_set["cases"] if case["case_id"] not in selected
    )
    return _episodes(
        "s1",
        manifest.campaign_id,
        manifest,
        stage1,
        range(8),
        model_digests,
        revision,
        fingerprint,
    ) + _episodes(
        "s2_additional",
        manifest.campaign_id,
        manifest,
        remaining,
        (0, 7),
        model_digests,
        revision,
        fingerprint,
    )


def _binding(snapshot: RuntimeSnapshot, case_set: Mapping[str, Any]) -> tuple[str, str]:
    fingerprint = case_fingerprint(case_set)
    if not isinstance(snapshot, RuntimeSnapshot):
        raise TypeError("runtime_snapshot must be a RuntimeSnapshot.")
    if snapshot.payload.get("case_set_fingerprint") != fingerprint:
        raise ValueError("Runtime snapshot case fingerprint drifted.")
    revision = snapshot.payload.get("code_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("Runtime snapshot revision is invalid.")
    return fingerprint, revision


def _episodes(
    stage: str,
    campaign: str,
    manifest: ExperimentManifest,
    cases: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
    digests: Mapping[str, str],
    revision: str,
    fingerprint: str,
) -> tuple[ScheduledEpisode, ...]:
    rows: list[ScheduledEpisode] = []
    for model in manifest.models:
        digest = digests.get(model.slot, "")
        require_model_digest(f"model_digests.{model.slot}", digest)
        for index in indexes:
            condition = build_harness_config(manifest, index)
            for case in cases:
                rows.append(
                    _episode(
                        stage,
                        campaign,
                        model,
                        digest,
                        condition,
                        case,
                        revision,
                        fingerprint,
                    )
                )
    return tuple(rows)


def _episode(
    stage: str,
    campaign: str,
    model: ModelSpec,
    digest: str,
    condition: Any,
    case: Mapping[str, Any],
    revision: str,
    fingerprint: str,
) -> ScheduledEpisode:
    case_id = case["case_id"]
    seed = case["seed"]
    pair = "pair-" + _digest(f"{campaign}|{case_id}|{seed}")
    template = "stress-v2-" + _digest(f"{fingerprint}|{case_id}")
    primary = "snapshot-" + _digest(f"{fingerprint}|{case_id}|{seed}")
    attempt = "episode-" + _digest(
        f"{campaign}|{model.tag}|{digest}|{condition.condition_id()}|{case_id}|{seed}|0"
    )
    return ScheduledEpisode(
        stage,
        campaign,
        model.slot,
        model.tag,
        digest,
        condition,
        condition.condition_id(),
        case_id,
        seed,
        attempt,
        pair,
        template,
        0,
        primary,
        fingerprint,
        revision,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


__all__ = [
    "ExperimentManifest",
    "ModelSpec",
    "RuntimeSnapshot",
    "ScheduledEpisode",
    "build_harness_config",
    "build_runtime_snapshot",
    "build_smoke_schedule",
    "build_union_schedule",
    "load_experiment_manifest",
    "select_smoke_case",
    "select_stage1_cases",
    "write_frozen_campaign_manifest",
]
