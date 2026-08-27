"""Manifest parsing for the ICLR 2027 grounded-decoding V8 campaign.

This mirrors the style of ``_minimal_factorial_manifest.py`` but defines a
dedicated, independently frozen manifest shape: V8 introduces a new
``conditions``/``selection``/``comparators`` structure that the V5/V7
``ExperimentManifest`` schema does not have, so it cannot be parsed through
``ExperimentManifest.from_mapping`` (which is additionally gated to only
accept the three already-registered V5/V7-family manifest hashes). Every
substructure that *is* shared with V5/V7 (transport, runtime sources, fixed
harness, simulation, scoring, bootstrap, outputs, and the plain
``slot``/``tag`` model spec) is imported and reused verbatim rather than
redefined.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ._harness_config_support import ExecutionMode, OutputEnforcement, PolicyContent, parse_enum
from ._minimal_factorial_manifest import ModelSpec, _items
from ._minimal_factorial_schedule_support import (
    BootstrapSpec,
    FixedHarnessSpec,
    FrozenSpec,
    OutputSpec,
    RuntimeSources,
    ScoringSpec,
    SimulationSpec,
    TransportSpec,
    canonical_sha256,
)

# Computed once from the exact committed content of
# configs/iclr2027/grounded_decoding_v8.yaml via canonical_sha256(yaml.safe_load(...)).
# Any edit to that file must be accompanied by recomputing and updating this constant;
# this is the "frozen manifest constants" gate for V8, mirroring
# ExperimentManifest.REGISTERED_MANIFEST_SHAS for V5/V7.
GROUNDED_DECODING_MANIFEST_SHA256 = (
    "34af6925a00609afd3ad4395865028f97e506838dafb367d1f468a98bc811a72"
)


@dataclass(frozen=True, init=False)
class GroundedConditionSpec(FrozenSpec):
    policy_content: str
    output_enforcement: str
    execution_modes: tuple[str, ...]

    def policy(self) -> PolicyContent:
        return parse_enum(PolicyContent, self.policy_content, "conditions.policy_content")

    def output(self) -> OutputEnforcement:
        return parse_enum(
            OutputEnforcement, self.output_enforcement, "conditions.output_enforcement"
        )

    def executions(self) -> tuple[ExecutionMode, ...]:
        modes = tuple(
            parse_enum(ExecutionMode, value, "conditions.execution_modes")
            for value in self.execution_modes
        )
        if len(modes) != len(set(modes)):
            raise ValueError("conditions.execution_modes must be unique.")
        return modes


@dataclass(frozen=True, init=False)
class GroundedSelectionSpec(FrozenSpec):
    stage1_hash_prefix: str
    stage1_cases_per_category: int
    stage2_cases_per_category: int
    stage2_models: tuple[str, ...]
    stage2_execution_mode: str

    def stage2_mode(self) -> ExecutionMode:
        return parse_enum(
            ExecutionMode, self.stage2_execution_mode, "selection.stage2_execution_mode"
        )


@dataclass(frozen=True, init=False)
class ComparatorPaths(FrozenSpec):
    v5_manifest: str
    v5_episodes: str
    v7_manifest: str
    v7_episodes: str


@dataclass(frozen=True)
class GroundedDecodingManifest:
    schema_version: str
    campaign_id: str
    smoke_campaign_id: str
    case_path: str
    models: tuple[ModelSpec, ...]
    transport: TransportSpec
    runtime_sources: RuntimeSources
    fixed_harness: FixedHarnessSpec
    simulation: SimulationSpec
    scoring: ScoringSpec
    conditions: GroundedConditionSpec
    selection: GroundedSelectionSpec
    comparators: ComparatorPaths
    bootstrap: BootstrapSpec
    outputs: OutputSpec
    source_path: Path

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], source_path: Path = Path(".")
    ) -> "GroundedDecodingManifest":
        if (
            not isinstance(value, Mapping)
            or canonical_sha256(value) != GROUNDED_DECODING_MANIFEST_SHA256
        ):
            raise ValueError("Frozen V8 manifest constants drifted.")
        models = tuple(ModelSpec(**item) for item in _items(value["models"]))
        return cls(
            value["schema_version"],
            value["campaign_id"],
            value["smoke_campaign_id"],
            value["case_set"],
            models,
            _spec(value["transport"], TransportSpec),
            _spec(value["runtime_sources"], RuntimeSources),
            _spec(value["fixed_harness"], FixedHarnessSpec),
            _spec(value["simulation"], SimulationSpec),
            _spec(value["scoring"], ScoringSpec),
            _spec(value["conditions"], GroundedConditionSpec),
            _spec(value["selection"], GroundedSelectionSpec),
            _spec(value["comparators"], ComparatorPaths),
            _spec(value["bootstrap"], BootstrapSpec),
            _spec(value["outputs"], OutputSpec),
            source_path,
        )

    def repo_root(self) -> Path:
        # configs/iclr2027/grounded_decoding_v8.yaml -> parents[2] is the repo root,
        # the same convention build_runtime_snapshot() uses for V5/V7 manifests.
        return self.source_path.parents[2]

    def model_slots(self) -> frozenset[str]:
        return frozenset(model.slot for model in self.models)


def load_grounded_decoding_manifest(path: str | Path) -> GroundedDecodingManifest:
    source_path = Path(path).resolve()
    return GroundedDecodingManifest.from_mapping(
        yaml.safe_load(source_path.read_text(encoding="utf-8")), source_path
    )


def _spec(value: Any, spec_type: type[FrozenSpec]) -> FrozenSpec:
    if not isinstance(value, Mapping):
        raise ValueError("Manifest nested value must be a mapping.")
    return spec_type(value)


__all__ = [
    "GROUNDED_DECODING_MANIFEST_SHA256",
    "ComparatorPaths",
    "GroundedConditionSpec",
    "GroundedDecodingManifest",
    "GroundedSelectionSpec",
    "load_grounded_decoding_manifest",
]
