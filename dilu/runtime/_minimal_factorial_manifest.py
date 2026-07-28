"""Manifest and runtime snapshot support for the minimal factorial campaign."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ._harness_config_support import (
    ConditionSpec,
    ExecutionMode,
    FallbackPolicy,
    OutputEnforcement,
    ParserMode,
    PolicyContent,
    ResolverMode,
    ThinkMode,
    TraceLevel,
    TransportProfile,
)
from ._minimal_factorial_provenance import validate_schedule_rows
from ._minimal_factorial_schedule_support import (
    BootstrapSpec,
    FixedHarnessSpec,
    FrozenSpec,
    OutputSpec,
    RuntimeSources,
    ScoringSpec,
    SelectionSpec,
    SimulationSpec,
    TransportSpec,
    canonical_sha256,
    freeze,
    plain,
    publish_once,
)
from .config_loader import load_runtime_config
from .dilu_scoring import BALANCED_DRIVING_SCORE_POLICY_VERSION
from .harness_config import (
    HarnessConfig,
    RetryPolicy,
    ShieldConfig,
    TransportConfig,
    resolve_main_conditions,
)
from .highway_env_config import resolve_simulation_env_bundle
from .task_benchmark import build_benchmark_case_set_fingerprint

SHORT_CASE_FINGERPRINT = "dilu_highway_reactive_stress_v2:ed2f63e396fc1b87"
CASE_FINGERPRINT = (
    "sha256:bd6d65d694a1452e0770e9854e478bb463be8302168e8c17396e86786401fd33"
)
MANIFEST_SHA = "39cc138fca22472a7f4c05586df65d0983e0ab0ff2d2af67c9b939ecf8f4782e"
REVISION_RE = re.compile(r"\A[0-9a-fA-F]{40}\Z")
SOURCE_SHA = {
    "configs/iclr2027/minimal_factorial_runtime.yaml": (
        "be5934115a63c3858b65504f9a0d0c05c071bf7b9cbd46daf9009dae77f41f77"
    ),
    "config.example.yaml": (
        "0b3efd696063d8b6cc8b99df9c89653d74cebd0c7c4da8d69fd03ac9f0d3a450"
    ),
    "configs/iclr2027/protocol_constants.yaml": (
        "6ff8e540496501d5e463569fdc8ac195e7dd94868f02b94812010a01869b6df8"
    ),
}
FINGERPRINTS = (
    "2f7b5369fcd2963472bb81c1437b9fe885362ec6ee027c686b5c6938d75dd248",
    "a49fc424bb3c00d3785d25477d7f5fe047058016b455b8f39e86cdb096796e96",
    "9cada55048a6f6bc40ea52de0a4719ae189fe979aa244f7eb5a01b2f657739ed",
    "749c02930dc4ea97cd61b166c1b9df1ce8c3376e848d29850da2db10229b0f6a",
)


@dataclass(frozen=True)
class ModelSpec:
    slot: str
    tag: str


@dataclass(frozen=True)
class ExperimentManifest:
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
    selection: SelectionSpec
    bootstrap: BootstrapSpec
    outputs: OutputSpec
    source_path: Path

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], source_path: Path = Path(".")
    ) -> "ExperimentManifest":
        if not isinstance(value, Mapping) or canonical_sha256(value) != MANIFEST_SHA:
            raise ValueError("Frozen manifest constants drifted.")
        models = tuple(ModelSpec(**item) for item in _items(value["models"], 2))
        specs = (
            _spec(value["transport"], TransportSpec),
            _spec(value["runtime_sources"], RuntimeSources),
            _spec(value["fixed_harness"], FixedHarnessSpec),
            _spec(value["simulation"], SimulationSpec),
            _spec(value["scoring"], ScoringSpec),
            _spec(value["selection"], SelectionSpec),
            _spec(value["bootstrap"], BootstrapSpec),
            _spec(value["outputs"], OutputSpec),
        )
        return cls(
            value["schema_version"],
            value["campaign_id"],
            value["smoke_campaign_id"],
            value["case_set"],
            models,
            *specs,
            source_path,
        )


@dataclass(frozen=True, init=False)
class RuntimeSnapshot:
    payload: Mapping[str, Any]
    sha256: str

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Use RuntimeSnapshot.create().")

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "RuntimeSnapshot":
        snapshot = object.__new__(cls)
        frozen = freeze(payload)
        object.__setattr__(snapshot, "payload", frozen)
        object.__setattr__(snapshot, "sha256", canonical_sha256(frozen))
        return snapshot


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    source_path = Path(path).resolve()
    return ExperimentManifest.from_mapping(
        yaml.safe_load(source_path.read_text(encoding="utf-8")), source_path
    )


def build_harness_config(
    manifest: ExperimentManifest, condition: int | ConditionSpec
) -> HarnessConfig:
    index = condition if isinstance(condition, int) else None
    if index is not None and index not in range(8):
        raise ValueError("Condition index must be in [0, 7].")
    spec = (
        condition
        if isinstance(condition, ConditionSpec)
        else ConditionSpec(
            PolicyContent.HISTORICAL_DILU_2024,
            OutputEnforcement.PROMPT_ONLY,
            ExecutionMode.UNSHIELDED_OPERATIONAL,
        )
    )
    config = HarnessConfig(
        spec,
        ParserMode(manifest.fixed_harness.parser_mode),
        ResolverMode(manifest.fixed_harness.resolver_mode),
        FallbackPolicy(manifest.fixed_harness.fallback_policy),
        ShieldConfig.implementation_defaults(),
        TransportConfig(
            TransportProfile.OLLAMA_NATIVE_CHAT,
            ThinkMode(manifest.transport.think_mode),
            manifest.transport.temperature,
            manifest.transport.context_tokens,
            manifest.transport.max_output_tokens,
            manifest.transport.timeout_sec,
            manifest.transport.generation_seed_master,
            False,
            False,
        ),
        RetryPolicy(**manifest.fixed_harness.retry_policy.to_dict()),
        TraceLevel(manifest.fixed_harness.trace_level),
    )
    config.validate_scientific()
    return resolve_main_conditions(config)[index] if index is not None else config


def case_fingerprint(case_set: Mapping[str, Any]) -> str:
    cases = case_set.get("cases") if isinstance(case_set, Mapping) else None
    if not isinstance(cases, list) or len(cases) != 120:
        raise ValueError("Case-set count drifted.")
    categories = {case.get("category") for case in cases}
    if len(categories) != 10 or any(
        sum(case.get("category") == category for case in cases) != 12
        for category in categories
    ):
        raise ValueError("Case categories drifted.")
    if build_benchmark_case_set_fingerprint(dict(case_set)) != SHORT_CASE_FINGERPRINT:
        raise ValueError("Short case fingerprint drifted.")
    if "sha256:" + canonical_sha256(case_set) != CASE_FINGERPRINT:
        raise ValueError("Full case fingerprint drifted.")
    return CASE_FINGERPRINT


def build_runtime_snapshot(
    manifest: ExperimentManifest, case_set: Mapping[str, Any]
) -> RuntimeSnapshot:
    fingerprint = case_fingerprint(case_set)
    root = manifest.source_path.parents[2]
    _clean_and_tracked(root, manifest.runtime_sources)
    revision = _revision(root)
    source_sha = {path: _file_sha(root / path) for path in SOURCE_SHA}
    runtime = load_runtime_config(str(root / manifest.runtime_sources.runtime_config))
    observed = (
        source_sha,
        canonical_sha256(runtime),
        _file_sha(Path(__file__).with_name("dilu_scoring.py")),
        canonical_sha256([case["success_criteria"] for case in case_set["cases"]]),
        _file_sha(Path(__file__).with_name("_scientific_trace_serialization.py")),
    )
    if observed != (SOURCE_SHA, *FINGERPRINTS):
        raise ValueError(f"Frozen provenance drifted: observed={observed}.")
    environment = resolve_simulation_env_bundle(
        runtime,
        show_trajectories=False,
        render_agent=False,
        env_id_override=manifest.simulation.target_env_id,
        native_env_defaults_override=True,
        require_discrete_meta_action=True,
    )
    return RuntimeSnapshot.create(
        {
            "code_revision": revision,
            "source_sha256": source_sha,
            "runtime_config": runtime,
            "environment_config": environment["env_config_snapshot"],
            "primary_metric_spec": {
                "metric": manifest.scoring.behavior_score,
                "version": BALANCED_DRIVING_SCORE_POLICY_VERSION,
            },
            "shield_config": asdict(ShieldConfig.implementation_defaults()),
            "scoring_fingerprint": observed[2],
            "predicate_fingerprint": observed[3],
            "simulator_versions": _versions(("gymnasium", "highway-env", "numpy")),
            "trace_schema_sha256": observed[4],
            "case_set_fingerprint": fingerprint,
        }
    )


def validate_schedule(
    manifest: ExperimentManifest, snapshot: RuntimeSnapshot, schedule: Sequence[Any]
) -> None:
    if canonical_sha256(snapshot.payload) != snapshot.sha256:
        raise ValueError("Runtime snapshot hash drifted.")
    validate_schedule_rows(manifest, snapshot, schedule)


def serialize_frozen_campaign(
    manifest: ExperimentManifest, snapshot: RuntimeSnapshot, schedule: Sequence[Any]
) -> bytes:
    validate_schedule(manifest, snapshot, schedule)
    payload = {
        "manifest": _manifest_payload(manifest),
        "runtime_snapshot": plain(snapshot.payload),
        "runtime_snapshot_sha256": snapshot.sha256,
        "schedule": [episode.to_payload() for episode in schedule],
    }
    return json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"


def write_frozen_campaign_manifest(
    path: str | Path,
    manifest: ExperimentManifest,
    snapshot: RuntimeSnapshot,
    schedule: Sequence[Any],
) -> None:
    publish_once(Path(path), serialize_frozen_campaign(manifest, snapshot, schedule))


def _clean_and_tracked(root: Path, sources: RuntimeSources) -> None:
    status = _git(
        root, ["status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"]
    )
    if sources.require_clean_git and (status.returncode != 0 or status.stdout):
        raise ValueError("Runtime snapshot requires a clean Git worktree.")
    for source in (
        sources.runtime_config,
        sources.base_runtime_config,
        sources.protocol_constants,
    ):
        tracked = _git(root, ["ls-files", "--error-unmatch", "--", source])
        if tracked.returncode != 0 or tracked.stdout.strip() != source:
            raise ValueError(f"Runtime source is not tracked: {source}.")


def _revision(root: Path) -> str:
    result = _git(root, ["rev-parse", "HEAD"])
    revision = result.stdout.strip()
    if result.returncode != 0 or not REVISION_RE.fullmatch(revision):
        raise ValueError("Git revision is not an exact commit SHA.")
    return revision


def _spec(value: Any, spec_type: type[FrozenSpec]) -> FrozenSpec:
    if not isinstance(value, Mapping):
        raise ValueError("Manifest nested value must be a mapping.")
    return spec_type(value)


def _items(value: Any, size: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("Model list drifted.")
    if not all(
        isinstance(item, Mapping) and set(item) == {"slot", "tag"} for item in value
    ):
        raise ValueError("Model list drifted.")
    return tuple(dict(item) for item in value)


def _manifest_payload(manifest: ExperimentManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "campaign_id": manifest.campaign_id,
        "smoke_campaign_id": manifest.smoke_campaign_id,
        "case_path": manifest.case_path,
        "models": [asdict(model) for model in manifest.models],
        "transport": manifest.transport.to_dict(),
        "runtime_sources": manifest.runtime_sources.to_dict(),
        "fixed_harness": manifest.fixed_harness.to_dict(),
        "simulation": manifest.simulation.to_dict(),
        "scoring": manifest.scoring.to_dict(),
        "selection": manifest.selection.to_dict(),
        "bootstrap": manifest.bootstrap.to_dict(),
        "outputs": manifest.outputs.to_dict(),
    }


def _git(root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _versions(packages: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result
