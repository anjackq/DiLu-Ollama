"""Frozen manifest and deterministic schedules for the ICLR 2027 factorial."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import yaml
from ._harness_config_support import (
    ConditionSpec, ExecutionMode, FallbackPolicy, OutputEnforcement, ParserMode,
    PolicyContent, ResolverMode, ThinkMode, TraceLevel, TransportProfile,
)
from ._scientific_runtime_binding import ScientificEpisodeIdentity
from ._scientific_transport_validation import require_model_digest
from .config_loader import load_runtime_config
from .dilu_scoring import BALANCED_DRIVING_SCORE_POLICY_VERSION
from .harness_config import (
    HarnessConfig, RetryPolicy, ShieldConfig, TransportConfig,
    resolve_main_conditions,
)
from .highway_env_config import resolve_simulation_env_bundle
from .task_benchmark import build_benchmark_case_set_fingerprint
_SHORT_CASE_FP = "dilu_highway_reactive_stress_v2:ed2f63e396fc1b87"
_CASE_FP = "sha256:bd6d65d694a1452e0770e9854e478bb463be8302168e8c17396e86786401fd33"
_SOURCE_SHA = {
    "configs/iclr2027/minimal_factorial_runtime.yaml": "be5934115a63c3858b65504f9a0d0c05c071bf7b9cbd46daf9009dae77f41f77",
    "config.example.yaml": "0b3efd696063d8b6cc8b99df9c89653d74cebd0c7c4da8d69fd03ac9f0d3a450",
    "configs/iclr2027/protocol_constants.yaml": "6ff8e540496501d5e463569fdc8ac195e7dd94868f02b94812010a01869b6df8",
}
_RUNTIME_SHA = "2f7b5369fcd2963472bb81c1437b9fe885362ec6ee027c686b5c6938d75dd248"
_SCORING_SHA = "a49fc424bb3c00d3785d25477d7f5fe047058016b455b8f39e86cdb096796e96"
_PREDICATE_SHA = "9cada55048a6f6bc40ea52de0a4719ae189fe979aa244f7eb5a01b2f657739ed"
_TRACE_SHA = "749c02930dc4ea97cd61b166c1b9df1ce8c3376e848d29850da2db10229b0f6a"
_MANIFEST_SHA = "39cc138fca22472a7f4c05586df65d0983e0ab0ff2d2af67c9b939ecf8f4782e"
_REVISION_RE = re.compile(r"\A[0-9a-fA-F]{40}\Z")
@dataclass(frozen=True)
class ModelSpec:
    slot: str
    tag: str
@dataclass(frozen=True)
class FrozenSpec:
    values: Mapping[str, Any]
    def __getattr__(self, name: str) -> Any:
        value = self.values[name]
        return FrozenSpec(value) if isinstance(value, Mapping) else value
    def __getitem__(self, name: str) -> Any:
        return self.values[name]
    def to_dict(self) -> dict[str, Any]:
        return _plain(self.values)
@dataclass(frozen=True)
class ExperimentManifest:
    schema_version: str
    campaign_id: str
    smoke_campaign_id: str
    case_path: str
    models: tuple[ModelSpec, ...]
    transport: FrozenSpec
    runtime_sources: FrozenSpec
    fixed_harness: FrozenSpec
    simulation: FrozenSpec
    scoring: FrozenSpec
    selection: FrozenSpec
    bootstrap: FrozenSpec
    outputs: FrozenSpec
    source_path: Path
    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], source_path: Path = Path(".")
    ) -> "ExperimentManifest":
        if not isinstance(value, Mapping) or _sha(value) != _MANIFEST_SHA:
            raise ValueError("Frozen manifest constants drifted.")
        models = tuple(ModelSpec(**item) for item in _items(value["models"], ModelSpec, 2))
        transport = _spec(value["transport"])
        runtime = _spec(value["runtime_sources"])
        fixed = _spec(value["fixed_harness"])
        simulation = _spec(value["simulation"])
        scoring = _spec(value["scoring"])
        selection = _spec(value["selection"])
        bootstrap = _spec(value["bootstrap"])
        outputs = _spec(value["outputs"])
        return cls(
            value["schema_version"], value["campaign_id"],
            value["smoke_campaign_id"], value["case_set"], models, transport,
            runtime, fixed, simulation, scoring, selection, bootstrap, outputs,
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
        frozen = _freeze(payload)
        snapshot = object.__new__(cls)
        object.__setattr__(snapshot, "payload", frozen)
        object.__setattr__(snapshot, "sha256", _sha(frozen))
        return snapshot
@dataclass(frozen=True)
class ScheduledEpisode:
    stage: str
    campaign_id: str
    model_slot: str
    model_tag: str
    model_digest: str
    condition: HarnessConfig
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
            self.campaign_id, self.episode_attempt_id, self.case_id, self.pair_id,
            self.template_id, self.replicate_id, self.simulator_seed,
            self.primary_snapshot_id, self.benchmark_fingerprint,
            self.code_revision,
        )
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
    spec = condition if isinstance(condition, ConditionSpec) else ConditionSpec(
        PolicyContent.HISTORICAL_DILU_2024,
        OutputEnforcement.PROMPT_ONLY,
        ExecutionMode.UNSHIELDED_OPERATIONAL,
    )
    config = HarnessConfig(
        spec, ParserMode(manifest.fixed_harness.parser_mode),
        ResolverMode(manifest.fixed_harness.resolver_mode),
        FallbackPolicy(manifest.fixed_harness.fallback_policy),
        ShieldConfig.implementation_defaults(),
        TransportConfig(
            TransportProfile.OLLAMA_NATIVE_CHAT, ThinkMode(manifest.transport.think_mode),
            manifest.transport.temperature, manifest.transport.context_tokens,
            manifest.transport.max_output_tokens, manifest.transport.timeout_sec,
            manifest.transport.generation_seed_master, False, False,
        ),
        RetryPolicy(**manifest.fixed_harness.retry_policy.to_dict()),
        TraceLevel(manifest.fixed_harness.trace_level),
    )
    config.validate_scientific()
    return resolve_main_conditions(config)[index] if index is not None else config
def select_smoke_case(
    case_set: Mapping[str, Any], campaign_id: str
) -> Mapping[str, Any]:
    _case_fingerprint(case_set)
    return min(case_set["cases"], key=lambda case: _digest(f"{campaign_id}|smoke|{case['case_id']}"))
def select_stage1_cases(
    case_set: Mapping[str, Any], campaign_id: str
) -> tuple[Mapping[str, Any], ...]:
    _case_fingerprint(case_set)
    selected: list[Mapping[str, Any]] = []
    for category in sorted({case["category"] for case in case_set["cases"]}):
        choices = [case for case in case_set["cases"] if case["category"] == category]
        selected.extend(sorted(choices, key=lambda case: _digest(f"{campaign_id}|{case['case_id']}"))[:3])
    return tuple(selected)
def build_smoke_schedule(
    manifest: ExperimentManifest, case_set: Mapping[str, Any],
    model_digests: Mapping[str, str], *, runtime_snapshot: RuntimeSnapshot,
) -> tuple[ScheduledEpisode, ...]:
    fingerprint, revision = _snapshot_binding(runtime_snapshot, case_set)
    return _episodes(
        "smoke", manifest.smoke_campaign_id, manifest,
        (select_smoke_case(case_set, manifest.campaign_id),), range(8),
        model_digests, revision, fingerprint,
    )
def build_union_schedule(
    manifest: ExperimentManifest, case_set: Mapping[str, Any],
    model_digests: Mapping[str, str], *, runtime_snapshot: RuntimeSnapshot,
) -> tuple[ScheduledEpisode, ...]:
    fingerprint, revision = _snapshot_binding(runtime_snapshot, case_set)
    stage1 = select_stage1_cases(case_set, manifest.campaign_id)
    selected = {case["case_id"] for case in stage1}
    remaining = tuple(case for case in case_set["cases"] if case["case_id"] not in selected)
    return _episodes("s1", manifest.campaign_id, manifest, stage1, range(8), model_digests, revision, fingerprint) + _episodes("s2_additional", manifest.campaign_id, manifest, remaining, (0, 7), model_digests, revision, fingerprint)
def build_runtime_snapshot(
    manifest: ExperimentManifest, case_set: Mapping[str, Any]
) -> RuntimeSnapshot:
    fingerprint = _case_fingerprint(case_set)
    root = manifest.source_path.parents[2]
    _clean_and_tracked(root, manifest.runtime_sources)
    revision = _revision(root)
    source_sha = {path: _file_sha(root / path) for path in _SOURCE_SHA}
    runtime = load_runtime_config(str(root / manifest.runtime_sources.runtime_config))
    runtime_sha = _sha(runtime)
    scoring_sha = _file_sha(Path(__file__).with_name("dilu_scoring.py"))
    predicate_sha = _sha([case["success_criteria"] for case in case_set["cases"]])
    trace_sha = _file_sha(Path(__file__).with_name("_scientific_trace_serialization.py"))
    observed = (source_sha, runtime_sha, scoring_sha, predicate_sha, trace_sha)
    expected = (_SOURCE_SHA, _RUNTIME_SHA, _SCORING_SHA, _PREDICATE_SHA, _TRACE_SHA)
    if observed != expected:
        raise ValueError(f"Frozen provenance drifted: expected={expected}, observed={observed}.")
    environment = resolve_simulation_env_bundle(
        runtime, show_trajectories=False, render_agent=False,
        env_id_override=manifest.simulation.target_env_id,
        native_env_defaults_override=True, require_discrete_meta_action=True,
    )
    return RuntimeSnapshot.create({
        "code_revision": revision,
        "source_sha256": source_sha,
        "runtime_config": runtime,
        "environment_config": environment["env_config_snapshot"],
        "primary_metric_spec": {
            "metric": manifest.scoring.behavior_score,
            "version": BALANCED_DRIVING_SCORE_POLICY_VERSION,
        },
        "shield_config": asdict(ShieldConfig.implementation_defaults()),
        "scoring_fingerprint": scoring_sha,
        "predicate_fingerprint": predicate_sha,
        "simulator_versions": _versions(("gymnasium", "highway-env", "numpy")),
        "trace_schema_sha256": trace_sha,
        "case_set_fingerprint": fingerprint,
    })
def write_frozen_campaign_manifest(
    path: str | Path, manifest: ExperimentManifest, snapshot: RuntimeSnapshot,
    schedule: Sequence[ScheduledEpisode],
) -> None:
    _validate_schedule(manifest, snapshot, schedule)
    manifest_payload = _manifest_payload(manifest)
    payload = {
        "manifest": manifest_payload,
        "runtime_snapshot": _plain(snapshot.payload),
        "runtime_snapshot_sha256": snapshot.sha256,
        "schedule": [_episode_payload(episode) for episode in schedule],
    }
    _write_once(Path(path), json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n")
def _episodes(
    stage: str, campaign: str, manifest: ExperimentManifest,
    cases: Sequence[Mapping[str, Any]], indexes: Sequence[int],
    model_digests: Mapping[str, str], revision: str, fingerprint: str,
) -> tuple[ScheduledEpisode, ...]:
    rows: list[ScheduledEpisode] = []
    for model in manifest.models:
        digest = model_digests.get(model.slot, "")
        require_model_digest(f"model_digests.{model.slot}", digest)
        for index in indexes:
            config = build_harness_config(manifest, index)
            for case in cases:
                rows.append(_episode(stage, campaign, model, digest, config, case, revision, fingerprint))
    return tuple(rows)
def _episode(
    stage: str, campaign: str, model: ModelSpec, digest: str, config: HarnessConfig,
    case: Mapping[str, Any], revision: str, fingerprint: str,
) -> ScheduledEpisode:
    case_id = case["case_id"]
    seed = case["seed"]
    pair = "pair-" + _digest(f"{campaign}|{case_id}|{seed}")
    template = "stress-v2-" + _digest(f"{fingerprint}|{case_id}")
    primary = "snapshot-" + _digest(f"{fingerprint}|{case_id}|{seed}")
    attempt = "episode-" + _digest(f"{campaign}|{model.tag}|{digest}|{config.condition_id()}|{case_id}|{seed}|0")
    return ScheduledEpisode(stage, campaign, model.slot, model.tag, digest, config, config.condition_id(), case_id, seed, attempt, pair, template, 0, primary, fingerprint, revision)
def _snapshot_binding(snapshot: RuntimeSnapshot, case_set: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(snapshot, RuntimeSnapshot):
        raise TypeError("runtime_snapshot must be a RuntimeSnapshot.")
    if _sha(snapshot.payload) != snapshot.sha256:
        raise ValueError("Runtime snapshot hash drifted.")
    fingerprint = _case_fingerprint(case_set)
    if snapshot.payload.get("case_set_fingerprint") != fingerprint:
        raise ValueError("Runtime snapshot case fingerprint drifted.")
    revision = snapshot.payload.get("code_revision")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError("Runtime snapshot revision is invalid.")
    return fingerprint, revision
def _validate_schedule(
    manifest: ExperimentManifest, snapshot: RuntimeSnapshot,
    schedule: Sequence[ScheduledEpisode],
) -> None:
    fingerprint = snapshot.payload.get("case_set_fingerprint")
    revision = snapshot.payload.get("code_revision")
    if not isinstance(fingerprint, str) or fingerprint != _CASE_FP:
        raise ValueError("Frozen manifest has invalid case fingerprint.")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError("Frozen manifest has invalid code revision.")
    for episode in schedule:
        allowed = manifest.smoke_campaign_id if episode.stage == "smoke" else manifest.campaign_id
        if episode.campaign_id != allowed or episode.code_revision != revision or episode.benchmark_fingerprint != fingerprint:
            raise ValueError("Scheduled episode does not match frozen snapshot.")
        require_model_digest("scheduled model_digest", episode.model_digest)
        episode.identity()
def _case_fingerprint(case_set: Mapping[str, Any]) -> str:
    cases = case_set.get("cases") if isinstance(case_set, Mapping) else None
    if not isinstance(cases, list) or len(cases) != 120:
        raise ValueError("Case-set count drifted.")
    categories = {case.get("category") for case in cases}
    if len(categories) != 10 or any(sum(case.get("category") == item for case in cases) != 12 for item in categories):
        raise ValueError("Case categories drifted.")
    if build_benchmark_case_set_fingerprint(dict(case_set)) != _SHORT_CASE_FP:
        raise ValueError("Short case fingerprint drifted.")
    if "sha256:" + _sha(case_set) != _CASE_FP:
        raise ValueError("Full case fingerprint drifted.")
    return _CASE_FP
def _clean_and_tracked(root: Path, sources: FrozenSpec) -> None:
    status = _git(root, ["status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"])
    if sources.require_clean_git and (status.returncode != 0 or status.stdout):
        raise ValueError("Runtime snapshot requires a clean Git worktree.")
    for source in (sources.runtime_config, sources.base_runtime_config, sources.protocol_constants):
        tracked = _git(root, ["ls-files", "--error-unmatch", "--", source])
        if tracked.returncode != 0 or tracked.stdout.strip() != source:
            raise ValueError(f"Runtime source is not tracked: {source}.")
def _revision(root: Path) -> str:
    result = _git(root, ["rev-parse", "HEAD"])
    revision = result.stdout.strip()
    if result.returncode != 0 or not _REVISION_RE.fullmatch(revision):
        raise ValueError("Git revision is not an exact commit SHA.")
    return revision
def _write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() == content:
            return
        raise ValueError(f"Frozen campaign manifest already exists: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
def _spec(value: Any) -> FrozenSpec:
    if not isinstance(value, Mapping):
        raise ValueError("Manifest nested value must be a mapping.")
    return FrozenSpec(_freeze(value))
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
def _items(value: Any, cls: type[Any], size: int) -> tuple[dict[str, Any], ...]:
    keys = {field.name for field in fields(cls)}
    if not isinstance(value, list) or len(value) != size or not all(isinstance(item, Mapping) and set(item) == keys for item in value):
        raise ValueError("Model list drifted.")
    return tuple(dict(item) for item in value)
def _git(root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, check=False, capture_output=True, text=True)
def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
def _versions(packages: Sequence[str]) -> dict[str, str]:
    return {package: _version(package) for package in packages}
def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
def _episode_payload(episode: ScheduledEpisode) -> dict[str, Any]:
    payload = asdict(episode)
    payload["condition"] = episode.condition.to_canonical_dict()
    return payload
