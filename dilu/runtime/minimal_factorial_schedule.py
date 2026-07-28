"""Frozen manifest and deterministic schedules for the ICLR 2027 factorial."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from ._harness_config_support import (ConditionSpec, ExecutionMode, FallbackPolicy,
                                      OutputEnforcement, ParserMode, PolicyContent,
                                      ResolverMode, ThinkMode, TraceLevel, TransportProfile)
from ._scientific_runtime_binding import ScientificEpisodeIdentity
from .config_loader import load_runtime_config
from .dilu_scoring import BALANCED_DRIVING_SCORE_POLICY_VERSION
from .harness_config import (HarnessConfig, RetryPolicy, ShieldConfig,
                             TransportConfig, resolve_main_conditions)
from .highway_env_config import resolve_simulation_env_bundle
from .task_benchmark import build_benchmark_case_set_fingerprint

_SHORT_CASE_FP = "dilu_highway_reactive_stress_v2:ed2f63e396fc1b87"
_CASE_FP = "sha256:bd6d65d694a1452e0770e9854e478bb463be8302168e8c17396e86786401fd33"
_SOURCE_SHA = {"configs/iclr2027/minimal_factorial_runtime.yaml": "be5934115a63c3858b65504f9a0d0c05c071bf7b9cbd46daf9009dae77f41f77", "config.example.yaml": "0b3efd696063d8b6cc8b99df9c89653d74cebd0c7c4da8d69fd03ac9f0d3a450", "configs/iclr2027/protocol_constants.yaml": "6ff8e540496501d5e463569fdc8ac195e7dd94868f02b94812010a01869b6df8"}
_RUNTIME_SHA = "2f7b5369fcd2963472bb81c1437b9fe885362ec6ee027c686b5c6938d75dd248"
_SCORING_SHA = "a49fc424bb3c00d3785d25477d7f5fe047058016b455b8f39e86cdb096796e96"
_PREDICATE_SHA = "9cada55048a6f6bc40ea52de0a4719ae189fe979aa244f7eb5a01b2f657739ed"
_TRACE_SHA = "749c02930dc4ea97cd61b166c1b9df1ce8c3376e848d29850da2db10229b0f6a"
_KEYS = {"schema_version", "campaign_id", "smoke_campaign_id", "case_set", "models", "transport", "runtime_sources", "fixed_harness", "simulation", "scoring", "selection", "bootstrap", "outputs"}

@dataclass(frozen=True)
class ModelSpec: slot: str; tag: str
@dataclass(frozen=True)
class TransportSpec:
    native_endpoint: str; think_mode: str; temperature: float; context_tokens: int
    max_output_tokens: int; timeout_sec: float; generation_seed_master: int
@dataclass(frozen=True)
class RuntimeSources:
    runtime_config: str; base_runtime_config: str; protocol_constants: str; require_clean_git: bool
@dataclass(frozen=True)
class RetrySpec:
    max_transport_unavailable_retries: int; retry_cooldown_sec: float
    retry_on_timeout: bool; retry_on_empty_output: bool; retry_on_schema_rejection: bool
@dataclass(frozen=True)
class FixedHarnessSpec:
    parser_mode: str; resolver_mode: str; fallback_policy: str; trace_level: str
    retry_policy: RetrySpec; shield_source: str
@dataclass(frozen=True)
class SimulationSpec: target_env_id: str; few_shot_num: int; memory_enabled: bool; reflection_enabled: bool
@dataclass(frozen=True)
class ScoringSpec: behavior_score: str; task_predicates: str; split_score_implementation: str
@dataclass(frozen=True)
class SelectionSpec:
    categories: int; stage1_cases_per_category: int; stage2_cases_per_category: int
    smoke_hash_prefix: str; stage1_hash_prefix: str
@dataclass(frozen=True)
class BootstrapSpec: draws: int; version: str
@dataclass(frozen=True)
class OutputSpec: root: str; s1: str; smoke: str; llm_campaign: str; baselines: str; analysis: str

@dataclass(frozen=True)
class ExperimentManifest:
    schema_version: str; campaign_id: str; smoke_campaign_id: str; case_path: str
    models: tuple[ModelSpec, ...]; transport: TransportSpec; runtime_sources: RuntimeSources
    fixed_harness: FixedHarnessSpec; simulation: SimulationSpec; scoring: ScoringSpec
    selection: SelectionSpec; bootstrap: BootstrapSpec; outputs: OutputSpec; source_path: Path
    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], source_path: Path = Path(".")) -> "ExperimentManifest":
        if not isinstance(value, Mapping) or set(value) != _KEYS: raise ValueError("Minimal factorial manifest keys drifted.")
        for key, expected in (("schema_version", "iclr2027_minimal_factorial_manifest_v1"), ("campaign_id", "iclr2027-minimal-factorial-v1"), ("smoke_campaign_id", "iclr2027-minimal-factorial-smoke-v1"), ("case_set", "benchmarks/dilu_highway_reactive_stress_v2/cases.json")):
            if value.get(key) != expected: raise ValueError(f"{key} drifted.")
        models = tuple(ModelSpec(**item) for item in _items(value["models"], ModelSpec, 2))
        transport = TransportSpec(**_mapping(value["transport"], TransportSpec))
        runtime = RuntimeSources(**_mapping(value["runtime_sources"], RuntimeSources))
        fixed_raw = _mapping(value["fixed_harness"], FixedHarnessSpec, frozenset({"retry_policy"}))
        fixed = FixedHarnessSpec(**fixed_raw, retry_policy=RetrySpec(**_mapping(value["fixed_harness"]["retry_policy"], RetrySpec)))
        simulation = SimulationSpec(**_mapping(value["simulation"], SimulationSpec))
        scoring = ScoringSpec(**_mapping(value["scoring"], ScoringSpec))
        selection = SelectionSpec(**_mapping(value["selection"], SelectionSpec))
        bootstrap = BootstrapSpec(**_mapping(value["bootstrap"], BootstrapSpec))
        outputs = OutputSpec(**_mapping(value["outputs"], OutputSpec))
        expected = ( (ModelSpec("qwen_06b", "qwen3:0.6b"), ModelSpec("llama_1b", "llama3.2:1b")), TransportSpec("http://localhost:11434/api/chat", "no_think", 0.0, 8192, 128, 30.0, 20270728), RuntimeSources("configs/iclr2027/minimal_factorial_runtime.yaml", "config.example.yaml", "configs/iclr2027/protocol_constants.yaml", True), FixedHarnessSpec("strict_only", "disabled", "fixed_idle", "mandatory_scientific", RetrySpec(1, 10.0, False, False, False), "implementation_defaults"), SimulationSpec("highway-fast-v0", 0, False, False), ScoringSpec("driving_score_balanced_v1", "stress_v2_case_success_criteria", "dilu.runtime.dilu_scoring"), SelectionSpec(10, 3, 12, "iclr2027-minimal-factorial-v1|smoke", "iclr2027-minimal-factorial-v1"), BootstrapSpec(20000, "bootstrap-v1"), OutputSpec("results/iclr2027_minimal_factorial", "s1", "smoke", "llm_campaign", "baselines", "analysis") )
        if (models, transport, runtime, fixed, simulation, scoring, selection, bootstrap, outputs) != expected: raise ValueError("Frozen manifest constants drifted.")
        return cls(value["schema_version"], value["campaign_id"], value["smoke_campaign_id"], value["case_set"], models, transport, runtime, fixed, simulation, scoring, selection, bootstrap, outputs, source_path)

@dataclass(frozen=True)
class RuntimeSnapshot:
    payload: Mapping[str, Any]; sha256: str
    def __post_init__(self) -> None:
        if not isinstance(self.payload, MappingProxyType) or _sha(_plain(self.payload)) != self.sha256: raise ValueError("Runtime snapshot is not immutable or hash-consistent.")
    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "RuntimeSnapshot":
        frozen = _freeze(payload); return cls(frozen, _sha(_plain(frozen)))

@dataclass(frozen=True)
class ScheduledEpisode:
    stage: str; campaign_id: str; model_slot: str; model_tag: str; model_digest: str
    condition: HarnessConfig; condition_id: str; case_id: str; simulator_seed: int
    episode_attempt_id: str; pair_id: str; template_id: str; replicate_id: int
    primary_snapshot_id: str; benchmark_fingerprint: str; code_revision: str
    def identity(self) -> ScientificEpisodeIdentity:
        return ScientificEpisodeIdentity(self.campaign_id, self.episode_attempt_id, self.case_id, self.pair_id, self.template_id, self.replicate_id, self.simulator_seed, self.primary_snapshot_id, self.benchmark_fingerprint, self.code_revision)

def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    path = Path(path).resolve(); return ExperimentManifest.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path)

def build_harness_config(manifest: ExperimentManifest, condition: int | ConditionSpec) -> HarnessConfig:
    index = condition if isinstance(condition, int) else None
    if index is not None and index not in range(8): raise ValueError("Condition index must be in [0, 7].")
    base = HarnessConfig(condition if isinstance(condition, ConditionSpec) else ConditionSpec(PolicyContent.HISTORICAL_DILU_2024, OutputEnforcement.PROMPT_ONLY, ExecutionMode.UNSHIELDED_OPERATIONAL), ParserMode(manifest.fixed_harness.parser_mode), ResolverMode(manifest.fixed_harness.resolver_mode), FallbackPolicy(manifest.fixed_harness.fallback_policy), ShieldConfig.implementation_defaults(), TransportConfig(TransportProfile.OLLAMA_NATIVE_CHAT, ThinkMode(manifest.transport.think_mode), manifest.transport.temperature, manifest.transport.context_tokens, manifest.transport.max_output_tokens, manifest.transport.timeout_sec, manifest.transport.generation_seed_master, False, False), RetryPolicy(**asdict(manifest.fixed_harness.retry_policy)), TraceLevel(manifest.fixed_harness.trace_level))
    base.validate_scientific(); return resolve_main_conditions(base)[index] if index is not None else base

def select_smoke_case(case_set: Mapping[str, Any], campaign_id: str) -> Mapping[str, Any]:
    _case_fp(case_set); return min(case_set["cases"], key=lambda case: _digest(f"{campaign_id}|smoke|{case['case_id']}"))
def select_stage1_cases(case_set: Mapping[str, Any], campaign_id: str) -> tuple[Mapping[str, Any], ...]:
    _case_fp(case_set); return tuple(case for category in sorted({case["category"] for case in case_set["cases"]}) for case in sorted((case for case in case_set["cases"] if case["category"] == category), key=lambda case: _digest(f"{campaign_id}|{case['case_id']}"))[:3])
def build_smoke_schedule(manifest: ExperimentManifest, case_set: Mapping[str, Any], model_digests: Mapping[str, str], *, code_revision: str) -> tuple[ScheduledEpisode, ...]:
    return _episodes("smoke", manifest.smoke_campaign_id, manifest, (select_smoke_case(case_set, manifest.campaign_id),), range(8), model_digests, code_revision, _case_fp(case_set))
def build_union_schedule(manifest: ExperimentManifest, case_set: Mapping[str, Any], model_digests: Mapping[str, str], *, code_revision: str) -> tuple[ScheduledEpisode, ...]:
    fingerprint = _case_fp(case_set); stage1 = select_stage1_cases(case_set, manifest.campaign_id); selected = {case["case_id"] for case in stage1}; rest = tuple(case for case in case_set["cases"] if case["case_id"] not in selected)
    return _episodes("s1", manifest.campaign_id, manifest, stage1, range(8), model_digests, code_revision, fingerprint) + _episodes("s2_additional", manifest.campaign_id, manifest, rest, (0, 7), model_digests, code_revision, fingerprint)

def build_runtime_snapshot(manifest: ExperimentManifest, case_set: Mapping[str, Any]) -> RuntimeSnapshot:
    fingerprint = _case_fp(case_set); root = manifest.source_path.parents[2]; _clean_and_tracked(root, manifest.runtime_sources)
    revision = _git(root, ["rev-parse", "HEAD"]).stdout.strip(); source_sha = {path: _file_sha(root / path) for path in _SOURCE_SHA}
    runtime = load_runtime_config(str(root / manifest.runtime_sources.runtime_config)); runtime_sha = _sha(runtime)
    scoring_sha = _file_sha(Path(__file__).with_name("dilu_scoring.py")); predicate_sha = _sha([case["success_criteria"] for case in case_set["cases"]]); trace_sha = _file_sha(Path(__file__).with_name("_scientific_trace_serialization.py"))
    if source_sha != _SOURCE_SHA or (runtime_sha, scoring_sha, predicate_sha, trace_sha) != (_RUNTIME_SHA, _SCORING_SHA, _PREDICATE_SHA, _TRACE_SHA): raise ValueError("Runtime source or fingerprint drifted.")
    environment = resolve_simulation_env_bundle(runtime, show_trajectories=False, render_agent=False, env_id_override=manifest.simulation.target_env_id, native_env_defaults_override=True, require_discrete_meta_action=True)
    return RuntimeSnapshot.create({"code_revision": revision, "source_sha256": source_sha, "runtime_config": runtime, "environment_config": environment["env_config_snapshot"], "primary_metric_spec": {"metric": manifest.scoring.behavior_score, "version": BALANCED_DRIVING_SCORE_POLICY_VERSION}, "shield_config": asdict(ShieldConfig.implementation_defaults()), "scoring_fingerprint": scoring_sha, "predicate_fingerprint": predicate_sha, "simulator_versions": _versions(("gymnasium", "highway-env", "numpy")), "trace_schema_sha256": trace_sha, "case_set_fingerprint": fingerprint})

def write_frozen_campaign_manifest(path: str | Path, manifest: ExperimentManifest, snapshot: RuntimeSnapshot, schedule: Sequence[ScheduledEpisode]) -> None:
    manifest_payload = _plain(asdict(manifest)); manifest_payload.pop("source_path", None)
    payload = {"manifest": manifest_payload, "runtime_snapshot": _plain(snapshot.payload), "runtime_snapshot_sha256": snapshot.sha256, "schedule": [_episode_payload(item) for item in schedule]}
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def _episodes(stage: str, campaign: str, manifest: ExperimentManifest, cases: Sequence[Mapping[str, Any]], indexes: Sequence[int], digests: Mapping[str, str], revision: str, fingerprint: str) -> tuple[ScheduledEpisode, ...]:
    rows = []
    for model in manifest.models:
        digest = digests.get(model.slot, "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest): raise ValueError(f"Missing or invalid digest for {model.slot}.")
        for index in indexes:
            config = build_harness_config(manifest, index)
            for case in cases:
                seed = case["seed"]; case_id = case["case_id"]
                pair = "pair-" + _digest(f"{campaign}|{case_id}|{seed}"); template = "stress-v2-" + _digest(f"{fingerprint}|{case_id}"); snapshot = "snapshot-" + _digest(f"{fingerprint}|{case_id}|{seed}")
                attempt = "episode-" + _digest(f"{campaign}|{model.tag}|{digest}|{config.condition_id()}|{case_id}|{seed}|0")
                rows.append(ScheduledEpisode(stage, campaign, model.slot, model.tag, digest, config, config.condition_id(), case_id, seed, attempt, pair, template, 0, snapshot, fingerprint, revision))
    return tuple(rows)

def _case_fp(case_set: Mapping[str, Any]) -> str:
    if not isinstance(case_set, Mapping) or not isinstance(case_set.get("cases"), list) or len(case_set["cases"]) != 120: raise ValueError("Case-set count drifted.")
    categories = {case.get("category") for case in case_set["cases"]}
    if len(categories) != 10 or any(sum(case.get("category") == category for case in case_set["cases"]) != 12 for category in categories): raise ValueError("Case categories drifted.")
    if build_benchmark_case_set_fingerprint(dict(case_set)) != _SHORT_CASE_FP or "sha256:" + _sha(case_set) != _CASE_FP: raise ValueError("Case fingerprint drifted.")
    return _CASE_FP
def _clean_and_tracked(root: Path, sources: RuntimeSources) -> None:
    if sources.require_clean_git and _git(root, ["status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"]).stdout: raise ValueError("Runtime snapshot requires a clean Git worktree.")
    for source in (sources.runtime_config, sources.base_runtime_config, sources.protocol_constants):
        result = _git(root, ["ls-files", "--error-unmatch", "--", source])
        if result.returncode != 0 or result.stdout.strip() != source: raise ValueError("Runtime source is not tracked.")
def _git(root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]: return subprocess.run(["git", *args], cwd=root, check=False, capture_output=True, text=True)
def _mapping(value: Any, cls: type[Any], excluded: frozenset[str] = frozenset()) -> dict[str, Any]:
    expected = {field.name for field in fields(cls)} - excluded
    if not isinstance(value, Mapping) or set(value) != expected | excluded: raise ValueError(f"{cls.__name__} keys drifted.")
    return {key: item for key, item in value.items() if key not in excluded}
def _items(value: Any, cls: type[Any], size: int) -> tuple[dict[str, Any], ...]:
    keys = {field.name for field in fields(cls)}
    if not isinstance(value, list) or len(value) != size or not all(isinstance(item, Mapping) and set(item) == keys for item in value): raise ValueError("Model list drifted.")
    return tuple(dict(item) for item in value)
def _digest(value: str) -> str: return hashlib.sha256(value.encode()).hexdigest()
def _sha(value: Any) -> str: return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
def _file_sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping): return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)): return tuple(_freeze(item) for item in value)
    return value
def _plain(value: Any) -> Any:
    if isinstance(value, Mapping): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple): return [_plain(item) for item in value]
    if isinstance(value, Path): return str(value)
    return value
def _versions(packages: Sequence[str]) -> dict[str, str]:
    return {package: _version(package) for package in packages}
def _version(package: str) -> str:
    try: return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: return "not-installed"
def _episode_payload(item: ScheduledEpisode) -> dict[str, Any]:
    payload = asdict(item); payload["condition"] = item.condition.to_canonical_dict(); return payload
