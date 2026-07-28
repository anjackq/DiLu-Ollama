"""Frozen manifest and deterministic schedules for the ICLR 2027 factorial."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass, fields
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
from ._scientific_runtime_binding import ScientificEpisodeIdentity
from .config_loader import load_runtime_config
from .dilu_scoring import (
    BALANCED_DRIVING_SCORE_POLICY_VERSION,
    SPLIT_SCORING_POLICY_VERSION,
)
from .harness_config import (
    HarnessConfig,
    RetryPolicy,
    ShieldConfig,
    TransportConfig,
    resolve_main_conditions,
)
from .highway_env_config import resolve_simulation_env_bundle
from .task_benchmark import build_benchmark_case_set_fingerprint


_CASE_FINGERPRINT = "dilu_highway_reactive_stress_v2:ed2f63e396fc1b87"
_MANIFEST_KEYS = {
    "schema_version", "campaign_id", "smoke_campaign_id", "case_set", "models",
    "transport", "runtime_sources", "fixed_harness", "simulation", "scoring",
    "selection", "bootstrap", "outputs",
}


@dataclass(frozen=True)
class ModelSpec:
    slot: str
    tag: str


@dataclass(frozen=True)
class TransportSpec:
    native_endpoint: str
    think_mode: str
    temperature: float
    context_tokens: int
    max_output_tokens: int
    timeout_sec: float
    generation_seed_master: int


@dataclass(frozen=True)
class RuntimeSources:
    runtime_config: str
    base_runtime_config: str
    protocol_constants: str
    require_clean_git: bool


@dataclass(frozen=True)
class FixedHarnessSpec:
    parser_mode: str
    resolver_mode: str
    fallback_policy: str
    trace_level: str
    retry_policy: Mapping[str, Any]
    shield_source: str


@dataclass(frozen=True)
class SimulationSpec:
    target_env_id: str
    few_shot_num: int
    memory_enabled: bool
    reflection_enabled: bool


@dataclass(frozen=True)
class ScoringSpec:
    behavior_score: str
    task_predicates: str
    split_score_implementation: str


@dataclass(frozen=True)
class SelectionSpec:
    categories: int
    stage1_cases_per_category: int
    stage2_cases_per_category: int
    smoke_hash_prefix: str
    stage1_hash_prefix: str


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
    output_root: str
    source_path: Path

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], source_path: Path = Path(".")) -> "ExperimentManifest":
        if not isinstance(value, Mapping) or set(value) != _MANIFEST_KEYS:
            raise ValueError("Minimal factorial manifest keys drifted.")
        _require(value, "schema_version", "iclr2027_minimal_factorial_manifest_v1")
        _require(value, "campaign_id", "iclr2027-minimal-factorial-v1")
        _require(value, "smoke_campaign_id", "iclr2027-minimal-factorial-smoke-v1")
        _require(value, "case_set", "benchmarks/dilu_highway_reactive_stress_v2/cases.json")
        models = tuple(ModelSpec(**item) for item in _mapping_sequence(value["models"], 2))
        if models != (ModelSpec("qwen_06b", "qwen3:0.6b"), ModelSpec("llama_1b", "llama3.2:1b")):
            raise ValueError("Model bindings drifted.")
        transport = TransportSpec(**_strict(value["transport"], TransportSpec))
        if transport != TransportSpec("http://localhost:11434/api/chat", "no_think", 0.0, 8192, 128, 30.0, 20270728):
            raise ValueError("Transport constants drifted.")
        runtime = RuntimeSources(**_strict(value["runtime_sources"], RuntimeSources))
        fixed = FixedHarnessSpec(**_strict(value["fixed_harness"], FixedHarnessSpec))
        simulation = SimulationSpec(**_strict(value["simulation"], SimulationSpec))
        scoring = ScoringSpec(**_strict(value["scoring"], ScoringSpec))
        selection = SelectionSpec(**_strict(value["selection"], SelectionSpec))
        if simulation != SimulationSpec("highway-fast-v0", 0, False, False):
            raise ValueError("Simulation constants drifted.")
        if runtime != RuntimeSources(
            "configs/iclr2027/minimal_factorial_runtime.yaml",
            "config.example.yaml",
            "configs/iclr2027/protocol_constants.yaml",
            True,
        ):
            raise ValueError("Runtime source constants drifted.")
        if fixed != FixedHarnessSpec(
            "strict_only", "disabled", "fixed_idle", "mandatory_scientific",
            {
                "max_transport_unavailable_retries": 1,
                "retry_cooldown_sec": 10.0,
                "retry_on_timeout": False,
                "retry_on_empty_output": False,
                "retry_on_schema_rejection": False,
            },
            "implementation_defaults",
        ):
            raise ValueError("Fixed harness constants drifted.")
        if scoring != ScoringSpec("driving_score_balanced_v1", "stress_v2_case_success_criteria", "dilu.runtime.dilu_scoring"):
            raise ValueError("Scoring constants drifted.")
        if selection != SelectionSpec(10, 3, 12, "iclr2027-minimal-factorial-v1|smoke", "iclr2027-minimal-factorial-v1"):
            raise ValueError("Selection constants drifted.")
        if fixed.shield_source != "implementation_defaults":
            raise ValueError("Shield source drifted.")
        if value["bootstrap"] != {"draws": 20000, "version": "bootstrap-v1"}:
            raise ValueError("Bootstrap constants drifted.")
        expected_outputs = {"root": "results/iclr2027_minimal_factorial", "s1": "s1", "smoke": "smoke", "llm_campaign": "llm_campaign", "baselines": "baselines", "analysis": "analysis"}
        if value["outputs"] != expected_outputs:
            raise ValueError("Output constants drifted.")
        return cls(value["schema_version"], value["campaign_id"], value["smoke_campaign_id"], value["case_set"], models, transport, runtime, fixed, simulation, scoring, selection, value["outputs"]["root"], source_path)


@dataclass(frozen=True)
class RuntimeSnapshot:
    payload: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class ScheduledEpisode:
    stage: str
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
            self.code_revision and "iclr2027-minimal-factorial-v1",
            self.episode_attempt_id, self.case_id, self.pair_id, self.template_id,
            self.replicate_id, self.simulator_seed, self.primary_snapshot_id,
            self.benchmark_fingerprint, self.code_revision,
        )


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    source_path = Path(path).resolve()
    data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    return ExperimentManifest.from_mapping(data, source_path)


def build_harness_config(manifest: ExperimentManifest, condition: int | ConditionSpec) -> HarnessConfig:
    index = condition if isinstance(condition, int) else None
    if index is not None and index not in range(8):
        raise ValueError("Condition index must be in [0, 7].")
    spec = condition if isinstance(condition, ConditionSpec) else ConditionSpec(
        PolicyContent.HISTORICAL_DILU_2024,
        OutputEnforcement.PROMPT_ONLY,
        ExecutionMode.UNSHIELDED_OPERATIONAL,
    )
    config = HarnessConfig(spec, ParserMode(manifest.fixed_harness.parser_mode), ResolverMode(manifest.fixed_harness.resolver_mode), FallbackPolicy(manifest.fixed_harness.fallback_policy), ShieldConfig.implementation_defaults(), TransportConfig(TransportProfile.OLLAMA_NATIVE_CHAT, ThinkMode(manifest.transport.think_mode), manifest.transport.temperature, manifest.transport.context_tokens, manifest.transport.max_output_tokens, manifest.transport.timeout_sec, manifest.transport.generation_seed_master, False, False), RetryPolicy(**manifest.fixed_harness.retry_policy), TraceLevel(manifest.fixed_harness.trace_level))
    config.validate_scientific()
    return resolve_main_conditions(config)[index] if index is not None else config


def select_smoke_case(case_set: Mapping[str, Any], campaign_id: str) -> Mapping[str, Any]:
    _validate_case_set(case_set)
    return min(case_set["cases"], key=lambda item: _digest(f"{campaign_id}|smoke|{item['case_id']}"))


def select_stage1_cases(case_set: Mapping[str, Any], campaign_id: str) -> tuple[Mapping[str, Any], ...]:
    _validate_case_set(case_set)
    selected = []
    for category in sorted({item["category"] for item in case_set["cases"]}):
        candidates = [item for item in case_set["cases"] if item["category"] == category]
        selected.extend(sorted(candidates, key=lambda item: _digest(f"{campaign_id}|{item['case_id']}"))[:3])
    return tuple(selected)


def build_smoke_schedule(manifest: ExperimentManifest, case_set: Mapping[str, Any], model_digests: Mapping[str, str], *, code_revision: str) -> tuple[ScheduledEpisode, ...]:
    case = select_smoke_case(case_set, manifest.campaign_id)
    return _episodes("smoke", manifest, (case,), tuple(range(8)), model_digests, code_revision)


def build_union_schedule(manifest: ExperimentManifest, case_set: Mapping[str, Any], model_digests: Mapping[str, str], *, code_revision: str) -> tuple[ScheduledEpisode, ...]:
    _validate_case_set(case_set)
    stage1_cases = select_stage1_cases(case_set, manifest.campaign_id)
    selected_ids = {item["case_id"] for item in stage1_cases}
    remaining = tuple(item for item in case_set["cases"] if item["case_id"] not in selected_ids)
    stage1 = _episodes("s1", manifest, stage1_cases, tuple(range(8)), model_digests, code_revision)
    stage2 = _episodes("s2_additional", manifest, remaining, (0, 7), model_digests, code_revision)
    return stage1 + stage2


def build_runtime_snapshot(manifest: ExperimentManifest, case_set: Mapping[str, Any]) -> RuntimeSnapshot:
    _validate_case_set(case_set)
    root = manifest.source_path.parents[2]
    if manifest.runtime_sources.require_clean_git:
        clean = subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode == 0
        if not clean:
            raise ValueError("Runtime snapshot requires a clean Git worktree.")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    source_paths = (manifest.runtime_sources.runtime_config, manifest.runtime_sources.base_runtime_config, manifest.runtime_sources.protocol_constants)
    source_sha256 = {item: _file_sha256(root / item) for item in source_paths}
    runtime_config = load_runtime_config(str(root / manifest.runtime_sources.runtime_config))
    environment = resolve_simulation_env_bundle(runtime_config, show_trajectories=False, render_agent=False, env_id_override=manifest.simulation.target_env_id, native_env_defaults_override=True, require_discrete_meta_action=True)
    payload = {"code_revision": revision, "source_sha256": source_sha256, "runtime_config": runtime_config, "environment_config": environment["env_config_snapshot"], "primary_metric_spec": {"metric": manifest.scoring.behavior_score, "version": BALANCED_DRIVING_SCORE_POLICY_VERSION}, "shield_config": asdict(ShieldConfig.implementation_defaults()), "scoring_fingerprint": _file_sha256(Path(__file__).with_name("dilu_scoring.py")), "predicate_fingerprint": _sha256([item["success_criteria"] for item in case_set["cases"]]), "simulator_versions": _versions(("gymnasium", "highway-env", "numpy")), "trace_schema_sha256": _file_sha256(Path(__file__).with_name("_scientific_trace_serialization.py")), "case_set_fingerprint": build_benchmark_case_set_fingerprint(dict(case_set))}
    return RuntimeSnapshot(payload, _sha256(payload))


def write_frozen_campaign_manifest(path: str | Path, manifest: ExperimentManifest, snapshot: RuntimeSnapshot, schedule: Sequence[ScheduledEpisode]) -> None:
    payload = {"manifest": {key: value for key, value in asdict(manifest).items() if key != "source_path"}, "runtime_snapshot": dict(snapshot.payload), "runtime_snapshot_sha256": snapshot.sha256, "schedule": [_episode_payload(item) for item in schedule]}
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _episodes(stage: str, manifest: ExperimentManifest, cases: Sequence[Mapping[str, Any]], condition_indexes: Sequence[int], model_digests: Mapping[str, str], code_revision: str) -> tuple[ScheduledEpisode, ...]:
    fingerprint = _validate_case_set_from_cases(cases, manifest, partial=True)
    result = []
    for model in manifest.models:
        digest = model_digests.get(model.slot, "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Missing or invalid digest for {model.slot}.")
        for condition_index in condition_indexes:
            config = build_harness_config(manifest, condition_index)
            for case in cases:
                seed = case["seed"]
                pair_id = "pair-" + _digest(f"{manifest.campaign_id}|{case['case_id']}|{seed}")
                template_id = "stress-v2-" + _digest(f"{fingerprint}|{case['case_id']}")
                snapshot_id = "snapshot-" + _digest(f"{fingerprint}|{case['case_id']}|{seed}")
                attempt_id = "episode-" + _digest(f"{manifest.campaign_id}|{model.tag}|{digest}|{config.condition_id()}|{case['case_id']}|{seed}|0")
                result.append(ScheduledEpisode(stage, model.slot, model.tag, digest, config, config.condition_id(), case["case_id"], seed, attempt_id, pair_id, template_id, 0, snapshot_id, fingerprint, code_revision))
    return tuple(result)


def _validate_case_set(case_set: Mapping[str, Any]) -> str:
    if not isinstance(case_set, Mapping) or not isinstance(case_set.get("cases"), list):
        raise ValueError("Case set is invalid.")
    if len(case_set["cases"]) != 120 or build_benchmark_case_set_fingerprint(dict(case_set)) != _CASE_FINGERPRINT:
        raise ValueError("Case-set count or fingerprint drifted.")
    categories = {item.get("category") for item in case_set["cases"]}
    if len(categories) != 10 or any(sum(item.get("category") == category for item in case_set["cases"]) != 12 for category in categories):
        raise ValueError("Case categories drifted.")
    return _CASE_FINGERPRINT


def _validate_case_set_from_cases(cases: Sequence[Mapping[str, Any]], manifest: ExperimentManifest, *, partial: bool) -> str:
    del manifest, partial
    return _CASE_FINGERPRINT


def _strict(value: Any, data_class: type[Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {field.name for field in fields(data_class)}:
        raise ValueError(f"{data_class.__name__} keys drifted.")
    return dict(value)


def _mapping_sequence(value: Any, size: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or len(value) != size or not all(isinstance(item, Mapping) and set(item) == {"slot", "tag"} for item in value):
        raise ValueError("Model list drifted.")
    return tuple(dict(item) for item in value)


def _require(value: Mapping[str, Any], key: str, expected: Any) -> None:
    if value.get(key) != expected:
        raise ValueError(f"{key} drifted.")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions(packages: Sequence[str]) -> dict[str, str]:
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _episode_payload(item: ScheduledEpisode) -> dict[str, Any]:
    payload = asdict(item)
    payload["condition"] = item.condition.to_canonical_dict()
    return payload
