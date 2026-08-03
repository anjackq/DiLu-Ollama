"""Read-only evidence loading for registered minimal-factorial analysis."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._campaign_attempt_serialization import hash_payload
from ._minimal_factorial_analysis_validation import (
    AnalysisValidation,
    validate_joined_rows,
)
from ._minimal_factorial_analysis_metrics import validate_analysis_metrics
from ._minimal_factorial_analysis_locks import validate_authorized_runtime_locks
from ._minimal_factorial_calibration_completion import (
    require_complete_claim_campaign,
)
from ._minimal_factorial_calibration_provenance import require_frozen_checkout
from ._minimal_factorial_runner_summaries import load_summary_records
from ._minimal_factorial_schedule_support import canonical_sha256
from ._minimal_factorial_manifest import _manifest_payload, load_experiment_manifest
from .minimal_factorial_calibration import (
    _claim_repo_root,
    build_calibration_contract,
    validate_baseline_bundle,
)
from .minimal_factorial_schedule import select_stage1_cases
from .scientific_trace import ScientificTraceWriter


@dataclass(frozen=True)
class AnalysisInputPaths:
    manifest: Path
    episodes: Path
    baseline_report: Path
    baseline_episodes: Path


@dataclass(frozen=True)
class ValidatedAnalysisInputs:
    claim: Mapping[str, Any]
    episodes: tuple[Mapping[str, Any], ...]
    baseline_rows: tuple[Mapping[str, Any], ...]
    manifest_sha256: str


def load_analysis_inputs(
    paths: AnalysisInputPaths,
) -> tuple[AnalysisValidation, ValidatedAnalysisInputs | None]:
    try:
        manifest_path = Path(paths.manifest).resolve(strict=True)
        manifest_bytes = manifest_path.read_bytes()
        claim = _object(json.loads(manifest_bytes.decode("utf-8")), "claim manifest")
        repo_root = _require_registered_paths(paths, manifest_path, claim)
        registered = _object(claim.get("manifest"), "registered manifest")
        case_set = _object(
            json.loads(
                (repo_root / str(registered["case_path"])).read_text(encoding="utf-8")
            ),
            "registered case set",
        )
        require_frozen_checkout(
            repo_root,
            _object(claim.get("runtime_snapshot"), "runtime snapshot"),
            _rows(case_set.get("cases"), "registered cases"),
        )
        snapshot = _object(claim.get("runtime_snapshot"), "runtime snapshot")
        if "sha256:" + canonical_sha256(case_set) != snapshot.get(
            "case_set_fingerprint"
        ):
            raise ValueError("Registered case-set fingerprint drifted.")
        selection = _object(registered.get("selection"), "registered selection")
        selected = {
            str(case["case_id"])
            for case in select_stage1_cases(
                case_set, str(selection["stage1_hash_prefix"])
            )
        }
        observed_selected = {
            str(row.get("case_id"))
            for row in _rows(claim.get("schedule"), "claim schedule")
            if row.get("stage") == "stage1"
        }
        if observed_selected != selected:
            raise ValueError("Selected-30 case identities drifted from registration.")
        require_complete_claim_campaign(manifest_path, claim)
        campaign_sha256 = _campaign_sha256(claim)
        loaded_episodes = load_summary_records(
            Path(paths.episodes).resolve(strict=True),
            expected_campaign_provenance_sha256=campaign_sha256,
        )
        transport = _object(registered.get("transport"), "registered transport")
        episodes = _enrich_trace_metrics(
            loaded_episodes,
            manifest_path.parent / "traces" / "decision_traces.jsonl",
            native_endpoint=str(transport["native_endpoint"]),
        )
        report_path = Path(paths.baseline_report).resolve(strict=True)
        report = _object(
            json.loads(report_path.read_text(encoding="utf-8")),
            "baseline report",
        )
        baseline_rows = _rows(report.get("episodes"), "baseline report episodes")
        schedule = _rows(claim.get("schedule"), "claim schedule")
        contract = build_calibration_contract(
            claim,
            tuple(row for row in schedule if row.get("condition_id") == "c111"),
        )
        baseline_validation = validate_baseline_bundle(
            report_path,
            Path(paths.baseline_episodes).resolve(strict=True),
            contract,
        )
        validation = validate_joined_rows(claim, episodes, baseline_rows)
        errors = list(validation.errors)
        errors.extend(validate_analysis_metrics(episodes, baseline_rows))
        outputs = _object(registered.get("outputs"), "registered outputs")
        lock_root = repo_root / str(outputs["root"]) / str(outputs["s1"]) / "locks"
        errors.extend(validate_authorized_runtime_locks(lock_root, claim, episodes))
        if not baseline_validation.valid:
            errors.extend(
                f"baseline bundle: {error}" for error in baseline_validation.errors
            )
        if errors:
            return _blocked(validation, errors), None
        return validation, ValidatedAnalysisInputs(
            claim,
            episodes,
            baseline_rows,
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        )
    except Exception as exc:
        validation = AnalysisValidation(
            "blocked",
            (f"analysis input validation failed: {exc}",),
            False,
            840,
            0,
            360,
            0,
        )
        return validation, None


def _require_registered_paths(
    paths: AnalysisInputPaths,
    manifest_path: Path,
    claim: Mapping[str, Any],
) -> Path:
    manifest = _object(claim.get("manifest"), "registered manifest")
    repo_root = _claim_repo_root(manifest_path, manifest)
    registered = load_experiment_manifest(
        repo_root / "configs" / "iclr2027" / "minimal_factorial.yaml"
    )
    if manifest != _manifest_payload(registered):
        raise ValueError("Frozen campaign manifest drifted from the registered source.")
    outputs = _object(manifest.get("outputs"), "registered outputs")
    result_root = repo_root / str(outputs["root"])
    expected = {
        "episodes": result_root / str(outputs["llm_campaign"]) / "episodes.jsonl",
        "baseline_report": result_root
        / str(outputs["baselines"])
        / "non_llm_baseline_report.json",
        "baseline_episodes": result_root
        / str(outputs["baselines"])
        / "episode_metrics.csv",
    }
    observed = {
        "episodes": Path(paths.episodes).resolve(),
        "baseline_report": Path(paths.baseline_report).resolve(),
        "baseline_episodes": Path(paths.baseline_episodes).resolve(),
    }
    if any(expected[name].resolve() != observed[name] for name in expected):
        raise ValueError("Analysis input path drifted from registered outputs.")
    return repo_root


def _campaign_sha256(claim: Mapping[str, Any]) -> str:
    runtime_sha256 = str(claim.get("runtime_snapshot_sha256") or "")
    if runtime_sha256.startswith("sha256:"):
        runtime_sha256 = runtime_sha256.removeprefix("sha256:")
    return hash_payload(
        {
            "schema_version": "iclr2027.campaign_provenance.v1",
            "runtime_snapshot_sha256": "sha256:" + runtime_sha256,
            "scheduled_denominator": list(
                _rows(claim.get("schedule"), "claim schedule")
            ),
        }
    )


def _enrich_trace_metrics(
    episodes: tuple[Mapping[str, Any], ...],
    trace_path: Path,
    *,
    native_endpoint: str,
) -> tuple[Mapping[str, Any], ...]:
    trace_writer = ScientificTraceWriter(
        trace_path,
        artifact_root=trace_path.parent.parent,
        resume=True,
    )
    references_by_line = {
        reference.line_number: reference.to_dict()
        for reference in trace_writer.reference_snapshot()
    }
    records = tuple(
        _object(json.loads(line), "scientific trace")
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    )
    output = []
    for episode in episodes:
        references = episode.get("scientific_trace_references")
        if not isinstance(references, list) or not references:
            raise ValueError("Completed analysis episode has no trace references.")
        selected = []
        for reference in references:
            item = _object(reference, "trace reference")
            line_number = item.get("line_number")
            if isinstance(line_number, bool) or not isinstance(line_number, int):
                raise ValueError("Trace reference line number is invalid.")
            if line_number < 1 or line_number > len(records):
                raise ValueError("Trace reference escapes the validated trace.")
            if dict(item) != references_by_line.get(line_number):
                raise ValueError("Trace reference does not match the validated record.")
            record = records[line_number - 1]
            key = _object(record.get("trace_key"), "trace key")
            if key.get("episode_attempt_id") != episode.get("episode_attempt_id"):
                raise ValueError("Trace reference episode identity drifted.")
            _validate_trace_join(record, episode, native_endpoint)
            selected.append(record)
        output.append({**dict(episode), **_trace_metrics(selected)})
    return tuple(output)


def _validate_trace_join(
    record: Mapping[str, Any],
    episode: Mapping[str, Any],
    native_endpoint: str,
) -> None:
    key = _object(record.get("trace_key"), "trace key")
    context = _object(record.get("context"), "trace context")
    generation = _object(record.get("generation"), "trace generation")
    request = _object(generation.get("request"), "trace request")
    evidence = _object(generation.get("transport_evidence"), "transport evidence")
    condition = _object(episode.get("condition"), "episode condition")
    transport = _object(condition.get("transport"), "episode transport")
    pairs = {
        "campaign_id": (key.get("campaign_id"), episode.get("campaign_id")),
        "condition_id": (key.get("condition_id"), episode.get("condition_id")),
        "case_id": (key.get("case_id"), episode.get("case_id")),
        "pair_id": (key.get("pair_id"), episode.get("pair_id")),
        "template_id": (key.get("template_id"), episode.get("template_id")),
        "replicate_id": (key.get("replicate_id"), episode.get("replicate_id")),
        "simulator_seed": (
            context.get("simulator_seed"),
            episode.get("simulator_seed"),
        ),
        "decision_snapshot_id": (
            context.get("decision_snapshot_id"),
            (
                episode.get("primary_snapshot_id")
                if key.get("decision_index") == 0
                else None
            ),
        ),
        "benchmark_fingerprint": (
            context.get("benchmark_fingerprint"),
            episode.get("benchmark_fingerprint"),
        ),
        "code_revision": (context.get("code_revision"), episode.get("code_revision")),
        "config_sha256": (record.get("config_sha256"), episode.get("config_sha256")),
        "prompt_sha256": (
            _object(record.get("prompt"), "trace prompt").get("prompt_sha256"),
            episode.get("prompt_sha256"),
        ),
        "model_tag": (request.get("model_tag"), episode.get("model_tag")),
        "model_digest": (request.get("model_digest"), episode.get("model_digest")),
        "native_endpoint": (request.get("native_endpoint"), native_endpoint),
        "think_mode": (request.get("think_mode"), transport.get("think_mode")),
        "capability_artifact_sha256": (
            evidence.get("capability_artifact_sha256"),
            episode.get("capability_artifact_sha256"),
        ),
        "capability_snapshot_sha256": (
            evidence.get("capability_snapshot_sha256"),
            episode.get("capability_snapshot_sha256"),
        ),
    }
    drifted = sorted(name for name, values in pairs.items() if values[0] != values[1])
    if drifted:
        raise ValueError(f"Scientific trace join drifted: {', '.join(drifted)}.")


def _trace_metrics(records: list[Mapping[str, Any]]) -> dict[str, int]:
    stage_counts = {
        "lane_change": 0,
        "longitudinal_safety": 0,
        "low_speed_recovery": 0,
    }
    any_intervention = 0
    proposal_change = 0
    for record in records:
        stack = _object(record.get("shield_stack"), "shield stack")
        stages = stack.get("stages")
        if not isinstance(stages, list) or len(stages) != 3:
            raise ValueError("Scientific trace shield stages drifted.")
        applied = False
        for stage in stages:
            item = _object(stage, "shield stage")
            name = str(item.get("stage_name") or "")
            if name not in stage_counts:
                raise ValueError("Scientific trace shield stage name drifted.")
            if item.get("applied") is True:
                stage_counts[name] += 1
                applied = True
        any_intervention += int(applied)
        proposal_change += int(
            stack.get("proposed_action_id") != stack.get("executed_action_id")
        )
    return {
        "analysis_any_shield_intervention_count": any_intervention,
        "analysis_lane_change_shield_count": stage_counts["lane_change"],
        "analysis_longitudinal_safety_shield_count": stage_counts[
            "longitudinal_safety"
        ],
        "analysis_low_speed_recovery_shield_count": stage_counts["low_speed_recovery"],
        "analysis_proposal_action_change_count": proposal_change,
    }


def _blocked(
    validation: AnalysisValidation,
    errors: list[str],
) -> AnalysisValidation:
    return dataclasses.replace(
        validation,
        status="blocked",
        errors=tuple(sorted(set(errors))),
        contrast_artifacts_written=False,
    )


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _rows(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError(f"{name} must be a list of objects.")
    return tuple(value)


__all__ = ["AnalysisInputPaths", "ValidatedAnalysisInputs", "load_analysis_inputs"]
