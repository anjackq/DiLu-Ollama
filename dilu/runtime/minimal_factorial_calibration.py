"""Matched non-LLM calibration orchestration for the minimal factorial."""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from evaluate_non_llm_baselines import (
    aggregate_results,
    run_baseline_episode,
    write_csv,
)

from ._minimal_factorial_calibration_completion import (
    require_complete_claim_campaign,
)
from ._minimal_factorial_calibration_provenance import require_frozen_checkout
from ._minimal_factorial_schedule_support import canonical_sha256
from ._minimal_factorial_calibration_validation import (
    CalibrationValidation,
    validate_baseline_bundle,
)
from .harness_config import ShieldConfig
from .highway_env_config import resolve_simulation_env_bundle
from .non_llm_baselines import BaselinePolicy
from .path_utils import write_json_atomic

CALIBRATION_POLICIES = ("always_left", "speed_hold_25", "idm_mobil")
_SHA256_RE = re.compile(r"\A(?:sha256:)?([0-9a-f]{64})\Z")


@dataclass(frozen=True)
class CalibrationContract:
    policies: tuple[str, ...]
    case_seeds: tuple[tuple[str, int], ...]
    shield_config: Mapping[str, Any]
    provenance: Mapping[str, str]
    scoring_version: str
    expected_rows: int


def build_calibration_contract(
    claim_manifest: Mapping[str, Any],
    c111_rows: Sequence[Mapping[str, Any]],
) -> CalibrationContract:
    claim = _mapping(claim_manifest, "claim manifest")
    schedule = _rows(claim.get("schedule"), "claim schedule")
    if len(schedule) != 840:
        raise ValueError("Calibration requires the complete 840-row claim campaign.")
    expected_c111 = tuple(row for row in schedule if row.get("condition_id") == "c111")
    supplied_c111 = tuple(_mapping(row, "c111 row") for row in c111_rows)
    if len(expected_c111) != 240 or _canonical_rows(supplied_c111) != _canonical_rows(
        expected_c111
    ):
        raise ValueError("Calibration requires the exact 240-row c111 denominator.")

    snapshot = _mapping(claim.get("runtime_snapshot"), "runtime snapshot")
    if claim.get("runtime_snapshot_sha256") != canonical_sha256(snapshot):
        raise ValueError("Claim runtime snapshot hash drifted.")
    defaults = json.loads(
        json.dumps(dataclasses.asdict(ShieldConfig.implementation_defaults()))
    )
    if snapshot.get("shield_config") != defaults:
        raise ValueError(
            "Frozen shield primitives drifted from implementation defaults."
        )

    by_model: dict[str, set[tuple[str, int]]] = {}
    benchmark_fingerprints: set[str] = set()
    revisions: set[str] = set()
    for row in supplied_c111:
        model = _text(row, "model_slot")
        pair = (_text(row, "case_id"), _integer(row, "simulator_seed"))
        by_model.setdefault(model, set()).add(pair)
        condition = _mapping(row.get("condition"), "c111 condition")
        factors = _mapping(condition.get("condition"), "c111 factors")
        if factors.get("execution_mode") != "shielded":
            raise ValueError("c111 execution mode must be shielded.")
        if condition.get("shield") != defaults:
            raise ValueError("c111 shield configuration drifted.")
        benchmark_fingerprints.add(_text(row, "benchmark_fingerprint"))
        revisions.add(_text(row, "code_revision"))
    if len(by_model) != 2 or any(len(values) != 120 for values in by_model.values()):
        raise ValueError("Each c111 model must contain exactly 120 case/seed pairs.")
    pair_sets = tuple(by_model.values())
    if pair_sets[0] != pair_sets[1]:
        raise ValueError("c111 model rows do not share identical case/seed pairs.")
    if len(benchmark_fingerprints) != 1 or len(revisions) != 1:
        raise ValueError("c111 benchmark or source revision drifted.")

    fingerprint = benchmark_fingerprints.pop()
    revision = revisions.pop()
    if fingerprint != _sha256(snapshot.get("case_set_fingerprint")):
        raise ValueError("c111 benchmark fingerprint drifted from the case set.")
    if revision != snapshot.get("code_revision"):
        raise ValueError("c111 source revision drifted from the runtime snapshot.")
    primary_metric = _mapping(snapshot.get("primary_metric_spec"), "primary metric")
    scoring_version = _text(primary_metric, "version")
    provenance = {
        "benchmark_fingerprint": fingerprint,
        "case_set_sha256": fingerprint,
        "environment_config_sha256": _digest(snapshot, "environment_config"),
        "predicate_sha256": _sha256(snapshot.get("predicate_fingerprint")),
        "resolved_runtime_config_sha256": _digest(snapshot, "runtime_config"),
        "scoring_sha256": _sha256(snapshot.get("scoring_fingerprint")),
        "primary_metric_spec_sha256": "sha256:" + canonical_sha256(primary_metric),
        "scoring_policy_version": scoring_version,
        "shield_config_sha256": "sha256:" + canonical_sha256(defaults),
        "source_revision": revision,
    }
    case_seeds = tuple(sorted(pair_sets[0]))
    return CalibrationContract(
        CALIBRATION_POLICIES,
        case_seeds,
        MappingProxyType(defaults),
        MappingProxyType(provenance),
        scoring_version,
        len(CALIBRATION_POLICIES) * len(case_seeds),
    )


def run_baseline_campaign(
    claim_manifest_path: Path,
    *,
    output_root: Path,
) -> Path:
    claim_path = Path(claim_manifest_path).resolve(strict=True)
    claim = _load_object(claim_path)
    schedule = _rows(claim.get("schedule"), "claim schedule")
    contract = build_calibration_contract(
        claim,
        tuple(row for row in schedule if row.get("condition_id") == "c111"),
    )
    require_complete_claim_campaign(claim_path, claim)
    nested_manifest = _mapping(claim.get("manifest"), "frozen manifest")
    repo_root = _claim_repo_root(claim_path, nested_manifest)
    case_path = (repo_root / _text(nested_manifest, "case_path")).resolve(strict=True)
    _require_contained(case_path, repo_root)
    case_set = _load_object(case_path)
    cases = _rows(case_set.get("cases"), "case set")
    observed_pairs = {
        (str(case.get("case_id")), int(case.get("seed"))) for case in cases
    }
    if observed_pairs != set(contract.case_seeds) or len(cases) != 120:
        raise ValueError("Checked-in cases drifted from the c111 denominator.")
    if "sha256:" + canonical_sha256(case_set) != contract.provenance["case_set_sha256"]:
        raise ValueError("Checked-in case-set hash drifted from c111.")

    snapshot = _mapping(claim.get("runtime_snapshot"), "runtime snapshot")
    require_frozen_checkout(repo_root, snapshot, cases)
    runtime = dict(_mapping(snapshot.get("runtime_config"), "runtime config"))
    expected_environment = _mapping(
        snapshot.get("environment_config"),
        "environment config",
    )
    simulation = _mapping(nested_manifest.get("simulation"), "simulation manifest")
    env_type = _text(simulation, "target_env_id")
    bundle = resolve_simulation_env_bundle(
        runtime,
        show_trajectories=False,
        render_agent=False,
        env_id_override=env_type,
        native_env_defaults_override=True,
        require_discrete_meta_action=True,
    )
    if canonical_sha256(bundle.get("env_config_snapshot")) != canonical_sha256(
        expected_environment
    ):
        raise ValueError("Resolved environment bundle drifted from c111.")
    env_config_map = _mapping(bundle.get("env_config_map"), "environment map")

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = _target_paths(root)
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("Baseline calibration artifacts already exist.")
    baseline_manifest = {
        "artifact_type": "matched_non_llm_calibration_v1",
        "policies": list(contract.policies),
        "case_seeds": [list(pair) for pair in contract.case_seeds],
        "expected_rows": contract.expected_rows,
        "shield_config": dict(contract.shield_config),
        "provenance": dict(contract.provenance),
    }

    case_by_id = {str(case["case_id"]): dict(case) for case in cases}
    episodes: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    primary_spec = snapshot.get("primary_metric_spec")
    for policy_name in contract.policies:
        policy = BaselinePolicy(policy_name, runtime)
        if not policy.spec.safety_shield_compatible:
            raise ValueError(
                f"Calibration policy is not shield-compatible: {policy_name}."
            )
        policy_rows: list[dict[str, Any]] = []
        for case_id, seed in contract.case_seeds:
            case = case_by_id[case_id]
            if int(case["seed"]) != seed:
                raise ValueError("Case seed drifted during calibration execution.")
            result = run_baseline_episode(
                config=runtime,
                env_config_map=dict(env_config_map),
                env_type=env_type,
                case=case,
                policy=policy,
                safety_shields_enabled=True,
            )
            if (
                result.get("error") not in (None, "")
                or result.get("episode_stop_reason") == "error"
            ):
                raise RuntimeError(f"Calibration evaluator failed: {policy_name}.")
            if result.get("baseline_safety_shields_enabled") is not True:
                raise ValueError(
                    f"Calibration evaluator did not apply shields: {policy_name}."
                )
            if (
                result.get("balanced_driving_score_policy_version")
                != contract.scoring_version
            ):
                raise ValueError(f"Calibration scoring version drifted: {policy_name}.")
            row = {
                **dict(result),
                **dict(contract.provenance),
                "baseline_policy": policy_name,
                "case_id": case_id,
                "simulator_seed": seed,
                "safety_shields_enabled": True,
            }
            policy_rows.append(row)
            episodes.append(row)
        aggregate = aggregate_results(
            policy_name,
            policy_rows,
            planned_episode_count=len(contract.case_seeds),
            primary_metric_spec=primary_spec,
        )
        aggregates.append({**aggregate, **dict(contract.provenance)})

    report = {
        "artifact_type": "matched_non_llm_calibration_report_v1",
        "baselines": list(contract.policies),
        "case_count": len(contract.case_seeds),
        "episode_count": len(episodes),
        "safety_shields_enabled": True,
        "provenance": dict(contract.provenance),
        "aggregates": aggregates,
        "episodes": episodes,
    }
    _write_json(targets["manifest"], baseline_manifest)
    _write_json(targets["report"], report)
    write_csv(targets["episodes"], episodes)
    validation = validate_baseline_bundle(
        targets["report"],
        targets["episodes"],
        contract,
    )
    _write_json(targets["validation"], dataclasses.asdict(validation))
    if not validation.valid:
        raise RuntimeError("Baseline calibration bundle failed validation.")
    return targets["validation"]


def _claim_repo_root(path: Path, manifest: Mapping[str, Any]) -> Path:
    outputs = _mapping(manifest.get("outputs"), "output manifest")
    relative_root = Path(_text(outputs, "root"))
    llm_directory = str(outputs.get("llm_campaign") or path.parent.name)
    root = path.parent
    for _ in range(len(relative_root.parts) + 1):
        root = root.parent
    expected = root / relative_root / llm_directory / "campaign_manifest.json"
    if expected.resolve() != path:
        raise ValueError("Claim manifest path drifted from frozen outputs.")
    return root.resolve()


def _target_paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / "baseline_manifest.json",
        "report": root / "non_llm_baseline_report.json",
        "episodes": root / "episode_metrics.csv",
        "validation": root / "calibration_validation.json",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(str(path), dict(payload))


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(_mapping(value, str(path)))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _rows(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError(f"{name} must be a list of objects.")
    return tuple(value)


def _canonical_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    values = tuple(
        sorted(
            (_text(row, "episode_attempt_id"), canonical_sha256(row)) for row in rows
        )
    )
    if len({attempt_id for attempt_id, _digest_value in values}) != len(values):
        raise ValueError("Claim schedule contains duplicate episode IDs.")
    return values


def _text(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item or item != item.strip():
        raise ValueError(f"{name} must be non-empty canonical text.")
    return item


def _integer(value: Mapping[str, Any], name: str) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"{name} must be an integer.")
    return item


def _sha256(value: Any) -> str:
    match = _SHA256_RE.fullmatch(str(value or ""))
    if match is None:
        raise ValueError("Frozen provenance digest is invalid.")
    return "sha256:" + match.group(1)


def _digest(snapshot: Mapping[str, Any], name: str) -> str:
    return "sha256:" + canonical_sha256(_mapping(snapshot.get(name), name))


def _require_contained(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Calibration case path escapes the repository.") from exc


__all__ = [
    "CALIBRATION_POLICIES",
    "CalibrationContract",
    "CalibrationValidation",
    "build_calibration_contract",
    "run_baseline_campaign",
    "validate_baseline_bundle",
]
