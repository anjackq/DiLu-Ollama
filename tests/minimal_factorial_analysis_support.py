from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from dilu.runtime._minimal_factorial_schedule_support import canonical_sha256
from dilu.runtime.harness_config import ShieldConfig
from dilu.runtime.minimal_factorial_calibration import build_calibration_contract
from dilu.runtime.minimal_factorial_schedule import (
    build_harness_config,
    load_experiment_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("qwen_06b", "qwen3:0.6b", "sha256:" + "a" * 64),
    ("llama_1b", "llama3.2:1b", "sha256:" + "b" * 64),
)


def synthetic_analysis_bundle() -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    manifest = load_experiment_manifest(
        ROOT / "configs/iclr2027/minimal_factorial.yaml"
    )
    conditions = {
        index: build_harness_config(manifest, index).to_canonical_dict()
        for index in range(8)
    }
    cases = [
        {
            "case_id": f"case-{index:03d}",
            "seed": 1_000 + index,
            "category": f"category-{index // 12:02d}",
        }
        for index in range(120)
    ]
    stage1_ids = {
        case["case_id"] for case in cases if int(str(case["case_id"])[-3:]) % 12 < 3
    }
    case_set = {"benchmark_name": "synthetic-stress-v2", "cases": cases}
    benchmark = "sha256:" + canonical_sha256(case_set)
    shield = json.loads(
        json.dumps(dataclasses.asdict(ShieldConfig.implementation_defaults()))
    )
    snapshot = {
        "code_revision": "c" * 40,
        "runtime_config": {"simulation_duration": 30},
        "environment_config": {"duration": 30, "vehicles_count": 10},
        "shield_config": shield,
        "predicate_fingerprint": "d" * 64,
        "scoring_fingerprint": "e" * 64,
        "trace_schema_sha256": "f" * 64,
        "case_set_fingerprint": benchmark,
        "primary_metric_spec": {
            "metric": "driving_score_balanced_v1",
            "version": "balanced_driving_score_policy_v1",
        },
    }
    schedule: list[dict[str, Any]] = []
    for slot, tag, model_digest in MODELS:
        for condition_index, condition in conditions.items():
            condition_id = f"c{condition_index:03b}"
            selected = (
                cases
                if condition_id in {"c000", "c111"}
                else [case for case in cases if case["case_id"] in stage1_ids]
            )
            for case in selected:
                schedule.append(
                    _schedule_row(
                        manifest.campaign_id,
                        slot,
                        tag,
                        model_digest,
                        condition_id,
                        condition,
                        case,
                        benchmark,
                    )
                )
    claim = {
        "manifest": {
            "campaign_id": manifest.campaign_id,
            "smoke_campaign_id": manifest.smoke_campaign_id,
            "models": [{"slot": slot, "tag": tag} for slot, tag, _digest in MODELS],
            "selection": {
                "categories": 10,
                "stage1_cases_per_category": 3,
                "stage2_cases_per_category": 12,
            },
        },
        "runtime_snapshot": snapshot,
        "runtime_snapshot_sha256": canonical_sha256(snapshot),
        "schedule": schedule,
    }
    episodes = [
        _episode_row(row, cases, claim["runtime_snapshot_sha256"], line)
        for line, row in enumerate(schedule, start=1)
    ]
    contract = build_calibration_contract(
        claim,
        [row for row in schedule if row["condition_id"] == "c111"],
    )
    category_by_case = {case["case_id"]: case["category"] for case in cases}
    baselines = [
        {
            **dict(contract.provenance),
            "baseline_policy": policy,
            "case_id": case_id,
            "simulator_seed": seed,
            "category": category_by_case[case_id],
            "safety_shields_enabled": True,
            "balanced_driving_score_policy_version": contract.scoring_version,
            "driving_score_balanced_v1": 0.5,
            "task_completed": True,
            "crashed": False,
            "error": None,
        }
        for policy in contract.policies
        for case_id, seed in contract.case_seeds
    ]
    return claim, episodes, baselines


def _schedule_row(
    campaign_id: str,
    slot: str,
    tag: str,
    model_digest: str,
    condition_id: str,
    condition: dict[str, Any],
    case: dict[str, Any],
    benchmark: str,
) -> dict[str, Any]:
    identity = f"{campaign_id}|{slot}|{condition_id}|{case['case_id']}"
    stage = "stage1" if int(str(case["case_id"])[-3:]) % 12 < 3 else "stage2_additional"
    return {
        "stage": stage,
        "campaign_id": campaign_id,
        "model_slot": slot,
        "model_tag": tag,
        "model_digest": model_digest,
        "condition": condition,
        "condition_id": condition_id,
        "case_id": case["case_id"],
        "simulator_seed": case["seed"],
        "episode_attempt_id": "episode-" + _hex(identity),
        "pair_id": "pair-" + _hex(f"{campaign_id}|{case['case_id']}"),
        "template_id": "stress-v2-" + _hex(str(case["case_id"])),
        "replicate_id": 0,
        "primary_snapshot_id": "snapshot-" + _hex(str(case["case_id"])),
        "benchmark_fingerprint": benchmark,
        "code_revision": "c" * 40,
    }


def _episode_row(
    scheduled: dict[str, Any],
    cases: list[dict[str, Any]],
    snapshot_sha256: str,
    line: int,
) -> dict[str, Any]:
    row = json.loads(json.dumps(scheduled))
    case = next(case for case in cases if case["case_id"] == row["case_id"])
    cell = f"{row['model_slot']}|{row['condition_id']}"
    trace_schema = "sha256:" + "f" * 64
    row.update(
        {
            "status": "completed",
            "category": case["category"],
            "seed": case["seed"],
            "runtime_snapshot_sha256": "sha256:" + snapshot_sha256,
            "config_sha256": "sha256:" + canonical_sha256(row["condition"]),
            "runtime_lock_source_artifact_sha256": _sha(cell + "|source"),
            "runtime_lock_authorization_artifact_sha256": _sha(cell + "|auth"),
            "runtime_lock_binding_sha256": _sha(cell + "|binding"),
            "prompt_sha256": _sha(cell + "|prompt"),
            "capability_artifact_sha256": _sha(row["model_slot"] + "|artifact"),
            "capability_snapshot_sha256": _sha(row["model_slot"] + "|snapshot"),
            "trace_schema_sha256": trace_schema,
            "scientific_trace_references": [
                {
                    "relative_path": "traces/decision_traces.jsonl",
                    "line_number": line,
                    "record_sha256": _sha(f"trace|{line}"),
                    "schema_version": "iclr2027.scientific_trace.v1",
                    "schema_sha256": trace_schema,
                }
            ],
            "decisions_made": 1,
            "decision_calls_total": 1,
            "responses_strict_format": 1,
            "fallback_action_count": 0,
            "decision_timeout_count": 0,
            "decision_latency_ms_avg": 10.0,
            "analysis_any_shield_intervention_count": 0,
            "analysis_lane_change_shield_count": 0,
            "analysis_longitudinal_safety_shield_count": 0,
            "analysis_low_speed_recovery_shield_count": 0,
            "analysis_proposal_action_change_count": 0,
            "episode_stop_reason": "completed",
            "error": None,
            "task_completed": True,
            "crashed": False,
            "driving_score_balanced_v1": 0.5,
        }
    )
    return row


def _sha(value: str) -> str:
    return "sha256:" + _hex(value)


def _hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
