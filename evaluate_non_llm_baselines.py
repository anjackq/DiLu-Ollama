"""Evaluate non-LLM behavior baselines on DiLu benchmark case sets.

The script reuses the DiLu-Ollama benchmark case loader, scenario-event
application, episode evaluator, and aggregate metric pipeline. It does not call
an LLM backend and it marks rows as behavior-reference baselines rather than
LLM-contract rows.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from evaluate_models_ollama import (
    _apply_reactive_safety_shields,
    aggregate_results,
    extract_step_traffic_metrics,
)
from dilu.runtime import (
    BenchmarkEpisodeEvaluator,
    augment_behavior_aware_benchmark_episode,
    benchmark_max_steps,
    build_benchmark_case_set_fingerprint,
    build_case_env_config,
    build_primary_metric_spec,
    compute_split_scores_for_episode,
    current_timestamp,
    ensure_dir,
    load_benchmark_case_set,
    load_runtime_config,
    resolve_simulation_env_bundle,
    validate_benchmark_case_set,
    write_json_atomic,
)
from dilu.runtime.highway_scenario_spec import (
    apply_highway_scenario_events,
    apply_highway_scenario_spec,
)
from dilu.runtime.non_llm_baselines import (
    DEFAULT_BASELINE_NAMES,
    EXPERT_CONTROL_MODE,
    BaselinePolicy,
    configure_true_idm_ego,
    iter_baseline_specs,
    parse_baseline_levels,
    resolve_baseline_names,
    stable_seed,
    vehicle_snapshot,
)
from dilu.runtime.safety_shields import (
    FASTER_ACTION_ID,
    IDLE_ACTION_ID,
    LANE_LEFT_ACTION_ID,
    LANE_RIGHT_ACTION_ID,
    SLOWER_ACTION_ID,
)
from dilu.scenario.envScenario import EnvScenario


LOGGER = logging.getLogger("evaluate_non_llm_baselines")
DEFAULT_BASELINES = DEFAULT_BASELINE_NAMES
LLM_SCORE_FIELDS = {
    "llm_driver_score_v1",
    "llm_output_contract_score_v1",
    "llm_runtime_reliability_score_v1",
    "llm_action_validity_score_v1",
    "llm_flow_recovery_independence_score_v1",
    "llm_safety_intervention_independence_score_v1",
    "llm_parser_independence_score_v1",
    "llm_intervention_independence_score_v1",
    "llm_latency_score_v1",
    "llm_resource_efficiency_score_v1",
    "dilu_joint_score_v1",
}


def filter_cases(case_set: dict[str, Any], categories: Optional[str], limit: Optional[int]) -> dict[str, Any]:
    selected = list(case_set.get("cases") or [])
    if categories:
        requested = {token.strip() for token in categories.split(",") if token.strip()}
        selected = [case for case in selected if str(case.get("category")) in requested]
        if not selected:
            raise ValueError(f"No cases matched categories: {sorted(requested)}")
    if limit is not None:
        selected = selected[: int(limit)]
    filtered = dict(case_set)
    filtered["cases"] = selected
    filtered["categories"] = sorted({str(case["category"]) for case in selected})
    return filtered


def metric_thresholds(config: dict[str, Any]) -> dict[str, float]:
    return {
        "ttc_threshold_sec": float(config.get("metrics_ttc_threshold_sec", 2.0)),
        "headway_threshold_m": float(config.get("metrics_headway_threshold_m", 15.0)),
        "rear_ttc_threshold_sec": float(config.get("metrics_rear_ttc_threshold_sec", 2.5)),
        "rear_headway_threshold_m": float(config.get("metrics_rear_headway_threshold_m", 12.0)),
        "low_speed_blocking_threshold_mps": float(config.get("metrics_low_speed_blocking_threshold_mps", 8.5)),
        "blocking_front_gap_safe_m": float(config.get("metrics_blocking_front_gap_safe_m", 25.0)),
        "blocking_front_ttc_safe_sec": float(config.get("metrics_blocking_front_ttc_safe_sec", 4.0)),
        "stop_threshold_mps": float(config.get("metrics_stop_threshold_mps", 0.5)),
        "near_stop_threshold_mps": float(config.get("metrics_near_stop_threshold_mps", 2.0)),
    }


def traffic_metrics(env: Any, thresholds: dict[str, float]) -> dict[str, Any]:
    return extract_step_traffic_metrics(env, **thresholds)


def scrub_non_llm_scores(episode: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(episode)
    for field in LLM_SCORE_FIELDS:
        cleaned[field] = None
    cleaned["baseline_llm_contract_applicable"] = False
    if cleaned.get("baseline_control_mode") == EXPERT_CONTROL_MODE:
        cleaned["baseline_claim_scope"] = "expert_behavior_reference_only"
    else:
        cleaned["baseline_claim_scope"] = "behavior_reference_only"
    return cleaned


def run_baseline_episode(
    *,
    config: dict[str, Any],
    env_config_map: dict[str, dict[str, Any]],
    env_type: str,
    case: dict[str, Any],
    policy: BaselinePolicy,
    safety_shields_enabled: bool,
) -> dict[str, Any]:
    seed = int(case["seed"])
    case_env_config, case_env_snapshot = build_case_env_config(env_config_map, env_type, case)
    max_steps = benchmark_max_steps(case, case_env_snapshot, int(config.get("simulation_duration", 30)))
    thresholds = metric_thresholds(config)
    rng = random.Random(stable_seed(policy.name, case.get("case_id"), seed))
    expert_mode = policy.control_mode == EXPERT_CONTROL_MODE
    effective_safety_shields = bool(safety_shields_enabled and policy.spec.safety_shield_compatible)
    baseline_metadata = policy.spec.to_metadata(safety_shields_enabled=effective_safety_shields)
    started = time.time()
    env = gym.make(env_type, render_mode="rgb_array")

    crashed = terminated = truncated = False
    error = None
    final_info: dict[str, Any] = {}
    steps = 0
    reward_sum = 0.0
    speed_sum = 0.0
    speed_count = 0
    min_speed: Optional[float] = None
    counts: Counter[str] = Counter()
    action_trace: list[dict[str, Any]] = []
    previous_action: Optional[int] = None
    benchmark_evaluator: Optional[BenchmarkEpisodeEvaluator] = None
    applied_event_ids: set[str] = set()
    expert_meta: dict[str, Any] = {}

    try:
        env.unwrapped.configure(case_env_config[env_type])
        _, info = env.reset(seed=seed)
        scenario_meta = apply_highway_scenario_spec(env, case)
        if expert_mode:
            expert_meta = configure_true_idm_ego(
                env,
                target_speed_mps=float(policy.target_speed),
                enable_lane_change=True,
            )
        scenario = EnvScenario(env, env_type, seed, database=None, enable_db=False)
        benchmark_evaluator = BenchmarkEpisodeEvaluator(
            case,
            env,
            scenario_spec_metadata={**scenario_meta, **expert_meta},
        )
        final_info = dict(info or {})

        for step_idx in range(1, max_steps + 1):
            event_meta = apply_highway_scenario_events(
                env,
                case,
                step_idx=step_idx,
                applied_event_ids=applied_event_ids,
            )
            decision_started = time.time()
            previous_lane_rank = vehicle_snapshot(getattr(env.unwrapped, "vehicle", None)).lane_rank
            if expert_mode:
                decision = policy.decide(env, case, step_idx, rng)
                decision_elapsed = max(0.0, time.time() - decision_started)
                proposed_action = None
                action = None
                shield_meta = {
                    "reactive_safety_shield_applied": False,
                    "reactive_safety_original_action_id": None,
                    "reactive_safety_final_action_id": None,
                    "lane_change_shield_applied": False,
                    "longitudinal_safety_shield_applied": False,
                    "flow_recovery_shield_applied": False,
                }
                decision_meta = {
                    "baseline_policy": policy.name,
                    "baseline_decision_reason": decision.reason,
                    "decision_elapsed_sec": round(decision_elapsed, 6),
                    "runtime_parse_path": "non_llm_expert_vehicle",
                    "timed_out": False,
                    "used_fallback": False,
                    **dict(expert_meta),
                    **dict(event_meta),
                }
            else:
                decision = policy.decide(env, case, step_idx, rng)
                decision_elapsed = max(0.0, time.time() - decision_started)
                proposed_action = int(decision.action_id)
                decision_meta = {
                    "baseline_policy": policy.name,
                    "baseline_decision_reason": decision.reason,
                    "decision_elapsed_sec": round(decision_elapsed, 6),
                    "runtime_parse_path": "non_llm_baseline",
                    "timed_out": False,
                    "used_fallback": False,
                    **dict(decision.metadata),
                    **dict(event_meta),
                }
                if effective_safety_shields:
                    action, shield_meta = _apply_reactive_safety_shields(proposed_action, scenario, decision_meta)
                else:
                    action = proposed_action
                    shield_meta = {
                        "reactive_safety_shield_applied": False,
                        "reactive_safety_original_action_id": int(proposed_action),
                        "reactive_safety_final_action_id": int(action),
                        "lane_change_shield_applied": False,
                        "longitudinal_safety_shield_applied": False,
                        "flow_recovery_shield_applied": False,
                    }
            _, reward, terminated, truncated, info = env.step(action)
            final_info = dict(info or {})
            crashed = bool(final_info.get("crashed", False))
            steps += 1
            reward_sum += float(reward)
            current_lane_rank = vehicle_snapshot(getattr(env.unwrapped, "vehicle", None)).lane_rank
            if expert_mode:
                if (
                    previous_lane_rank is not None
                    and current_lane_rank is not None
                    and current_lane_rank < previous_lane_rank
                ):
                    inferred_action = LANE_LEFT_ACTION_ID
                elif (
                    previous_lane_rank is not None
                    and current_lane_rank is not None
                    and current_lane_rank > previous_lane_rank
                ):
                    inferred_action = LANE_RIGHT_ACTION_ID
                else:
                    inferred_action = IDLE_ACTION_ID
                action_context = {
                    **shield_meta,
                    **event_meta,
                    "action_id": int(inferred_action),
                    "final_action_id": int(inferred_action),
                    "expert_vehicle_autonomous": True,
                }
                counts["lane_change"] += int(inferred_action in (LANE_LEFT_ACTION_ID, LANE_RIGHT_ACTION_ID))
                if previous_action is not None and {previous_action, inferred_action} == {FASTER_ACTION_ID, SLOWER_ACTION_ID}:
                    counts["flap_accel_decel"] += 1
                previous_action = int(inferred_action)
            else:
                action_context = {**shield_meta, **event_meta, "action_id": int(action), "final_action_id": int(action)}
                counts["lane_change_shield"] += int(bool(shield_meta.get("lane_change_shield_applied", False)))
                counts["longitudinal_safety_shield"] += int(bool(shield_meta.get("longitudinal_safety_shield_applied", False)))
                counts["flow_recovery_shield"] += int(bool(shield_meta.get("flow_recovery_shield_applied", False)))
                counts["lane_change"] += int(action in (LANE_LEFT_ACTION_ID, LANE_RIGHT_ACTION_ID))
                if previous_action is not None and {previous_action, action} == {FASTER_ACTION_ID, SLOWER_ACTION_ID}:
                    counts["flap_accel_decel"] += 1
                previous_action = int(action)
            step_metrics = traffic_metrics(env, thresholds)
            if step_metrics.get("ego_speed_mps") is not None:
                speed = float(step_metrics["ego_speed_mps"])
                speed_sum += speed
                speed_count += 1
                min_speed = speed if min_speed is None else min(min_speed, speed)
            for key in (
                "ttc_danger",
                "headway_violation",
                "rear_ttc_danger",
                "rear_headway_violation",
                "low_speed_blocking",
                "stopped",
                "near_stop",
            ):
                counts[key] += int(bool(step_metrics.get(key, False)))
            if benchmark_evaluator is not None:
                benchmark_evaluator.update(
                    env,
                    step_idx=steps,
                    step_metrics=step_metrics,
                    crashed=crashed,
                    info=final_info,
                    action_context=action_context,
                )
            action_trace.append(
                {
                    "step_idx": step_idx,
                    "proposed_action_id": None if proposed_action is None else int(proposed_action),
                    "action_id": int(action_context["final_action_id"]),
                    "decision_reason": decision.reason,
                    "decision_elapsed_sec": round(decision_elapsed, 6),
                    "shield_applied": bool(shield_meta.get("reactive_safety_shield_applied", False)),
                    "baseline_control_mode": policy.control_mode,
                }
            )
            if terminated or truncated:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    if error is not None:
        stop_reason = "error"
    elif crashed:
        stop_reason = "crash"
    elif truncated:
        stop_reason = "truncated"
    elif terminated:
        stop_reason = "terminated"
    else:
        stop_reason = "completed"

    duration_sec = time.time() - started
    benchmark_metrics = benchmark_evaluator.finalize(crashed=crashed, episode_stop_reason=stop_reason) if benchmark_evaluator is not None else {}
    decisions = max(steps, 1)
    episode = {
        "seed": seed,
        "case_id": case.get("case_id"),
        "category": case.get("category"),
        "baseline_policy": policy.name,
        "baseline_llm_contract_applicable": False,
        "baseline_claim_scope": (
            "expert_behavior_reference_only"
            if expert_mode
            else "behavior_reference_only"
        ),
        **baseline_metadata,
        "baseline_requested_safety_shields_enabled": bool(safety_shields_enabled),
        "safety_shields_enabled": bool(effective_safety_shields),
        "steps": int(steps),
        "max_steps": int(max_steps),
        "crashed": bool(crashed),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success_no_collision": bool(error is None and not crashed),
        "episode_runtime_sec": round(duration_sec, 3),
        "avg_step_runtime_sec": round(duration_sec / max(steps, 1), 3),
        "episode_stop_reason": stop_reason,
        "timeout_triggered": False,
        "timeout_early_stop_triggered": False,
        "timeout_early_stop_reason": None,
        "timeout_early_stop_step": None,
        "first_timeout_step": None,
        "decision_calls_total": int(steps),
        "decisions_made": int(steps),
        "decision_timeout_count": 0,
        "decision_timeout_rate": 0.0,
        "fallback_action_count": 0,
        "fallback_action_rate": 0.0,
        "responses_with_delimiter": 0,
        "responses_strict_format": 0,
        "responses_direct_parseable": 0,
        "format_failure_count": 0,
        "format_failure_rate": 0.0,
        "fallback_reason_counts": {},
        "runtime_parse_path_counts": {
            "non_llm_expert_vehicle" if expert_mode else "non_llm_baseline": int(steps)
        },
        "semantic_recovery_count": 0,
        "semantic_recovery_rate": 0.0,
        "semantic_recovery_label_counts": {},
        "intent_resolver_used_count": 0,
        "intent_resolver_used_rate": 0.0,
        "intent_resolver_recovery_count": 0,
        "intent_resolver_recovery_rate": 0.0,
        "intent_resolver_abstain_count": 0,
        "intent_resolver_abstain_rate": 0.0,
        "episode_reward_sum": round(reward_sum, 4),
        "episode_reward_avg": round(reward_sum / max(steps, 1), 4),
        "avg_ego_speed_mps": round(speed_sum / max(speed_count, 1), 4),
        "ttc_danger_steps": int(counts["ttc_danger"]),
        "ttc_danger_rate": round(counts["ttc_danger"] / decisions, 4),
        "headway_violation_steps": int(counts["headway_violation"]),
        "headway_violation_rate": round(counts["headway_violation"] / decisions, 4),
        "rear_ttc_danger_steps": int(counts["rear_ttc_danger"]),
        "rear_ttc_danger_rate": round(counts["rear_ttc_danger"] / decisions, 4),
        "rear_headway_violation_steps": int(counts["rear_headway_violation"]),
        "rear_headway_violation_rate": round(counts["rear_headway_violation"] / decisions, 4),
        "low_speed_blocking_steps": int(counts["low_speed_blocking"]),
        "low_speed_blocking_rate": round(counts["low_speed_blocking"] / decisions, 4),
        "min_ego_speed_mps": round(float(min_speed), 4) if min_speed is not None else None,
        "stopped_ever": bool(counts["stopped"] > 0),
        "stop_steps": int(counts["stopped"]),
        "stop_rate": round(counts["stopped"] / decisions, 4),
        "near_stop_steps": int(counts["near_stop"]),
        "near_stop_rate": round(counts["near_stop"] / decisions, 4),
        "lane_change_count": int(counts["lane_change"]),
        "lane_change_rate": round(counts["lane_change"] / decisions, 4),
        "lane_change_shield_count": int(counts["lane_change_shield"]),
        "lane_change_shield_rate": round(counts["lane_change_shield"] / decisions, 4),
        "lane_change_shield_reason_counts": {},
        "unsafe_lane_change_attempt_count": int(counts["lane_change_shield"]),
        "longitudinal_safety_shield_count": int(counts["longitudinal_safety_shield"]),
        "longitudinal_safety_shield_rate": round(counts["longitudinal_safety_shield"] / decisions, 4),
        "longitudinal_safety_shield_reason_counts": {},
        "unsafe_longitudinal_action_attempt_count": int(counts["longitudinal_safety_shield"]),
        "flow_recovery_shield_count": int(counts["flow_recovery_shield"]),
        "flow_recovery_shield_rate": round(counts["flow_recovery_shield"] / decisions, 4),
        "flow_recovery_reason_counts": {},
        "flap_accel_decel_count": int(counts["flap_accel_decel"]),
        "flap_accel_decel_rate": round(counts["flap_accel_decel"] / decisions, 4),
        "decision_latency_ms_avg": 0.0,
        "p95_decision_latency_sec": 0.0,
        "slow_decision_count": 0,
        "error": error,
        "final_info": copy.deepcopy(final_info),
        "action_sequence": [item["action_id"] for item in action_trace],
        "action_trace": action_trace,
        **expert_meta,
        **benchmark_metrics,
    }
    episode = augment_behavior_aware_benchmark_episode(episode)
    episode = compute_split_scores_for_episode(episode)
    return scrub_non_llm_scores(episode)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            flattened = {}
            for key in keys:
                value = row.get(key)
                flattened[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            writer.writerow(flattened)


def category_summary(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[(str(episode["baseline_policy"]), str(episode.get("category")))].append(episode)
    rows = []
    for (baseline, category), subset in sorted(grouped.items()):
        total = len(subset)
        rows.append(
            {
                "baseline_policy": baseline,
                "category": category,
                "episodes": total,
                "completed": sum(1 for episode in subset if bool(episode.get("task_completed"))),
                "crashes": sum(1 for episode in subset if bool(episode.get("crashed"))),
                "errors": sum(1 for episode in subset if episode.get("error")),
                "avg_ego_speed_mps": round(sum(float(episode.get("avg_ego_speed_mps", 0.0) or 0.0) for episode in subset) / max(total, 1), 4),
                "task_completion_rate": round(sum(1 for episode in subset if bool(episode.get("task_completed"))) / max(total, 1), 4),
                "no_collision_rate": round(sum(1 for episode in subset if bool(episode.get("success_no_collision"))) / max(total, 1), 4),
                "driving_score_balanced_v1": round(sum(float(episode.get("driving_score_balanced_v1", 0.0) or 0.0) for episode in subset) / max(total, 1), 4),
                "driving_task_score_v2": round(sum(float(episode.get("driving_task_score_v2", 0.0) or 0.0) for episode in subset) / max(total, 1), 4),
                "driving_behavior_task_gap_v1": round(sum(float(episode.get("driving_behavior_task_gap_v1", 0.0) or 0.0) for episode in subset) / max(total, 1), 4),
                "driving_score_behavior_v1": round(sum(float(episode.get("driving_score_behavior_v1", 0.0) or 0.0) for episode in subset) / max(total, 1), 4),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Runtime YAML config.")
    parser.add_argument("--benchmark-case-set", default="dilu_highway_reactive_stress_v1")
    parser.add_argument("--benchmark-categories", default=None)
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument(
        "--baseline-levels",
        default=None,
        help="Comma-separated baseline levels to run, e.g. 1,2,3. Overrides --baselines when set.",
    )
    parser.add_argument("--list-baselines", action="store_true", help="List available baselines and exit.")
    parser.add_argument("--limit", type=int, default=None, help="Limit cases after optional category filtering.")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-safety-shields", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Show CLI progress bars.")
    parser.add_argument("--no-progress", action="store_true", help="Disable CLI progress bars.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _is_interactive_output() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _resolve_progress_enabled(args: argparse.Namespace) -> bool:
    if bool(args.progress) and bool(args.no_progress):
        raise ValueError("Use only one of --progress or --no-progress.")
    if bool(args.progress):
        return True
    if bool(args.no_progress):
        return False
    return _is_interactive_output()


def print_baseline_registry() -> None:
    print("Available non-LLM baselines:")
    for spec in iter_baseline_specs():
        print(
            f"- {spec.name} | level={spec.level} | family={spec.family} | "
            f"control_mode={spec.control_mode} | "
            f"category_aware={spec.uses_case_category} | "
            f"criteria_aware={spec.uses_success_criteria} | "
            f"oracle_hidden_spec={spec.uses_hidden_scenario_spec}"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s: %(message)s")
    if bool(args.list_baselines):
        print_baseline_registry()
        return

    progress_enabled = _resolve_progress_enabled(args)
    baseline_levels = parse_baseline_levels(args.baseline_levels)
    baseline_names = resolve_baseline_names(args.baselines, levels=baseline_levels)
    config = load_runtime_config(args.config)
    case_set = filter_cases(load_benchmark_case_set(args.benchmark_case_set), args.benchmark_categories, args.limit)
    env_bundle = resolve_simulation_env_bundle(
        config,
        show_trajectories=False,
        render_agent=False,
        env_id_override=str(case_set.get("target_env_id") or "highway-fast-v0"),
        env_config_overrides=(case_set.get("defaults") or {}).get("env_overrides") or {},
        require_discrete_meta_action=True,
    )
    env_type = str(env_bundle["env_id"])
    validation = validate_benchmark_case_set(case_set, env_bundle["env_config_map"], env_type)
    if not validation.get("passed"):
        raise ValueError(f"Benchmark validation failed: {validation['summary']}")

    experiment_id = args.experiment_id or f"non_llm_baselines_{current_timestamp()}"
    output_root = Path(args.output_root or Path("results") / "baselines" / experiment_id)
    ensure_dir(str(output_root))
    primary_spec = build_primary_metric_spec({**config, "scientific_min_response_strict_format_rate": 0.0})
    fingerprint = build_benchmark_case_set_fingerprint(case_set)
    all_episodes: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []

    progress_context = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        if progress_enabled
        else nullcontext(None)
    )
    case_count = len(case_set["cases"])
    with progress_context as progress:
        baseline_task = progress.add_task("Baselines", total=len(baseline_names)) if progress else None
        overall_task = progress.add_task("Episodes", total=len(baseline_names) * case_count) if progress else None
        for baseline_name in baseline_names:
            policy = BaselinePolicy(baseline_name, config)
            effective_safety_shields = bool((not args.no_safety_shields) and policy.spec.safety_shield_compatible)
            episodes = []
            case_task = progress.add_task(f"Cases ({baseline_name})", total=case_count) if progress else None
            if not progress:
                LOGGER.info("Evaluating baseline=%s cases=%s", baseline_name, case_count)
            for idx, case in enumerate(case_set["cases"], start=1):
                if not progress:
                    LOGGER.info("  case %s/%s: %s", idx, case_count, case["case_id"])
                episode = run_baseline_episode(
                    config=config,
                    env_config_map=env_bundle["env_config_map"],
                    env_type=env_type,
                    case=case,
                    policy=policy,
                    safety_shields_enabled=not bool(args.no_safety_shields),
                )
                episodes.append(episode)
                all_episodes.append(episode)
                if progress:
                    progress.advance(case_task)
                    progress.advance(overall_task)
            if progress:
                progress.remove_task(case_task)
                progress.advance(baseline_task)
            aggregate = aggregate_results(
                baseline_name,
                episodes,
                planned_episode_count=case_count,
                primary_metric_spec=primary_spec,
            )
            aggregate.update(
                {
                    "baseline_policy": baseline_name,
                    "baseline_llm_contract_applicable": False,
                    "baseline_claim_scope": (
                        "expert_behavior_reference_only"
                        if policy.control_mode == EXPERT_CONTROL_MODE
                        else "behavior_reference_only"
                    ),
                    **policy.spec.to_metadata(safety_shields_enabled=effective_safety_shields),
                    "baseline_requested_safety_shields_enabled": not bool(args.no_safety_shields),
                    "benchmark_case_set": case_set["benchmark_name"],
                    "benchmark_fingerprint": fingerprint,
                    "safety_shields_enabled": effective_safety_shields,
                }
            )
            for field in LLM_SCORE_FIELDS:
                aggregate[field] = None
            aggregates.append(aggregate)

    report = {
        "created_at": current_timestamp(),
        "source": "evaluate_non_llm_baselines.py",
        "experiment_id": experiment_id,
        "benchmark_case_set": case_set["benchmark_name"],
        "benchmark_fingerprint": fingerprint,
        "env_id": env_type,
        "baseline_claim_scope": "behavior_reference_only",
        "llm_contract_applicable": False,
        "safety_shields_enabled": not bool(args.no_safety_shields),
        "case_count": len(case_set["cases"]),
        "baselines": baseline_names,
        "baseline_levels_requested": baseline_levels,
        "baseline_registry": [
            {
                "baseline_name": spec.name,
                "baseline_level": spec.level,
                "baseline_family": spec.family,
                "baseline_control_mode": spec.control_mode,
                "uses_case_category": spec.uses_case_category,
                "uses_success_criteria": spec.uses_success_criteria,
                "uses_hidden_scenario_spec": spec.uses_hidden_scenario_spec,
                "uses_future_events": spec.uses_future_events,
                "safety_shield_compatible": spec.safety_shield_compatible,
            }
            for spec in iter_baseline_specs()
        ],
        "validation": validation,
        "aggregates": aggregates,
        "episodes": all_episodes,
    }
    write_json_atomic(str(output_root / "non_llm_baseline_report.json"), json_safe(report))
    write_csv(output_root / "baseline_summary.csv", aggregates)
    write_csv(output_root / "episode_metrics.csv", all_episodes)
    write_csv(output_root / "category_summary.csv", category_summary(all_episodes))
    LOGGER.info("Wrote baseline bundle: %s", output_root)


if __name__ == "__main__":
    main()
