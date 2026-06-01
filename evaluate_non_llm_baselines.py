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
import hashlib
import json
import logging
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import gymnasium as gym
import numpy as np

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
from dilu.runtime.safety_shields import (
    FASTER_ACTION_ID,
    IDLE_ACTION_ID,
    LANE_LEFT_ACTION_ID,
    LANE_RIGHT_ACTION_ID,
    SLOWER_ACTION_ID,
)
from dilu.scenario.envScenario import EnvScenario


LOGGER = logging.getLogger("evaluate_non_llm_baselines")
DEFAULT_BASELINES = ("idle_always", "random_seeded", "keep_lane_cruise", "idm_mobil")
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


@dataclass(frozen=True)
class VehicleSnapshot:
    lane_rank: Optional[int]
    x: Optional[float]
    speed: float


@dataclass(frozen=True)
class LaneSnapshot:
    front_gap_m: Optional[float]
    rear_gap_m: Optional[float]
    front_ttc_sec: Optional[float]
    rear_ttc_sec: Optional[float]
    front_speed_mps: Optional[float]


@dataclass(frozen=True)
class BaselineDecision:
    action_id: int
    reason: str
    metadata: dict[str, Any]


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def available_actions(env: Any) -> list[int]:
    try:
        actions = env.unwrapped.get_available_actions()
        return [int(action_id) for action_id in actions]
    except Exception:
        return [0, 1, 2, 3, 4]


def choose_available(action_id: int, available: Iterable[int], fallback_order: Iterable[int]) -> int:
    available_set = {int(action) for action in available}
    if int(action_id) in available_set:
        return int(action_id)
    for fallback in fallback_order:
        if int(fallback) in available_set:
            return int(fallback)
    return min(available_set) if available_set else IDLE_ACTION_ID


def vehicle_snapshot(vehicle: Any) -> VehicleSnapshot:
    lane_rank = None
    x = None
    try:
        lane_rank = int(getattr(vehicle, "lane_index")[2])
    except Exception:
        pass
    try:
        x = float(getattr(vehicle, "position")[0])
    except Exception:
        pass
    try:
        speed = float(getattr(vehicle, "speed", 0.0) or 0.0)
    except Exception:
        speed = 0.0
    return VehicleSnapshot(lane_rank=lane_rank, x=x, speed=speed)


def lane_count(env: Any) -> int:
    try:
        return int(env.unwrapped.config.get("lanes_count", 3))
    except Exception:
        return 3


def lane_snapshot(env: Any, target_lane_rank: int) -> LaneSnapshot:
    ego = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    ego_state = vehicle_snapshot(ego)
    front_gap = rear_gap = front_ttc = rear_ttc = front_speed = None
    if ego is None or road is None or ego_state.x is None:
        return LaneSnapshot(front_gap, rear_gap, front_ttc, rear_ttc, front_speed)

    vehicles = [vehicle for vehicle in list(getattr(road, "vehicles", []) or []) if vehicle is not ego]
    front_candidates: list[tuple[float, Any]] = []
    rear_candidates: list[tuple[float, Any]] = []
    for vehicle in vehicles:
        state = vehicle_snapshot(vehicle)
        if state.lane_rank != int(target_lane_rank) or state.x is None:
            continue
        delta = float(state.x - ego_state.x)
        if delta >= 0:
            front_candidates.append((delta, vehicle))
        else:
            rear_candidates.append((-delta, vehicle))

    if front_candidates:
        front_gap, front_vehicle = min(front_candidates, key=lambda item: item[0])
        front_state = vehicle_snapshot(front_vehicle)
        front_speed = front_state.speed
        closing_speed = ego_state.speed - front_state.speed
        if closing_speed > 1e-6:
            front_ttc = front_gap / closing_speed
    if rear_candidates:
        rear_gap, rear_vehicle = min(rear_candidates, key=lambda item: item[0])
        rear_state = vehicle_snapshot(rear_vehicle)
        closing_speed = rear_state.speed - ego_state.speed
        if closing_speed > 1e-6:
            rear_ttc = rear_gap / closing_speed

    return LaneSnapshot(front_gap, rear_gap, front_ttc, rear_ttc, front_speed)


def lane_is_safe(snapshot: LaneSnapshot) -> bool:
    front_safe = snapshot.front_gap_m is None or snapshot.front_gap_m >= 14.0
    rear_gap_safe = snapshot.rear_gap_m is None or snapshot.rear_gap_m >= 10.0
    rear_ttc_safe = snapshot.rear_ttc_sec is None or snapshot.rear_ttc_sec >= 2.5
    return bool(front_safe and rear_gap_safe and rear_ttc_safe)


def target_speed_mps(config: dict[str, Any]) -> float:
    reward_range = config.get("reward_speed_range")
    if isinstance(reward_range, list) and len(reward_range) >= 2:
        return float(sum(float(value) for value in reward_range[:2]) / 2.0)
    return 25.0


class BaselinePolicy:
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = str(name)
        self.config = dict(config)
        self.target_speed = target_speed_mps(config)

    def decide(self, env: Any, case: dict[str, Any], step_idx: int, rng: random.Random) -> BaselineDecision:
        available = available_actions(env)
        if self.name == "idle_always":
            return BaselineDecision(
                choose_available(IDLE_ACTION_ID, available, (SLOWER_ACTION_ID, FASTER_ACTION_ID)),
                "fixed_idle",
                {},
            )
        if self.name == "random_seeded":
            return BaselineDecision(int(rng.choice(available)), "seeded_uniform_available_action", {})
        if self.name == "keep_lane_cruise":
            return self._keep_lane_cruise(env, available)
        if self.name == "idm_mobil":
            return self._idm_mobil_style(env, case, available)
        raise ValueError(f"Unknown baseline policy: {self.name}")

    def _keep_lane_cruise(self, env: Any, available: list[int]) -> BaselineDecision:
        ego = getattr(env.unwrapped, "vehicle", None)
        ego_state = vehicle_snapshot(ego)
        current_lane = ego_state.lane_rank if ego_state.lane_rank is not None else 1
        current_lane_snapshot = lane_snapshot(env, current_lane)
        action = IDLE_ACTION_ID
        reason = "target_speed_hold"
        if (
            current_lane_snapshot.front_gap_m is not None
            and current_lane_snapshot.front_gap_m < 12.0
        ) or (
            current_lane_snapshot.front_ttc_sec is not None
            and current_lane_snapshot.front_ttc_sec < 3.0
        ):
            action = SLOWER_ACTION_ID
            reason = "front_gap_or_ttc_caution"
        elif ego_state.speed < self.target_speed - 1.0:
            action = FASTER_ACTION_ID
            reason = "below_target_speed"
        elif ego_state.speed > self.target_speed + 2.0:
            action = SLOWER_ACTION_ID
            reason = "above_target_speed"
        return BaselineDecision(
            choose_available(action, available, (IDLE_ACTION_ID, SLOWER_ACTION_ID, FASTER_ACTION_ID)),
            reason,
            {"target_speed_mps": round(self.target_speed, 3)},
        )

    def _target_lane_action_from_case(self, env: Any, case: dict[str, Any]) -> Optional[int]:
        criteria = case.get("success_criteria") or {}
        if str(criteria.get("type") or "").strip().lower() not in {
            "lane_change",
            "overtake",
            "safe_overtake",
            "delayed_overtake_gap",
            "right_lane_opening_discipline",
        }:
            return None
        direction = str(criteria.get("direction") or criteria.get("target_direction") or "").strip().lower()
        offset = int(criteria.get("target_lane_offset", 0) or 0)
        if direction in {"left", "lane_left"} or offset < 0:
            return LANE_LEFT_ACTION_ID
        if direction in {"right", "lane_right"} or offset > 0:
            return LANE_RIGHT_ACTION_ID
        return None

    def _best_safe_lane_action(self, env: Any, available: list[int], preferred: Optional[int] = None) -> Optional[int]:
        ego = getattr(env.unwrapped, "vehicle", None)
        ego_state = vehicle_snapshot(ego)
        if ego_state.lane_rank is None:
            return None
        candidates: list[tuple[float, int]] = []
        action_to_offset = {LANE_LEFT_ACTION_ID: -1, LANE_RIGHT_ACTION_ID: 1}
        ordered_actions = [preferred] if preferred is not None else []
        ordered_actions.extend([LANE_LEFT_ACTION_ID, LANE_RIGHT_ACTION_ID])
        for action in ordered_actions:
            if action is None or action not in action_to_offset or action not in available:
                continue
            target_lane = ego_state.lane_rank + action_to_offset[action]
            if target_lane < 0 or target_lane >= lane_count(env):
                continue
            snapshot = lane_snapshot(env, target_lane)
            if not lane_is_safe(snapshot):
                continue
            front_gap = snapshot.front_gap_m if snapshot.front_gap_m is not None else 1e6
            rear_gap = snapshot.rear_gap_m if snapshot.rear_gap_m is not None else 1e6
            candidates.append((min(float(front_gap), 120.0) + 0.25 * min(float(rear_gap), 80.0), int(action)))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _idm_mobil_style(self, env: Any, case: dict[str, Any], available: list[int]) -> BaselineDecision:
        ego = getattr(env.unwrapped, "vehicle", None)
        ego_state = vehicle_snapshot(ego)
        current_lane = ego_state.lane_rank if ego_state.lane_rank is not None else 1
        current = lane_snapshot(env, current_lane)
        preferred_lane_action = self._target_lane_action_from_case(env, case)
        safe_target_action = self._best_safe_lane_action(env, available, preferred_lane_action)
        metadata = {
            "target_speed_mps": round(self.target_speed, 3),
            "front_gap_m": None if current.front_gap_m is None else round(current.front_gap_m, 3),
            "front_ttc_sec": None if current.front_ttc_sec is None else round(current.front_ttc_sec, 3),
        }

        severe_front = (
            current.front_gap_m is not None
            and current.front_gap_m < 10.0
        ) or (
            current.front_ttc_sec is not None
            and current.front_ttc_sec < 2.5
        )
        slow_front = (
            current.front_gap_m is not None
            and current.front_gap_m < 28.0
            and current.front_speed_mps is not None
            and current.front_speed_mps + 1.0 < ego_state.speed
        )

        if preferred_lane_action is not None and safe_target_action == preferred_lane_action:
            return BaselineDecision(preferred_lane_action, "case_target_lane_safe", metadata)
        if severe_front and safe_target_action is not None:
            return BaselineDecision(safe_target_action, "mobil_gap_escape", metadata)
        if severe_front:
            return BaselineDecision(
                choose_available(SLOWER_ACTION_ID, available, (IDLE_ACTION_ID,)),
                "idm_brake_for_front_risk",
                metadata,
            )
        if slow_front and safe_target_action is not None:
            return BaselineDecision(safe_target_action, "mobil_overtake_slow_front_vehicle", metadata)
        if ego_state.speed < self.target_speed - 1.0 and (current.front_gap_m is None or current.front_gap_m > 18.0):
            return BaselineDecision(
                choose_available(FASTER_ACTION_ID, available, (IDLE_ACTION_ID,)),
                "idm_accelerate_to_desired_speed",
                metadata,
            )
        if ego_state.speed > self.target_speed + 2.0:
            return BaselineDecision(
                choose_available(SLOWER_ACTION_ID, available, (IDLE_ACTION_ID,)),
                "idm_decelerate_to_desired_speed",
                metadata,
            )
        return BaselineDecision(choose_available(IDLE_ACTION_ID, available, (SLOWER_ACTION_ID,)), "idm_hold_speed", metadata)


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

    try:
        env.unwrapped.configure(case_env_config[env_type])
        _, info = env.reset(seed=seed)
        scenario_meta = apply_highway_scenario_spec(env, case)
        scenario = EnvScenario(env, env_type, seed, database=None, enable_db=False)
        benchmark_evaluator = BenchmarkEpisodeEvaluator(case, env, scenario_spec_metadata=scenario_meta)
        final_info = dict(info or {})

        for step_idx in range(1, max_steps + 1):
            event_meta = apply_highway_scenario_events(
                env,
                case,
                step_idx=step_idx,
                applied_event_ids=applied_event_ids,
            )
            decision_started = time.time()
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
            if safety_shields_enabled:
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
            action_context = {**shield_meta, **event_meta, "action_id": int(action), "final_action_id": int(action)}
            counts["lane_change_shield"] += int(bool(shield_meta.get("lane_change_shield_applied", False)))
            counts["longitudinal_safety_shield"] += int(bool(shield_meta.get("longitudinal_safety_shield_applied", False)))
            counts["flow_recovery_shield"] += int(bool(shield_meta.get("flow_recovery_shield_applied", False)))
            counts["lane_change"] += int(action in (LANE_LEFT_ACTION_ID, LANE_RIGHT_ACTION_ID))
            if previous_action is not None and {previous_action, action} == {FASTER_ACTION_ID, SLOWER_ACTION_ID}:
                counts["flap_accel_decel"] += 1
            previous_action = int(action)

            _, reward, terminated, truncated, info = env.step(action)
            final_info = dict(info or {})
            crashed = bool(final_info.get("crashed", False))
            steps += 1
            reward_sum += float(reward)
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
                    "proposed_action_id": int(proposed_action),
                    "action_id": int(action),
                    "decision_reason": decision.reason,
                    "decision_elapsed_sec": round(decision_elapsed, 6),
                    "shield_applied": bool(shield_meta.get("reactive_safety_shield_applied", False)),
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
        "baseline_claim_scope": "behavior_reference_only",
        "safety_shields_enabled": bool(safety_shields_enabled),
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
        "runtime_parse_path_counts": {"non_llm_baseline": int(steps)},
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
    parser.add_argument("--limit", type=int, default=None, help="Limit cases after optional category filtering.")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-safety-shields", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s: %(message)s")
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

    baseline_names = [name.strip() for name in str(args.baselines).split(",") if name.strip()]
    unknown = sorted(set(baseline_names) - set(DEFAULT_BASELINES))
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}; available={list(DEFAULT_BASELINES)}")

    experiment_id = args.experiment_id or f"non_llm_baselines_{current_timestamp()}"
    output_root = Path(args.output_root or Path("results") / "baselines" / experiment_id)
    ensure_dir(str(output_root))
    primary_spec = build_primary_metric_spec({**config, "scientific_min_response_strict_format_rate": 0.0})
    fingerprint = build_benchmark_case_set_fingerprint(case_set)
    all_episodes: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []

    for baseline_name in baseline_names:
        policy = BaselinePolicy(baseline_name, config)
        episodes = []
        LOGGER.info("Evaluating baseline=%s cases=%s", baseline_name, len(case_set["cases"]))
        for idx, case in enumerate(case_set["cases"], start=1):
            LOGGER.info("  case %s/%s: %s", idx, len(case_set["cases"]), case["case_id"])
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
        aggregate = aggregate_results(
            baseline_name,
            episodes,
            planned_episode_count=len(case_set["cases"]),
            primary_metric_spec=primary_spec,
        )
        aggregate.update(
            {
                "baseline_policy": baseline_name,
                "baseline_llm_contract_applicable": False,
                "baseline_claim_scope": "behavior_reference_only",
                "benchmark_case_set": case_set["benchmark_name"],
                "benchmark_fingerprint": fingerprint,
                "safety_shields_enabled": not bool(args.no_safety_shields),
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
