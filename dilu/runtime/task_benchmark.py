import copy
import hashlib
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from dilu.runtime.highway_scenario_spec import (
    apply_highway_scenario_events,
    apply_highway_scenario_spec,
    scenario_spec_summary,
)
from dilu.runtime.dilu_scoring import SPLIT_SCORE_FIELDS
from dilu.runtime.safety_shields import (
    LANE_CHANGE_ACTIONS,
    TARGET_FRONT_GAP_REQUIRED_M,
    TARGET_REAR_GAP_REQUIRED_M,
    TARGET_REAR_TTC_REQUIRED_SEC,
)


DEFAULT_BENCHMARK_CASE_SET = "lampilot_highway_v1"
DEFAULT_TARGET_ENV_ID = "highway-fast-v0"
BENCHMARK_TTC_SAFE_THRESHOLD_SEC = 2.0
BENCHMARK_SPEED_STD_SAFE_MPS = 4.0
BENCHMARK_OVERALL_WEIGHTS = {
    "ttc": 0.5,
    "speed_variance": 0.3,
    "time_efficiency": 0.2,
}
BENCHMARK_SCORING_POLICY_VERSION = "v2_behavior_aware"
BENCHMARK_RECOMMENDED_HEADLINE_METRIC = "driving_score_v2"
BENCHMARK_BOOTSTRAP_ITERATIONS = 2000
BENCHMARK_BOOTSTRAP_SEED = 20260326
BENCHMARK_V2_ASSERTIVE_CATEGORIES = (
    "follow_gap_decrease",
    "lane_change_left",
    "lane_change_right",
    "overtake_left",
    "overtake_right",
    "speed_increase",
)
BENCHMARK_V2_DEFENSIVE_CATEGORIES = (
    "follow_gap_increase",
    "speed_decrease",
)
BENCHMARK_V2_CONSERVATIVE_PROFILES = {
    "assertive": {
        "stop_rate": {"weight": 0.40, "grace": 0.02},
        "near_stop_rate": {"weight": 0.20, "grace": 0.05},
        "low_speed_blocking_rate": {"weight": 0.40, "grace": 0.05},
    },
    "defensive": {
        "stop_rate": {"weight": 0.20, "grace": 0.10},
        "near_stop_rate": {"weight": 0.10, "grace": 0.20},
        "low_speed_blocking_rate": {"weight": 0.70, "grace": 0.10},
    },
}
BENCHMARK_V2_RUNTIME_PENALTY = {
    "decision_timeout_rate": {"weight": 0.40},
    "fallback_action_rate": {"weight": 0.60},
    "grace": 0.01,
    "cap": 0.25,
}

_ENV_OVERRIDE_ALIASES = {
    "simulation_duration": "duration",
    "vehicle_count": "vehicles_count",
    "other_vehicle_type": "other_vehicles_type",
}
_ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
_STRESS_CRITERIA_TYPES = {
    "cut_in_brake_response",
    "delayed_overtake_gap",
    "closing_rear_lane_change",
    "multi_hazard_recovery",
    "dense_dynamic_flow",
    "right_lane_opening_discipline",
    "squeeze_box_patience",
    "false_alarm_stability",
    "mandatory_overtake_slow_lead",
    "timed_gap_overtake",
    "traffic_jam_escape",
    "traffic_jam_patience",
    "multi_lane_route_discipline",
    "bottleneck_merge_pressure",
    "cut_in_then_recover",
    "false_opening_stability",
    "dense_four_lane_flow",
    "stop_go_wave_response",
}
_STRESS_RATE_FIELDS = {
    "cut_in_brake_response": "cut_in_response_success_rate",
    "delayed_overtake_gap": "delayed_overtake_success_rate",
    "closing_rear_lane_change": "closing_rear_avoidance_success_rate",
    "multi_hazard_recovery": "multi_hazard_recovery_success_rate",
    "dense_dynamic_flow": "dynamic_dense_flow_success_rate",
    "right_lane_opening_discipline": "right_lane_opening_discipline_success_rate",
    "squeeze_box_patience": "squeeze_box_patience_success_rate",
    "false_alarm_stability": "false_alarm_stability_success_rate",
    "mandatory_overtake_slow_lead": "mandatory_overtake_success_rate",
    "timed_gap_overtake": "timed_gap_overtake_success_rate",
    "traffic_jam_escape": "traffic_jam_escape_success_rate",
    "traffic_jam_patience": "traffic_jam_patience_success_rate",
    "multi_lane_route_discipline": "multi_lane_route_discipline_success_rate",
    "bottleneck_merge_pressure": "bottleneck_merge_success_rate",
    "cut_in_then_recover": "cut_in_recovery_success_rate",
    "false_opening_stability": "false_opening_stability_success_rate",
    "dense_four_lane_flow": "dense_four_lane_flow_success_rate",
    "stop_go_wave_response": "stop_go_wave_response_success_rate",
}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _infer_scenario_family_from_env_id(env_id: str) -> str:
    text = str(env_id or "").strip().lower()
    if text.startswith("merge-"):
        return "merge"
    if text.startswith("intersection-"):
        return "intersection"
    if text.startswith("parking-"):
        return "parking"
    return "highway"


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _normalize_env_overrides(raw_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw_overrides = raw_overrides or {}
    normalized: Dict[str, Any] = {}
    for key, value in raw_overrides.items():
        mapped_key = _ENV_OVERRIDE_ALIASES.get(str(key), str(key))
        if isinstance(value, dict):
            normalized[mapped_key] = _normalize_env_overrides(value)
        else:
            normalized[mapped_key] = value
    return normalized


def resolve_benchmark_case_set_path(identifier: str) -> str:
    raw = str(identifier or "").strip()
    if not raw:
        raise ValueError("Benchmark case set identifier cannot be empty.")
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    candidate = os.path.join(_repo_root(), "benchmarks", raw, "cases.json")
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Benchmark case set not found: {identifier}. "
        f"Expected a JSON file path or benchmarks/<name>/cases.json."
    )


def load_benchmark_case_set(identifier: str) -> Dict[str, Any]:
    case_set_path = resolve_benchmark_case_set_path(identifier)
    with open(case_set_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Benchmark case set must be a JSON object.")

    defaults = raw.get("defaults") or {}
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("Benchmark case set must define a non-empty `cases` list.")

    normalized_cases: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases_raw, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Benchmark case #{idx} must be a JSON object.")
        case_id = str(case.get("case_id") or f"case_{idx:03d}").strip()
        category = str(case.get("category") or "").strip()
        instruction = str(case.get("instruction") or "").strip()
        if not category:
            raise ValueError(f"Benchmark case `{case_id}` is missing `category`.")
        if not instruction:
            raise ValueError(f"Benchmark case `{case_id}` is missing `instruction`.")
        if case.get("seed") is None:
            raise ValueError(f"Benchmark case `{case_id}` is missing `seed`.")

        success_criteria = copy.deepcopy(defaults.get("success_criteria") or {})
        if case.get("success_criteria"):
            success_criteria = _deep_update(success_criteria, dict(case["success_criteria"]))

        env_overrides = copy.deepcopy(defaults.get("env_overrides") or {})
        if case.get("env_overrides"):
            env_overrides = _deep_update(env_overrides, dict(case["env_overrides"]))
        env_overrides = _normalize_env_overrides(env_overrides)

        difficulty = str(case.get("difficulty") or defaults.get("difficulty") or "medium").strip().lower()
        if difficulty not in _ALLOWED_DIFFICULTIES:
            raise ValueError(
                f"Benchmark case `{case_id}` has invalid difficulty `{difficulty}`. "
                f"Allowed: {sorted(_ALLOWED_DIFFICULTIES)}"
            )
        case_group = str(case.get("case_group") or defaults.get("case_group") or category).strip() or category

        normalized_case = {
            "case_id": case_id,
            "category": category,
            "instruction": instruction,
            "seed": int(case["seed"]),
            "time_limit_sec": float(case.get("time_limit_sec", defaults.get("time_limit_sec", 12.0))),
            "success_criteria": success_criteria,
            "env_overrides": env_overrides,
            "tags": [str(tag) for tag in (case.get("tags") or [])],
            "difficulty": difficulty,
            "case_group": case_group,
        }
        if case.get("scenario_spec"):
            normalized_case["scenario_spec"] = copy.deepcopy(case["scenario_spec"])
        normalized_cases.append(normalized_case)

    categories = sorted({case["category"] for case in normalized_cases})
    benchmark_name = str(raw.get("benchmark_name") or os.path.basename(os.path.dirname(case_set_path)) or DEFAULT_BENCHMARK_CASE_SET)
    target_env_id = str(raw.get("target_env_id") or DEFAULT_TARGET_ENV_ID).strip() or DEFAULT_TARGET_ENV_ID
    scenario_family = str(raw.get("scenario_family") or _infer_scenario_family_from_env_id(target_env_id)).strip().lower()
    return {
        "benchmark_name": benchmark_name,
        "case_set_path": case_set_path,
        "version": str(raw.get("version") or "1.0"),
        "description": str(raw.get("description") or "").strip(),
        "target_env_id": target_env_id,
        "scenario_family": scenario_family,
        "defaults": defaults,
        "categories": categories,
        "cases": normalized_cases,
    }


def build_benchmark_case_set_fingerprint(case_set: Dict[str, Any]) -> str:
    benchmark_name = str(case_set.get("benchmark_name") or DEFAULT_BENCHMARK_CASE_SET).strip()
    normalized_cases = []
    for case in case_set.get("cases") or []:
        normalized_cases.append(
            {
                "case_id": str(case.get("case_id") or "").strip(),
                "category": str(case.get("category") or "").strip(),
                "instruction": str(case.get("instruction") or "").strip(),
                "seed": int(case.get("seed") or 0),
                "time_limit_sec": float(case.get("time_limit_sec") or 0.0),
                "difficulty": str(case.get("difficulty") or "").strip().lower(),
                "case_group": str(case.get("case_group") or "").strip(),
                "env_overrides": copy.deepcopy(case.get("env_overrides") or {}),
                "success_criteria": copy.deepcopy(case.get("success_criteria") or {}),
                "scenario_spec": copy.deepcopy(case.get("scenario_spec") or {}),
            }
        )
    payload = {
        "benchmark_name": benchmark_name,
        "version": str(case_set.get("version") or ""),
        "target_env_id": str(case_set.get("target_env_id") or DEFAULT_TARGET_ENV_ID).strip(),
        "scenario_family": str(case_set.get("scenario_family") or "").strip().lower(),
        "categories": list(case_set.get("categories") or []),
        "case_count": len(normalized_cases),
        "cases": normalized_cases,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{benchmark_name}:{digest}"


def build_case_env_config(
    base_env_config_map: Dict[str, Dict[str, Any]],
    env_type: str,
    case: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    env_config_map = copy.deepcopy(base_env_config_map)
    env_cfg = copy.deepcopy(env_config_map[env_type])
    overrides = dict((case or {}).get("env_overrides") or {})
    if overrides:
        _deep_update(env_cfg, overrides)

    time_limit_sec = float((case or {}).get("time_limit_sec") or 0.0)
    if time_limit_sec > 0:
        env_cfg["duration"] = max(float(env_cfg.get("duration", 0.0) or 0.0), time_limit_sec)

    if isinstance(env_cfg.get("observation"), dict) and env_cfg.get("vehicles_count") is not None:
        env_cfg["observation"] = dict(env_cfg["observation"])
        env_cfg["observation"]["vehicles_count"] = int(env_cfg["vehicles_count"])

    env_config_map[env_type] = env_cfg
    return env_config_map, env_cfg


def benchmark_max_steps(case: Optional[Dict[str, Any]], env_config_snapshot: Dict[str, Any], default_steps: int) -> int:
    if not case:
        return int(default_steps)
    policy_frequency = float(env_config_snapshot.get("policy_frequency", 1) or 1)
    time_limit_sec = float(case.get("time_limit_sec") or env_config_snapshot.get("duration") or default_steps)
    return max(1, int(math.ceil(time_limit_sec * max(policy_frequency, 1.0))))


def build_benchmark_instruction(case: Dict[str, Any]) -> str:
    return (
        f"Primary task: {case['instruction']} "
        "Complete the task while driving safely, obeying lane boundaries, and avoiding collisions."
    )


def benchmark_metric_config(scenario_family: str = "highway") -> Dict[str, Any]:
    scenario_family = str(scenario_family or "highway").strip().lower()
    recommended_headline_metric = (
        BENCHMARK_RECOMMENDED_HEADLINE_METRIC if scenario_family == "highway" else "driving_score"
    )
    return {
        "ttc_safe_threshold_sec": BENCHMARK_TTC_SAFE_THRESHOLD_SEC,
        "speed_std_safe_mps": BENCHMARK_SPEED_STD_SAFE_MPS,
        "overall_weights": dict(BENCHMARK_OVERALL_WEIGHTS),
        "driving_score_formula": "0 if crashed else completion_rate * overall_score",
        "benchmark_scoring_policy_version": BENCHMARK_SCORING_POLICY_VERSION,
        "recommended_headline_metric": recommended_headline_metric,
        "scenario_family": scenario_family,
        "behavior_aware_v2": {
            "formula": "overall_score_v2 = overall_score * (1 - conservative_penalty_severity_v2) * (1 - runtime_penalty_severity_v2); driving_score_v2 = driving_score * (1 - conservative_penalty_severity_v2) * (1 - runtime_penalty_severity_v2)",
            "category_groups": {
                "assertive": list(BENCHMARK_V2_ASSERTIVE_CATEGORIES),
                "defensive": list(BENCHMARK_V2_DEFENSIVE_CATEGORIES),
            },
            "conservative_profiles": copy.deepcopy(BENCHMARK_V2_CONSERVATIVE_PROFILES),
            "runtime_penalty": copy.deepcopy(BENCHMARK_V2_RUNTIME_PENALTY),
        },
        "bootstrap_iterations": int(BENCHMARK_BOOTSTRAP_ITERATIONS),
        "bootstrap_seed": int(BENCHMARK_BOOTSTRAP_SEED),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm_with_grace(rate: Any, grace: float) -> float:
    return _clamp01((float(rate or 0.0) - float(grace)) / max(1e-9, 1.0 - float(grace)))


def _runtime_norm(rate: Any) -> float:
    grace = float(BENCHMARK_V2_RUNTIME_PENALTY["grace"])
    cap = float(BENCHMARK_V2_RUNTIME_PENALTY["cap"])
    return _clamp01((float(rate or 0.0) - grace) / max(1e-9, cap - grace))


def _conservative_profile_name(category: Any) -> str:
    category_name = str(category or "").strip()
    if category_name in BENCHMARK_V2_DEFENSIVE_CATEGORIES:
        return "defensive"
    return "assertive"


def compute_behavior_aware_penalty_v2(
    *,
    category: Any,
    stop_rate: Any,
    near_stop_rate: Any,
    low_speed_blocking_rate: Any,
    decision_timeout_rate: Any,
    fallback_action_rate: Any,
) -> Dict[str, Any]:
    profile_name = _conservative_profile_name(category)
    profile = BENCHMARK_V2_CONSERVATIVE_PROFILES[profile_name]

    conservative_penalty_severity_v2 = 0.0
    for metric_name, metric_cfg in profile.items():
        conservative_penalty_severity_v2 += float(metric_cfg["weight"]) * _norm_with_grace(
            {
                "stop_rate": stop_rate,
                "near_stop_rate": near_stop_rate,
                "low_speed_blocking_rate": low_speed_blocking_rate,
            }[metric_name],
            float(metric_cfg["grace"]),
        )

    runtime_penalty_severity_v2 = (
        float(BENCHMARK_V2_RUNTIME_PENALTY["decision_timeout_rate"]["weight"]) * _runtime_norm(decision_timeout_rate)
        + float(BENCHMARK_V2_RUNTIME_PENALTY["fallback_action_rate"]["weight"]) * _runtime_norm(fallback_action_rate)
    )

    conservative_penalty_severity_v2 = _clamp01(conservative_penalty_severity_v2)
    runtime_penalty_severity_v2 = _clamp01(runtime_penalty_severity_v2)
    conservative_factor_v2 = 1.0 - conservative_penalty_severity_v2
    runtime_factor_v2 = 1.0 - runtime_penalty_severity_v2
    behavior_penalty_factor_v2 = conservative_factor_v2 * runtime_factor_v2

    return {
        "behavior_penalty_profile_v2": profile_name,
        "conservative_penalty_severity_v2": round(conservative_penalty_severity_v2, 4),
        "runtime_penalty_severity_v2": round(runtime_penalty_severity_v2, 4),
        "behavior_penalty_factor_v2": round(behavior_penalty_factor_v2, 4),
    }


def augment_behavior_aware_benchmark_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    if "task_completed" not in episode:
        return dict(episode)
    if str(episode.get("scenario_family") or "highway").strip().lower() != "highway":
        return dict(episode)

    scored = dict(episode)
    penalty_metrics = compute_behavior_aware_penalty_v2(
        category=scored.get("category"),
        stop_rate=scored.get("stop_rate"),
        near_stop_rate=scored.get("near_stop_rate"),
        low_speed_blocking_rate=scored.get("low_speed_blocking_rate"),
        decision_timeout_rate=scored.get("decision_timeout_rate"),
        fallback_action_rate=scored.get("fallback_action_rate"),
    )
    behavior_penalty_factor_v2 = float(penalty_metrics["behavior_penalty_factor_v2"])
    overall_score = float(scored.get("overall_score", 0.0) or 0.0)
    driving_score = float(scored.get("driving_score", 0.0) or 0.0)

    scored.update(penalty_metrics)
    scored["overall_score_v2"] = round(overall_score * behavior_penalty_factor_v2, 4)
    scored["driving_score_v2"] = round(driving_score * behavior_penalty_factor_v2, 4)
    return scored


def _lane_rank(vehicle) -> Optional[int]:
    lane_index = getattr(vehicle, "lane_index", None)
    if not lane_index or len(lane_index) < 3:
        return None
    try:
        return int(lane_index[2])
    except Exception:
        return None


def _vehicle_x(vehicle) -> Optional[float]:
    if vehicle is None:
        return None
    try:
        return float(vehicle.position[0])
    except Exception:
        return None


def _vehicle_speed(vehicle) -> float:
    try:
        return float(getattr(vehicle, "speed", 0.0) or 0.0)
    except Exception:
        return 0.0


def _ttc_from_gap(gap_m: Optional[float], closing_speed_mps: float) -> Optional[float]:
    if gap_m is None:
        return None
    if float(gap_m) <= 0:
        return 0.0
    if float(closing_speed_mps) <= 1e-6:
        return math.inf
    return float(gap_m) / float(closing_speed_mps)


def _vehicle_by_runtime_id(road, runtime_id: Optional[int]):
    if road is None or runtime_id is None:
        return None
    for vehicle in getattr(road, "vehicles", []):
        if id(vehicle) == runtime_id:
            return vehicle
    return None


def _resolve_direction_offset(criteria: Dict[str, Any]) -> int:
    if "target_lane_offset" in criteria:
        return int(criteria.get("target_lane_offset") or 0)
    direction = str(criteria.get("direction") or criteria.get("target_lane") or "").strip().lower()
    if direction == "left":
        return -1
    if direction == "right":
        return 1
    return 0


def _lane_change_action_for_offset(offset: int) -> Optional[int]:
    for action_id, action_offset in LANE_CHANGE_ACTIONS.items():
        if int(action_offset) == int(offset):
            return int(action_id)
    return None


def _lane_count_from_env(env, vehicles: List[Any]) -> Optional[int]:
    uenv = getattr(env, "unwrapped", env)
    cfg = dict(getattr(uenv, "config", {}) or {})
    if cfg.get("lanes_count") is not None:
        try:
            return int(cfg["lanes_count"])
        except Exception:
            pass
    ranks = [_lane_rank(vehicle) for vehicle in vehicles]
    numeric_ranks = [rank for rank in ranks if rank is not None]
    return max(numeric_ranks) + 1 if numeric_ranks else None


def _nearest_front_rear_gaps(
    *,
    vehicles: List[Any],
    ego,
    target_lane_rank: int,
) -> Tuple[Optional[Any], Optional[Any], Optional[float], Optional[float]]:
    ego_x = _vehicle_x(ego)
    if ego_x is None:
        return None, None, None, None
    front_vehicle = None
    rear_vehicle = None
    front_gap = None
    rear_gap = None
    for vehicle in vehicles:
        if vehicle is ego or _lane_rank(vehicle) != target_lane_rank:
            continue
        vehicle_x = _vehicle_x(vehicle)
        if vehicle_x is None:
            continue
        dx = float(vehicle_x - ego_x)
        if dx >= 0 and (front_gap is None or dx < front_gap):
            front_vehicle = vehicle
            front_gap = dx
        if dx < 0 and (rear_gap is None or abs(dx) < rear_gap):
            rear_vehicle = vehicle
            rear_gap = abs(dx)
    return front_vehicle, rear_vehicle, front_gap, rear_gap


def _optional_float(criteria: Dict[str, Any], key: str) -> Optional[float]:
    value = criteria.get(key)
    if value is None:
        return None
    return float(value)


def _speed_within_optional_band(speed: float, criteria: Dict[str, Any]) -> bool:
    min_speed = _optional_float(criteria, "min_speed_mps")
    max_speed = _optional_float(criteria, "max_speed_mps")
    if min_speed is not None and float(speed) < min_speed:
        return False
    if max_speed is not None and float(speed) > max_speed:
        return False
    return True


def inspect_benchmark_initial_state(env) -> Dict[str, Any]:
    uenv = env.unwrapped
    ego = getattr(uenv, "vehicle", None)
    road = getattr(uenv, "road", None)
    available_actions = list(getattr(uenv, "get_available_actions", lambda: [])())
    front_vehicle = None
    front_gap_m = None
    front_is_ahead = False
    if ego is not None and road is not None:
        front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
        if front_vehicle is not None:
            front_gap_m = float(np.linalg.norm(ego.position - front_vehicle.position))
            ego_x = _vehicle_x(ego)
            front_x = _vehicle_x(front_vehicle)
            front_is_ahead = bool(
                ego_x is not None and front_x is not None and front_x > ego_x
            )
    return {
        "initial_lane_rank": _lane_rank(ego),
        "initial_speed_mps": float(getattr(ego, "speed", 0.0) or 0.0) if ego is not None else None,
        "initial_front_vehicle_exists": bool(front_vehicle is not None),
        "initial_front_gap_m": front_gap_m,
        "initial_front_vehicle_is_ahead": bool(front_is_ahead),
        "available_actions": available_actions,
        "can_change_left": 0 in available_actions,
        "can_change_right": 2 in available_actions,
    }


def validate_benchmark_case(case: Dict[str, Any], initial_state: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    criteria = dict(case.get("success_criteria") or {})
    criteria_type = str(criteria.get("type") or "").strip().lower()
    max_case_steps = max(1, int(math.ceil(float(case.get("time_limit_sec") or 1.0))))
    for event in ((case.get("scenario_spec") or {}).get("events") or []):
        try:
            if int(event.get("step", 0) or 0) > max_case_steps:
                reasons.append("scenario_event_step_out_of_range")
                break
        except Exception:
            reasons.append("invalid_scenario_event_step")
            break

    if criteria_type == "speed_band":
        speed = initial_state.get("initial_speed_mps")
        min_speed = float(criteria.get("min_speed_mps", 0.0))
        max_speed = float(criteria.get("max_speed_mps", 999.0))
        if speed is None:
            reasons.append("missing_initial_speed")
        elif min_speed <= float(speed) <= max_speed:
            reasons.append("initial_speed_inside_target_band")

    elif criteria_type == "front_gap_band":
        if not initial_state.get("initial_front_vehicle_exists"):
            reasons.append("missing_initial_front_vehicle")
        else:
            front_gap_m = initial_state.get("initial_front_gap_m")
            min_gap = float(criteria.get("min_gap_m", 0.0))
            max_gap = float(criteria.get("max_gap_m", 1e9))
            if front_gap_m is None:
                reasons.append("missing_initial_front_gap")
            elif min_gap <= float(front_gap_m) <= max_gap:
                reasons.append("initial_front_gap_inside_target_band")

    elif criteria_type == "lane_change":
        lane_rank = initial_state.get("initial_lane_rank")
        target_offset = _resolve_direction_offset(criteria)
        if target_offset == 0:
            reasons.append("invalid_target_lane_offset")
        elif target_offset < 0 and not initial_state.get("can_change_left"):
            reasons.append("target_left_lane_unavailable")
        elif target_offset > 0 and not initial_state.get("can_change_right"):
            reasons.append("target_right_lane_unavailable")
        elif lane_rank is None:
            reasons.append("missing_initial_lane_rank")
        else:
            target_lane_rank = int(lane_rank) + int(target_offset)
            if target_lane_rank == int(lane_rank):
                reasons.append("ego_already_in_target_lane")

    elif criteria_type == "overtake":
        target_offset = _resolve_direction_offset(criteria)
        if target_offset == 0:
            reasons.append("invalid_target_lane_offset")
        elif target_offset < 0 and not initial_state.get("can_change_left"):
            reasons.append("target_left_lane_unavailable")
        elif target_offset > 0 and not initial_state.get("can_change_right"):
            reasons.append("target_right_lane_unavailable")
        if not initial_state.get("initial_front_vehicle_exists"):
            reasons.append("missing_initial_front_vehicle")
        elif not initial_state.get("initial_front_vehicle_is_ahead"):
            reasons.append("initial_front_vehicle_not_ahead")

    elif criteria_type == "merge_complete":
        lane_rank = initial_state.get("initial_lane_rank")
        target_offset = _resolve_direction_offset(criteria)
        if target_offset == 0:
            reasons.append("invalid_target_lane_offset")
        elif target_offset < 0 and not initial_state.get("can_change_left"):
            reasons.append("target_left_lane_unavailable")
        elif target_offset > 0 and not initial_state.get("can_change_right"):
            reasons.append("target_right_lane_unavailable")
        elif lane_rank is None:
            reasons.append("missing_initial_lane_rank")
        min_speed = _optional_float(criteria, "min_speed_mps")
        max_speed = _optional_float(criteria, "max_speed_mps")
        if min_speed is not None and max_speed is not None and min_speed > max_speed:
            reasons.append("invalid_speed_band")

    elif criteria_type == "arrive":
        if bool(criteria.get("requires_yield", False)):
            min_yield_steps = int(criteria.get("min_yield_steps", 1) or 1)
            yield_speed_mps = _optional_float(criteria, "yield_speed_mps")
            if min_yield_steps < 1:
                reasons.append("invalid_min_yield_steps")
            if yield_speed_mps is None or yield_speed_mps < 0:
                reasons.append("invalid_yield_speed_mps")

    elif criteria_type == "flow_cruise":
        min_speed = _optional_float(criteria, "min_speed_mps")
        max_speed = _optional_float(criteria, "max_speed_mps")
        if min_speed is not None and max_speed is not None and min_speed > max_speed:
            reasons.append("invalid_speed_band")

    elif criteria_type == "safe_overtake":
        target_offset = _resolve_direction_offset(criteria)
        if target_offset == 0:
            reasons.append("invalid_target_lane_offset")
        elif target_offset < 0 and not initial_state.get("can_change_left"):
            reasons.append("target_left_lane_unavailable")
        elif target_offset > 0 and not initial_state.get("can_change_right"):
            reasons.append("target_right_lane_unavailable")
        if not initial_state.get("initial_front_vehicle_exists"):
            reasons.append("missing_initial_front_vehicle")
        elif not initial_state.get("initial_front_vehicle_is_ahead"):
            reasons.append("initial_front_vehicle_not_ahead")

    elif criteria_type == "blocked_lane_patience":
        if not initial_state.get("initial_front_vehicle_exists"):
            reasons.append("missing_initial_front_vehicle")

    elif criteria_type == "post_brake_recovery":
        if not initial_state.get("initial_front_vehicle_exists"):
            reasons.append("missing_initial_front_vehicle")
        min_recovery_speed = _optional_float(criteria, "min_recovery_speed_mps")
        if min_recovery_speed is not None and min_recovery_speed < 0:
            reasons.append("invalid_min_recovery_speed_mps")

    elif criteria_type == "dense_flow":
        min_steps = int(criteria.get("min_survival_steps", 1) or 1)
        if min_steps < 1:
            reasons.append("invalid_min_survival_steps")

    elif criteria_type == "lane_discipline":
        target_offset = _resolve_direction_offset(criteria)
        expect_move = bool(criteria.get("expect_move", target_offset != 0))
        if expect_move and target_offset == 0:
            reasons.append("invalid_target_lane_offset")
        elif target_offset > 0 and not initial_state.get("can_change_right"):
            reasons.append("target_right_lane_unavailable")
        elif target_offset < 0 and not initial_state.get("can_change_left"):
            reasons.append("target_left_lane_unavailable")

    elif criteria_type in _STRESS_CRITERIA_TYPES:
        if criteria_type in {
            "delayed_overtake_gap",
            "closing_rear_lane_change",
            "right_lane_opening_discipline",
            "mandatory_overtake_slow_lead",
            "timed_gap_overtake",
            "traffic_jam_escape",
            "multi_lane_route_discipline",
            "bottleneck_merge_pressure",
        }:
            target_offset = _resolve_direction_offset(criteria)
            if target_offset == 0:
                reasons.append("invalid_target_lane_offset")
            elif target_offset > 0 and not initial_state.get("can_change_right"):
                reasons.append("target_right_lane_unavailable")
            elif target_offset < 0 and not initial_state.get("can_change_left"):
                reasons.append("target_left_lane_unavailable")
        if criteria_type in {
            "delayed_overtake_gap",
            "multi_hazard_recovery",
            "mandatory_overtake_slow_lead",
            "timed_gap_overtake",
            "traffic_jam_escape",
            "traffic_jam_patience",
            "bottleneck_merge_pressure",
            "cut_in_then_recover",
            "stop_go_wave_response",
        }:
            if not initial_state.get("initial_front_vehicle_exists"):
                reasons.append("missing_initial_front_vehicle")
        min_steps = int(criteria.get("min_survival_steps", 1) or 1)
        if min_steps < 1:
            reasons.append("invalid_min_survival_steps")
        opportunity_start = _optional_float(criteria, "opportunity_start_step")
        opportunity_end = _optional_float(criteria, "opportunity_end_step")
        if opportunity_start is not None and opportunity_end is not None and opportunity_start > opportunity_end:
            reasons.append("invalid_opportunity_window")
        min_progress = _optional_float(criteria, "min_progress_m")
        if min_progress is not None and min_progress < 0:
            reasons.append("invalid_min_progress_m")

    else:
        reasons.append(f"unsupported_success_criteria_type:{criteria_type or 'missing'}")

    return reasons


def _validate_scheduled_scenario_events(env: Any, case: Dict[str, Any]) -> Tuple[bool, List[str]]:
    events = list(((case or {}).get("scenario_spec") or {}).get("events") or [])
    if not events:
        return False, []

    applied_event_ids: set = set()
    errors: List[str] = []
    try:
        event_steps = sorted({int(event.get("step", -1)) for event in events})
        for step_idx in event_steps:
            apply_highway_scenario_events(
                env,
                case,
                step_idx=step_idx,
                applied_event_ids=applied_event_ids,
            )
    except Exception as exc:
        errors.append(f"scheduled_event_validation_error:{type(exc).__name__}:{exc}")
    return True, errors


def validate_benchmark_case_set(
    case_set: Dict[str, Any],
    base_env_config_map: Dict[str, Dict[str, Any]],
    env_type: str,
    validate_scheduled_events: bool = True,
) -> Dict[str, Any]:
    target_env_id = str(case_set.get("target_env_id") or "").strip()
    if target_env_id and target_env_id != str(env_type):
        raise ValueError(
            f"Benchmark case set {case_set.get('benchmark_name')!r} targets env_id={target_env_id!r}, "
            f"but evaluation resolved env_id={env_type!r}."
        )
    invalid_cases: List[Dict[str, Any]] = []
    valid_cases: List[Dict[str, Any]] = []
    scheduled_event_validated_case_count = 0

    for case in case_set.get("cases", []):
        case_env_config_map, _ = build_case_env_config(base_env_config_map, env_type, case)
        env = gym.make(env_type, render_mode="rgb_array")
        initial_state: Dict[str, Any] = {}
        reasons: List[str] = []
        try:
            try:
                env.unwrapped.configure(case_env_config_map[env_type])
                env.reset(seed=int(case["seed"]))
                apply_highway_scenario_spec(env, case)
                initial_state = inspect_benchmark_initial_state(env)
                reasons = validate_benchmark_case(case, initial_state)
                if validate_scheduled_events:
                    has_events, event_reasons = _validate_scheduled_scenario_events(env, case)
                    if has_events:
                        scheduled_event_validated_case_count += 1
                    reasons.extend(event_reasons)
            except Exception as exc:
                reasons = [f"scenario_validation_error:{type(exc).__name__}:{exc}"]
        finally:
            env.close()

        item = {
            "case_id": case["case_id"],
            "category": case["category"],
            "seed": int(case["seed"]),
            "difficulty": case.get("difficulty"),
            "case_group": case.get("case_group"),
            "reasons": reasons,
            "initial_state": initial_state,
        }
        if reasons:
            invalid_cases.append(item)
        else:
            valid_cases.append(item)

    summary = {
        "benchmark_name": str(case_set.get("benchmark_name") or ""),
        "total_cases": len(case_set.get("cases", [])),
        "valid_case_count": len(valid_cases),
        "invalid_case_count": len(invalid_cases),
        "valid_categories": sorted({item["category"] for item in valid_cases}),
        "invalid_categories": sorted({item["category"] for item in invalid_cases}),
        "case_group_count": len({str(case.get("case_group") or case.get("category") or "") for case in case_set.get("cases", [])}),
        "scheduled_event_validation_enabled": bool(validate_scheduled_events),
        "scheduled_event_validated_case_count": scheduled_event_validated_case_count,
    }
    return {
        "passed": len(invalid_cases) == 0,
        "invalid_cases": invalid_cases,
        "valid_cases": valid_cases,
        "summary": summary,
    }


def bootstrap_ci95(
    values: List[float],
    *,
    iterations: int = BENCHMARK_BOOTSTRAP_ITERATIONS,
    seed: int = BENCHMARK_BOOTSTRAP_SEED,
) -> Optional[List[float]]:
    if not values:
        return None
    values_arr = np.array(list(values), dtype=float)
    if values_arr.size == 1:
        only = round(float(values_arr[0]), 4)
        return [only, only]
    rng = random.Random(int(seed))
    values_list = [float(value) for value in values_arr.tolist()]
    means = []
    for _ in range(max(1, int(iterations))):
        sample = [values_list[rng.randrange(len(values_list))] for _ in range(len(values_list))]
        means.append(float(sum(sample) / len(sample)))
    lower, upper = np.percentile(np.array(means, dtype=float), [2.5, 97.5])
    return [round(float(lower), 4), round(float(upper), 4)]


def compute_benchmark_case_scores(
    *,
    task_completed: bool,
    crashed: bool,
    min_positive_ttc_sec: Optional[float],
    speed_history: List[float],
    completion_time_sec: Optional[float],
    time_limit_sec: float,
) -> Dict[str, Any]:
    completion_rate = 1.0 if bool(task_completed) else 0.0
    if crashed:
        ttc_score = 0.0
    elif min_positive_ttc_sec is None:
        ttc_score = 1.0
    else:
        ttc_score = max(
            0.0,
            min(1.0, float(min_positive_ttc_sec) / BENCHMARK_TTC_SAFE_THRESHOLD_SEC),
        )

    if len(speed_history) <= 1:
        speed_std_mps = 0.0
    else:
        speed_std_mps = float(np.std(np.array(speed_history, dtype=float)))
    speed_variance_score = max(
        0.0,
        min(1.0, 1.0 - (speed_std_mps / BENCHMARK_SPEED_STD_SAFE_MPS)),
    )

    if completion_time_sec is None or float(time_limit_sec) <= 0:
        time_efficiency_score = 0.0
    else:
        time_efficiency_score = max(
            0.0,
            min(1.0, 1.0 - (float(completion_time_sec) / float(time_limit_sec))),
        )

    overall_score = (
        BENCHMARK_OVERALL_WEIGHTS["ttc"] * ttc_score
        + BENCHMARK_OVERALL_WEIGHTS["speed_variance"] * speed_variance_score
        + BENCHMARK_OVERALL_WEIGHTS["time_efficiency"] * time_efficiency_score
    )
    driving_score = 0.0 if crashed else (completion_rate * overall_score)
    return {
        "completion_rate": round(completion_rate, 4),
        "ttc_score": round(ttc_score, 4),
        "speed_std_mps": round(speed_std_mps, 4),
        "speed_variance_score": round(speed_variance_score, 4),
        "time_efficiency_score": round(time_efficiency_score, 4),
        "overall_score": round(overall_score, 4),
        "driving_score": round(driving_score, 4),
    }


def _mean_metric(episodes: List[Dict[str, Any]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0) or 0.0) for item in episodes) / max(len(episodes), 1))


def _metric_values(episodes: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for episode in episodes:
        if episode.get(key) is None:
            continue
        values.append(float(episode.get(key)))
    return values


def _failure_reason_counts(episodes: List[Dict[str, Any]]) -> Dict[str, int]:
    failure_reasons: Dict[str, int] = {}
    for episode in episodes:
        reason = str(episode.get("benchmark_failure_reason") or "").strip()
        if reason:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    return failure_reasons


def _criteria_subset(episodes: List[Dict[str, Any]], criteria_type: str) -> List[Dict[str, Any]]:
    target = str(criteria_type or "").strip().lower()
    return [
        episode
        for episode in episodes
        if str((episode.get("benchmark_success_criteria") or {}).get("type") or "").strip().lower() == target
    ]


def _task_completion_rate(episodes: List[Dict[str, Any]]) -> Optional[float]:
    if not episodes:
        return None
    return round(sum(1 for episode in episodes if episode.get("task_completed")) / len(episodes), 4)


def _missed_overtake_opportunity_summary(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    opportunity_steps = sum(
        int(episode.get("benchmark_safe_overtake_opportunity_steps", 0) or 0)
        for episode in episodes
    )
    missed_steps = sum(
        int(episode.get("benchmark_missed_overtake_opportunity_steps", 0) or 0)
        for episode in episodes
    )
    return {
        "safe_overtake_opportunity_steps_total": int(opportunity_steps),
        "missed_overtake_opportunity_steps_total": int(missed_steps),
        "missed_overtake_opportunity_rate": (
            round(float(missed_steps) / float(opportunity_steps), 4)
            if opportunity_steps > 0
            else None
        ),
    }


def _v2_decision_pressure_summary(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {}
    passive_trap_count = sum(1 for episode in episodes if bool(episode.get("benchmark_passive_trap_failed", False)))
    opportunity_cases = [
        episode
        for episode in episodes
        if episode.get("benchmark_valid_opportunity_step") is not None
    ]
    timely_cases = [
        episode
        for episode in opportunity_cases
        if bool(episode.get("benchmark_maneuver_in_window", False))
    ]
    return {
        "passive_trap_failure_count": int(passive_trap_count),
        "passive_trap_failure_rate": round(float(passive_trap_count) / max(len(episodes), 1), 4),
        "timely_maneuver_opportunity_count": int(len(opportunity_cases)),
        "timely_maneuver_success_count": int(len(timely_cases)),
        "timely_maneuver_success_rate": (
            round(float(len(timely_cases)) / float(len(opportunity_cases)), 4)
            if opportunity_cases
            else None
        ),
    }


def summarize_benchmark_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {}

    benchmark_total = len(episodes)
    completion_values = [1.0 if episode.get("task_completed") else 0.0 for episode in episodes]
    driving_values = [float(episode.get("driving_score", 0.0) or 0.0) for episode in episodes]
    driving_v2_values = [
        float(episode.get("driving_score_v2"))
        for episode in episodes
        if episode.get("driving_score_v2") is not None
    ]
    overall_v2_values = [
        float(episode.get("overall_score_v2"))
        for episode in episodes
        if episode.get("overall_score_v2") is not None
    ]
    behavior_penalty_values = [
        float(episode.get("behavior_penalty_factor_v2"))
        for episode in episodes
        if episode.get("behavior_penalty_factor_v2") is not None
    ]
    conservative_penalty_values = [
        float(episode.get("conservative_penalty_severity_v2"))
        for episode in episodes
        if episode.get("conservative_penalty_severity_v2") is not None
    ]
    runtime_penalty_values = [
        float(episode.get("runtime_penalty_severity_v2"))
        for episode in episodes
        if episode.get("runtime_penalty_severity_v2") is not None
    ]

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in sorted({str(episode.get("category") or "uncategorized") for episode in episodes}):
        subset = [episode for episode in episodes if str(episode.get("category") or "uncategorized") == category]
        category_summary = {
            "benchmark_case_count": len(subset),
            "task_completion_count": sum(1 for episode in subset if episode.get("task_completed")),
            "task_completion_rate": round(sum(1 for episode in subset if episode.get("task_completed")) / max(len(subset), 1), 4),
            "ttc_score_mean": round(_mean_metric(subset, "ttc_score"), 4),
            "speed_variance_score_mean": round(_mean_metric(subset, "speed_variance_score"), 4),
            "time_efficiency_score_mean": round(_mean_metric(subset, "time_efficiency_score"), 4),
            "overall_score_mean": round(_mean_metric(subset, "overall_score"), 4),
            "driving_score": round(_mean_metric(subset, "driving_score"), 4),
            "benchmark_failure_reasons": _failure_reason_counts(subset),
        }
        category_summary.update(_missed_overtake_opportunity_summary(subset))
        category_summary.update(_v2_decision_pressure_summary(subset))
        category_driving_v2_values = [
            float(episode.get("driving_score_v2"))
            for episode in subset
            if episode.get("driving_score_v2") is not None
        ]
        category_overall_v2_values = [
            float(episode.get("overall_score_v2"))
            for episode in subset
            if episode.get("overall_score_v2") is not None
        ]
        category_behavior_penalty_values = [
            float(episode.get("behavior_penalty_factor_v2"))
            for episode in subset
            if episode.get("behavior_penalty_factor_v2") is not None
        ]
        if category_driving_v2_values:
            category_summary["driving_score_v2"] = round(
                float(np.mean(np.array(category_driving_v2_values, dtype=float))),
                4,
            )
        if category_overall_v2_values:
            category_summary["overall_score_v2_mean"] = round(
                float(np.mean(np.array(category_overall_v2_values, dtype=float))),
                4,
            )
        if category_behavior_penalty_values:
            category_summary["behavior_penalty_factor_v2_mean"] = round(
                float(np.mean(np.array(category_behavior_penalty_values, dtype=float))),
                4,
            )
        for split_field in SPLIT_SCORE_FIELDS:
            split_values = _metric_values(subset, split_field)
            if split_values:
                category_summary[split_field] = round(
                    float(np.mean(np.array(split_values, dtype=float))),
                    4,
                )
        by_category[category] = category_summary

    summary = {
        "benchmark_case_count": benchmark_total,
        "task_completion_count": int(sum(completion_values)),
        "task_completion_rate": round(float(np.mean(np.array(completion_values, dtype=float))), 4),
        "task_completion_rate_ci95": bootstrap_ci95(
            completion_values,
            iterations=BENCHMARK_BOOTSTRAP_ITERATIONS,
            seed=BENCHMARK_BOOTSTRAP_SEED,
        ),
        "ttc_score_mean": round(_mean_metric(episodes, "ttc_score"), 4),
        "speed_variance_score_mean": round(_mean_metric(episodes, "speed_variance_score"), 4),
        "time_efficiency_score_mean": round(_mean_metric(episodes, "time_efficiency_score"), 4),
        "overall_score_mean": round(_mean_metric(episodes, "overall_score"), 4),
        "driving_score": round(float(np.mean(np.array(driving_values, dtype=float))), 4),
        "driving_score_ci95": bootstrap_ci95(
            driving_values,
            iterations=BENCHMARK_BOOTSTRAP_ITERATIONS,
            seed=BENCHMARK_BOOTSTRAP_SEED + 1,
        ),
        "benchmark_failure_reasons": _failure_reason_counts(episodes),
        "benchmark_by_category": by_category,
    }
    summary.update(_missed_overtake_opportunity_summary(episodes))
    summary.update(_v2_decision_pressure_summary(episodes))
    if overall_v2_values:
        summary["overall_score_v2_mean"] = round(
            float(np.mean(np.array(overall_v2_values, dtype=float))),
            4,
        )
    if driving_v2_values:
        summary["driving_score_v2"] = round(
            float(np.mean(np.array(driving_v2_values, dtype=float))),
            4,
        )
        summary["driving_score_v2_ci95"] = bootstrap_ci95(
            driving_v2_values,
            iterations=BENCHMARK_BOOTSTRAP_ITERATIONS,
            seed=BENCHMARK_BOOTSTRAP_SEED + 2,
        )
    if behavior_penalty_values:
        summary["behavior_penalty_factor_v2_mean"] = round(
            float(np.mean(np.array(behavior_penalty_values, dtype=float))),
            4,
        )
    if conservative_penalty_values:
        summary["conservative_penalty_severity_v2_mean"] = round(
            float(np.mean(np.array(conservative_penalty_values, dtype=float))),
            4,
        )
    if runtime_penalty_values:
        summary["runtime_penalty_severity_v2_mean"] = round(
            float(np.mean(np.array(runtime_penalty_values, dtype=float))),
            4,
        )
    for split_index, split_field in enumerate(SPLIT_SCORE_FIELDS):
        split_values = _metric_values(episodes, split_field)
        if split_values:
            summary[split_field] = round(
                float(np.mean(np.array(split_values, dtype=float))),
                4,
            )
            summary[f"{split_field}_ci95"] = bootstrap_ci95(
                split_values,
                iterations=BENCHMARK_BOOTSTRAP_ITERATIONS,
                seed=BENCHMARK_BOOTSTRAP_SEED + 20 + split_index,
            )
    safe_overtake_cases = _criteria_subset(episodes, "safe_overtake")
    post_brake_cases = _criteria_subset(episodes, "post_brake_recovery")
    lane_discipline_cases = _criteria_subset(episodes, "lane_discipline")
    dense_flow_cases = _criteria_subset(episodes, "dense_flow")
    overtake_rate = _task_completion_rate(safe_overtake_cases)
    if overtake_rate is not None:
        summary["overtake_success_rate"] = overtake_rate
        summary["missed_overtake_rate"] = round(1.0 - overtake_rate, 4)
    recovery_rate = _task_completion_rate(post_brake_cases)
    if recovery_rate is not None:
        summary["post_brake_recovery_success_rate"] = recovery_rate
    lane_discipline_rate = _task_completion_rate(lane_discipline_cases)
    if lane_discipline_rate is not None:
        summary["lane_discipline_success_rate"] = lane_discipline_rate
    dense_flow_rate = _task_completion_rate(dense_flow_cases)
    if dense_flow_rate is not None:
        summary["dense_flow_success_rate"] = dense_flow_rate
    for criteria_type, field_name in _STRESS_RATE_FIELDS.items():
        stress_cases = _criteria_subset(episodes, criteria_type)
        stress_rate = _task_completion_rate(stress_cases)
        if stress_rate is not None:
            summary[field_name] = stress_rate
    summary["unsafe_lane_change_attempt_rate"] = round(
        sum(1 for episode in episodes if int(episode.get("benchmark_unsafe_lane_change_attempts", 0) or 0) > 0)
        / max(len(episodes), 1),
        4,
    )
    return summary


def benchmark_result_validity(
    *,
    decision_timeout_rate_mean: Optional[float],
    fallback_action_rate_mean: Optional[float],
    timeout_episode_rate: Optional[float],
) -> Tuple[bool, Optional[str]]:
    reasons: List[str] = []
    if decision_timeout_rate_mean is not None and float(decision_timeout_rate_mean) >= 0.5:
        reasons.append("decision_timeout_rate_mean>=0.5")
    if fallback_action_rate_mean is not None and float(fallback_action_rate_mean) >= 0.5:
        reasons.append("fallback_action_rate_mean>=0.5")
    if timeout_episode_rate is not None and float(timeout_episode_rate) >= 0.5:
        reasons.append("timeout_episode_rate>=0.5")
    if reasons:
        return False, "; ".join(reasons)
    return True, None


class BenchmarkEpisodeEvaluator:
    def __init__(
        self,
        case: Dict[str, Any],
        env,
        scenario_spec_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.case = copy.deepcopy(case)
        self.case_id = str(case["case_id"])
        self.category = str(case["category"])
        self.instruction = str(case["instruction"])
        self.success_criteria = dict(case.get("success_criteria") or {})
        self.time_limit_sec = float(case.get("time_limit_sec") or 0.0)
        self.difficulty = str(case.get("difficulty") or "medium")
        self.case_group = str(case.get("case_group") or self.category)
        self.scenario_family = str(case.get("scenario_family") or _infer_scenario_family_from_env_id(getattr(env.unwrapped, "spec", None).id if getattr(getattr(env.unwrapped, "spec", None), "id", None) else "") or "highway")

        uenv = env.unwrapped
        env_cfg = dict(getattr(uenv, "config", {}) or {})
        self.policy_frequency = float(env_cfg.get("policy_frequency", 1) or 1)
        self.max_steps = benchmark_max_steps(case, env_cfg, default_steps=int(math.ceil(self.time_limit_sec or 1.0)))

        initial_state = inspect_benchmark_initial_state(env)
        self.initial_lane_rank = initial_state.get("initial_lane_rank")
        self.initial_speed_mps = float(initial_state.get("initial_speed_mps") or 0.0)
        self.initial_front_gap_m = initial_state.get("initial_front_gap_m")
        self.initial_front_vehicle_exists = bool(initial_state.get("initial_front_vehicle_exists"))
        self.initial_front_vehicle_is_ahead = bool(initial_state.get("initial_front_vehicle_is_ahead"))
        self.available_actions = list(initial_state.get("available_actions") or [])
        self.initial_x = _vehicle_x(getattr(uenv, "vehicle", None)) or 0.0
        self.initial_front_vehicle_id = None
        self.initial_front_x = None

        ego = getattr(uenv, "vehicle", None)
        road = getattr(uenv, "road", None)
        if ego is not None and road is not None:
            front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
            if front_vehicle is not None:
                self.initial_front_vehicle_id = id(front_vehicle)
                self.initial_front_x = _vehicle_x(front_vehicle)

        self.hold_steps_required = max(1, int(self.success_criteria.get("hold_steps", 2)))
        self.hold_streak = 0
        self.completion_step = None
        self.completion_time_sec = None
        self.task_completed = False
        self.failure_reason = None
        self.visited_left_lane = False
        self.visited_right_lane = False
        self.speed_history: List[float] = []
        self.front_gap_history: List[float] = []
        self.min_positive_ttc_sec = None
        self.max_progress_m = 0.0
        self.last_info: Dict[str, Any] = {}
        self.yield_observed_steps = 0
        self.completion_speed_mps = None
        self.completion_progress_m = None
        self.last_criteria_status: Dict[str, Any] = {}
        self.scenario_spec_report = scenario_spec_summary(self.case)
        self.scenario_spec_metadata = dict(scenario_spec_metadata or {})
        self.overtake_latency_steps = None
        self.recovery_clear_step = None
        self.recovery_time_steps = None
        self.unsafe_lane_change_attempts = 0
        self.benchmark_lane_change_count = 0
        self.flap_accel_decel_count = 0
        self.low_speed_blocking_steps = 0
        self.ttc_danger_steps = 0
        self.headway_violation_steps = 0
        self.previous_final_action_id = None
        self.safe_overtake_opportunity_steps = 0
        self.missed_overtake_opportunity_steps = 0
        self.first_safe_overtake_opportunity_step = None
        self.first_lane_change_attempt_step = None
        self.applied_benchmark_event_ids: List[str] = []
        self.applied_benchmark_event_types: List[str] = []
        self.first_benchmark_event_step = None
        self.first_event_step_by_type: Dict[str, int] = {}
        self.valid_opportunity_step = None
        self.first_maneuver_step = None
        self.maneuver_in_window = False
        self.jam_exit_step = None
        self.bottleneck_avoidance_step = None
        self.recovery_after_wave = False
        self.passive_trap_failed = False
        self.first_brake_action_step = None
        self.first_accel_action_step = None

    def _safe_overtake_opportunity_available(self, env) -> bool:
        if str(self.success_criteria.get("type") or "").strip().lower() != "safe_overtake":
            return False
        if self.task_completed or self.initial_lane_rank is None:
            return False

        ego = getattr(env.unwrapped, "vehicle", None)
        road = getattr(env.unwrapped, "road", None)
        if ego is None or road is None:
            return False
        ego_lane_rank = _lane_rank(ego)
        if ego_lane_rank != self.initial_lane_rank:
            return False

        target_offset = _resolve_direction_offset(self.success_criteria)
        target_lane_rank = int(self.initial_lane_rank) + int(target_offset)
        target_action_id = _lane_change_action_for_offset(target_offset)
        if target_action_id is None or target_action_id not in self.available_actions:
            return False

        vehicles = list(getattr(road, "vehicles", []) or [])
        lane_count = _lane_count_from_env(env, vehicles)
        if lane_count is None or target_lane_rank < 0 or target_lane_rank >= lane_count:
            return False

        target_vehicle = _vehicle_by_runtime_id(road, self.initial_front_vehicle_id)
        ego_x = _vehicle_x(ego)
        target_x = _vehicle_x(target_vehicle)
        if ego_x is None or target_x is None or target_x <= ego_x:
            return False

        front_vehicle, rear_vehicle, front_gap, rear_gap = _nearest_front_rear_gaps(
            vehicles=vehicles,
            ego=ego,
            target_lane_rank=target_lane_rank,
        )
        rear_ttc = (
            _ttc_from_gap(rear_gap, _vehicle_speed(rear_vehicle) - _vehicle_speed(ego))
            if rear_vehicle is not None
            else None
        )
        front_safe = front_gap is None or front_gap >= TARGET_FRONT_GAP_REQUIRED_M
        rear_gap_safe = rear_gap is None or rear_gap >= TARGET_REAR_GAP_REQUIRED_M
        rear_ttc_safe = rear_ttc is None or rear_ttc >= TARGET_REAR_TTC_REQUIRED_SEC
        return bool(front_safe and rear_gap_safe and rear_ttc_safe)

    def _opportunity_window(self) -> Tuple[Optional[int], Optional[int]]:
        start = self.success_criteria.get("opportunity_start_step")
        end = self.success_criteria.get("opportunity_end_step")
        try:
            start_step = int(start) if start is not None else None
        except Exception:
            start_step = None
        try:
            end_step = int(end) if end is not None else None
        except Exception:
            end_step = None
        return start_step, end_step

    def _step_in_opportunity_window(self, step_idx: int) -> bool:
        start_step, end_step = self._opportunity_window()
        if start_step is None and end_step is None:
            return False
        if start_step is not None and int(step_idx) < start_step:
            return False
        if end_step is not None and int(step_idx) > end_step:
            return False
        return True

    def _target_lane_reached(self, lane_rank: Optional[int]) -> bool:
        if lane_rank is None or self.initial_lane_rank is None:
            return False
        target_offset = _resolve_direction_offset(self.success_criteria)
        if target_offset == 0:
            return False
        return int(lane_rank) == int(self.initial_lane_rank) + int(target_offset)

    def _required_lane_used(self) -> bool:
        target_offset = _resolve_direction_offset(self.success_criteria)
        if target_offset < 0:
            return bool(self.visited_left_lane)
        if target_offset > 0:
            return bool(self.visited_right_lane)
        return False

    def _initial_front_passed(self, ego: Any, road: Any, pass_margin_m: float) -> bool:
        target_vehicle = _vehicle_by_runtime_id(road, self.initial_front_vehicle_id)
        target_x = _vehicle_x(target_vehicle)
        ego_x = _vehicle_x(ego)
        return bool(
            target_x is not None
            and ego_x is not None
            and target_x <= (ego_x - float(pass_margin_m))
        )

    def _arrived(self, env, ego) -> bool:
        has_arrived = getattr(env.unwrapped, "has_arrived", None)
        if callable(has_arrived) and ego is not None:
            try:
                return bool(has_arrived(ego))
            except TypeError:
                return bool(has_arrived())
        return bool((self.last_info or {}).get("is_success"))

    def _criteria_status(self, env, step_metrics: Dict[str, Any]) -> Dict[str, Any]:
        ego = getattr(env.unwrapped, "vehicle", None)
        road = getattr(env.unwrapped, "road", None)
        lane_rank = _lane_rank(ego)
        current_speed = float(getattr(ego, "speed", 0.0) or 0.0) if ego is not None else 0.0
        front_gap_m = step_metrics.get("front_gap_m")
        criteria_type = str(self.success_criteria.get("type") or "").strip().lower()
        status: Dict[str, Any] = {
            "criteria_type": criteria_type,
            "task_predicate_satisfied": False,
        }

        if criteria_type == "speed_band":
            min_speed = float(self.success_criteria.get("min_speed_mps", 0.0))
            max_speed = float(self.success_criteria.get("max_speed_mps", 999.0))
            speed_in_band = min_speed <= current_speed <= max_speed
            status.update({"speed_in_band": bool(speed_in_band), "task_predicate_satisfied": bool(speed_in_band)})
            return status

        if criteria_type == "front_gap_band":
            if front_gap_m is None:
                status.update({"front_gap_in_band": False})
                return status
            min_gap = float(self.success_criteria.get("min_gap_m", 0.0))
            max_gap = float(self.success_criteria.get("max_gap_m", 1e9))
            front_gap_in_band = min_gap <= float(front_gap_m) <= max_gap
            status.update(
                {"front_gap_in_band": bool(front_gap_in_band), "task_predicate_satisfied": bool(front_gap_in_band)}
            )
            return status

        if criteria_type == "lane_change":
            if lane_rank is None or self.initial_lane_rank is None:
                status.update({"target_lane_reached": False})
                return status
            target_offset = _resolve_direction_offset(self.success_criteria)
            target_lane_rank = self.initial_lane_rank + target_offset
            target_lane_reached = lane_rank == target_lane_rank
            status.update(
                {"target_lane_reached": bool(target_lane_reached), "task_predicate_satisfied": bool(target_lane_reached)}
            )
            return status

        if criteria_type == "overtake":
            if lane_rank is None or self.initial_lane_rank is None:
                status.update({"required_lane_used": False, "pass_margin_satisfied": False})
                return status
            target_offset = _resolve_direction_offset(self.success_criteria)
            used_required_lane = (
                self.visited_left_lane if target_offset < 0 else self.visited_right_lane
            )
            target_vehicle = _vehicle_by_runtime_id(road, self.initial_front_vehicle_id)
            if target_vehicle is None or ego is None:
                status.update({"required_lane_used": bool(used_required_lane), "pass_margin_satisfied": False})
                return status
            target_x = _vehicle_x(target_vehicle)
            ego_x = _vehicle_x(ego)
            if target_x is None or ego_x is None:
                status.update({"required_lane_used": bool(used_required_lane), "pass_margin_satisfied": False})
                return status
            pass_margin_m = float(self.success_criteria.get("pass_margin_m", 5.0))
            pass_margin_satisfied = target_x <= (ego_x - pass_margin_m)
            task_satisfied = bool(used_required_lane and pass_margin_satisfied)
            status.update(
                {
                    "required_lane_used": bool(used_required_lane),
                    "pass_margin_satisfied": bool(pass_margin_satisfied),
                    "task_predicate_satisfied": task_satisfied,
                }
            )
            return status

        if criteria_type == "merge_complete":
            if lane_rank is None or self.initial_lane_rank is None:
                status.update(
                    {
                        "target_lane_reached": False,
                        "merge_progress_satisfied": False,
                        "speed_band_satisfied": False,
                    }
                )
                return status
            target_offset = _resolve_direction_offset(self.success_criteria)
            target_lane_rank = self.initial_lane_rank + target_offset
            min_progress_m = float(self.success_criteria.get("min_progress_m", 0.0) or 0.0)
            target_lane_reached = lane_rank == target_lane_rank
            progress_satisfied = float(self.max_progress_m) >= min_progress_m
            speed_satisfied = _speed_within_optional_band(current_speed, self.success_criteria)
            task_satisfied = bool(target_lane_reached and progress_satisfied and speed_satisfied)
            status.update(
                {
                    "target_lane_reached": bool(target_lane_reached),
                    "merge_progress_satisfied": bool(progress_satisfied),
                    "speed_band_satisfied": bool(speed_satisfied),
                    "task_predicate_satisfied": task_satisfied,
                }
            )
            return status

        if criteria_type == "arrive":
            arrived = self._arrived(env, ego)
            requires_yield = bool(self.success_criteria.get("requires_yield", False))
            min_yield_steps = max(1, int(self.success_criteria.get("min_yield_steps", 1) or 1))
            yield_satisfied = (not requires_yield) or self.yield_observed_steps >= min_yield_steps
            task_satisfied = bool(arrived and yield_satisfied)
            status.update(
                {
                    "arrived": bool(arrived),
                    "requires_yield": bool(requires_yield),
                    "yield_satisfied": bool(yield_satisfied),
                    "task_predicate_satisfied": task_satisfied,
                }
            )
            return status

        if criteria_type == "flow_cruise":
            min_steps = max(1, int(self.success_criteria.get("min_survival_steps", self.hold_steps_required) or 1))
            max_lane_changes = int(self.success_criteria.get("max_lane_changes", 0) or 0)
            max_low_speed_steps = int(self.success_criteria.get("max_low_speed_blocking_steps", 0) or 0)
            speed_satisfied = _speed_within_optional_band(current_speed, self.success_criteria)
            survival_satisfied = len(self.speed_history) >= min_steps
            lane_satisfied = self.benchmark_lane_change_count <= max_lane_changes
            low_speed_satisfied = self.low_speed_blocking_steps <= max_low_speed_steps
            task_satisfied = bool(speed_satisfied and survival_satisfied and lane_satisfied and low_speed_satisfied)
            status.update(
                {
                    "speed_band_satisfied": bool(speed_satisfied),
                    "survival_satisfied": bool(survival_satisfied),
                    "lane_change_satisfied": bool(lane_satisfied),
                    "low_speed_blocking_satisfied": bool(low_speed_satisfied),
                    "task_predicate_satisfied": task_satisfied,
                }
            )
            return status

        if criteria_type == "safe_overtake":
            if lane_rank is None or self.initial_lane_rank is None:
                status.update({"required_lane_used": False, "pass_margin_satisfied": False})
                return status
            target_offset = _resolve_direction_offset(self.success_criteria)
            used_required_lane = self.visited_left_lane if target_offset < 0 else self.visited_right_lane
            target_vehicle = _vehicle_by_runtime_id(road, self.initial_front_vehicle_id)
            if target_vehicle is None or ego is None:
                status.update({"required_lane_used": bool(used_required_lane), "pass_margin_satisfied": False})
                return status
            target_x = _vehicle_x(target_vehicle)
            ego_x = _vehicle_x(ego)
            if target_x is None or ego_x is None:
                status.update({"required_lane_used": bool(used_required_lane), "pass_margin_satisfied": False})
                return status
            pass_margin_m = float(self.success_criteria.get("pass_margin_m", 8.0))
            min_final_speed = float(self.success_criteria.get("min_final_speed_mps", 0.0) or 0.0)
            pass_margin_satisfied = target_x <= (ego_x - pass_margin_m)
            speed_satisfied = current_speed >= min_final_speed
            shield_satisfied = self.unsafe_lane_change_attempts == 0
            task_satisfied = bool(used_required_lane and pass_margin_satisfied and speed_satisfied and shield_satisfied)
            status.update(
                {
                    "required_lane_used": bool(used_required_lane),
                    "pass_margin_satisfied": bool(pass_margin_satisfied),
                    "speed_band_satisfied": bool(speed_satisfied),
                    "unsafe_attempt_satisfied": bool(shield_satisfied),
                    "task_predicate_satisfied": task_satisfied,
                }
            )
            return status

        if criteria_type == "blocked_lane_patience":
            min_steps = max(1, int(self.success_criteria.get("min_survival_steps", self.hold_steps_required) or 1))
            min_speed = float(self.success_criteria.get("min_speed_mps", 0.0) or 0.0)
            max_unsafe = int(self.success_criteria.get("max_unsafe_lane_change_attempts", 0) or 0)
            survival_satisfied = len(self.speed_history) >= min_steps
            speed_satisfied = current_speed >= min_speed
            unsafe_satisfied = self.unsafe_lane_change_attempts <= max_unsafe
            task_satisfied = bool(survival_satisfied and speed_satisfied and unsafe_satisfied)
            status.update(
                {
                    "survival_satisfied": bool(survival_satisfied),
                    "speed_band_satisfied": bool(speed_satisfied),
                    "unsafe_attempt_satisfied": bool(unsafe_satisfied),
                    "task_predicate_satisfied": task_satisfied,
                }
            )
            return status

        if criteria_type == "post_brake_recovery":
            min_recovery_speed = float(self.success_criteria.get("min_recovery_speed_mps", 22.0) or 22.0)
            recovery_satisfied = self.recovery_clear_step is not None and current_speed >= min_recovery_speed
            task_satisfied = bool(recovery_satisfied and self.unsafe_lane_change_attempts == 0)
            status.update(
                {
                    "recovery_clear_observed": bool(self.recovery_clear_step is not None),
                    "recovery_speed_satisfied": bool(current_speed >= min_recovery_speed),
                    "unsafe_attempt_satisfied": bool(self.unsafe_lane_change_attempts == 0),
                    "task_predicate_satisfied": task_satisfied,
                }
            )
            return status

        if criteria_type == "dense_flow":
            min_steps = max(1, int(self.success_criteria.get("min_survival_steps", self.hold_steps_required) or 1))
            min_avg_speed = float(self.success_criteria.get("min_avg_speed_mps", 18.0) or 18.0)
            max_flaps = int(self.success_criteria.get("max_flap_accel_decel_count", 2) or 2)
            max_ttc_rate = float(self.success_criteria.get("max_ttc_danger_rate", 0.2) or 0.2)
            max_headway_rate = float(self.success_criteria.get("max_headway_violation_rate", 0.4) or 0.4)
            steps = max(len(self.speed_history), 1)
            avg_speed = float(np.mean(np.array(self.speed_history, dtype=float))) if self.speed_history else 0.0
            ttc_rate = float(self.ttc_danger_steps) / steps
            headway_rate = float(self.headway_violation_steps) / steps
            task_satisfied = bool(
                len(self.speed_history) >= min_steps
                and avg_speed >= min_avg_speed
                and self.flap_accel_decel_count <= max_flaps
                and ttc_rate <= max_ttc_rate
                and headway_rate <= max_headway_rate
            )
            status.update(
                {
                    "survival_satisfied": bool(len(self.speed_history) >= min_steps),
                    "avg_speed_satisfied": bool(avg_speed >= min_avg_speed),
                    "flap_satisfied": bool(self.flap_accel_decel_count <= max_flaps),
                    "ttc_rate_satisfied": bool(ttc_rate <= max_ttc_rate),
                    "headway_rate_satisfied": bool(headway_rate <= max_headway_rate),
                    "task_predicate_satisfied": task_satisfied,
                }
            )
            return status

        if criteria_type == "lane_discipline":
            if lane_rank is None or self.initial_lane_rank is None:
                status.update({"target_lane_reached": False})
                return status
            target_offset = _resolve_direction_offset(self.success_criteria)
            expect_move = bool(self.success_criteria.get("expect_move", target_offset != 0))
            max_unsafe = int(self.success_criteria.get("max_unsafe_lane_change_attempts", 0) or 0)
            min_speed = float(self.success_criteria.get("min_speed_mps", 0.0) or 0.0)
            target_lane_reached = lane_rank == (self.initial_lane_rank + target_offset)
            lane_satisfied = target_lane_reached if expect_move else lane_rank == self.initial_lane_rank
            unsafe_satisfied = self.unsafe_lane_change_attempts <= max_unsafe
            speed_satisfied = current_speed >= min_speed
            task_satisfied = bool(lane_satisfied and unsafe_satisfied and speed_satisfied)
            status.update(
                {
                    "target_lane_reached": bool(target_lane_reached),
                    "lane_discipline_satisfied": bool(lane_satisfied),
                    "unsafe_attempt_satisfied": bool(unsafe_satisfied),
                    "speed_band_satisfied": bool(speed_satisfied),
                    "task_predicate_satisfied": task_satisfied,
                }
            )
            return status

        if criteria_type in _STRESS_CRITERIA_TYPES:
            steps = max(len(self.speed_history), 1)
            min_steps = max(1, int(self.success_criteria.get("min_survival_steps", self.hold_steps_required) or 1))
            max_ttc_rate = float(self.success_criteria.get("max_ttc_danger_rate", 0.25) or 0.25)
            max_headway_rate = float(self.success_criteria.get("max_headway_violation_rate", 0.4) or 0.4)
            max_flaps = int(self.success_criteria.get("max_flap_accel_decel_count", 4))
            max_lane_changes = int(self.success_criteria.get("max_lane_changes", 99))
            max_unsafe = int(self.success_criteria.get("max_unsafe_lane_change_attempts", 0) or 0)
            min_avg_speed = float(self.success_criteria.get("min_avg_speed_mps", 0.0) or 0.0)
            avg_speed = float(np.mean(np.array(self.speed_history, dtype=float))) if self.speed_history else 0.0
            ttc_rate = float(self.ttc_danger_steps) / steps
            headway_rate = float(self.headway_violation_steps) / steps
            event_satisfied = bool(self.applied_benchmark_event_ids) or not bool(
                self.success_criteria.get("requires_event", True)
            )
            survival_satisfied = len(self.speed_history) >= min_steps
            safety_satisfied = (
                self.unsafe_lane_change_attempts <= max_unsafe
                and ttc_rate <= max_ttc_rate
                and headway_rate <= max_headway_rate
            )
            smooth_satisfied = self.flap_accel_decel_count <= max_flaps
            lane_change_satisfied = self.benchmark_lane_change_count <= max_lane_changes
            avg_speed_satisfied = avg_speed >= min_avg_speed
            speed_band_satisfied = _speed_within_optional_band(current_speed, self.success_criteria)
            task_satisfied = bool(
                event_satisfied
                and survival_satisfied
                and safety_satisfied
                and smooth_satisfied
                and lane_change_satisfied
                and avg_speed_satisfied
                and speed_band_satisfied
            )

            if criteria_type == "cut_in_brake_response":
                max_current_speed = _optional_float(self.success_criteria, "max_post_event_speed_mps")
                if max_current_speed is not None and self.applied_benchmark_event_ids:
                    speed_band_satisfied = current_speed <= max_current_speed
                    task_satisfied = bool(task_satisfied and speed_band_satisfied)
            elif criteria_type == "delayed_overtake_gap":
                target_offset = _resolve_direction_offset(self.success_criteria)
                used_required_lane = self.visited_left_lane if target_offset < 0 else self.visited_right_lane
                target_vehicle = _vehicle_by_runtime_id(road, self.initial_front_vehicle_id)
                target_x = _vehicle_x(target_vehicle)
                ego_x = _vehicle_x(ego)
                pass_margin_m = float(self.success_criteria.get("pass_margin_m", 8.0) or 8.0)
                pass_margin_satisfied = bool(
                    target_x is not None and ego_x is not None and target_x <= (ego_x - pass_margin_m)
                )
                min_final_speed = float(self.success_criteria.get("min_final_speed_mps", 20.0) or 20.0)
                final_speed_satisfied = current_speed >= min_final_speed
                task_satisfied = bool(
                    event_satisfied
                    and used_required_lane
                    and pass_margin_satisfied
                    and final_speed_satisfied
                    and safety_satisfied
                )
                status.update(
                    {
                        "required_lane_used": bool(used_required_lane),
                        "pass_margin_satisfied": bool(pass_margin_satisfied),
                        "final_speed_satisfied": bool(final_speed_satisfied),
                    }
                )
            elif criteria_type == "closing_rear_lane_change":
                task_satisfied = bool(task_satisfied and self.benchmark_lane_change_count <= max_lane_changes)
            elif criteria_type == "multi_hazard_recovery":
                min_recovery_speed = float(self.success_criteria.get("min_recovery_speed_mps", 20.0) or 20.0)
                recovery_satisfied = self.recovery_clear_step is not None and current_speed >= min_recovery_speed
                task_satisfied = bool(task_satisfied and recovery_satisfied)
                status.update(
                    {
                        "recovery_clear_observed": bool(self.recovery_clear_step is not None),
                        "recovery_speed_satisfied": bool(current_speed >= min_recovery_speed),
                    }
                )
            elif criteria_type == "right_lane_opening_discipline":
                if lane_rank is None or self.initial_lane_rank is None:
                    task_satisfied = False
                    target_lane_reached = False
                else:
                    target_offset = _resolve_direction_offset(self.success_criteria)
                    target_lane_reached = lane_rank == (self.initial_lane_rank + target_offset)
                    task_satisfied = bool(task_satisfied and target_lane_reached)
                status.update({"target_lane_reached": bool(target_lane_reached)})
            elif criteria_type == "false_alarm_stability":
                task_satisfied = bool(task_satisfied and self.benchmark_lane_change_count == 0)
            elif criteria_type in {"mandatory_overtake_slow_lead", "timed_gap_overtake"}:
                pass_margin_m = float(self.success_criteria.get("pass_margin_m", 10.0) or 10.0)
                min_final_speed = float(self.success_criteria.get("min_final_speed_mps", 22.0) or 22.0)
                min_progress_m = float(self.success_criteria.get("min_progress_m", 80.0) or 80.0)
                required_lane_used = self._required_lane_used()
                pass_margin_satisfied = self._initial_front_passed(ego, road, pass_margin_m)
                final_speed_satisfied = current_speed >= min_final_speed
                progress_satisfied = float(self.max_progress_m) >= min_progress_m
                window_satisfied = (
                    bool(self.maneuver_in_window)
                    if criteria_type == "timed_gap_overtake"
                    else True
                )
                task_satisfied = bool(
                    event_satisfied
                    and safety_satisfied
                    and required_lane_used
                    and pass_margin_satisfied
                    and final_speed_satisfied
                    and progress_satisfied
                    and window_satisfied
                )
                status.update(
                    {
                        "required_lane_used": bool(required_lane_used),
                        "pass_margin_satisfied": bool(pass_margin_satisfied),
                        "final_speed_satisfied": bool(final_speed_satisfied),
                        "progress_satisfied": bool(progress_satisfied),
                        "maneuver_in_window_satisfied": bool(window_satisfied),
                    }
                )
            elif criteria_type == "traffic_jam_escape":
                min_final_speed = float(self.success_criteria.get("min_final_speed_mps", 18.0) or 18.0)
                min_progress_m = float(self.success_criteria.get("min_progress_m", 65.0) or 65.0)
                jam_exit_satisfied = self.jam_exit_step is not None
                final_speed_satisfied = current_speed >= min_final_speed
                progress_satisfied = float(self.max_progress_m) >= min_progress_m
                task_satisfied = bool(
                    event_satisfied
                    and safety_satisfied
                    and jam_exit_satisfied
                    and final_speed_satisfied
                    and progress_satisfied
                )
                status.update(
                    {
                        "jam_exit_satisfied": bool(jam_exit_satisfied),
                        "final_speed_satisfied": bool(final_speed_satisfied),
                        "progress_satisfied": bool(progress_satisfied),
                    }
                )
            elif criteria_type == "traffic_jam_patience":
                safe_start = int(self.success_criteria.get("safe_window_start_step", 9999) or 9999)
                no_early_maneuver = (
                    self.first_maneuver_step is None
                    or int(self.first_maneuver_step) >= safe_start
                )
                min_progress_m = float(self.success_criteria.get("min_progress_m", 35.0) or 35.0)
                progress_satisfied = float(self.max_progress_m) >= min_progress_m
                task_satisfied = bool(
                    event_satisfied
                    and safety_satisfied
                    and no_early_maneuver
                    and progress_satisfied
                    and lane_change_satisfied
                )
                status.update(
                    {
                        "no_early_maneuver_satisfied": bool(no_early_maneuver),
                        "progress_satisfied": bool(progress_satisfied),
                    }
                )
            elif criteria_type == "multi_lane_route_discipline":
                min_progress_m = float(self.success_criteria.get("min_progress_m", 70.0) or 70.0)
                target_lane_reached = self._target_lane_reached(lane_rank)
                progress_satisfied = float(self.max_progress_m) >= min_progress_m
                task_satisfied = bool(
                    event_satisfied
                    and safety_satisfied
                    and target_lane_reached
                    and progress_satisfied
                    and avg_speed_satisfied
                )
                status.update(
                    {
                        "target_lane_reached": bool(target_lane_reached),
                        "progress_satisfied": bool(progress_satisfied),
                    }
                )
            elif criteria_type == "bottleneck_merge_pressure":
                latest_step = int(self.success_criteria.get("latest_maneuver_step", 10) or 10)
                min_progress_m = float(self.success_criteria.get("min_progress_m", 70.0) or 70.0)
                bottleneck_avoidance_satisfied = (
                    self.bottleneck_avoidance_step is not None
                    and int(self.bottleneck_avoidance_step) <= latest_step
                )
                progress_satisfied = float(self.max_progress_m) >= min_progress_m
                task_satisfied = bool(
                    event_satisfied
                    and safety_satisfied
                    and bottleneck_avoidance_satisfied
                    and progress_satisfied
                )
                status.update(
                    {
                        "bottleneck_avoidance_satisfied": bool(bottleneck_avoidance_satisfied),
                        "progress_satisfied": bool(progress_satisfied),
                    }
                )
            elif criteria_type == "cut_in_then_recover":
                min_recovery_speed = float(self.success_criteria.get("min_recovery_speed_mps", 20.0) or 20.0)
                requires_brake_action = bool(self.success_criteria.get("requires_brake_action", True))
                brake_satisfied = (not requires_brake_action) or self.first_brake_action_step is not None
                recovery_satisfied = self.recovery_clear_step is not None and current_speed >= min_recovery_speed
                task_satisfied = bool(event_satisfied and safety_satisfied and brake_satisfied and recovery_satisfied)
                status.update(
                    {
                        "brake_action_satisfied": bool(brake_satisfied),
                        "recovery_clear_observed": bool(self.recovery_clear_step is not None),
                        "recovery_speed_satisfied": bool(current_speed >= min_recovery_speed),
                    }
                )
            elif criteria_type == "false_opening_stability":
                task_satisfied = bool(
                    event_satisfied
                    and safety_satisfied
                    and smooth_satisfied
                    and self.benchmark_lane_change_count == 0
                    and avg_speed_satisfied
                )
            elif criteria_type == "dense_four_lane_flow":
                task_satisfied = bool(
                    event_satisfied
                    and survival_satisfied
                    and safety_satisfied
                    and smooth_satisfied
                    and avg_speed_satisfied
                    and lane_change_satisfied
                )
            elif criteria_type == "stop_go_wave_response":
                min_recovery_speed = float(self.success_criteria.get("min_recovery_speed_mps", 18.0) or 18.0)
                min_progress_m = float(self.success_criteria.get("min_progress_m", 45.0) or 45.0)
                recovery_satisfied = self.recovery_clear_step is not None and current_speed >= min_recovery_speed
                progress_satisfied = float(self.max_progress_m) >= min_progress_m
                task_satisfied = bool(
                    event_satisfied
                    and safety_satisfied
                    and recovery_satisfied
                    and progress_satisfied
                    and smooth_satisfied
                )
                status.update(
                    {
                        "recovery_clear_observed": bool(self.recovery_clear_step is not None),
                        "recovery_speed_satisfied": bool(current_speed >= min_recovery_speed),
                        "progress_satisfied": bool(progress_satisfied),
                    }
                )

            status.update(
                {
                    "event_satisfied": bool(event_satisfied),
                    "survival_satisfied": bool(survival_satisfied),
                    "safety_satisfied": bool(safety_satisfied),
                    "smooth_satisfied": bool(smooth_satisfied),
                    "lane_change_satisfied": bool(lane_change_satisfied),
                    "avg_speed_satisfied": bool(avg_speed_satisfied),
                    "speed_band_satisfied": bool(speed_band_satisfied),
                    "ttc_rate_satisfied": bool(ttc_rate <= max_ttc_rate),
                    "headway_rate_satisfied": bool(headway_rate <= max_headway_rate),
                    "task_predicate_satisfied": bool(task_satisfied),
                }
            )
            return status

        return status

    def _completion_predicate(self, env, step_metrics: Dict[str, Any]) -> bool:
        self.last_criteria_status = self._criteria_status(env, step_metrics)
        return bool(self.last_criteria_status.get("task_predicate_satisfied", False))

    def update(
        self,
        env,
        step_idx: int,
        step_metrics: Dict[str, Any],
        crashed: bool,
        info: Optional[Dict[str, Any]] = None,
        action_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ego = getattr(env.unwrapped, "vehicle", None)
        self.last_info = dict(info or {})
        action_context = dict(action_context or {})
        if bool(action_context.get("benchmark_events_applied", False)):
            event_ids = [str(item) for item in (action_context.get("benchmark_event_ids") or [])]
            event_types = [str(item) for item in (action_context.get("benchmark_event_types") or [])]
            for event_id in event_ids:
                if event_id not in self.applied_benchmark_event_ids:
                    self.applied_benchmark_event_ids.append(event_id)
            self.applied_benchmark_event_types.extend(event_types)
            if self.first_benchmark_event_step is None:
                self.first_benchmark_event_step = int(action_context.get("benchmark_event_step") or step_idx)
            for event_type in event_types:
                if event_type not in self.first_event_step_by_type:
                    self.first_event_step_by_type[event_type] = int(
                        action_context.get("benchmark_event_step") or step_idx
                    )
        lane_rank = _lane_rank(ego)
        if lane_rank is not None and self.initial_lane_rank is not None:
            if lane_rank < self.initial_lane_rank:
                self.visited_left_lane = True
            if lane_rank > self.initial_lane_rank:
                self.visited_right_lane = True
            criteria_type = str(self.success_criteria.get("type") or "").strip().lower()
            target_offset = _resolve_direction_offset(self.success_criteria)
            if (
                criteria_type in {"overtake", "safe_overtake"}
                and self.overtake_latency_steps is None
                and target_offset != 0
                and ((target_offset < 0 and lane_rank < self.initial_lane_rank) or (target_offset > 0 and lane_rank > self.initial_lane_rank))
            ):
                self.overtake_latency_steps = int(step_idx)

        final_action_id = action_context.get("final_action_id", action_context.get("action_id"))
        try:
            final_action_id = int(final_action_id) if final_action_id is not None else None
        except Exception:
            final_action_id = None
        lane_change_attempt_action_id = action_context.get(
            "lane_change_original_action_id",
            action_context.get("original_action_id", action_context.get("model_action_id", final_action_id)),
        )
        try:
            lane_change_attempt_action_id = (
                int(lane_change_attempt_action_id)
                if lane_change_attempt_action_id is not None
                else None
            )
        except Exception:
            lane_change_attempt_action_id = None
        if lane_change_attempt_action_id in (0, 2) and self.first_lane_change_attempt_step is None:
            self.first_lane_change_attempt_step = int(step_idx)
        if lane_change_attempt_action_id in (0, 2) and self.first_maneuver_step is None:
            self.first_maneuver_step = int(step_idx)
        if final_action_id == 4 and self.first_brake_action_step is None:
            self.first_brake_action_step = int(step_idx)
        if final_action_id == 3 and self.first_accel_action_step is None:
            self.first_accel_action_step = int(step_idx)
        if self._step_in_opportunity_window(step_idx):
            if self.valid_opportunity_step is None:
                self.valid_opportunity_step = int(step_idx)
            if lane_change_attempt_action_id in (0, 2):
                self.maneuver_in_window = True
        if final_action_id in (0, 2):
            self.benchmark_lane_change_count += 1
        if (
            self.previous_final_action_id is not None
            and ((self.previous_final_action_id == 3 and final_action_id == 4) or (self.previous_final_action_id == 4 and final_action_id == 3))
        ):
            self.flap_accel_decel_count += 1
        if final_action_id is not None:
            self.previous_final_action_id = final_action_id
        if bool(action_context.get("lane_change_shield_applied", False)):
            self.unsafe_lane_change_attempts += 1
        target_offset = _resolve_direction_offset(self.success_criteria)
        target_lane_change_action_id = _lane_change_action_for_offset(target_offset)
        if self._safe_overtake_opportunity_available(env):
            self.safe_overtake_opportunity_steps += 1
            if self.first_safe_overtake_opportunity_step is None:
                self.first_safe_overtake_opportunity_step = int(step_idx)
            if lane_change_attempt_action_id != target_lane_change_action_id:
                self.missed_overtake_opportunity_steps += 1

        current_speed = float(getattr(ego, "speed", 0.0) or 0.0) if ego is not None else 0.0
        self.speed_history.append(current_speed)
        criteria_type = str(self.success_criteria.get("type") or "").strip().lower()
        if (
            criteria_type == "traffic_jam_escape"
            and self.jam_exit_step is None
            and lane_rank is not None
            and self.initial_lane_rank is not None
            and int(lane_rank) != int(self.initial_lane_rank)
        ):
            self.jam_exit_step = int(step_idx)
        if (
            criteria_type == "bottleneck_merge_pressure"
            and self.bottleneck_avoidance_step is None
            and lane_rank is not None
            and self.initial_lane_rank is not None
            and int(lane_rank) != int(self.initial_lane_rank)
        ):
            self.bottleneck_avoidance_step = int(step_idx)
        if bool(self.success_criteria.get("requires_yield", False)):
            yield_speed_mps = float(self.success_criteria.get("yield_speed_mps", 2.0) or 2.0)
            if current_speed <= yield_speed_mps:
                self.yield_observed_steps += 1
        front_gap_m = step_metrics.get("front_gap_m")
        if front_gap_m is not None:
            self.front_gap_history.append(float(front_gap_m))
        self.low_speed_blocking_steps += int(bool(step_metrics.get("low_speed_blocking", False)))
        self.ttc_danger_steps += int(bool(step_metrics.get("ttc_danger", False)))
        self.headway_violation_steps += int(bool(step_metrics.get("headway_violation", False)))
        ttc_sec = step_metrics.get("ttc_sec")
        if ttc_sec is not None and float(ttc_sec) > 0:
            positive_ttc = float(ttc_sec)
            if self.min_positive_ttc_sec is None:
                self.min_positive_ttc_sec = positive_ttc
            else:
                self.min_positive_ttc_sec = min(self.min_positive_ttc_sec, positive_ttc)

        ego_x = _vehicle_x(ego)
        if ego_x is not None:
            self.max_progress_m = max(self.max_progress_m, float(ego_x - self.initial_x))

        if criteria_type in {
            "post_brake_recovery",
            "multi_hazard_recovery",
            "cut_in_then_recover",
            "stop_go_wave_response",
        }:
            clear_gap = float(self.success_criteria.get("clear_front_gap_m", 25.0) or 25.0)
            clear_ttc = float(self.success_criteria.get("clear_front_ttc_sec", 4.0) or 4.0)
            front_clear = front_gap_m is None or float(front_gap_m) >= clear_gap
            ttc_clear = ttc_sec is None or float(ttc_sec) >= clear_ttc
            if self.recovery_clear_step is None and front_clear and ttc_clear:
                self.recovery_clear_step = int(step_idx)
            min_recovery_speed = float(self.success_criteria.get("min_recovery_speed_mps", 22.0) or 22.0)
            if (
                self.recovery_clear_step is not None
                and self.recovery_time_steps is None
                and current_speed >= min_recovery_speed
            ):
                self.recovery_time_steps = int(step_idx - self.recovery_clear_step)
                self.recovery_after_wave = True

        if crashed:
            self.failure_reason = self.failure_reason or "crash"
            self.hold_streak = 0
            return

        if self._completion_predicate(env, step_metrics):
            self.hold_streak += 1
            if (not self.task_completed) and self.hold_streak >= self.hold_steps_required:
                self.task_completed = True
                self.completion_step = int(step_idx)
                self.completion_time_sec = round(step_idx / max(self.policy_frequency, 1.0), 3)
                self.completion_speed_mps = round(float(current_speed), 4)
                self.completion_progress_m = round(float(self.max_progress_m), 4)
        else:
            self.hold_streak = 0

    def finalize(self, crashed: bool, episode_stop_reason: str) -> Dict[str, Any]:
        score_metrics = compute_benchmark_case_scores(
            task_completed=bool(self.task_completed),
            crashed=bool(crashed),
            min_positive_ttc_sec=self.min_positive_ttc_sec,
            speed_history=self.speed_history,
            completion_time_sec=self.completion_time_sec,
            time_limit_sec=self.time_limit_sec,
        )

        if self.failure_reason is None and not self.task_completed:
            if episode_stop_reason == "crash":
                self.failure_reason = "crash"
            elif episode_stop_reason == "episode_timeout_cap":
                self.failure_reason = "episode_timeout_cap"
            elif self.initial_front_vehicle_id is None and str(self.success_criteria.get("type") or "").lower() in {
                "front_gap_band",
                "overtake",
            }:
                self.failure_reason = "missing_initial_front_vehicle"
            else:
                self.failure_reason = "task_not_completed"

        criteria_type = str(self.success_criteria.get("type") or "").strip().lower()
        if bool(self.success_criteria.get("passive_trap", False)) and not self.task_completed:
            maneuver_required = criteria_type in {
                "mandatory_overtake_slow_lead",
                "timed_gap_overtake",
                "traffic_jam_escape",
                "multi_lane_route_discipline",
                "bottleneck_merge_pressure",
            }
            brake_or_recover_required = criteria_type in {"cut_in_then_recover", "stop_go_wave_response"}
            min_progress_m = float(self.success_criteria.get("min_progress_m", 0.0) or 0.0)
            self.passive_trap_failed = bool(
                (maneuver_required and self.first_maneuver_step is None)
                or (brake_or_recover_required and self.recovery_time_steps is None)
                or (min_progress_m > 0.0 and float(self.max_progress_m) < min_progress_m)
            )

        return {
            "case_id": self.case_id,
            "instruction": self.instruction,
            "category": self.category,
            "scenario_family": self.scenario_family,
            "tags": list(self.case.get("tags") or []),
            "difficulty": self.difficulty,
            "case_group": self.case_group,
            "time_limit_sec": round(float(self.time_limit_sec), 3),
            "benchmark_case_env_overrides": copy.deepcopy(self.case.get("env_overrides") or {}),
            "benchmark_success_criteria": copy.deepcopy(self.success_criteria),
            "benchmark_initial_lane_rank": self.initial_lane_rank,
            "benchmark_initial_front_gap_m": (
                round(float(self.initial_front_gap_m), 4)
                if self.initial_front_gap_m is not None
                else None
            ),
            "benchmark_completion_step": self.completion_step,
            "benchmark_completion_time_sec": self.completion_time_sec,
            "benchmark_completion_speed_mps": self.completion_speed_mps,
            "benchmark_completion_progress_m": self.completion_progress_m,
            "benchmark_yield_observed_steps": int(self.yield_observed_steps),
            "benchmark_criteria_status": copy.deepcopy(self.last_criteria_status),
            "benchmark_scenario_spec": copy.deepcopy(self.scenario_spec_report),
            "benchmark_scenario_spec_applied": bool(
                self.scenario_spec_metadata.get("benchmark_scenario_spec_applied", bool(self.scenario_spec_report))
            ),
            "benchmark_event_ids_applied": list(self.applied_benchmark_event_ids),
            "benchmark_event_types_applied": list(self.applied_benchmark_event_types),
            "benchmark_event_count_applied": int(len(self.applied_benchmark_event_ids)),
            "benchmark_first_event_step": self.first_benchmark_event_step,
            "benchmark_overtake_latency_steps": self.overtake_latency_steps,
            "benchmark_recovery_clear_step": self.recovery_clear_step,
            "benchmark_recovery_time_steps": self.recovery_time_steps,
            "benchmark_unsafe_lane_change_attempts": int(self.unsafe_lane_change_attempts),
            "benchmark_lane_change_count": int(self.benchmark_lane_change_count),
            "benchmark_safe_overtake_opportunity_steps": int(self.safe_overtake_opportunity_steps),
            "benchmark_missed_overtake_opportunity_steps": int(self.missed_overtake_opportunity_steps),
            "benchmark_first_safe_overtake_opportunity_step": self.first_safe_overtake_opportunity_step,
            "benchmark_first_lane_change_attempt_step": self.first_lane_change_attempt_step,
            "benchmark_valid_opportunity_step": self.valid_opportunity_step,
            "benchmark_first_maneuver_step": self.first_maneuver_step,
            "benchmark_maneuver_in_window": bool(self.maneuver_in_window),
            "benchmark_jam_exit_step": self.jam_exit_step,
            "benchmark_bottleneck_avoidance_step": self.bottleneck_avoidance_step,
            "benchmark_recovery_after_wave": bool(self.recovery_after_wave),
            "benchmark_first_brake_action_step": self.first_brake_action_step,
            "benchmark_first_accel_action_step": self.first_accel_action_step,
            "benchmark_passive_trap_failed": bool(self.passive_trap_failed),
            "benchmark_missed_overtake_opportunity_rate": (
                round(
                    float(self.missed_overtake_opportunity_steps)
                    / max(int(self.safe_overtake_opportunity_steps), 1),
                    4,
                )
                if self.safe_overtake_opportunity_steps > 0
                else 0.0
            ),
            "benchmark_flap_accel_decel_count": int(self.flap_accel_decel_count),
            "benchmark_low_speed_blocking_steps": int(self.low_speed_blocking_steps),
            "task_completed": bool(self.task_completed),
            "completion_rate": score_metrics["completion_rate"],
            "ttc_score": score_metrics["ttc_score"],
            "speed_variance_score": score_metrics["speed_variance_score"],
            "time_efficiency_score": score_metrics["time_efficiency_score"],
            "overall_score": score_metrics["overall_score"],
            "driving_score": score_metrics["driving_score"],
            "benchmark_failure_reason": self.failure_reason,
            "benchmark_speed_std_mps": score_metrics["speed_std_mps"],
            "benchmark_min_positive_ttc_sec": (
                round(float(self.min_positive_ttc_sec), 4)
                if self.min_positive_ttc_sec is not None
                else None
            ),
            "benchmark_max_progress_m": round(float(self.max_progress_m), 4),
        }
