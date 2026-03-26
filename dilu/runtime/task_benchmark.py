import copy
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_BENCHMARK_CASE_SET = "lampilot_highway_v1"
BENCHMARK_TTC_SAFE_THRESHOLD_SEC = 2.0
BENCHMARK_SPEED_STD_SAFE_MPS = 4.0
BENCHMARK_OVERALL_WEIGHTS = {
    "ttc": 0.5,
    "speed_variance": 0.3,
    "time_efficiency": 0.2,
}

_ENV_OVERRIDE_ALIASES = {
    "simulation_duration": "duration",
    "vehicle_count": "vehicles_count",
    "other_vehicle_type": "other_vehicles_type",
}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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

        normalized_case = {
            "case_id": case_id,
            "category": category,
            "instruction": instruction,
            "seed": int(case["seed"]),
            "time_limit_sec": float(case.get("time_limit_sec", defaults.get("time_limit_sec", 12.0))),
            "success_criteria": success_criteria,
            "env_overrides": env_overrides,
            "tags": [str(tag) for tag in (case.get("tags") or [])],
        }
        normalized_cases.append(normalized_case)

    categories = sorted({case["category"] for case in normalized_cases})
    benchmark_name = str(raw.get("benchmark_name") or os.path.basename(os.path.dirname(case_set_path)) or DEFAULT_BENCHMARK_CASE_SET)
    return {
        "benchmark_name": benchmark_name,
        "case_set_path": case_set_path,
        "version": str(raw.get("version") or "1.0"),
        "description": str(raw.get("description") or "").strip(),
        "defaults": defaults,
        "categories": categories,
        "cases": normalized_cases,
    }


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


def benchmark_metric_config() -> Dict[str, Any]:
    return {
        "ttc_safe_threshold_sec": BENCHMARK_TTC_SAFE_THRESHOLD_SEC,
        "speed_std_safe_mps": BENCHMARK_SPEED_STD_SAFE_MPS,
        "overall_weights": dict(BENCHMARK_OVERALL_WEIGHTS),
        "driving_score_formula": "0 if crashed else 0.5*completion_rate + 0.5*overall_score",
    }


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


class BenchmarkEpisodeEvaluator:
    def __init__(self, case: Dict[str, Any], env) -> None:
        self.case = copy.deepcopy(case)
        self.case_id = str(case["case_id"])
        self.category = str(case["category"])
        self.instruction = str(case["instruction"])
        self.success_criteria = dict(case.get("success_criteria") or {})
        self.time_limit_sec = float(case.get("time_limit_sec") or 0.0)

        uenv = env.unwrapped
        env_cfg = dict(getattr(uenv, "config", {}) or {})
        self.policy_frequency = float(env_cfg.get("policy_frequency", 1) or 1)
        self.max_steps = benchmark_max_steps(case, env_cfg, default_steps=int(math.ceil(self.time_limit_sec or 1.0)))
        self.initial_lane_rank = _lane_rank(getattr(uenv, "vehicle", None))
        self.initial_x = _vehicle_x(getattr(uenv, "vehicle", None)) or 0.0
        self.initial_speed_mps = float(getattr(getattr(uenv, "vehicle", None), "speed", 0.0) or 0.0)
        self.initial_front_vehicle_id = None
        self.initial_front_gap_m = None
        self.initial_front_x = None

        ego = getattr(uenv, "vehicle", None)
        road = getattr(uenv, "road", None)
        if ego is not None and road is not None:
            front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
            if front_vehicle is not None:
                self.initial_front_vehicle_id = id(front_vehicle)
                self.initial_front_gap_m = float(np.linalg.norm(ego.position - front_vehicle.position))
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

    def _completion_predicate(self, env, step_metrics: Dict[str, Any]) -> bool:
        ego = getattr(env.unwrapped, "vehicle", None)
        road = getattr(env.unwrapped, "road", None)
        lane_rank = _lane_rank(ego)
        current_speed = float(getattr(ego, "speed", 0.0) or 0.0) if ego is not None else 0.0
        front_gap_m = step_metrics.get("front_gap_m")
        criteria_type = str(self.success_criteria.get("type") or "").strip().lower()

        if criteria_type == "speed_band":
            min_speed = float(self.success_criteria.get("min_speed_mps", 0.0))
            max_speed = float(self.success_criteria.get("max_speed_mps", 999.0))
            return min_speed <= current_speed <= max_speed

        if criteria_type == "front_gap_band":
            if front_gap_m is None:
                return False
            min_gap = float(self.success_criteria.get("min_gap_m", 0.0))
            max_gap = float(self.success_criteria.get("max_gap_m", 1e9))
            return min_gap <= float(front_gap_m) <= max_gap

        if criteria_type == "lane_change":
            if lane_rank is None or self.initial_lane_rank is None:
                return False
            target_offset = _resolve_direction_offset(self.success_criteria)
            target_lane_rank = self.initial_lane_rank + target_offset
            return lane_rank == target_lane_rank

        if criteria_type == "overtake":
            if lane_rank is None or self.initial_lane_rank is None:
                return False
            target_offset = _resolve_direction_offset(self.success_criteria)
            used_required_lane = (
                self.visited_left_lane if target_offset < 0 else self.visited_right_lane
            )
            target_vehicle = _vehicle_by_runtime_id(road, self.initial_front_vehicle_id)
            if target_vehicle is None or ego is None:
                return False
            target_x = _vehicle_x(target_vehicle)
            ego_x = _vehicle_x(ego)
            if target_x is None or ego_x is None:
                return False
            pass_margin_m = float(self.success_criteria.get("pass_margin_m", 5.0))
            return bool(used_required_lane and target_x <= (ego_x - pass_margin_m))

        return False

    def update(self, env, step_idx: int, step_metrics: Dict[str, Any], crashed: bool) -> None:
        ego = getattr(env.unwrapped, "vehicle", None)
        lane_rank = _lane_rank(ego)
        if lane_rank is not None and self.initial_lane_rank is not None:
            if lane_rank < self.initial_lane_rank:
                self.visited_left_lane = True
            if lane_rank > self.initial_lane_rank:
                self.visited_right_lane = True

        current_speed = float(getattr(ego, "speed", 0.0) or 0.0) if ego is not None else 0.0
        self.speed_history.append(current_speed)
        front_gap_m = step_metrics.get("front_gap_m")
        if front_gap_m is not None:
            self.front_gap_history.append(float(front_gap_m))
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
        else:
            self.hold_streak = 0

    def finalize(self, crashed: bool, episode_stop_reason: str) -> Dict[str, Any]:
        completion_rate = 1.0 if self.task_completed else 0.0
        if crashed:
            ttc_score = 0.0
        elif self.min_positive_ttc_sec is None:
            ttc_score = 1.0
        else:
            ttc_score = max(
                0.0,
                min(1.0, float(self.min_positive_ttc_sec) / BENCHMARK_TTC_SAFE_THRESHOLD_SEC),
            )

        if len(self.speed_history) <= 1:
            speed_std_mps = 0.0
        else:
            speed_std_mps = float(np.std(np.array(self.speed_history, dtype=float)))
        speed_variance_score = max(
            0.0,
            min(1.0, 1.0 - (speed_std_mps / BENCHMARK_SPEED_STD_SAFE_MPS)),
        )

        if self.completion_time_sec is None or self.time_limit_sec <= 0:
            time_efficiency_score = 0.0
        else:
            time_efficiency_score = max(
                0.0,
                min(1.0, 1.0 - (float(self.completion_time_sec) / float(self.time_limit_sec))),
            )

        overall_score = (
            BENCHMARK_OVERALL_WEIGHTS["ttc"] * ttc_score
            + BENCHMARK_OVERALL_WEIGHTS["speed_variance"] * speed_variance_score
            + BENCHMARK_OVERALL_WEIGHTS["time_efficiency"] * time_efficiency_score
        )
        driving_score = 0.0 if crashed else (0.5 * completion_rate + 0.5 * overall_score)

        if self.failure_reason is None and not self.task_completed:
            if episode_stop_reason == "crash":
                self.failure_reason = "crash"
            elif self.initial_front_vehicle_id is None and str(self.success_criteria.get("type") or "").lower() in {
                "front_gap_band",
                "overtake",
            }:
                self.failure_reason = "missing_initial_front_vehicle"
            else:
                self.failure_reason = "task_not_completed"

        return {
            "case_id": self.case_id,
            "instruction": self.instruction,
            "category": self.category,
            "tags": list(self.case.get("tags") or []),
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
            "task_completed": bool(self.task_completed),
            "completion_rate": round(completion_rate, 4),
            "ttc_score": round(ttc_score, 4),
            "speed_variance_score": round(speed_variance_score, 4),
            "time_efficiency_score": round(time_efficiency_score, 4),
            "overall_score": round(overall_score, 4),
            "driving_score": round(driving_score, 4),
            "benchmark_failure_reason": self.failure_reason,
            "benchmark_speed_std_mps": round(speed_std_mps, 4),
            "benchmark_min_positive_ttc_sec": (
                round(float(self.min_positive_ttc_sec), 4)
                if self.min_positive_ttc_sec is not None
                else None
            ),
            "benchmark_max_progress_m": round(float(self.max_progress_m), 4),
        }

