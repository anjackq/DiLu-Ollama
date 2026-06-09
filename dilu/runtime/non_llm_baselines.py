from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from highway_env.vehicle.behavior import IDMVehicle

from dilu.runtime.safety_shields import (
    FASTER_ACTION_ID,
    IDLE_ACTION_ID,
    LANE_LEFT_ACTION_ID,
    LANE_RIGHT_ACTION_ID,
    SLOWER_ACTION_ID,
)


EXPERT_BASELINE_NAME = "true_idm_ego"
DISCRETE_CONTROL_MODE = "discrete_action"
EXPERT_CONTROL_MODE = "expert_vehicle"


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    level: int
    family: str
    control_mode: str
    uses_case_category: bool = False
    uses_success_criteria: bool = False
    uses_hidden_scenario_spec: bool = False
    uses_future_events: bool = False
    safety_shield_compatible: bool = True

    def to_metadata(self, *, safety_shields_enabled: bool) -> dict[str, Any]:
        return {
            "baseline_level": int(self.level),
            "baseline_family": self.family,
            "baseline_control_mode": self.control_mode,
            "baseline_uses_case_category": bool(self.uses_case_category),
            "baseline_uses_success_criteria": bool(self.uses_success_criteria),
            "baseline_uses_hidden_scenario_spec": bool(self.uses_hidden_scenario_spec),
            "baseline_uses_future_events": bool(self.uses_future_events),
            "baseline_safety_shields_enabled": bool(
                safety_shields_enabled and self.safety_shield_compatible
            ),
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


_BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec("idle_always", 1, "legacy_fixed", DISCRETE_CONTROL_MODE),
    BaselineSpec("random_seeded", 1, "legacy_random", DISCRETE_CONTROL_MODE),
    BaselineSpec("always_faster", 1, "fixed_action", DISCRETE_CONTROL_MODE),
    BaselineSpec("always_slower", 1, "fixed_action", DISCRETE_CONTROL_MODE),
    BaselineSpec("always_left", 1, "fixed_action", DISCRETE_CONTROL_MODE),
    BaselineSpec("always_right", 1, "fixed_action", DISCRETE_CONTROL_MODE),
    BaselineSpec("safe_stop", 1, "fixed_action", DISCRETE_CONTROL_MODE),
    BaselineSpec("speed_hold_20", 1, "speed_hold", DISCRETE_CONTROL_MODE),
    BaselineSpec("speed_hold_25", 1, "speed_hold", DISCRETE_CONTROL_MODE),
    BaselineSpec("speed_hold_30", 1, "speed_hold", DISCRETE_CONTROL_MODE),
    BaselineSpec("keep_lane_cruise", 2, "legacy_rule", DISCRETE_CONTROL_MODE),
    BaselineSpec(
        "idm_mobil",
        2,
        "legacy_rule",
        DISCRETE_CONTROL_MODE,
        uses_success_criteria=True,
    ),
    BaselineSpec("defensive_rule_driver", 2, "rule_driver", DISCRETE_CONTROL_MODE),
    BaselineSpec(
        "overtake_rule_driver",
        2,
        "rule_driver",
        DISCRETE_CONTROL_MODE,
        uses_success_criteria=True,
    ),
    BaselineSpec(
        "scenario_aware_rule_driver",
        2,
        "rule_driver",
        DISCRETE_CONTROL_MODE,
        uses_case_category=True,
        uses_success_criteria=True,
    ),
    BaselineSpec(
        EXPERT_BASELINE_NAME,
        3,
        "expert_vehicle",
        EXPERT_CONTROL_MODE,
        safety_shield_compatible=False,
    ),
)

BASELINE_REGISTRY: dict[str, BaselineSpec] = {spec.name: spec for spec in _BASELINE_SPECS}
DEFAULT_BASELINE_NAMES = ("idle_always", "random_seeded", "keep_lane_cruise", "idm_mobil")


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def iter_baseline_specs() -> tuple[BaselineSpec, ...]:
    return _BASELINE_SPECS


def get_baseline_spec(name: str) -> BaselineSpec:
    key = str(name or "").strip()
    try:
        return BASELINE_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown baseline `{key}`; available={list(BASELINE_REGISTRY)}"
        ) from exc


def baseline_names_for_levels(levels: Sequence[int]) -> list[str]:
    requested = {int(level) for level in levels}
    invalid = sorted(level for level in requested if level not in {1, 2, 3})
    if invalid:
        raise ValueError(f"Invalid baseline levels: {invalid}; allowed=[1, 2, 3]")
    return [spec.name for spec in _BASELINE_SPECS if spec.level in requested]


def parse_baseline_levels(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None or str(raw).strip() == "":
        return None
    levels: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            raise ValueError("Empty token in --baseline-levels.")
        levels.append(int(token))
    return levels


def resolve_baseline_names(raw_baselines: Optional[str], levels: Optional[Sequence[int]]) -> list[str]:
    if levels is not None:
        return baseline_names_for_levels(levels)

    raw = str(raw_baselines or ",".join(DEFAULT_BASELINE_NAMES)).strip()
    if raw.lower() == "all":
        return [spec.name for spec in _BASELINE_SPECS]
    names = [token.strip() for token in raw.split(",") if token.strip()]
    if not names:
        names = list(DEFAULT_BASELINE_NAMES)
    unknown = sorted({name for name in names if name not in BASELINE_REGISTRY})
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}; available={list(BASELINE_REGISTRY)}")
    return names


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


def _speed_hold_target_from_name(name: str, default: float) -> float:
    if str(name).startswith("speed_hold_"):
        try:
            return float(str(name).rsplit("_", 1)[-1])
        except Exception:
            return float(default)
    return float(default)


def _front_risk(snapshot: LaneSnapshot, *, gap_m: float = 12.0, ttc_sec: float = 3.0) -> bool:
    return (
        snapshot.front_gap_m is not None
        and snapshot.front_gap_m < gap_m
    ) or (
        snapshot.front_ttc_sec is not None
        and snapshot.front_ttc_sec < ttc_sec
    )


def _front_slow(snapshot: LaneSnapshot, ego_speed: float, *, gap_m: float = 32.0) -> bool:
    return (
        snapshot.front_gap_m is not None
        and snapshot.front_gap_m < gap_m
        and snapshot.front_speed_mps is not None
        and snapshot.front_speed_mps + 1.0 < ego_speed
    )


class BaselinePolicy:
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = str(name)
        self.spec = get_baseline_spec(self.name)
        self.config = dict(config)
        self.target_speed = _speed_hold_target_from_name(self.name, target_speed_mps(config))

    @property
    def control_mode(self) -> str:
        return self.spec.control_mode

    def decide(
        self,
        env: Any,
        case: dict[str, Any],
        step_idx: int,
        rng: Optional[random.Random] = None,
    ) -> BaselineDecision:
        available = available_actions(env)
        rng = rng or random.Random(stable_seed(self.name, case.get("case_id"), step_idx))
        if self.name == EXPERT_BASELINE_NAME:
            return BaselineDecision(IDLE_ACTION_ID, "expert_vehicle_autonomous_control", {})
        if self.name == "idle_always":
            return BaselineDecision(
                choose_available(IDLE_ACTION_ID, available, (SLOWER_ACTION_ID, FASTER_ACTION_ID)),
                "fixed_idle",
                {},
            )
        if self.name == "random_seeded":
            return BaselineDecision(int(rng.choice(available)), "seeded_uniform_available_action", {})
        if self.name in {
            "always_faster",
            "always_slower",
            "always_left",
            "always_right",
        }:
            return self._fixed_action(available)
        if self.name == "safe_stop":
            return self._safe_stop(env, available)
        if self.name.startswith("speed_hold_"):
            return self._speed_hold(env, available, reason_prefix=self.name)
        if self.name == "keep_lane_cruise":
            return self._keep_lane_cruise(env, available)
        if self.name == "defensive_rule_driver":
            return self._defensive_rule_driver(env, available)
        if self.name == "overtake_rule_driver":
            return self._overtake_rule_driver(env, case, available)
        if self.name == "scenario_aware_rule_driver":
            return self._scenario_aware_rule_driver(env, case, available)
        if self.name == "idm_mobil":
            return self._idm_mobil_style(env, case, available)
        raise ValueError(f"Unknown baseline policy: {self.name}")

    def _fixed_action(self, available: list[int]) -> BaselineDecision:
        action_map = {
            "always_faster": FASTER_ACTION_ID,
            "always_slower": SLOWER_ACTION_ID,
            "always_left": LANE_LEFT_ACTION_ID,
            "always_right": LANE_RIGHT_ACTION_ID,
        }
        requested = action_map[self.name]
        return BaselineDecision(
            choose_available(requested, available, (IDLE_ACTION_ID, SLOWER_ACTION_ID, FASTER_ACTION_ID)),
            f"fixed_{self.name.removeprefix('always_')}",
            {},
        )

    def _safe_stop(self, env: Any, available: list[int]) -> BaselineDecision:
        speed = vehicle_snapshot(getattr(env.unwrapped, "vehicle", None)).speed
        if speed <= 0.5:
            return BaselineDecision(
                choose_available(IDLE_ACTION_ID, available, (SLOWER_ACTION_ID,)),
                "safe_stop_hold",
                {},
            )
        return BaselineDecision(
            choose_available(SLOWER_ACTION_ID, available, (IDLE_ACTION_ID,)),
            "safe_stop_brake",
            {},
        )

    def _speed_hold(
        self,
        env: Any,
        available: list[int],
        *,
        reason_prefix: str = "speed_hold",
    ) -> BaselineDecision:
        ego_state = vehicle_snapshot(getattr(env.unwrapped, "vehicle", None))
        action = IDLE_ACTION_ID
        reason = f"{reason_prefix}_idle_near_target"
        if ego_state.speed < self.target_speed - 1.0:
            action = FASTER_ACTION_ID
            reason = f"{reason_prefix}_accelerate"
        elif ego_state.speed > self.target_speed + 1.0:
            action = SLOWER_ACTION_ID
            reason = f"{reason_prefix}_decelerate"
        return BaselineDecision(
            choose_available(action, available, (IDLE_ACTION_ID, SLOWER_ACTION_ID, FASTER_ACTION_ID)),
            reason,
            {"target_speed_mps": round(float(self.target_speed), 3)},
        )

    def _keep_lane_cruise(self, env: Any, available: list[int]) -> BaselineDecision:
        ego = getattr(env.unwrapped, "vehicle", None)
        ego_state = vehicle_snapshot(ego)
        current_lane = ego_state.lane_rank if ego_state.lane_rank is not None else 1
        current = lane_snapshot(env, current_lane)
        if _front_risk(current):
            return BaselineDecision(
                choose_available(SLOWER_ACTION_ID, available, (IDLE_ACTION_ID,)),
                "front_gap_or_ttc_caution",
                {"target_speed_mps": round(float(self.target_speed), 3)},
            )
        return self._speed_hold(env, available, reason_prefix="target_speed_hold")

    def _defensive_rule_driver(self, env: Any, available: list[int]) -> BaselineDecision:
        ego = getattr(env.unwrapped, "vehicle", None)
        ego_state = vehicle_snapshot(ego)
        current_lane = ego_state.lane_rank if ego_state.lane_rank is not None else 1
        current = lane_snapshot(env, current_lane)
        metadata = self._front_metadata(current)
        metadata["target_speed_mps"] = round(float(self.target_speed), 3)
        if _front_risk(current, gap_m=14.0, ttc_sec=3.5):
            return BaselineDecision(
                choose_available(SLOWER_ACTION_ID, available, (IDLE_ACTION_ID,)),
                "defensive_front_risk_brake",
                metadata,
            )
        if ego_state.speed < self.target_speed - 1.0 and (
            current.front_gap_m is None or current.front_gap_m >= 25.0
        ):
            return BaselineDecision(
                choose_available(FASTER_ACTION_ID, available, (IDLE_ACTION_ID,)),
                "defensive_recover_to_flow",
                metadata,
            )
        if ego_state.speed > self.target_speed + 2.0:
            return BaselineDecision(
                choose_available(SLOWER_ACTION_ID, available, (IDLE_ACTION_ID,)),
                "defensive_above_flow_speed",
                metadata,
            )
        return BaselineDecision(choose_available(IDLE_ACTION_ID, available, (SLOWER_ACTION_ID,)), "defensive_hold_lane", metadata)

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

    def _best_safe_lane_action(
        self,
        env: Any,
        available: list[int],
        preferred: Optional[int] = None,
    ) -> Optional[int]:
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

    def _overtake_rule_driver(self, env: Any, case: dict[str, Any], available: list[int]) -> BaselineDecision:
        ego = getattr(env.unwrapped, "vehicle", None)
        ego_state = vehicle_snapshot(ego)
        current_lane = ego_state.lane_rank if ego_state.lane_rank is not None else 1
        current = lane_snapshot(env, current_lane)
        preferred = self._target_lane_action_from_case(env, case)
        safe_target_action = self._best_safe_lane_action(env, available, preferred)
        metadata = self._front_metadata(current)
        metadata.update({"target_speed_mps": round(float(self.target_speed), 3)})
        if _front_slow(current, ego_state.speed, gap_m=36.0) and safe_target_action is not None:
            return BaselineDecision(safe_target_action, "overtake_safe_target_lane", metadata)
        if _front_risk(current, gap_m=12.0, ttc_sec=3.0):
            return BaselineDecision(
                choose_available(SLOWER_ACTION_ID, available, (IDLE_ACTION_ID,)),
                "overtake_front_risk_brake",
                metadata,
            )
        if _front_slow(current, ego_state.speed, gap_m=36.0):
            return BaselineDecision(
                choose_available(IDLE_ACTION_ID, available, (SLOWER_ACTION_ID,)),
                "overtake_wait_for_safe_gap",
                metadata,
            )
        return self._speed_hold(env, available, reason_prefix="overtake_flow_speed")

    def _scenario_aware_rule_driver(self, env: Any, case: dict[str, Any], available: list[int]) -> BaselineDecision:
        category = str(case.get("category") or "").strip().lower()
        criteria_type = str((case.get("success_criteria") or {}).get("type") or "").strip().lower()
        if category in {
            "slow_lead_overtake",
            "delayed_overtake_gap",
        } or criteria_type in {"safe_overtake", "delayed_overtake_gap"}:
            decision = self._overtake_rule_driver(env, case, available)
        elif category in {
            "cut_in_brake_response",
            "blocked_lane_patience",
            "closing_rear_lane_change",
            "squeeze_box_patience",
            "multi_hazard_recovery",
        } or criteria_type in {
            "cut_in_brake_response",
            "blocked_lane_patience",
            "closing_rear_lane_change",
            "squeeze_box_patience",
            "multi_hazard_recovery",
        }:
            decision = self._defensive_rule_driver(env, available)
        elif category in {"false_alarm_stability", "free_flow_cruise", "lane_discipline"}:
            decision = self._keep_lane_cruise(env, available)
        else:
            decision = self._idm_mobil_style(env, case, available)
        metadata = {
            **decision.metadata,
            "scenario_aware_category": category,
            "scenario_aware_success_criteria_type": criteria_type,
            "uses_hidden_scenario_spec": False,
            "uses_future_events": False,
        }
        return BaselineDecision(decision.action_id, decision.reason, metadata)

    def _idm_mobil_style(self, env: Any, case: dict[str, Any], available: list[int]) -> BaselineDecision:
        ego = getattr(env.unwrapped, "vehicle", None)
        ego_state = vehicle_snapshot(ego)
        current_lane = ego_state.lane_rank if ego_state.lane_rank is not None else 1
        current = lane_snapshot(env, current_lane)
        preferred_lane_action = self._target_lane_action_from_case(env, case)
        safe_target_action = self._best_safe_lane_action(env, available, preferred_lane_action)
        metadata = {
            "target_speed_mps": round(float(self.target_speed), 3),
            **self._front_metadata(current),
        }

        severe_front = _front_risk(current, gap_m=10.0, ttc_sec=2.5)
        slow_front = _front_slow(current, ego_state.speed, gap_m=28.0)

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

    @staticmethod
    def _front_metadata(snapshot: LaneSnapshot) -> dict[str, Any]:
        return {
            "front_gap_m": None if snapshot.front_gap_m is None else round(snapshot.front_gap_m, 3),
            "front_ttc_sec": None if snapshot.front_ttc_sec is None else round(snapshot.front_ttc_sec, 3),
        }


def configure_true_idm_ego(
    env: Any,
    *,
    target_speed_mps: float = 25.0,
    enable_lane_change: bool = True,
) -> dict[str, Any]:
    uenv = env.unwrapped
    old_ego = getattr(uenv, "vehicle", None)
    road = getattr(uenv, "road", None)
    if old_ego is None or road is None:
        raise ValueError("true_idm_ego requires a highway env with an ego vehicle and road.")

    lane_index = getattr(old_ego, "lane_index", None)
    if lane_index is None:
        raise ValueError("true_idm_ego requires the ego vehicle to have a lane_index.")
    position = getattr(old_ego, "position", None)
    heading = float(getattr(old_ego, "heading", 0.0) or 0.0)
    speed = float(getattr(old_ego, "speed", 0.0) or 0.0)
    expert = IDMVehicle(
        road,
        position.copy() if hasattr(position, "copy") else position,
        heading=heading,
        speed=speed,
        target_lane_index=lane_index,
        target_speed=float(target_speed_mps),
        enable_lane_change=bool(enable_lane_change),
    )
    expert.dilu_benchmark_id = "true_idm_ego"
    expert.dilu_benchmark_role = "expert_ego"
    for attr_name in ("speed_index", "target_speeds"):
        if hasattr(old_ego, attr_name):
            try:
                setattr(expert, attr_name, getattr(old_ego, attr_name))
            except Exception:
                pass

    road.vehicles = [expert if vehicle is old_ego else vehicle for vehicle in list(road.vehicles or [])]
    if expert not in road.vehicles:
        road.vehicles.insert(0, expert)
    if hasattr(uenv, "controlled_vehicles"):
        controlled = list(getattr(uenv, "controlled_vehicles") or [])
        if controlled:
            controlled[0] = expert
        else:
            controlled = [expert]
        uenv.controlled_vehicles = controlled
    action_type = getattr(uenv, "action_type", None)
    if action_type is not None:
        try:
            action_type.controlled_vehicle = expert
        except Exception:
            pass
    return {
        "true_idm_ego_enabled": True,
        "true_idm_ego_target_speed_mps": round(float(target_speed_mps), 4),
        "true_idm_ego_lane_change_enabled": bool(enable_lane_change),
        "baseline_expert_initial_lane_rank": int(lane_index[2]),
        "baseline_expert_initial_speed_mps": round(float(speed), 4),
    }
