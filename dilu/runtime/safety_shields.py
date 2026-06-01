from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

LANE_LEFT_ACTION_ID = 0
IDLE_ACTION_ID = 1
LANE_RIGHT_ACTION_ID = 2
FASTER_ACTION_ID = 3
SLOWER_ACTION_ID = 4

LANE_CHANGE_ACTIONS = {
    LANE_LEFT_ACTION_ID: -1,
    LANE_RIGHT_ACTION_ID: 1,
}

TARGET_FRONT_GAP_REQUIRED_M = 14.0
TARGET_REAR_GAP_REQUIRED_M = 10.0
TARGET_REAR_TTC_REQUIRED_SEC = 2.5
FRONT_CRITICAL_GAP_M = 8.0
FRONT_CAUTION_GAP_M = 12.0
ACCELERATION_CAUTION_GAP_M = 18.0
FRONT_CRITICAL_TTC_SEC = 2.0
FRONT_CAUTION_TTC_SEC = 3.0
ACCELERATION_CAUTION_TTC_SEC = 5.0
ACCELERATION_PROJECTED_SPEED_GAIN_MPS = 2.0
PROJECTED_TIME_HORIZON_SEC = 1.0
LOW_SPEED_RECOVERY_FLOOR_MPS = 16.7
LOW_SPEED_RECOVERY_FRONT_GAP_M = 25.0
LOW_SPEED_RECOVERY_TTC_SEC = 4.0

@dataclass(frozen=True)
class SafetyShieldResult:
    original_action_id: int
    action_id: int
    applied: bool
    reason: str
    shield_type: str
    front_gap_m: Optional[float] = None
    rear_gap_m: Optional[float] = None
    front_ttc_sec: Optional[float] = None
    rear_ttc_sec: Optional[float] = None
    target_lane_rank: Optional[int] = None
    required_front_gap_m: Optional[float] = None
    required_rear_gap_m: Optional[float] = None
    required_front_ttc_sec: Optional[float] = None
    required_rear_ttc_sec: Optional[float] = None
    projected_front_gap_m: Optional[float] = None
    projected_front_ttc_sec: Optional[float] = None
    projected_ego_speed_mps: Optional[float] = None
    projection_horizon_sec: Optional[float] = None

    def to_metadata(self, prefix: str) -> dict[str, Any]:
        metadata = {
            f"{prefix}_shield_applied": bool(self.applied),
            f"{prefix}_shield_reason": self.reason,
            f"{prefix}_original_action_id": int(self.original_action_id),
            f"{prefix}_final_action_id": int(self.action_id),
            f"{prefix}_front_gap_m": self.front_gap_m,
            f"{prefix}_rear_gap_m": self.rear_gap_m,
            f"{prefix}_front_ttc_sec": self.front_ttc_sec,
            f"{prefix}_rear_ttc_sec": self.rear_ttc_sec,
            f"{prefix}_target_lane_rank": self.target_lane_rank,
            f"{prefix}_required_front_gap_m": self.required_front_gap_m,
            f"{prefix}_required_rear_gap_m": self.required_rear_gap_m,
            f"{prefix}_required_front_ttc_sec": self.required_front_ttc_sec,
            f"{prefix}_required_rear_ttc_sec": self.required_rear_ttc_sec,
            f"{prefix}_projected_front_gap_m": self.projected_front_gap_m,
            f"{prefix}_projected_front_ttc_sec": self.projected_front_ttc_sec,
            f"{prefix}_projected_ego_speed_mps": self.projected_ego_speed_mps,
            f"{prefix}_projection_horizon_sec": self.projection_horizon_sec,
        }
        if prefix == "lane_change":
            metadata.update(
                {
                    "lane_change_target_front_gap_m": self.front_gap_m,
                    "lane_change_target_front_ttc_sec": self.front_ttc_sec,
                    "lane_change_target_rear_gap_m": self.rear_gap_m,
                    "lane_change_target_rear_ttc_sec": self.rear_ttc_sec,
                }
            )
        if prefix == "longitudinal_safety":
            metadata.update(
                {
                    "longitudinal_safety_current_front_gap_m": self.front_gap_m,
                    "longitudinal_safety_current_front_ttc_sec": self.front_ttc_sec,
                }
            )
        return metadata

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)

def _rounded(value: Optional[float]) -> Optional[float]:
    if value is None or value == math.inf:
        return value
    return round(float(value), 4)

def _result(
    *,
    original_action_id: int,
    action_id: int,
    applied: bool,
    reason: str,
    shield_type: str,
    front_gap_m: Optional[float] = None,
    rear_gap_m: Optional[float] = None,
    front_ttc_sec: Optional[float] = None,
    rear_ttc_sec: Optional[float] = None,
    target_lane_rank: Optional[int] = None,
    required_front_gap_m: Optional[float] = None,
    required_rear_gap_m: Optional[float] = None,
    required_front_ttc_sec: Optional[float] = None,
    required_rear_ttc_sec: Optional[float] = None,
    projected_front_gap_m: Optional[float] = None,
    projected_front_ttc_sec: Optional[float] = None,
    projected_ego_speed_mps: Optional[float] = None,
    projection_horizon_sec: Optional[float] = None,
) -> SafetyShieldResult:
    return SafetyShieldResult(
        original_action_id=int(original_action_id),
        action_id=int(action_id),
        applied=bool(applied),
        reason=str(reason),
        shield_type=str(shield_type),
        front_gap_m=_rounded(front_gap_m),
        rear_gap_m=_rounded(rear_gap_m),
        front_ttc_sec=_rounded(front_ttc_sec),
        rear_ttc_sec=_rounded(rear_ttc_sec),
        target_lane_rank=target_lane_rank,
        required_front_gap_m=_rounded(required_front_gap_m),
        required_rear_gap_m=_rounded(required_rear_gap_m),
        required_front_ttc_sec=_rounded(required_front_ttc_sec),
        required_rear_ttc_sec=_rounded(required_rear_ttc_sec),
        projected_front_gap_m=_rounded(projected_front_gap_m),
        projected_front_ttc_sec=_rounded(projected_front_ttc_sec),
        projected_ego_speed_mps=_rounded(projected_ego_speed_mps),
        projection_horizon_sec=_rounded(projection_horizon_sec),
    )

def _scenario_family(scenario: Any) -> str:
    try:
        return str(scenario.scenario_family()).strip().lower()
    except Exception:
        return "highway"

def _available_actions(scenario: Any) -> set[int]:
    try:
        return {int(action_id) for action_id in scenario.available_action_ids()}
    except Exception:
        return {0, 1, 2, 3, 4}

def _fallback_action(scenario: Any, available_actions: set[int]) -> int:
    try:
        fallback = int(scenario.preferred_fallback_action_id())
        if fallback in available_actions:
            return fallback
    except Exception:
        pass
    for action_id in (SLOWER_ACTION_ID, IDLE_ACTION_ID, FASTER_ACTION_ID):
        if action_id in available_actions:
            return action_id
    return min(available_actions) if available_actions else IDLE_ACTION_ID

def _prefer_action(scenario: Any, preferred: Iterable[int]) -> int:
    available = _available_actions(scenario)
    for action_id in preferred:
        if int(action_id) in available:
            return int(action_id)
    return _fallback_action(scenario, available)


def _ego_vehicle(scenario: Any) -> Any:
    ego = getattr(scenario, "ego", None)
    if ego is not None:
        return ego
    env = getattr(scenario, "env", None)
    return getattr(getattr(env, "unwrapped", None), "vehicle", None)


def _road_vehicles(scenario: Any, ego: Any) -> list[Any]:
    env = getattr(scenario, "env", None)
    road = getattr(getattr(env, "unwrapped", None), "road", None)
    vehicles = getattr(road, "vehicles", None)
    if vehicles is None:
        vehicles = getattr(getattr(scenario, "road", None), "vehicles", None)
    return [vehicle for vehicle in list(vehicles or []) if vehicle is not ego]


def _lane_rank(vehicle: Any) -> Optional[int]:
    try:
        return int(getattr(vehicle, "lane_index")[2])
    except Exception:
        return None


def _position_x(vehicle: Any) -> Optional[float]:
    try:
        return float(getattr(vehicle, "position")[0])
    except Exception:
        return None


def _speed(vehicle: Any) -> float:
    try:
        return float(getattr(vehicle, "speed", 0.0))
    except Exception:
        return 0.0


def _lane_count(scenario: Any, ego: Any, vehicles: list[Any]) -> Optional[int]:
    network = getattr(scenario, "network", None)
    lane_index = getattr(ego, "lane_index", None)
    if network is not None and lane_index is not None:
        try:
            side_lanes = network.all_side_lanes(lane_index)
            if side_lanes:
                return len(side_lanes)
        except Exception:
            pass
    ranks = [_lane_rank(vehicle) for vehicle in [ego, *vehicles]]
    numeric_ranks = [rank for rank in ranks if rank is not None]
    return max(numeric_ranks) + 1 if numeric_ranks else None


def _ttc_from_gap(gap_m: Optional[float], closing_speed_mps: float) -> Optional[float]:
    if gap_m is None:
        return None
    if gap_m <= 0:
        return 0.0
    if closing_speed_mps <= 1e-6:
        return math.inf
    return float(gap_m) / float(closing_speed_mps)


def _front_risk_is_severe(
    *,
    front_gap_m: Optional[float],
    front_ttc_sec: Optional[float],
    projected_front_gap_m: Optional[float] = None,
    projected_front_ttc_sec: Optional[float] = None,
) -> bool:
    gap_values = [front_gap_m, projected_front_gap_m]
    ttc_values = [front_ttc_sec, projected_front_ttc_sec]
    if any(value is not None and value < FRONT_CRITICAL_GAP_M for value in gap_values):
        return True
    return any(value is not None and value < FRONT_CRITICAL_TTC_SEC for value in ttc_values)


def _nearest_front_and_rear(
    *,
    vehicles: list[Any],
    ego: Any,
    lane_rank: int,
) -> tuple[Optional[Any], Optional[Any], Optional[float], Optional[float]]:
    ego_x = _position_x(ego)
    if ego_x is None:
        return None, None, None, None
    nearest_front = None
    nearest_rear = None
    nearest_front_gap = None
    nearest_rear_gap = None
    for vehicle in vehicles:
        if _lane_rank(vehicle) != lane_rank:
            continue
        vehicle_x = _position_x(vehicle)
        if vehicle_x is None:
            continue
        dx = vehicle_x - ego_x
        if dx >= 0 and (nearest_front_gap is None or dx < nearest_front_gap):
            nearest_front = vehicle
            nearest_front_gap = float(dx)
        if dx < 0 and (nearest_rear_gap is None or abs(float(dx)) < nearest_rear_gap):
            nearest_rear = vehicle
            nearest_rear_gap = abs(float(dx))
    return nearest_front, nearest_rear, nearest_front_gap, nearest_rear_gap


def _non_applied(action_id: int, reason: str, shield_type: str) -> SafetyShieldResult:
    return _result(
        original_action_id=action_id,
        action_id=action_id,
        applied=False,
        reason=reason,
        shield_type=shield_type,
    )


def apply_lane_change_safety_shield(scenario: Any, proposed_action_id: int) -> SafetyShieldResult:
    original_action_id = _safe_int(proposed_action_id, default=IDLE_ACTION_ID)
    shield_type = "lane_change"
    if _scenario_family(scenario) != "highway":
        return _non_applied(original_action_id, "non_highway_scenario", shield_type)
    if original_action_id not in LANE_CHANGE_ACTIONS:
        return _non_applied(original_action_id, "not_lane_change", shield_type)

    replacement = _prefer_action(scenario, (IDLE_ACTION_ID, SLOWER_ACTION_ID, FASTER_ACTION_ID))
    if original_action_id not in _available_actions(scenario):
        return _result(
            original_action_id=original_action_id,
            action_id=replacement,
            applied=True,
            reason="unavailable_lane_action",
            shield_type=shield_type,
        )

    ego = _ego_vehicle(scenario)
    if ego is None:
        return _result(
            original_action_id=original_action_id,
            action_id=replacement,
            applied=True,
            reason="missing_ego_vehicle",
            shield_type=shield_type,
        )
    vehicles = _road_vehicles(scenario, ego)
    ego_lane_rank = _lane_rank(ego)
    lane_count = _lane_count(scenario, ego, vehicles)
    if ego_lane_rank is None or lane_count is None:
        return _result(
            original_action_id=original_action_id,
            action_id=replacement,
            applied=True,
            reason="unknown_lane_topology",
            shield_type=shield_type,
        )

    target_lane_rank = ego_lane_rank + LANE_CHANGE_ACTIONS[original_action_id]
    context: dict[str, Any] = {"target_lane_rank": target_lane_rank}
    if target_lane_rank < 0 or target_lane_rank >= lane_count:
        return _result(
            original_action_id=original_action_id,
            action_id=replacement,
            applied=True,
            reason="unavailable_lane_action",
            shield_type=shield_type,
            **context,
        )

    front_vehicle, rear_vehicle, front_gap, rear_gap = _nearest_front_and_rear(
        vehicles=vehicles,
        ego=ego,
        lane_rank=target_lane_rank,
    )
    front_ttc = _ttc_from_gap(front_gap, _speed(ego) - _speed(front_vehicle)) if front_vehicle is not None else None
    rear_ttc = _ttc_from_gap(rear_gap, _speed(rear_vehicle) - _speed(ego)) if rear_vehicle is not None else None
    context.update(
        {
            "front_gap_m": front_gap,
            "rear_gap_m": rear_gap,
            "front_ttc_sec": front_ttc,
            "rear_ttc_sec": rear_ttc,
            "required_front_gap_m": TARGET_FRONT_GAP_REQUIRED_M,
            "required_rear_gap_m": TARGET_REAR_GAP_REQUIRED_M,
            "required_rear_ttc_sec": TARGET_REAR_TTC_REQUIRED_SEC,
        }
    )
    if front_gap is not None and front_gap < TARGET_FRONT_GAP_REQUIRED_M:
        reason = "target_front_gap_below_required"
    elif rear_gap is not None and rear_gap < TARGET_REAR_GAP_REQUIRED_M:
        reason = "target_rear_gap_below_required"
    elif rear_ttc is not None and rear_ttc < TARGET_REAR_TTC_REQUIRED_SEC:
        reason = "target_rear_ttc_below_required"
    else:
        return _result(
            original_action_id=original_action_id,
            action_id=original_action_id,
            applied=False,
            reason="safe",
            shield_type=shield_type,
            **context,
        )
    return _result(
        original_action_id=original_action_id,
        action_id=replacement,
        applied=True,
        reason=reason,
        shield_type=shield_type,
        **context,
    )


def apply_longitudinal_safety_shield(scenario: Any, proposed_action_id: int) -> SafetyShieldResult:
    original_action_id = _safe_int(proposed_action_id, default=IDLE_ACTION_ID)
    shield_type = "longitudinal"
    if _scenario_family(scenario) != "highway":
        return _non_applied(original_action_id, "non_highway_scenario", shield_type)
    if original_action_id not in {FASTER_ACTION_ID, IDLE_ACTION_ID}:
        return _non_applied(original_action_id, "not_longitudinal_risk_action", shield_type)

    ego = _ego_vehicle(scenario)
    replacement = _prefer_action(scenario, (SLOWER_ACTION_ID, IDLE_ACTION_ID))
    if ego is None:
        return _result(
            original_action_id=original_action_id,
            action_id=replacement,
            applied=True,
            reason="missing_ego_vehicle",
            shield_type=shield_type,
        )
    ego_lane_rank = _lane_rank(ego)
    if ego_lane_rank is None:
        return _result(
            original_action_id=original_action_id,
            action_id=replacement,
            applied=True,
            reason="unknown_ego_lane",
            shield_type=shield_type,
        )

    front_vehicle, _rear_vehicle, front_gap, _rear_gap = _nearest_front_and_rear(
        vehicles=_road_vehicles(scenario, ego),
        ego=ego,
        lane_rank=ego_lane_rank,
    )
    ego_speed = _speed(ego)
    projected_ego_speed = ego_speed + (
        ACCELERATION_PROJECTED_SPEED_GAIN_MPS
        if original_action_id == FASTER_ACTION_ID
        else 0.0
    )
    if front_vehicle is None or front_gap is None:
        return _result(
            original_action_id=original_action_id,
            action_id=original_action_id,
            applied=False,
            reason="safe",
            shield_type=shield_type,
            front_ttc_sec=math.inf,
            projected_front_ttc_sec=math.inf,
            projected_ego_speed_mps=projected_ego_speed,
            projection_horizon_sec=PROJECTED_TIME_HORIZON_SEC,
            required_front_gap_m=(
                ACCELERATION_CAUTION_GAP_M
                if original_action_id == FASTER_ACTION_ID
                else FRONT_CRITICAL_GAP_M
            ),
            required_front_ttc_sec=(
                ACCELERATION_CAUTION_TTC_SEC
                if original_action_id == FASTER_ACTION_ID
                else FRONT_CRITICAL_TTC_SEC
            ),
        )

    closing_speed = ego_speed - _speed(front_vehicle)
    front_ttc = _ttc_from_gap(front_gap, closing_speed)
    projected_closing_speed = closing_speed + (
        ACCELERATION_PROJECTED_SPEED_GAIN_MPS
        if original_action_id == FASTER_ACTION_ID
        else 0.0
    )
    projected_front_gap = front_gap - max(0.0, projected_closing_speed) * PROJECTED_TIME_HORIZON_SEC
    projected_ttc = _ttc_from_gap(projected_front_gap, projected_closing_speed)
    context = {
        "front_gap_m": front_gap,
        "front_ttc_sec": front_ttc,
        "projected_front_gap_m": projected_front_gap,
        "projected_front_ttc_sec": projected_ttc,
        "projected_ego_speed_mps": projected_ego_speed,
        "projection_horizon_sec": PROJECTED_TIME_HORIZON_SEC,
    }

    if original_action_id == IDLE_ACTION_ID and (
        front_gap < FRONT_CRITICAL_GAP_M
        or (front_ttc is not None and front_ttc < FRONT_CRITICAL_TTC_SEC)
    ):
        final_action = _prefer_action(scenario, (SLOWER_ACTION_ID, IDLE_ACTION_ID))
        reason = "front_clearance_blocks_idle"
        context.update(
            {
                "required_front_gap_m": FRONT_CRITICAL_GAP_M,
                "required_front_ttc_sec": FRONT_CRITICAL_TTC_SEC,
            }
        )
    elif original_action_id == IDLE_ACTION_ID and (
        projected_front_gap < FRONT_CRITICAL_GAP_M
        or (projected_ttc is not None and projected_ttc < FRONT_CRITICAL_TTC_SEC)
    ):
        final_action = _prefer_action(scenario, (SLOWER_ACTION_ID, IDLE_ACTION_ID))
        reason = "projected_front_clearance_blocks_idle"
        context.update(
            {
                "required_front_gap_m": FRONT_CRITICAL_GAP_M,
                "required_front_ttc_sec": FRONT_CRITICAL_TTC_SEC,
            }
        )
    elif original_action_id == FASTER_ACTION_ID and (
        front_gap < FRONT_CAUTION_GAP_M
        or (front_ttc is not None and front_ttc < FRONT_CAUTION_TTC_SEC)
    ):
        if _front_risk_is_severe(
            front_gap_m=front_gap,
            front_ttc_sec=front_ttc,
            projected_front_gap_m=projected_front_gap,
            projected_front_ttc_sec=projected_ttc,
        ):
            final_action = _prefer_action(scenario, (SLOWER_ACTION_ID, IDLE_ACTION_ID))
        else:
            final_action = _prefer_action(scenario, (IDLE_ACTION_ID, SLOWER_ACTION_ID))
        reason = "front_clearance_blocks_accelerate"
        context.update(
            {
                "required_front_gap_m": FRONT_CAUTION_GAP_M,
                "required_front_ttc_sec": FRONT_CAUTION_TTC_SEC,
            }
        )
    elif original_action_id == FASTER_ACTION_ID and (
        projected_front_gap < ACCELERATION_CAUTION_GAP_M
        or (projected_ttc is not None and projected_ttc < ACCELERATION_CAUTION_TTC_SEC)
    ):
        if _front_risk_is_severe(
            front_gap_m=front_gap,
            front_ttc_sec=front_ttc,
            projected_front_gap_m=projected_front_gap,
            projected_front_ttc_sec=projected_ttc,
        ):
            final_action = _prefer_action(scenario, (SLOWER_ACTION_ID, IDLE_ACTION_ID))
        else:
            final_action = _prefer_action(scenario, (IDLE_ACTION_ID, SLOWER_ACTION_ID))
        reason = "projected_front_gap_blocks_accelerate"
        context.update(
            {
                "required_front_gap_m": ACCELERATION_CAUTION_GAP_M,
                "required_front_ttc_sec": ACCELERATION_CAUTION_TTC_SEC,
            }
        )
    else:
        context.update(
            {
                "required_front_gap_m": (
                    ACCELERATION_CAUTION_GAP_M
                    if original_action_id == FASTER_ACTION_ID
                    else FRONT_CRITICAL_GAP_M
                ),
                "required_front_ttc_sec": (
                    ACCELERATION_CAUTION_TTC_SEC
                    if original_action_id == FASTER_ACTION_ID
                    else FRONT_CRITICAL_TTC_SEC
                ),
            }
        )
        return _result(
            original_action_id=original_action_id,
            action_id=original_action_id,
            applied=False,
            reason="safe",
            shield_type=shield_type,
            **context,
        )
    return _result(
        original_action_id=original_action_id,
        action_id=final_action,
        applied=True,
        reason=reason,
        shield_type=shield_type,
        **context,
    )


def apply_low_speed_recovery_shield(
    scenario: Any,
    proposed_action_id: int,
    safety_shield_applied: bool = False,
) -> SafetyShieldResult:
    original_action_id = _safe_int(proposed_action_id, default=IDLE_ACTION_ID)
    shield_type = "flow_recovery"
    if _scenario_family(scenario) != "highway":
        return _non_applied(original_action_id, "non_highway_scenario", shield_type)
    if safety_shield_applied:
        return _non_applied(original_action_id, "safety_shield_already_applied", shield_type)
    if original_action_id != IDLE_ACTION_ID:
        return _non_applied(original_action_id, "not_idle_action", shield_type)

    available_actions = _available_actions(scenario)
    if FASTER_ACTION_ID not in available_actions:
        return _non_applied(original_action_id, "faster_unavailable", shield_type)

    ego = _ego_vehicle(scenario)
    if ego is None:
        return _non_applied(original_action_id, "missing_ego_vehicle", shield_type)
    ego_lane_rank = _lane_rank(ego)
    if ego_lane_rank is None:
        return _non_applied(original_action_id, "unknown_ego_lane", shield_type)
    ego_speed = _speed(ego)
    if ego_speed >= LOW_SPEED_RECOVERY_FLOOR_MPS:
        return _non_applied(original_action_id, "speed_above_recovery_floor", shield_type)

    front_vehicle, _rear_vehicle, front_gap, _rear_gap = _nearest_front_and_rear(
        vehicles=_road_vehicles(scenario, ego),
        ego=ego,
        lane_rank=ego_lane_rank,
    )
    front_ttc = (
        _ttc_from_gap(front_gap, ego_speed - _speed(front_vehicle))
        if front_vehicle is not None
        else math.inf
    )
    context = {"front_gap_m": front_gap, "front_ttc_sec": front_ttc}
    front_gap_safe = front_vehicle is None or (
        front_gap is not None
        and front_gap >= LOW_SPEED_RECOVERY_FRONT_GAP_M
        and (front_ttc is None or front_ttc >= LOW_SPEED_RECOVERY_TTC_SEC)
    )
    if not front_gap_safe:
        return _result(
            original_action_id=original_action_id,
            action_id=original_action_id,
            applied=False,
            reason="front_gap_not_safe_for_recovery",
            shield_type=shield_type,
            **context,
        )

    return _result(
        original_action_id=original_action_id,
        action_id=FASTER_ACTION_ID,
        applied=True,
        reason="low_speed_recovery_after_front_risk",
        shield_type=shield_type,
        **context,
    )
