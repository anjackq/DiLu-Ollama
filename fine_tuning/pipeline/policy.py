from typing import Dict, Tuple

import numpy as np


def _ttc(distance_m: float, closing_speed_mps: float) -> float:
    if closing_speed_mps <= 0:
        return float("inf")
    return distance_m / max(closing_speed_mps, 1e-6)


def init_expert_state() -> Dict[str, int]:
    return {
        "lane_change_cooldown": 0,
        "post_brake_cooldown": 0,
        "blocked_steps": 0,
        "last_action": 1,
        "steps_since_last_faster": 999,
        "steps_since_last_slower": 999,
    }


def _decrement_cooldowns(state: Dict[str, int]) -> None:
    state["lane_change_cooldown"] = max(0, int(state.get("lane_change_cooldown", 0)) - 1)
    state["post_brake_cooldown"] = max(0, int(state.get("post_brake_cooldown", 0)) - 1)
    state["steps_since_last_faster"] = int(state.get("steps_since_last_faster", 999)) + 1
    state["steps_since_last_slower"] = int(state.get("steps_since_last_slower", 999)) + 1


def _apply_state_transition(state: Dict[str, int], action_id: int, emergency_or_protective: bool) -> None:
    state["last_action"] = int(action_id)
    if action_id in (0, 2):
        state["lane_change_cooldown"] = 4
    if action_id == 4:
        state["steps_since_last_slower"] = 0
        if not emergency_or_protective:
            state["post_brake_cooldown"] = 2
    if action_id == 3:
        state["steps_since_last_faster"] = 0


def _lane_safety_metrics(vehicle, road, target_lane_id: int):
    current_lane = vehicle.lane_index
    target_lane_index = (current_lane[0], current_lane[1], target_lane_id)
    target_front, target_rear = road.neighbour_vehicles(vehicle, target_lane_index)

    front_gap = float("inf")
    rear_gap = float("inf")
    rear_ttc = float("inf")
    rear_closing = 0.0

    if target_front is not None:
        front_gap = float(np.linalg.norm(vehicle.position - target_front.position))
    if target_rear is not None:
        rear_gap = float(np.linalg.norm(vehicle.position - target_rear.position))
        rear_closing = float(target_rear.speed - vehicle.speed)
        rear_ttc = _ttc(rear_gap, rear_closing)

    safe = front_gap >= 28 and rear_gap >= 20 and not (rear_closing > 0 and rear_ttc < 3.5)
    return safe, front_gap, rear_gap, rear_ttc


def expert_decision_v3_balanced(sce, state: Dict[str, int]) -> Tuple[int, str, Dict[str, int]]:
    vehicle = sce.env.unwrapped.vehicle
    road = sce.env.unwrapped.road
    cfg = getattr(sce.env.unwrapped, "config", {}) or {}
    lane_count = int(cfg.get("lanes_count", 4))
    lane_id = int(vehicle.lane_index[2])

    _decrement_cooldowns(state)

    front_vehicle, _ = road.neighbour_vehicles(vehicle, vehicle.lane_index)
    front_gap = float("inf")
    rel_speed = 0.0
    ttc_front = float("inf")
    if front_vehicle is not None:
        front_gap = float(np.linalg.norm(vehicle.position - front_vehicle.position))
        rel_speed = float(vehicle.speed - front_vehicle.speed)
        ttc_front = _ttc(front_gap, rel_speed)

    emergency = front_gap < 12 or ttc_front < 1.8
    protective = front_gap < 22 or ttc_front < 3.0
    blocked = front_gap < 65 and rel_speed > -0.5
    if blocked:
        state["blocked_steps"] = int(state.get("blocked_steps", 0)) + 1
    else:
        state["blocked_steps"] = max(0, int(state.get("blocked_steps", 0)) - 1)

    left_safe = False
    right_safe = False
    left_metrics = None
    right_metrics = None
    if lane_id > 0:
        left_safe, lf_gap, lr_gap, lr_ttc = _lane_safety_metrics(vehicle, road, lane_id - 1)
        left_metrics = (lf_gap, lr_gap, lr_ttc)
    if lane_id < lane_count - 1:
        right_safe, rf_gap, rr_gap, rr_ttc = _lane_safety_metrics(vehicle, road, lane_id + 1)
        right_metrics = (rf_gap, rr_gap, rr_ttc)

    action = 1
    reason = "Maintain lane and speed under nominal conditions."
    emergency_or_protective = False

    if emergency:
        action = 4
        reason = f"Emergency decel: front_gap={front_gap:.1f}m, ttc={ttc_front:.2f}s."
        emergency_or_protective = True
    elif protective:
        if state["lane_change_cooldown"] == 0 and left_safe:
            action = 0
            reason = "Protective escape: left lane fully safe, change left."
        elif state["lane_change_cooldown"] == 0 and right_safe:
            action = 2
            reason = "Protective escape: right lane fully safe, change right."
        else:
            action = 4
            reason = f"Protective decel: front_gap={front_gap:.1f}m, ttc={ttc_front:.2f}s."
            emergency_or_protective = True
    else:
        if blocked and state["lane_change_cooldown"] == 0:
            if left_safe:
                action = 0
                reason = "Blocked mode: safe left overtake opportunity."
            elif right_safe:
                action = 2
                reason = "Blocked mode: left unavailable, safe right merge."
            else:
                action = 4
                reason = "Blocked mode: no safe lane change, controlled deceleration."
        else:
            strategic_overtake = (
                front_vehicle is not None
                and front_gap < 90
                and rel_speed > 1.0
                and state["lane_change_cooldown"] == 0
            )
            if strategic_overtake and left_safe:
                action = 0
                reason = "Strategic overtake: left lane safe and front traffic is slower."
            elif strategic_overtake and right_safe:
                action = 2
                reason = "Strategic overtake: right lane safe and front traffic is slower."
            else:
                target_speed = 28.0
                accel_allowed = (
                    vehicle.speed < target_speed
                    and front_gap > 55
                    and ttc_front > 6.0
                    and state["post_brake_cooldown"] == 0
                )
                # Suppress immediate FASTER->SLOWER->FASTER oscillation.
                if (
                    accel_allowed
                    and int(state.get("steps_since_last_slower", 999)) <= 3
                    and int(state.get("steps_since_last_faster", 999)) <= 3
                ):
                    accel_allowed = False
                    reason = "Hysteresis: suppress accel after recent brake cycle."

                if accel_allowed:
                    action = 3
                    reason = f"Clear road: gap={front_gap:.1f}m, ttc={ttc_front:.2f}s, accelerate."
                else:
                    action = 1
                    if not reason.startswith("Hysteresis"):
                        reason = "No high-confidence maneuver; prefer idle."

    available_actions = list(sce.env.unwrapped.get_available_actions())
    if action not in available_actions:
        if 4 in available_actions:
            action = 4
            reason += " Action unavailable fallback: SLOWER."
        elif 1 in available_actions:
            action = 1
            reason += " Action unavailable fallback: IDLE."
        else:
            action = int(available_actions[0])
            reason += f" Action unavailable fallback: {action}."

    _apply_state_transition(state, action, emergency_or_protective)
    return int(action), reason, state


def expert_decision_v2_left_pass_preferred(sce) -> Tuple[int, str]:
    # Backward-compatible stateless wrapper.
    action, reasoning, _ = expert_decision_v3_balanced(sce, init_expert_state())
    return int(action), reasoning
