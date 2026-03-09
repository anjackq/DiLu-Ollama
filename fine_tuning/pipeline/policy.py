import numpy as np
from typing import Tuple


def expert_decision_v2_left_pass_preferred(sce) -> Tuple[int, str]:
    """
    Rule-based expert with right-hand traffic semantics:
    prefer left lane for overtaking when blocked by slower front vehicle.

    Actions:
      0: LANE_LEFT
      1: IDLE
      2: LANE_RIGHT
      3: FASTER
      4: SLOWER
    """
    vehicle = sce.env.unwrapped.vehicle
    road = sce.env.unwrapped.road

    front_vehicle, _ = road.neighbour_vehicles(vehicle, vehicle.lane_index)
    front_dist = float("inf")
    if front_vehicle is not None:
        front_dist = float(np.linalg.norm(vehicle.position - front_vehicle.position))

    def lane_change_safe(target_lane_id: int):
        current_lane = vehicle.lane_index
        target_lane_index = (current_lane[0], current_lane[1], target_lane_id)
        try:
            target_front, target_rear = road.neighbour_vehicles(vehicle, target_lane_index)
        except Exception:
            return False

        if target_front is not None:
            gap = float(np.linalg.norm(vehicle.position - target_front.position))
            rel = vehicle.speed - target_front.speed
            if gap < 15:
                return False
            if gap < 25 and rel > 0:
                return False

        if target_rear is not None:
            gap = float(np.linalg.norm(vehicle.position - target_rear.position))
            rear_closing = target_rear.speed - vehicle.speed
            if gap < 10:
                return False
            if gap < 20 and rear_closing > 2.0:
                return False

        return True

    target_speed = 28.0
    lane_id = int(vehicle.lane_index[2])
    cfg = getattr(sce.env.unwrapped, "config", {}) or {}
    lane_count = int(cfg.get("lanes_count", 4))

    # Hard safety first.
    if front_vehicle is not None:
        relative_speed = vehicle.speed - front_vehicle.speed
        if (front_dist < 25 and relative_speed > 0) or front_dist < 10:
            return 4, f"Safety critical: front vehicle {front_dist:.1f}m ahead; decelerate."

    # Overtake policy: prefer LEFT lane on right-hand traffic roads.
    if front_vehicle is not None and front_dist < 40 and vehicle.speed < target_speed:
        if lane_id > 0 and lane_change_safe(lane_id - 1):
            return 0, "Blocked by slower lead vehicle; safe gap on left; change left to overtake."
        if lane_id < lane_count - 1 and lane_change_safe(lane_id + 1):
            return 2, "Blocked by slower lead vehicle; left unavailable; safe right merge."
        if front_dist < 25:
            return 4, "No safe lane change gap while blocked; decelerate."

    if (front_vehicle is None or front_dist > 50) and vehicle.speed < target_speed:
        return 3, f"Road ahead is clear and speed {vehicle.speed:.1f}m/s is below target; accelerate."

    return 1, "Maintain lane and speed under nominal conditions."

