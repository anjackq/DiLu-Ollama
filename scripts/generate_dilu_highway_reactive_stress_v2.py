"""Generate the deterministic DiLu highway reactive stress-v2 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "benchmarks" / "dilu_highway_reactive_stress_v2" / "cases.json"
TARGET_SPEEDS = [0, 5, 10, 15, 20, 25, 30]


def vehicle(
    vehicle_id: str,
    role: str,
    lane_offset: int,
    x_offset_m: float,
    speed_mps: float,
    target_speed_mps: float | None = None,
    *,
    enable_lane_change: bool = False,
) -> Dict[str, Any]:
    return {
        "id": vehicle_id,
        "role": role,
        "lane_offset": lane_offset,
        "x_offset_m": round(float(x_offset_m), 3),
        "speed_mps": round(float(speed_mps), 3),
        "target_speed_mps": round(float(target_speed_mps if target_speed_mps is not None else speed_mps), 3),
        "enable_lane_change": bool(enable_lane_change),
    }


def event(
    event_id: str,
    step: int,
    event_type: str,
    vehicle_id: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "id": event_id,
        "step": int(step),
        "type": event_type,
        "vehicle_id": vehicle_id,
    }
    item.update(kwargs)
    return item


def spawn_event(event_id: str, step: int, spec: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": event_id, "step": int(step), "type": "spawn_vehicle", "vehicle": spec}


def base_case(
    *,
    category: str,
    idx: int,
    instruction: str,
    criteria: Dict[str, Any],
    vehicles: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    lanes_count: int = 3,
    time_limit_sec: int = 26,
    ego_lane_rank: int | None = None,
    ego_speed_mps: float = 24.0,
    tags: List[str] | None = None,
) -> Dict[str, Any]:
    if ego_lane_rank is None:
        ego_lane_rank = 2 if lanes_count == 4 else 1
    env_overrides: Dict[str, Any] = {}
    if lanes_count != 3:
        env_overrides["lanes_count"] = int(lanes_count)
        env_overrides["vehicles_count"] = 24
        env_overrides["vehicles_density"] = 0.75
    if time_limit_sec != 26:
        env_overrides["duration"] = max(30, int(time_limit_sec) + 6)
    case: Dict[str, Any] = {
        "case_id": f"{category}_{idx:03d}",
        "category": category,
        "instruction": instruction,
        "seed": 32000 + idx + 1000 * CATEGORY_ORDER[category],
        "time_limit_sec": float(time_limit_sec),
        "success_criteria": {
            "hold_steps": 2,
            "requires_event": True,
            **criteria,
        },
        "scenario_spec": {
            "clear_existing_vehicles": True,
            "ego": {
                "lane_rank": int(ego_lane_rank),
                "x_m": 100.0,
                "speed_mps": round(float(ego_speed_mps), 3),
            },
            "vehicles": vehicles,
            "events": events,
        },
        "tags": tags or ["stress_v2", "passive_trap"],
        "difficulty": "hard",
        "case_group": category,
    }
    if env_overrides:
        case["env_overrides"] = env_overrides
    return case


def front_pack(idx: int, *, lead_gap: float, lead_speed: float, target_offset: int = -1) -> List[Dict[str, Any]]:
    side = target_offset
    opposite = 1 if side < 0 else -1
    return [
        vehicle("slow_lead", "lead", 0, lead_gap, lead_speed, lead_speed),
        vehicle("target_front", "left_front" if side < 0 else "right_front", side, 82 + idx, 25.0, 25.0),
        vehicle("target_rear", "left_rear" if side < 0 else "right_rear", side, -42 - idx, 25.0, 25.0),
        vehicle("opposite_front", "right_front" if opposite > 0 else "left_front", opposite, 48 + idx, 22.0, 22.0),
        vehicle("opposite_rear", "right_rear" if opposite > 0 else "left_rear", opposite, -28 - idx, 24.0, 24.0),
    ]


def mandatory_overtake(idx: int) -> Dict[str, Any]:
    lead_gap = 30 + (idx % 4) * 3
    lead_speed = 11.0 + (idx % 3)
    return base_case(
        category="mandatory_overtake_slow_lead",
        idx=idx,
        instruction="Pass the persistently slower vehicle ahead when the adjacent lane is safe.",
        criteria={
            "type": "mandatory_overtake_slow_lead",
            "direction": "left",
            "pass_margin_m": 12.0,
            "min_final_speed_mps": 22.0,
            "min_progress_m": 92.0,
            "min_survival_steps": 16,
            "max_ttc_danger_rate": 0.25,
            "max_headway_violation_rate": 0.35,
            "max_unsafe_lane_change_attempts": 0,
            "passive_trap": True,
        },
        vehicles=front_pack(idx, lead_gap=lead_gap, lead_speed=lead_speed, target_offset=-1),
        events=[event("lead_settles_slow", 3, "set_target_speed", "slow_lead", target_speed_mps=lead_speed)],
    )


def timed_gap(idx: int) -> Dict[str, Any]:
    vehicles = front_pack(idx, lead_gap=32 + idx % 3, lead_speed=13.0, target_offset=-1)
    vehicles.extend(
        [
            vehicle("early_blocker", "left_rear", -1, -14.0, 30.0, 30.0),
            vehicle("closing_blocker", "left_rear", -1, -80.0, 31.0, 31.0),
        ]
    )
    return base_case(
        category="timed_gap_overtake",
        idx=idx,
        instruction="Wait for the safe passing gap, then use it before it closes.",
        criteria={
            "type": "timed_gap_overtake",
            "direction": "left",
            "opportunity_start_step": 5,
            "opportunity_end_step": 10,
            "pass_margin_m": 10.0,
            "min_final_speed_mps": 22.0,
            "min_progress_m": 86.0,
            "min_survival_steps": 16,
            "max_ttc_danger_rate": 0.25,
            "max_headway_violation_rate": 0.35,
            "max_unsafe_lane_change_attempts": 0,
            "passive_trap": True,
        },
        vehicles=vehicles,
        events=[
            event("gap_opens", 5, "reposition_vehicle", "early_blocker", lane_offset=-1, x_offset_m=-55.0, speed_mps=25.0),
            event("gap_closes", 11, "reposition_vehicle", "closing_blocker", lane_offset=-1, x_offset_m=-14.0, speed_mps=31.0),
        ],
    )


def traffic_jam_escape(idx: int) -> Dict[str, Any]:
    vehicles = [
        vehicle("jam_lead_1", "lead", 0, 20.0, 8.0, 7.0),
        vehicle("jam_lead_2", "front", 0, 38.0, 7.0, 7.0),
        vehicle("jam_lead_3", "front", 0, 58.0, 7.0, 7.0),
        vehicle("escape_front", "left_front", -1, 78.0, 24.0, 24.0),
        vehicle("escape_rear", "left_rear", -1, -46.0, 24.0, 24.0),
        vehicle("right_front", "right_front", 1, 28.0, 10.0, 10.0),
        vehicle("right_rear", "right_rear", 1, -22.0, 18.0, 18.0),
    ]
    return base_case(
        category="traffic_jam_escape",
        idx=idx,
        instruction="Escape the slow traffic jam lane when a safe adjacent gap is available.",
        criteria={
            "type": "traffic_jam_escape",
            "direction": "left",
            "opportunity_start_step": 3,
            "opportunity_end_step": 12,
            "min_final_speed_mps": 18.0,
            "min_progress_m": 66.0,
            "min_survival_steps": 18,
            "max_ttc_danger_rate": 0.30,
            "max_headway_violation_rate": 0.40,
            "max_unsafe_lane_change_attempts": 0,
            "passive_trap": True,
        },
        vehicles=vehicles,
        events=[
            event("jam_wave_slows", 3, "set_speed", "jam_lead_1", speed_mps=5.0),
            event("escape_lane_clear", 4, "reposition_vehicle", "escape_rear", lane_offset=-1, x_offset_m=-60.0, speed_mps=23.0),
        ],
    )


def traffic_jam_patience(idx: int) -> Dict[str, Any]:
    vehicles = [
        vehicle("jam_lead", "lead", 0, 24.0, 8.0, 7.0),
        vehicle("false_gap_rear", "left_rear", -1, -10.0, 31.0, 31.0),
        vehicle("left_front", "left_front", -1, 64.0, 22.0, 22.0),
        vehicle("right_front", "right_front", 1, 34.0, 11.0, 11.0),
        vehicle("right_rear", "right_rear", 1, -18.0, 20.0, 20.0),
    ]
    return base_case(
        category="traffic_jam_patience",
        idx=idx,
        instruction="Stay patient in slow traffic and avoid forcing an unsafe lane change.",
        criteria={
            "type": "traffic_jam_patience",
            "safe_window_start_step": 12,
            "min_progress_m": 32.0,
            "min_survival_steps": 18,
            "max_lane_changes": 1,
            "max_ttc_danger_rate": 0.35,
            "max_headway_violation_rate": 0.45,
            "max_unsafe_lane_change_attempts": 0,
            "passive_trap": True,
        },
        vehicles=vehicles,
        events=[
            event("rear_closes_false_gap", 4, "reposition_vehicle", "false_gap_rear", lane_offset=-1, x_offset_m=-8.0, speed_mps=32.0),
            event("gap_finally_safe", 12, "reposition_vehicle", "false_gap_rear", lane_offset=-1, x_offset_m=-55.0, speed_mps=23.0),
        ],
    )


def multi_lane_route(idx: int) -> Dict[str, Any]:
    vehicles = [
        vehicle("left_fast", "left_front", -1, 50.0, 28.0, 28.0),
        vehicle("right_lane_front", "right_front", 1, 86.0, 25.0, 25.0),
        vehicle("right_lane_rear", "right_rear", 1, -48.0, 25.0, 25.0),
        vehicle("far_lead", "lead", 0, 80.0, 24.0, 24.0),
        vehicle("outer_right_front", "right_front", -2, 60.0, 23.0, 23.0),
        vehicle("outer_right_rear", "right_rear", -2, -35.0, 24.0, 24.0),
    ]
    return base_case(
        category="multi_lane_route_discipline",
        idx=idx,
        instruction="Move toward the appropriate cruising lane while maintaining highway flow.",
        lanes_count=4,
        ego_lane_rank=2,
        criteria={
            "type": "multi_lane_route_discipline",
            "direction": "right",
            "min_avg_speed_mps": 20.0,
            "min_progress_m": 82.0,
            "min_survival_steps": 18,
            "max_lane_changes": 2,
            "max_ttc_danger_rate": 0.25,
            "max_headway_violation_rate": 0.35,
            "max_unsafe_lane_change_attempts": 0,
            "passive_trap": True,
        },
        vehicles=vehicles,
        events=[event("right_lane_confirmed_clear", 5, "reposition_vehicle", "right_lane_rear", lane_offset=1, x_offset_m=-58.0, speed_mps=24.0)],
    )


def bottleneck(idx: int) -> Dict[str, Any]:
    vehicles = front_pack(idx, lead_gap=44.0, lead_speed=18.0, target_offset=-1)
    vehicles.append(vehicle("bottleneck_blocker", "lead", 0, 76.0, 15.0, 8.0))
    return base_case(
        category="bottleneck_merge_pressure",
        idx=idx,
        instruction="Anticipate the lane bottleneck ahead and merge before late braking is required.",
        criteria={
            "type": "bottleneck_merge_pressure",
            "direction": "left",
            "latest_maneuver_step": 10,
            "min_progress_m": 76.0,
            "min_survival_steps": 18,
            "max_ttc_danger_rate": 0.25,
            "max_headway_violation_rate": 0.35,
            "max_unsafe_lane_change_attempts": 0,
            "passive_trap": True,
        },
        vehicles=vehicles,
        events=[
            event("bottleneck_stops", 6, "set_speed", "bottleneck_blocker", speed_mps=2.0),
            event("lead_slows_for_bottleneck", 7, "set_target_speed", "slow_lead", target_speed_mps=8.0),
        ],
    )


def cut_in_recover(idx: int) -> Dict[str, Any]:
    vehicles = [
        vehicle("far_lead", "lead", 0, 95.0, 24.0, 24.0),
        vehicle("cutter", "left_front", -1, 42.0, 24.0, 16.0),
        vehicle("left_rear", "left_rear", -1, -42.0, 24.0, 24.0),
        vehicle("right_front", "right_front", 1, 58.0, 22.0, 22.0),
    ]
    return base_case(
        category="cut_in_then_recover",
        idx=idx,
        instruction="Brake for the cut-in hazard, then recover to normal flow when the gap clears.",
        criteria={
            "type": "cut_in_then_recover",
            "clear_front_gap_m": 26.0,
            "clear_front_ttc_sec": 4.0,
            "min_recovery_speed_mps": 20.0,
            "min_survival_steps": 18,
            "max_ttc_danger_rate": 0.35,
            "max_headway_violation_rate": 0.45,
            "max_flap_accel_decel_count": 5,
            "requires_brake_action": True,
            "passive_trap": True,
        },
        vehicles=vehicles,
        events=[
            event("cutter_enters", 3, "reposition_vehicle", "cutter", lane_offset=0, lane_reference="current_ego", x_offset_m=19.0, speed_mps=15.0, target_speed_mps=15.0),
            event("cutter_clears", 10, "reposition_vehicle", "cutter", lane_offset=-1, x_offset_m=70.0, speed_mps=24.0, target_speed_mps=24.0),
        ],
    )


def false_opening(idx: int) -> Dict[str, Any]:
    vehicles = [
        vehicle("far_lead", "lead", 0, 88.0, 24.0, 24.0),
        vehicle("tempting_front", "left_front", -1, 64.0, 22.0, 22.0),
        vehicle("closing_rear", "left_rear", -1, -18.0, 31.0, 31.0),
        vehicle("right_front", "right_front", 1, 48.0, 22.0, 22.0),
        vehicle("right_rear", "right_rear", 1, -26.0, 23.0, 23.0),
    ]
    return base_case(
        category="false_opening_stability",
        idx=idx,
        instruction="Hold a stable course when a nearby lane opening is not actually useful.",
        criteria={
            "type": "false_opening_stability",
            "min_avg_speed_mps": 20.0,
            "min_survival_steps": 18,
            "max_lane_changes": 0,
            "max_flap_accel_decel_count": 2,
            "max_ttc_danger_rate": 0.25,
            "max_headway_violation_rate": 0.35,
            "passive_trap": False,
        },
        vehicles=vehicles,
        events=[
            event("false_gap_appears", 4, "reposition_vehicle", "tempting_front", lane_offset=-1, x_offset_m=76.0, speed_mps=23.0),
            event("rear_closes_gap", 6, "reposition_vehicle", "closing_rear", lane_offset=-1, x_offset_m=-12.0, speed_mps=32.0),
        ],
    )


def dense_four_lane(idx: int) -> Dict[str, Any]:
    vehicles: List[Dict[str, Any]] = []
    for lane_offset in [-2, -1, 0, 1]:
        for pos_idx, x_offset in enumerate([-58.0, -28.0, 34.0, 68.0, 105.0], start=1):
            if lane_offset == 0 and x_offset < 0:
                continue
            vehicles.append(
                vehicle(
                    f"dense_{lane_offset}_{pos_idx}".replace("-", "m"),
                    "traffic" if x_offset > 0 else "rear",
                    lane_offset,
                    x_offset + (idx % 3) * 2,
                    18.0 + ((pos_idx + idx) % 5),
                    20.0 + ((pos_idx + idx) % 4),
                    enable_lane_change=pos_idx in {2, 4},
                )
            )
    return base_case(
        category="dense_four_lane_flow",
        idx=idx,
        instruction="Make smooth safe progress through dense four-lane highway traffic.",
        lanes_count=4,
        ego_lane_rank=2,
        time_limit_sec=28,
        criteria={
            "type": "dense_four_lane_flow",
            "min_avg_speed_mps": 19.5,
            "min_survival_steps": 20,
            "max_lane_changes": 3,
            "max_flap_accel_decel_count": 4,
            "max_ttc_danger_rate": 0.25,
            "max_headway_violation_rate": 0.40,
            "passive_trap": True,
        },
        vehicles=vehicles[:18],
        events=[
            spawn_event("dynamic_rear_appears", 5, vehicle("dynamic_rear", "rear", -1, -46.0, 25.0, 25.0, enable_lane_change=True)),
            event("front_flow_recovers", 12, "set_target_speed", "dense_0_3", target_speed_mps=24.0),
        ],
    )


def stop_go_wave(idx: int) -> Dict[str, Any]:
    vehicles = [
        vehicle("wave_lead_1", "lead", 0, 28.0, 18.0, 18.0),
        vehicle("wave_lead_2", "front", 0, 52.0, 18.0, 18.0),
        vehicle("wave_lead_3", "front", 0, 78.0, 20.0, 20.0),
        vehicle("left_front", "left_front", -1, 46.0, 19.0, 19.0),
        vehicle("left_rear", "left_rear", -1, -26.0, 24.0, 24.0),
        vehicle("right_front", "right_front", 1, 48.0, 19.0, 19.0),
        vehicle("right_rear", "right_rear", 1, -22.0, 24.0, 24.0),
    ]
    return base_case(
        category="stop_go_wave_response",
        idx=idx,
        instruction="Respond smoothly to a stop-and-go traffic wave and recover when flow resumes.",
        time_limit_sec=28,
        criteria={
            "type": "stop_go_wave_response",
            "clear_front_gap_m": 24.0,
            "clear_front_ttc_sec": 4.0,
            "min_recovery_speed_mps": 18.0,
            "min_progress_m": 48.0,
            "min_survival_steps": 20,
            "max_flap_accel_decel_count": 5,
            "max_ttc_danger_rate": 0.35,
            "max_headway_violation_rate": 0.45,
            "passive_trap": True,
        },
        vehicles=vehicles,
        events=[
            event("wave_brakes", 3, "set_speed", "wave_lead_1", speed_mps=4.0),
            event("wave_rolls", 8, "set_speed", "wave_lead_2", speed_mps=7.0),
            event("wave_clears", 14, "set_speed", "wave_lead_1", speed_mps=22.0),
        ],
    )


CATEGORY_BUILDERS: Dict[str, Callable[[int], Dict[str, Any]]] = {
    "mandatory_overtake_slow_lead": mandatory_overtake,
    "timed_gap_overtake": timed_gap,
    "traffic_jam_escape": traffic_jam_escape,
    "traffic_jam_patience": traffic_jam_patience,
    "multi_lane_route_discipline": multi_lane_route,
    "bottleneck_merge_pressure": bottleneck,
    "cut_in_then_recover": cut_in_recover,
    "false_opening_stability": false_opening,
    "dense_four_lane_flow": dense_four_lane,
    "stop_go_wave_response": stop_go_wave,
}
CATEGORY_ORDER = {category: idx for idx, category in enumerate(CATEGORY_BUILDERS, start=1)}


def build_case_set() -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for category, builder in CATEGORY_BUILDERS.items():
        for idx in range(1, 13):
            cases.append(builder(idx))
    return {
        "benchmark_name": "dilu_highway_reactive_stress_v2",
        "version": "2.0",
        "description": (
            "Decision-forcing highway stress benchmark with passive-baseline traps, "
            "3+4 lane layouts, dense traffic, and deterministic traffic-jam events."
        ),
        "target_env_id": "highway-fast-v0",
        "scenario_family": "highway",
        "defaults": {
            "time_limit_sec": 26,
            "difficulty": "hard",
            "case_group": "stress_highway_v2",
            "env_overrides": {
                "lanes_count": 3,
                "vehicles_count": 16,
                "vehicles_density": 0.65,
                "duration": 32,
                "simulation_frequency": 5,
                "policy_frequency": 1,
                "ego_spacing": 1.5,
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": TARGET_SPEEDS,
                },
            },
            "success_criteria": {
                "hold_steps": 2,
                "requires_event": True,
            },
        },
        "cases": cases,
    }


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the generated case set to benchmarks/.")
    parser.add_argument("--check", action="store_true", help="Fail if the checked-in case set differs.")
    args = parser.parse_args()

    data = build_case_set()
    rendered = canonical_json(data)
    if args.write:
        TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
        TARGET_PATH.write_text(rendered, encoding="utf-8")
        print(f"wrote {TARGET_PATH.relative_to(REPO_ROOT)} with {len(data['cases'])} cases")
    if args.check:
        if not TARGET_PATH.exists():
            raise SystemExit(f"missing {TARGET_PATH}")
        current = TARGET_PATH.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit(f"{TARGET_PATH.relative_to(REPO_ROOT)} is out of date; run with --write")
        print(f"ok {TARGET_PATH.relative_to(REPO_ROOT)}")
    if not args.write and not args.check:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
