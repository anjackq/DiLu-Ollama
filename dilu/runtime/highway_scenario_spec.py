import copy
from typing import Any, Dict, List, MutableSet, Optional, Tuple

import numpy as np
from highway_env.vehicle.behavior import IDMVehicle


LaneIndex = Tuple[str, str, int]
_EVENT_TYPES = {
    "reposition_vehicle",
    "set_speed",
    "set_target_speed",
    "set_lane_change",
    "spawn_vehicle",
}
_LANE_REFERENCES = {"scenario_ego", "current_ego", "vehicle_current"}


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"scenario_spec.{field_name} must be numeric.") from exc


def _as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"scenario_spec.{field_name} must be an integer.") from exc


def _validate_speed(value: Any, field_name: str) -> float:
    speed = _as_float(value, field_name)
    if speed < 0:
        raise ValueError(f"scenario_spec.{field_name} must be non-negative.")
    return speed


def _normalize_lane_reference(value: Any, field_name: str) -> str:
    lane_reference = str(value or "scenario_ego").strip().lower()
    if lane_reference not in _LANE_REFERENCES:
        raise ValueError(
            f"scenario_spec.{field_name} has unsupported lane_reference `{lane_reference}`; "
            f"allowed={sorted(_LANE_REFERENCES)}."
        )
    return lane_reference


def _normalize_vehicle_spec(item: Dict[str, Any], field_prefix: str, seen_ids: Optional[set] = None) -> Dict[str, Any]:
    vehicle_id = str(item.get("id") or "").strip()
    if not vehicle_id:
        raise ValueError(f"scenario_spec.{field_prefix}.id cannot be empty.")
    if seen_ids is not None:
        if vehicle_id in seen_ids:
            raise ValueError(f"scenario_spec contains duplicate vehicle id `{vehicle_id}`.")
        seen_ids.add(vehicle_id)

    vehicle: Dict[str, Any] = {
        "id": vehicle_id,
        "role": str(item.get("role") or "traffic").strip() or "traffic",
        "speed_mps": _validate_speed(item.get("speed_mps", 22.0), f"{field_prefix}.speed_mps"),
        "enable_lane_change": bool(item.get("enable_lane_change", False)),
    }
    vehicle["target_speed_mps"] = _validate_speed(
        item.get("target_speed_mps", vehicle["speed_mps"]),
        f"{field_prefix}.target_speed_mps",
    )

    if "lane_rank" in item:
        vehicle["lane_rank"] = _as_int(item["lane_rank"], f"{field_prefix}.lane_rank")
    elif "lane_offset" in item:
        vehicle["lane_offset"] = _as_int(item["lane_offset"], f"{field_prefix}.lane_offset")
    else:
        vehicle["lane_offset"] = 0

    if "x_m" in item:
        vehicle["x_m"] = _as_float(item["x_m"], f"{field_prefix}.x_m")
    elif "x_offset_m" in item:
        x_offset = _as_float(item["x_offset_m"], f"{field_prefix}.x_offset_m")
        role = vehicle["role"].lower()
        if role in {"lead", "front", "left_front", "right_front"} and x_offset <= 0:
            raise ValueError(f"scenario_spec vehicle `{vehicle_id}` front-role x_offset_m must be positive.")
        if role in {"rear", "left_rear", "right_rear"} and x_offset >= 0:
            raise ValueError(f"scenario_spec vehicle `{vehicle_id}` rear-role x_offset_m must be negative.")
        vehicle["x_offset_m"] = x_offset
    else:
        raise ValueError(f"scenario_spec vehicle `{vehicle_id}` must define x_m or x_offset_m.")
    return vehicle


def _normalize_event(
    item: Dict[str, Any],
    field_prefix: str,
    initial_vehicle_ids: set,
    all_vehicle_ids: set,
    seen_event_ids: set,
) -> Dict[str, Any]:
    event_id = str(item.get("id") or "").strip()
    if not event_id:
        raise ValueError(f"scenario_spec.{field_prefix}.id cannot be empty.")
    if event_id in seen_event_ids:
        raise ValueError(f"scenario_spec contains duplicate event id `{event_id}`.")
    seen_event_ids.add(event_id)

    event_type = str(item.get("type") or "").strip().lower()
    if event_type not in _EVENT_TYPES:
        raise ValueError(
            f"scenario_spec event `{event_id}` has unsupported type `{event_type or 'missing'}`."
        )
    step = _as_int(item.get("step"), f"{field_prefix}.step")
    if step < 1:
        raise ValueError(f"scenario_spec event `{event_id}` step must be >= 1.")

    event: Dict[str, Any] = {"id": event_id, "type": event_type, "step": step}
    if event_type == "spawn_vehicle":
        vehicle_raw = item.get("vehicle")
        if not isinstance(vehicle_raw, dict):
            raise ValueError(f"scenario_spec event `{event_id}` must define a vehicle object.")
        vehicle = _normalize_vehicle_spec(vehicle_raw, f"{field_prefix}.vehicle")
        if "lane_offset" in vehicle:
            vehicle["lane_reference"] = _normalize_lane_reference(
                vehicle_raw.get("lane_reference", item.get("lane_reference", "scenario_ego")),
                f"{field_prefix}.vehicle.lane_reference",
            )
        vehicle_id = vehicle["id"]
        if vehicle_id in all_vehicle_ids:
            raise ValueError(f"scenario_spec contains duplicate vehicle id `{vehicle_id}`.")
        all_vehicle_ids.add(vehicle_id)
        event["vehicle"] = vehicle
        return event

    vehicle_id = str(item.get("vehicle_id") or "").strip()
    if not vehicle_id:
        raise ValueError(f"scenario_spec event `{event_id}` must define vehicle_id.")
    if vehicle_id not in all_vehicle_ids and vehicle_id not in initial_vehicle_ids:
        raise ValueError(f"scenario_spec event `{event_id}` references unknown vehicle `{vehicle_id}`.")
    event["vehicle_id"] = vehicle_id

    if "lane_rank" in item:
        event["lane_rank"] = _as_int(item["lane_rank"], f"{field_prefix}.lane_rank")
    if "lane_offset" in item:
        event["lane_offset"] = _as_int(item["lane_offset"], f"{field_prefix}.lane_offset")
        event["lane_reference"] = _normalize_lane_reference(
            item.get("lane_reference", "scenario_ego"),
            f"{field_prefix}.lane_reference",
        )
    elif "lane_reference" in item:
        event["lane_reference"] = _normalize_lane_reference(
            item["lane_reference"],
            f"{field_prefix}.lane_reference",
        )
    if "x_m" in item:
        event["x_m"] = _as_float(item["x_m"], f"{field_prefix}.x_m")
    if "x_offset_m" in item:
        event["x_offset_m"] = _as_float(item["x_offset_m"], f"{field_prefix}.x_offset_m")
    if "speed_mps" in item:
        event["speed_mps"] = _validate_speed(item["speed_mps"], f"{field_prefix}.speed_mps")
    if "target_speed_mps" in item:
        event["target_speed_mps"] = _validate_speed(
            item["target_speed_mps"],
            f"{field_prefix}.target_speed_mps",
        )
    if "enable_lane_change" in item:
        event["enable_lane_change"] = bool(item["enable_lane_change"])
    if event_type == "set_speed" and "speed_mps" not in event:
        raise ValueError(f"scenario_spec event `{event_id}` must define speed_mps.")
    if event_type == "set_target_speed" and "target_speed_mps" not in event:
        raise ValueError(f"scenario_spec event `{event_id}` must define target_speed_mps.")
    if event_type == "set_lane_change" and "enable_lane_change" not in event:
        raise ValueError(f"scenario_spec event `{event_id}` must define enable_lane_change.")
    return event


def normalize_scenario_spec(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("scenario_spec must be a JSON object.")

    spec = copy.deepcopy(raw)
    spec["clear_existing_vehicles"] = bool(spec.get("clear_existing_vehicles", True))

    ego = spec.get("ego") or {}
    if not isinstance(ego, dict):
        raise ValueError("scenario_spec.ego must be a JSON object.")
    normalized_ego: Dict[str, Any] = {}
    if "lane_rank" in ego:
        normalized_ego["lane_rank"] = _as_int(ego["lane_rank"], "ego.lane_rank")
    if "x_m" in ego:
        normalized_ego["x_m"] = _as_float(ego["x_m"], "ego.x_m")
    if "speed_mps" in ego:
        normalized_ego["speed_mps"] = _validate_speed(ego["speed_mps"], "ego.speed_mps")
    spec["ego"] = normalized_ego

    vehicles = spec.get("vehicles") or []
    if not isinstance(vehicles, list):
        raise ValueError("scenario_spec.vehicles must be a list.")
    seen_ids: set = set()
    normalized_vehicles: List[Dict[str, Any]] = []
    for idx, item in enumerate(vehicles, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"scenario_spec.vehicles[{idx}] must be a JSON object.")
        item = dict(item)
        item.setdefault("id", f"vehicle_{idx:02d}")
        vehicle = _normalize_vehicle_spec(item, f"vehicles[{idx}]", seen_ids)
        normalized_vehicles.append(vehicle)

    spec["vehicles"] = normalized_vehicles

    events = spec.get("events") or []
    if not isinstance(events, list):
        raise ValueError("scenario_spec.events must be a list.")
    all_vehicle_ids = set(seen_ids)
    seen_event_ids: set = set()
    normalized_events: List[Dict[str, Any]] = []
    for idx, item in enumerate(events, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"scenario_spec.events[{idx}] must be a JSON object.")
        normalized_events.append(
            _normalize_event(
                dict(item),
                f"events[{idx}]",
                initial_vehicle_ids=seen_ids,
                all_vehicle_ids=all_vehicle_ids,
                seen_event_ids=seen_event_ids,
            )
        )
    spec["events"] = normalized_events
    return spec


def scenario_spec_summary(case: Dict[str, Any]) -> Dict[str, Any]:
    spec = normalize_scenario_spec((case or {}).get("scenario_spec"))
    if not spec:
        return {}
    return {
        "clear_existing_vehicles": bool(spec.get("clear_existing_vehicles", True)),
        "ego": copy.deepcopy(spec.get("ego") or {}),
        "vehicles": copy.deepcopy(spec.get("vehicles") or []),
        "events": copy.deepcopy(spec.get("events") or []),
        "vehicle_count": len(spec.get("vehicles") or []),
        "event_count": len(spec.get("events") or []),
    }


def _lane_graph_endpoints(env: Any) -> Tuple[str, str]:
    ego = getattr(env.unwrapped, "vehicle", None)
    lane_index = getattr(ego, "lane_index", None)
    if lane_index and len(lane_index) >= 2:
        return str(lane_index[0]), str(lane_index[1])
    graph = getattr(getattr(env.unwrapped, "road", None), "network", None).graph
    lane_from = next(iter(graph))
    lane_to = next(iter(graph[lane_from]))
    return str(lane_from), str(lane_to)


def _lane_count(env: Any, lane_from: str, lane_to: str) -> int:
    graph = getattr(env.unwrapped.road.network, "graph", {})
    try:
        return len(graph[lane_from][lane_to])
    except Exception as exc:
        raise ValueError(f"Cannot resolve lanes for highway segment {lane_from!r}->{lane_to!r}.") from exc


def _lane_index(env: Any, lane_rank: int) -> LaneIndex:
    lane_from, lane_to = _lane_graph_endpoints(env)
    count = _lane_count(env, lane_from, lane_to)
    if lane_rank < 0 or lane_rank >= count:
        raise ValueError(f"scenario_spec lane_rank {lane_rank} is outside available lanes 0..{count - 1}.")
    return lane_from, lane_to, int(lane_rank)


def _vehicle_x(vehicle: Any) -> Optional[float]:
    try:
        return float(vehicle.position[0])
    except Exception:
        return None


def _set_vehicle_state(vehicle: Any, env: Any, lane_rank: int, x_m: float, speed_mps: float) -> LaneIndex:
    lane_index = _lane_index(env, lane_rank)
    lane = env.unwrapped.road.network.get_lane(lane_index)
    vehicle.lane_index = lane_index
    if hasattr(vehicle, "target_lane_index"):
        vehicle.target_lane_index = lane_index
    vehicle.position = np.array(lane.position(float(x_m), 0.0), dtype=float)
    vehicle.heading = float(lane.heading_at(float(x_m)))
    vehicle.speed = float(speed_mps)
    if hasattr(vehicle, "target_speed"):
        vehicle.target_speed = float(speed_mps)
    return lane_index


def _move_vehicle(
    vehicle: Any,
    env: Any,
    lane_rank: int,
    x_m: float,
    speed_mps: Optional[float] = None,
    target_speed_mps: Optional[float] = None,
) -> LaneIndex:
    lane_index = _lane_index(env, lane_rank)
    lane = env.unwrapped.road.network.get_lane(lane_index)
    vehicle.lane_index = lane_index
    if hasattr(vehicle, "target_lane_index"):
        vehicle.target_lane_index = lane_index
    vehicle.position = np.array(lane.position(float(x_m), 0.0), dtype=float)
    vehicle.heading = float(lane.heading_at(float(x_m)))
    if speed_mps is not None:
        vehicle.speed = float(speed_mps)
    if target_speed_mps is not None and hasattr(vehicle, "target_speed"):
        vehicle.target_speed = float(target_speed_mps)
    return lane_index


def _add_idm_vehicle(env: Any, vehicle_spec: Dict[str, Any], ego_lane_rank: int, ego_x: float) -> IDMVehicle:
    lane_rank = int(vehicle_spec.get("lane_rank", ego_lane_rank + int(vehicle_spec.get("lane_offset", 0))))
    if "x_m" in vehicle_spec:
        x_m = float(vehicle_spec["x_m"])
    else:
        x_m = float(ego_x + float(vehicle_spec["x_offset_m"]))
    lane_index = _lane_index(env, lane_rank)
    lane = env.unwrapped.road.network.get_lane(lane_index)
    vehicle = IDMVehicle(
        env.unwrapped.road,
        lane.position(x_m, 0.0),
        heading=lane.heading_at(x_m),
        speed=float(vehicle_spec["speed_mps"]),
        target_lane_index=lane_index,
        target_speed=float(vehicle_spec["target_speed_mps"]),
        enable_lane_change=bool(vehicle_spec.get("enable_lane_change", False)),
    )
    vehicle.dilu_benchmark_id = vehicle_spec["id"]
    vehicle.dilu_benchmark_role = vehicle_spec.get("role", "traffic")
    return vehicle


def _scenario_ego_lane_rank(spec: Dict[str, Any], current_ego_lane_rank: int) -> int:
    ego_spec = spec.get("ego") or {}
    if "lane_rank" in ego_spec:
        return int(ego_spec["lane_rank"])
    return int(current_ego_lane_rank)


def _resolve_lane_offset_reference(
    event: Dict[str, Any],
    *,
    scenario_ego_lane_rank: int,
    current_ego_lane_rank: int,
    vehicle_current_lane_rank: Optional[int],
) -> Tuple[str, int]:
    lane_reference = str(event.get("lane_reference") or "scenario_ego")
    if lane_reference == "scenario_ego":
        return lane_reference, int(scenario_ego_lane_rank)
    if lane_reference == "current_ego":
        return lane_reference, int(current_ego_lane_rank)
    if lane_reference == "vehicle_current":
        return lane_reference, int(
            vehicle_current_lane_rank
            if vehicle_current_lane_rank is not None
            else current_ego_lane_rank
        )
    raise ValueError(
        f"scenario_spec event `{event.get('id', '')}` has unsupported lane_reference `{lane_reference}`; "
        f"allowed={sorted(_LANE_REFERENCES)}."
    )


def _event_with_lane_resolution(
    event: Dict[str, Any],
    *,
    lane_reference: Optional[str],
    reference_lane_rank: Optional[int],
    resolved_lane_rank: Optional[int],
) -> Dict[str, Any]:
    item = copy.deepcopy(event)
    if lane_reference is not None:
        item["resolved_lane_reference"] = lane_reference
    if reference_lane_rank is not None:
        item["reference_lane_rank"] = int(reference_lane_rank)
    if resolved_lane_rank is not None:
        item["resolved_lane_rank"] = int(resolved_lane_rank)
    if "lane_offset" in event:
        item["original_lane_offset"] = int(event["lane_offset"])
    return item


def _vehicle_by_benchmark_id(env: Any, vehicle_id: str) -> Any:
    road = getattr(env.unwrapped, "road", None)
    for vehicle in list(getattr(road, "vehicles", []) or []):
        if str(getattr(vehicle, "dilu_benchmark_id", "") or "") == str(vehicle_id):
            return vehicle
    return None


def apply_highway_scenario_spec(env: Any, case: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    spec = normalize_scenario_spec((case or {}).get("scenario_spec"))
    if not spec:
        return {
            "benchmark_scenario_spec_applied": False,
            "benchmark_scenario_spec": {},
            "benchmark_scenario_vehicle_count": None,
        }

    uenv = env.unwrapped
    ego = getattr(uenv, "vehicle", None)
    road = getattr(uenv, "road", None)
    if ego is None or road is None:
        raise ValueError("scenario_spec requires an env with ego vehicle and road.")

    current_lane_rank = int(getattr(ego, "lane_index", (None, None, 0))[2])
    current_x = _vehicle_x(ego) or 0.0
    ego_spec = spec.get("ego") or {}
    ego_lane_rank = int(ego_spec.get("lane_rank", current_lane_rank))
    ego_x = float(ego_spec.get("x_m", current_x))
    ego_speed = float(ego_spec.get("speed_mps", getattr(ego, "speed", 0.0) or 0.0))
    _set_vehicle_state(ego, env, ego_lane_rank, ego_x, ego_speed)

    if spec.get("clear_existing_vehicles", True):
        road.vehicles = [ego]
    elif ego not in road.vehicles:
        road.vehicles.insert(0, ego)

    added = []
    for vehicle_spec in spec.get("vehicles") or []:
        vehicle = _add_idm_vehicle(env, vehicle_spec, ego_lane_rank, ego_x)
        road.vehicles.append(vehicle)
        added.append(vehicle)

    return {
        "benchmark_scenario_spec_applied": True,
        "benchmark_scenario_spec": scenario_spec_summary({"scenario_spec": spec}),
        "benchmark_scenario_vehicle_count": len(added),
        "benchmark_scenario_ego_lane_rank": ego_lane_rank,
        "benchmark_scenario_ego_speed_mps": round(float(ego_speed), 4),
        "benchmark_scenario_ego_x_m": round(float(ego_x), 4),
    }


def apply_highway_scenario_events(
    env: Any,
    case: Optional[Dict[str, Any]],
    step_idx: int,
    applied_event_ids: Optional[MutableSet[str]] = None,
) -> Dict[str, Any]:
    spec = normalize_scenario_spec((case or {}).get("scenario_spec"))
    events = list(spec.get("events") or [])
    if not events:
        return {
            "benchmark_events_applied": False,
            "benchmark_event_ids": [],
            "benchmark_event_types": [],
            "benchmark_event_step": int(step_idx),
            "benchmark_events": [],
        }

    applied_event_ids = applied_event_ids if applied_event_ids is not None else set()
    due_events = [
        event
        for event in events
        if int(event.get("step", -1)) == int(step_idx) and str(event.get("id")) not in applied_event_ids
    ]
    if not due_events:
        return {
            "benchmark_events_applied": False,
            "benchmark_event_ids": [],
            "benchmark_event_types": [],
            "benchmark_event_step": int(step_idx),
            "benchmark_events": [],
        }

    uenv = env.unwrapped
    ego = getattr(uenv, "vehicle", None)
    road = getattr(uenv, "road", None)
    if ego is None or road is None:
        raise ValueError("scenario_spec events require an env with ego vehicle and road.")
    ego_lane_rank = int(getattr(ego, "lane_index", (None, None, 0))[2])
    scenario_ego_lane_rank = _scenario_ego_lane_rank(spec, ego_lane_rank)
    ego_x = _vehicle_x(ego) or 0.0

    applied: List[Dict[str, Any]] = []
    for event in due_events:
        event_type = str(event["type"])
        event_id = str(event["id"])
        resolved_lane_reference = None
        reference_lane_rank = None
        resolved_lane_rank = None
        if event_type == "spawn_vehicle":
            vehicle_spec = dict(event["vehicle"])
            if "lane_rank" in vehicle_spec:
                resolved_lane_rank = int(vehicle_spec["lane_rank"])
            elif "lane_offset" in vehicle_spec:
                resolved_lane_reference, reference_lane_rank = _resolve_lane_offset_reference(
                    vehicle_spec,
                    scenario_ego_lane_rank=scenario_ego_lane_rank,
                    current_ego_lane_rank=ego_lane_rank,
                    vehicle_current_lane_rank=None,
                )
                resolved_lane_rank = int(reference_lane_rank) + int(vehicle_spec["lane_offset"])
                vehicle_spec["lane_rank"] = resolved_lane_rank
                vehicle_spec.pop("lane_offset", None)
            vehicle = _add_idm_vehicle(env, vehicle_spec, ego_lane_rank, ego_x)
            road.vehicles.append(vehicle)
        else:
            vehicle = _vehicle_by_benchmark_id(env, str(event["vehicle_id"]))
            if vehicle is None:
                raise ValueError(
                    f"scenario_spec event `{event_id}` references missing vehicle `{event['vehicle_id']}`."
                )
            if event_type == "reposition_vehicle":
                current_lane_rank = int(getattr(vehicle, "lane_index", (None, None, ego_lane_rank))[2])
                if "lane_rank" in event:
                    lane_rank = int(event["lane_rank"])
                    resolved_lane_rank = lane_rank
                elif "lane_offset" in event:
                    resolved_lane_reference, reference_lane_rank = _resolve_lane_offset_reference(
                        event,
                        scenario_ego_lane_rank=scenario_ego_lane_rank,
                        current_ego_lane_rank=ego_lane_rank,
                        vehicle_current_lane_rank=current_lane_rank,
                    )
                    lane_rank = int(reference_lane_rank) + int(event["lane_offset"])
                    resolved_lane_rank = lane_rank
                else:
                    lane_rank = current_lane_rank
                    resolved_lane_rank = lane_rank
                current_x = _vehicle_x(vehicle) or ego_x
                x_m = float(event.get("x_m", ego_x + float(event["x_offset_m"]) if "x_offset_m" in event else current_x))
                _move_vehicle(
                    vehicle,
                    env,
                    lane_rank,
                    x_m,
                    speed_mps=event.get("speed_mps"),
                    target_speed_mps=event.get("target_speed_mps"),
                )
                if "enable_lane_change" in event:
                    vehicle.enable_lane_change = bool(event["enable_lane_change"])
            elif event_type == "set_speed":
                vehicle.speed = float(event["speed_mps"])
            elif event_type == "set_target_speed":
                if hasattr(vehicle, "target_speed"):
                    vehicle.target_speed = float(event["target_speed_mps"])
            elif event_type == "set_lane_change":
                vehicle.enable_lane_change = bool(event["enable_lane_change"])
        applied_event_ids.add(event_id)
        applied.append(
            _event_with_lane_resolution(
                event,
                lane_reference=resolved_lane_reference,
                reference_lane_rank=reference_lane_rank,
                resolved_lane_rank=resolved_lane_rank,
            )
        )

    return {
        "benchmark_events_applied": bool(applied),
        "benchmark_event_ids": [str(event["id"]) for event in applied],
        "benchmark_event_types": [str(event["type"]) for event in applied],
        "benchmark_event_step": int(step_idx),
        "benchmark_events": applied,
    }
