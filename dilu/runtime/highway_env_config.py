import copy
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from highway_env.vehicle.controller import MDPVehicle


DEFAULT_STOP_CAPABLE_TARGET_SPEEDS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
DEFAULT_NATIVE_TARGET_SPEEDS = [float(x) for x in MDPVehicle.DEFAULT_TARGET_SPEEDS.tolist()]


def _normalize_action_target_speeds(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        tokens = [token.strip() for token in raw.split(",")]
        values = [float(token) for token in tokens if token]
    else:
        try:
            values = [float(item) for item in list(raw)]
        except Exception as exc:
            raise ValueError("sim_action_target_speeds must be a list or comma-separated string.") from exc
    if not values:
        raise ValueError("sim_action_target_speeds cannot be empty.")
    deduped = []
    for value in values:
        if not deduped or float(value) != float(deduped[-1]):
            deduped.append(float(value))
    if any(value < 0 for value in deduped):
        raise ValueError("sim_action_target_speeds must be non-negative.")
    if len(deduped) < 2:
        raise ValueError("sim_action_target_speeds must contain at least two speeds.")
    if deduped != sorted(deduped):
        raise ValueError("sim_action_target_speeds must be sorted ascending.")
    return deduped


def _resolve_action_target_speeds(
    config: Dict[str, Any],
    override: Optional[Any] = None,
) -> Optional[List[float]]:
    if override is not None:
        return _normalize_action_target_speeds(override)
    return _normalize_action_target_speeds(config.get("sim_action_target_speeds"))


def _derive_env_profile_label(action_target_speeds: Optional[List[float]]) -> str:
    if not action_target_speeds:
        return "default"
    rounded = [round(float(value), 6) for value in action_target_speeds]
    if rounded == [round(value, 6) for value in DEFAULT_STOP_CAPABLE_TARGET_SPEEDS]:
        return "default_stop_capable"
    if rounded == [round(value, 6) for value in DEFAULT_NATIVE_TARGET_SPEEDS]:
        return "default"
    return "custom_action_target_speeds"


def _deep_update(base: Dict[str, Any], updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def build_highway_env_config(
    config: Dict[str, Any],
    *,
    show_trajectories: bool,
    render_agent: bool,
    lanes_count: int = 4,
    action_target_speeds_override: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    resolved_lanes_count = int(config.get("lanes_count", lanes_count))
    resolved_ego_spacing = float(config.get("ego_spacing", 4))
    resolved_scaling = float(config.get("scaling", 5))
    explicit_target_speeds = (
        _normalize_action_target_speeds(action_target_speeds_override)
        if action_target_speeds_override is not None
        else None
    )
    config_target_speeds = _resolve_action_target_speeds(config)
    resolved_target_speeds = explicit_target_speeds if explicit_target_speeds is not None else config_target_speeds

    env_cfg: Dict[str, Any] = {
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "normalize": False,
            "vehicles_count": config["vehicle_count"],
            "see_behind": True,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": np.array(
                resolved_target_speeds if resolved_target_speeds is not None else np.linspace(5, 32, 9),
                dtype=float,
            ),
        },
        "lanes_count": resolved_lanes_count,
        "vehicles_count": config["vehicle_count"],
        "other_vehicles_type": config["other_vehicle_type"],
        "duration": config["simulation_duration"],
        "vehicles_density": config["vehicles_density"],
        "show_trajectories": show_trajectories,
        "render_agent": render_agent,
        "scaling": resolved_scaling,
        "initial_lane_id": None,
        "ego_spacing": resolved_ego_spacing,
    }

    # Optional passthrough knobs to support matching presets such as highway-fast-v0.
    optional_top_level_keys = [
        "simulation_frequency",
        "policy_frequency",
        "collision_reward",
        "high_speed_reward",
        "right_lane_reward",
        "lane_change_reward",
        "normalize_reward",
        "offroad_terminal",
    ]
    for key in optional_top_level_keys:
        if key in config and config[key] is not None:
            env_cfg[key] = config[key]

    if "reward_speed_range" in config and config["reward_speed_range"] is not None:
        env_cfg["reward_speed_range"] = config["reward_speed_range"]

    return {
        "highway-v0": env_cfg
    }


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_simulation_env_mode(
    config: Dict[str, Any],
    *,
    env_id_override: Optional[str] = None,
    native_env_defaults_override: Optional[bool] = None,
    fallback_env_id: str = "highway-fast-v0",
    fallback_native_env_defaults: bool = True,
) -> Dict[str, Any]:
    warnings = []

    env_id: Optional[str] = None
    env_source = ""
    if env_id_override and str(env_id_override).strip():
        env_id = str(env_id_override).strip()
        env_source = "cli"
    elif config.get("sim_env_id"):
        env_id = str(config.get("sim_env_id")).strip()
        env_source = "config.sim_env_id"
    elif config.get("rl_env_id"):
        env_id = str(config.get("rl_env_id")).strip()
        env_source = "config.rl_env_id (deprecated)"
        warnings.append(
            "Deprecated config key `rl_env_id` is being used. Prefer `sim_env_id`."
        )
    else:
        env_id = str(fallback_env_id).strip()
        env_source = "fallback"

    if not env_id:
        env_id = str(fallback_env_id).strip() or "highway-fast-v0"
        env_source = "fallback"

    if native_env_defaults_override is not None:
        use_native = bool(native_env_defaults_override)
        native_source = "cli"
    elif config.get("sim_use_native_env_defaults") is not None:
        use_native = _to_bool(config.get("sim_use_native_env_defaults"), bool(fallback_native_env_defaults))
        native_source = "config.sim_use_native_env_defaults"
    elif config.get("rl_use_native_env_defaults") is not None:
        use_native = _to_bool(config.get("rl_use_native_env_defaults"), bool(fallback_native_env_defaults))
        native_source = "config.rl_use_native_env_defaults (deprecated)"
        warnings.append(
            "Deprecated config key `rl_use_native_env_defaults` is being used. Prefer `sim_use_native_env_defaults`."
        )
    else:
        use_native = bool(fallback_native_env_defaults)
        native_source = "fallback"

    return {
        "env_id": env_id,
        "use_native_env_defaults": bool(use_native),
        "env_source": env_source,
        "native_source": native_source,
        "warnings": warnings,
    }


def build_native_highway_env_config(
    config: Dict[str, Any],
    *,
    env_id: str,
    show_trajectories: bool,
    render_agent: bool,
    lanes_count: int = 4,
    action_target_speeds_override: Optional[Any] = None,
    env_config_overrides: Optional[Dict[str, Any]] = None,
    require_discrete_meta_action: bool = False,
) -> Dict[str, Any]:
    probe = gym.make(env_id)
    env_cfg = dict(probe.unwrapped.config)
    _deep_update(env_cfg, env_config_overrides)
    action_cfg_probe = dict(env_cfg.get("action") or {})
    if require_discrete_meta_action:
        action_type_name = str(action_cfg_probe.get("type") or "").strip()
        if action_type_name != "DiscreteMetaAction":
            probe.close()
            raise ValueError(
                f"unsupported continuous or non-DiscreteMetaAction env '{env_id}' "
                f"(action.type={action_type_name or 'unknown'}) for the current DiLu discrete framework"
            )
    probe.close()

    resolved_lanes_count = int(config.get("lanes_count", lanes_count))
    simulation_duration = int(config.get("simulation_duration", env_cfg.get("duration", 30)))
    vehicle_count = int(config.get("vehicle_count", env_cfg.get("vehicles_count", 20)))
    vehicles_density = float(config.get("vehicles_density", env_cfg.get("vehicles_density", 1.0)))

    env_cfg["duration"] = simulation_duration
    env_cfg["vehicles_count"] = vehicle_count
    env_cfg["vehicles_density"] = vehicles_density
    env_cfg["lanes_count"] = resolved_lanes_count
    env_cfg["show_trajectories"] = bool(show_trajectories)
    env_cfg["render_agent"] = bool(render_agent)
    if "other_vehicle_type" in config and config["other_vehicle_type"] is not None:
        env_cfg["other_vehicles_type"] = config["other_vehicle_type"]
    explicit_target_speeds = (
        _normalize_action_target_speeds(action_target_speeds_override)
        if action_target_speeds_override is not None
        else None
    )
    config_target_speeds = _resolve_action_target_speeds(config)

    if isinstance(env_cfg.get("observation"), dict):
        env_cfg["observation"] = dict(env_cfg["observation"])
        env_cfg["observation"]["vehicles_count"] = vehicle_count
    action_cfg = dict(env_cfg.get("action") or {})
    if action_cfg.get("type") == "DiscreteMetaAction":
        benchmark_target_speeds = action_cfg.get("target_speeds")
        action_cfg["target_speeds"] = list(
            explicit_target_speeds
            if explicit_target_speeds is not None
            else benchmark_target_speeds
            if benchmark_target_speeds is not None
            else config_target_speeds
            if config_target_speeds is not None
            else DEFAULT_NATIVE_TARGET_SPEEDS
        )
        env_cfg["action"] = action_cfg

    optional_top_level_keys = [
        "simulation_frequency",
        "policy_frequency",
        "collision_reward",
        "high_speed_reward",
        "right_lane_reward",
        "lane_change_reward",
        "normalize_reward",
        "offroad_terminal",
        "ego_spacing",
        "scaling",
    ]
    for key in optional_top_level_keys:
        if key in config and config[key] is not None:
            env_cfg[key] = config[key]

    if "reward_speed_range" in config and config["reward_speed_range"] is not None:
        env_cfg["reward_speed_range"] = config["reward_speed_range"]

    return env_cfg


def resolve_simulation_env_bundle(
    config: Dict[str, Any],
    *,
    show_trajectories: bool,
    render_agent: bool,
    env_id_override: Optional[str] = None,
    native_env_defaults_override: Optional[bool] = None,
    lanes_count: int = 4,
    action_target_speeds_override: Optional[Any] = None,
    env_config_overrides: Optional[Dict[str, Any]] = None,
    require_discrete_meta_action: bool = False,
) -> Dict[str, Any]:
    mode = resolve_simulation_env_mode(
        config,
        env_id_override=env_id_override,
        native_env_defaults_override=native_env_defaults_override,
    )
    warnings = list(mode.get("warnings", []))

    if mode["use_native_env_defaults"]:
        env_id = str(mode["env_id"])
        env_cfg = build_native_highway_env_config(
            config,
            env_id=env_id,
            show_trajectories=show_trajectories,
            render_agent=render_agent,
            lanes_count=lanes_count,
            action_target_speeds_override=action_target_speeds_override,
            env_config_overrides=env_config_overrides,
            require_discrete_meta_action=require_discrete_meta_action,
        )
        resolved_action_target_speeds = _normalize_action_target_speeds(
            (env_cfg.get("action") or {}).get("target_speeds")
        )
        return {
            "requested_env_id": env_id,
            "env_id": env_id,
            "use_native_env_defaults": True,
            "env_config_map": {env_id: env_cfg},
            "env_config_snapshot": env_cfg,
            "resolved_action_target_speeds": resolved_action_target_speeds,
            "env_profile_label": _derive_env_profile_label(resolved_action_target_speeds),
            "env_source": mode.get("env_source"),
            "native_source": mode.get("native_source"),
            "warnings": warnings,
        }

    env_config_map = build_highway_env_config(
        config,
        show_trajectories=show_trajectories,
        render_agent=render_agent,
        lanes_count=lanes_count,
        action_target_speeds_override=action_target_speeds_override,
    )
    requested_env_id = str(mode["env_id"])
    if requested_env_id in env_config_map:
        _deep_update(env_config_map[requested_env_id], env_config_overrides)
    if requested_env_id in env_config_map:
        env_id = requested_env_id
    else:
        env_id = "highway-v0" if "highway-v0" in env_config_map else next(iter(env_config_map.keys()))
        warnings.append(
            f"Legacy builder does not support env_id='{requested_env_id}'. Falling back to env_id='{env_id}'."
        )
    return {
        "requested_env_id": requested_env_id,
        "env_id": env_id,
        "use_native_env_defaults": False,
        "env_config_map": env_config_map,
        "env_config_snapshot": env_config_map[env_id],
        "resolved_action_target_speeds": _normalize_action_target_speeds(
            (env_config_map[env_id].get("action") or {}).get("target_speeds")
        ),
        "env_profile_label": _derive_env_profile_label(
            _normalize_action_target_speeds((env_config_map[env_id].get("action") or {}).get("target_speeds"))
        ),
        "env_source": mode.get("env_source"),
        "native_source": mode.get("native_source"),
        "warnings": warnings,
    }
