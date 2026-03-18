from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np


def build_highway_env_config(
    config: Dict[str, Any],
    *,
    show_trajectories: bool,
    render_agent: bool,
    lanes_count: int = 4,
) -> Dict[str, Dict[str, Any]]:
    resolved_lanes_count = int(config.get("lanes_count", lanes_count))
    resolved_ego_spacing = float(config.get("ego_spacing", 4))
    resolved_scaling = float(config.get("scaling", 5))

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
            "target_speeds": np.linspace(5, 32, 9),
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
) -> Dict[str, Any]:
    probe = gym.make(env_id)
    env_cfg = dict(probe.unwrapped.config)
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

    if isinstance(env_cfg.get("observation"), dict):
        env_cfg["observation"] = dict(env_cfg["observation"])
        env_cfg["observation"]["vehicles_count"] = vehicle_count

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
        )
        return {
            "requested_env_id": env_id,
            "env_id": env_id,
            "use_native_env_defaults": True,
            "env_config_map": {env_id: env_cfg},
            "env_config_snapshot": env_cfg,
            "env_source": mode.get("env_source"),
            "native_source": mode.get("native_source"),
            "warnings": warnings,
        }

    env_config_map = build_highway_env_config(
        config,
        show_trajectories=show_trajectories,
        render_agent=render_agent,
        lanes_count=lanes_count,
    )
    requested_env_id = str(mode["env_id"])
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
        "env_source": mode.get("env_source"),
        "native_source": mode.get("native_source"),
        "warnings": warnings,
    }
