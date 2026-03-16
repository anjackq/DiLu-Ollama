from typing import Any, Dict

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
