import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.wrappers import RecordVideo
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from dilu.runtime import build_highway_env_config, current_timestamp, ensure_dir, write_json_atomic

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SB3 highway policy (PPO/DQN) and optionally capture simulation videos."
    )
    parser.add_argument("--config", default="config.yaml", help="Runtime config path.")
    parser.add_argument(
        "--rl-model-path",
        default="fine_tuning/rl_models/highway_dqn.zip",
        help="Path to trained SB3 model zip.",
    )
    parser.add_argument(
        "--rl-algorithm",
        choices=["auto", "ppo", "dqn"],
        default="auto",
        help="Model algorithm. auto -> infer from sidecar metadata, then fallback load attempts.",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic replay.")
    parser.add_argument(
        "--output",
        default="",
        help="Metrics JSON output path. Default: results/diagnostics/rl_eval_<timestamp>.json",
    )
    parser.add_argument(
        "--video-dir",
        default="",
        help="Video output directory. Default: results/diagnostics/rl_eval_videos_<timestamp>",
    )
    parser.add_argument("--video-prefix", default="rl_eval", help="Prefix for generated video file names.")
    parser.add_argument("--no-video", action="store_true", help="Disable video capture.")
    parser.add_argument(
        "--collect-simulation-duration",
        type=int,
        default=0,
        help="Optional override for episode duration (0 -> use config).",
    )
    parser.add_argument(
        "--collect-vehicle-count",
        type=int,
        default=0,
        help="Optional override for vehicles_count (0 -> use config/model shape).",
    )
    parser.add_argument(
        "--collect-vehicles-density",
        type=float,
        default=0.0,
        help="Optional override for vehicles_density (0 -> use config).",
    )
    parser.add_argument(
        "--env-id",
        default="",
        help="Gym env id for evaluation (default: config rl_env_id or highway-v0).",
    )
    native_group = parser.add_mutually_exclusive_group()
    native_group.add_argument(
        "--native-env-defaults",
        dest="native_env_defaults",
        action="store_true",
        help="Evaluate using native env defaults + top-level scenario overrides (recommended for RL parity).",
    )
    native_group.add_argument(
        "--no-native-env-defaults",
        dest="native_env_defaults",
        action="store_false",
        help="Use DiLu custom env builder instead of native env defaults.",
    )
    parser.set_defaults(native_env_defaults=None)
    return parser.parse_args()


def _infer_algorithm_from_metadata(model_path: str) -> str:
    base, ext = os.path.splitext(model_path)
    if ext.lower() != ".zip":
        return ""
    meta_path = f"{base}.meta.json"
    if not os.path.exists(meta_path):
        return ""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        algo = str(payload.get("algorithm", "")).strip().lower()
        return algo if algo in {"ppo", "dqn"} else ""
    except Exception:
        return ""


def _load_model(model_path: str, rl_algorithm: str):
    from stable_baselines3 import DQN, PPO

    mode = str(rl_algorithm).strip().lower()
    if mode == "ppo":
        return PPO.load(model_path), "ppo"
    if mode == "dqn":
        return DQN.load(model_path), "dqn"

    inferred = _infer_algorithm_from_metadata(model_path)
    if inferred == "ppo":
        return PPO.load(model_path), "ppo"
    if inferred == "dqn":
        return DQN.load(model_path), "dqn"

    load_errors: List[Tuple[str, str]] = []
    for name, loader in (("ppo", PPO.load), ("dqn", DQN.load)):
        try:
            return loader(model_path), name
        except Exception as exc:
            load_errors.append((name, f"{exc.__class__.__name__}: {exc}"))
    detail = "; ".join(f"{n} -> {m}" for n, m in load_errors)
    raise RuntimeError(f"Unable to load RL model with auto mode. Tried PPO and DQN. Details: {detail}")


def _to_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_env_mode(config_data: Dict, args: argparse.Namespace) -> Tuple[str, bool]:
    env_id = str(args.env_id or config_data.get("rl_env_id", "highway-v0")).strip() or "highway-v0"
    if args.native_env_defaults is None:
        native = _to_bool(config_data.get("rl_use_native_env_defaults", False), default=False)
    else:
        native = bool(args.native_env_defaults)
    return env_id, native


def _build_env_cfg(
    config_data: Dict,
    args: argparse.Namespace,
    env_id: str,
    native_env_defaults: bool,
) -> Dict:
    if native_env_defaults:
        probe = gym.make(env_id)
        env_cfg = dict(probe.unwrapped.config)
        probe.close()

        duration = int(args.collect_simulation_duration) if int(args.collect_simulation_duration) > 0 else int(config_data.get("simulation_duration", env_cfg.get("duration", 30)))
        vehicles_count = int(args.collect_vehicle_count) if int(args.collect_vehicle_count) > 0 else int(config_data.get("vehicle_count", env_cfg.get("vehicles_count", 20)))
        vehicles_density = float(args.collect_vehicles_density) if float(args.collect_vehicles_density) > 0 else float(config_data.get("vehicles_density", env_cfg.get("vehicles_density", 1.0)))

        env_cfg["duration"] = duration
        env_cfg["vehicles_count"] = vehicles_count
        env_cfg["vehicles_density"] = vehicles_density
        if "other_vehicle_type" in config_data:
            env_cfg["other_vehicles_type"] = config_data["other_vehicle_type"]

        for key in [
            "lanes_count",
            "simulation_frequency",
            "policy_frequency",
            "collision_reward",
            "high_speed_reward",
            "right_lane_reward",
            "lane_change_reward",
            "normalize_reward",
            "offroad_terminal",
            "ego_spacing",
        ]:
            if key in config_data and config_data[key] is not None:
                env_cfg[key] = config_data[key]
        if "reward_speed_range" in config_data and config_data["reward_speed_range"] is not None:
            env_cfg["reward_speed_range"] = config_data["reward_speed_range"]

        return env_cfg

    cfg = dict(config_data)

    if int(args.collect_simulation_duration) > 0:
        cfg["simulation_duration"] = int(args.collect_simulation_duration)
    if int(args.collect_vehicle_count) > 0:
        cfg["vehicle_count"] = int(args.collect_vehicle_count)
    if float(args.collect_vehicles_density) > 0:
        cfg["vehicles_density"] = float(args.collect_vehicles_density)

    return build_highway_env_config(
        cfg,
        show_trajectories=True,
        render_agent=False,
    )["highway-v0"]


def _align_vehicle_count_to_model(env_cfg: Dict, model) -> None:
    shape = tuple(int(x) for x in model.observation_space.shape)
    if len(shape) >= 2:
        expected_vehicle_count = int(shape[0])
        current_vehicle_count = int(env_cfg.get("observation", {}).get("vehicles_count", expected_vehicle_count))
        if current_vehicle_count != expected_vehicle_count:
            console.print(
                f"[yellow]Model observation expects vehicles_count={expected_vehicle_count}; "
                f"overriding from {current_vehicle_count} for compatibility.[/yellow]"
            )
            env_cfg["observation"]["vehicles_count"] = expected_vehicle_count


def _collect_new_files(video_dir: str, before: set[str], prefix: str) -> List[str]:
    if not os.path.isdir(video_dir):
        return []
    after = set(os.listdir(video_dir))
    new_files = sorted(name for name in (after - before) if name.startswith(prefix))
    return [os.path.join(video_dir, name) for name in new_files]


def _run_episode(
    model,
    env_id: str,
    env_cfg: Dict,
    seed: int,
    save_video: bool,
    video_dir: str,
    video_prefix: str,
) -> Dict:
    raw_env = gym.make(env_id, render_mode="rgb_array")
    raw_env.unwrapped.configure(env_cfg)

    before_files = set(os.listdir(video_dir)) if (save_video and os.path.isdir(video_dir)) else set()
    env = raw_env
    prefix = f"{video_prefix}_seed{seed}"
    if save_video:
        env = RecordVideo(
            raw_env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix=prefix,
            disable_logger=True,
        )

    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    steps = 0
    total_reward = 0.0
    crashed = False
    action_hist = {str(i): 0 for i in range(5)}

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action_id = int(np.asarray(action).item())
        action_hist[str(action_id)] += 1
        obs, reward, done, truncated, info = env.step(action_id)
        total_reward += float(reward)
        steps += 1
        if info.get("crashed", False):
            crashed = True

    env.close()
    created_videos = _collect_new_files(video_dir, before_files, prefix) if save_video else []
    return {
        "seed": int(seed),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "crashed": bool(crashed),
        "success": not bool(crashed),
        "action_hist": action_hist,
        "video_files": created_videos,
    }


def _aggregate(episodes: List[Dict]) -> Dict:
    n = len(episodes)
    if n == 0:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "crash_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "action_hist": {str(i): 0 for i in range(5)},
        }
    success_count = sum(int(e.get("success", False)) for e in episodes)
    crash_count = sum(int(e.get("crashed", False)) for e in episodes)
    avg_reward = sum(float(e.get("total_reward", 0.0)) for e in episodes) / n
    avg_steps = sum(int(e.get("steps", 0)) for e in episodes) / n
    hist = {str(i): 0 for i in range(5)}
    for ep in episodes:
        ep_hist = ep.get("action_hist", {})
        for k in hist:
            hist[k] += int(ep_hist.get(k, 0))
    return {
        "episodes": n,
        "success_rate": success_count / n,
        "crash_rate": crash_count / n,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "action_hist": hist,
    }


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import DQN, PPO  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 is required for RL evaluation. Install with `pip install stable-baselines3`."
        ) from exc

    if not os.path.exists(args.rl_model_path):
        raise FileNotFoundError(f"RL model not found: {args.rl_model_path}")

    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    ts = current_timestamp()
    output_path = args.output.strip() or os.path.join("results", "diagnostics", f"rl_eval_{ts}.json")
    video_dir = args.video_dir.strip() or os.path.join("results", "diagnostics", f"rl_eval_videos_{ts}")
    save_video = not bool(args.no_video)
    if save_video:
        ensure_dir(video_dir)

    model, resolved_algo = _load_model(args.rl_model_path, args.rl_algorithm)
    env_id, native_env_defaults = _resolve_env_mode(config_data, args)
    env_cfg = _build_env_cfg(config_data, args, env_id, native_env_defaults)
    _align_vehicle_count_to_model(env_cfg, model)

    console.print(
        f"[cyan]Evaluating RL policy[/cyan] model={args.rl_model_path} | algo={resolved_algo} "
        f"| env={env_id} | native_env_defaults={native_env_defaults} "
        f"| episodes={int(args.episodes)} | seed={int(args.seed)} | save_video={save_video}"
    )
    if save_video:
        console.print(f"[cyan]Video dir:[/cyan] {video_dir}")

    episodes: List[Dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]RL eval episodes[/cyan]", total=int(args.episodes))
        for i in range(int(args.episodes)):
            seed_i = int(args.seed) + i
            ep = _run_episode(
                model=model,
                env_id=env_id,
                env_cfg=env_cfg,
                seed=seed_i,
                save_video=save_video,
                video_dir=video_dir,
                video_prefix=args.video_prefix,
            )
            episodes.append(ep)
            progress.advance(task_id, 1)

    aggregate = _aggregate(episodes)
    payload = {
        "rl_model_path": args.rl_model_path,
        "rl_algorithm_requested": args.rl_algorithm,
        "rl_algorithm_resolved": resolved_algo,
        "env_id": env_id,
        "native_env_defaults": bool(native_env_defaults),
        "config_path": args.config,
        "episodes_requested": int(args.episodes),
        "seed_base": int(args.seed),
        "save_video": save_video,
        "video_dir": video_dir if save_video else None,
        "env_config": env_cfg,
        "aggregate": aggregate,
        "episodes": episodes,
    }
    write_json_atomic(output_path, payload)

    console.print(f"[green]Saved RL evaluation report:[/green] {output_path}")
    console.print(
        f"[green]Summary:[/green] success_rate={aggregate['success_rate']:.2%}, "
        f"crash_rate={aggregate['crash_rate']:.2%}, avg_reward={aggregate['avg_reward']:.3f}, "
        f"avg_steps={aggregate['avg_steps']:.2f}"
    )


if __name__ == "__main__":
    main()
