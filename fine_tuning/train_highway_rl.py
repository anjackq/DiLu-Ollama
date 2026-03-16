import argparse
import copy
import json
import os
import sys
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from dilu.runtime import ensure_parent_dir

console = Console()

IGNORED_NON_DQN_FLAGS = {
    "rl_n_envs": 6,
    "rl_vec_env": "subproc",
    "rl_n_epochs": 10,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train RL policy for highway driving (DQN repo profile only).")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--save-path", default="fine_tuning/rl_models/highway_dqn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--device", default="auto")

    p.add_argument("--rl-train-profile", choices=["dqn_repo"], default="dqn_repo")
    p.add_argument("--rl-env-id", default="")
    p.add_argument("--rl-n-envs", type=int, default=6)
    p.add_argument("--rl-vec-env", choices=["subproc", "dummy"], default="subproc")
    p.add_argument("--rl-batch-size", type=int, default=32)
    p.add_argument("--rl-n-epochs", type=int, default=10)
    p.add_argument("--rl-learning-rate", type=float, default=5e-4)
    p.add_argument("--rl-gamma", type=float, default=0.8)
    p.add_argument("--rl-tensorboard-log", default="highway_dqn/")
    p.add_argument("--rl-dqn-buffer-size", type=int, default=15000)
    p.add_argument("--rl-dqn-learning-starts", type=int, default=200)
    p.add_argument("--rl-dqn-train-freq", type=int, default=1)
    p.add_argument("--rl-dqn-gradient-steps", type=int, default=1)
    p.add_argument("--rl-dqn-target-update-interval", type=int, default=50)

    # Deprecated/ignored scalar flags kept for CLI compatibility.
    p.add_argument("--reward-env-weight", type=float, default=0.4)
    p.add_argument("--reward-speed-scale", type=float, default=0.8)
    p.add_argument("--reward-progress-scale", type=float, default=0.2)
    p.add_argument("--reward-crash-penalty", type=float, default=-2.5)
    p.add_argument("--reward-risk-brake-bonus", type=float, default=0.15)
    p.add_argument("--reward-early-brake-penalty", type=float, default=-0.35)
    p.add_argument("--reward-unjustified-brake-penalty", type=float, default=-0.25)
    p.add_argument("--risk-front-gap-m", type=float, default=22.0)
    p.add_argument("--risk-ttc-sec", type=float, default=3.0)
    p.add_argument("--early-brake-steps", type=int, default=5)
    return p.parse_args()


def _warn_ignored_legacy_flags(args) -> List[str]:
    legacy_reward_flag_defaults = {
        "reward_env_weight": 0.4,
        "reward_speed_scale": 0.8,
        "reward_progress_scale": 0.2,
        "reward_crash_penalty": -2.5,
        "reward_risk_brake_bonus": 0.15,
        "reward_early_brake_penalty": -0.35,
        "reward_unjustified_brake_penalty": -0.25,
    }
    ignored = []
    for k, v in legacy_reward_flag_defaults.items():
        if abs(float(getattr(args, k)) - float(v)) > 1e-9:
            ignored.append(k)
    for k, default_v in IGNORED_NON_DQN_FLAGS.items():
        if str(getattr(args, k)) != str(default_v):
            ignored.append(k)
    if ignored:
        console.print("[yellow]Ignored non-DQN/legacy flags:[/yellow] " + ", ".join(sorted(ignored)))
    return ignored


def _budget_warning(timesteps: int) -> str:
    if timesteps < 10_000:
        return "SEVERE: timesteps < 10,000. RL metrics are non-comparable and likely unstable."
    if timesteps < 100_000:
        return "WARNING: timesteps < 100,000. RL metrics are non-comparable and likely unstable."
    return ""


def _tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except Exception:
        return False


def build_runtime_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_native_env_cfg(env_id: str, config_data: Dict) -> Dict:
    probe = gym.make(env_id)
    env_cfg = dict(probe.unwrapped.config)
    probe.close()

    env_cfg["duration"] = int(config_data.get("simulation_duration", env_cfg.get("duration", 30)))
    env_cfg["vehicles_count"] = int(config_data.get("vehicle_count", env_cfg.get("vehicles_count", 20)))
    env_cfg["vehicles_density"] = float(config_data.get("vehicles_density", env_cfg.get("vehicles_density", 1.0)))
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


def _extract_speed_and_x(env):
    try:
        v = env.unwrapped.vehicle
        return float(v.speed), float(v.position[0])
    except Exception:
        return 0.0, 0.0


def _align_vehicle_count_to_model(env_cfg: Dict, model) -> Dict:
    out = {"requested_vehicle_count": None, "effective_vehicle_count": None, "adjusted_for_model_observation": False}
    obs = env_cfg.get("observation", {})
    if "vehicles_count" not in obs:
        return out
    req = int(obs["vehicles_count"])
    out["requested_vehicle_count"] = req
    out["effective_vehicle_count"] = req
    shape = getattr(model.observation_space, "shape", None) or ()
    if len(shape) >= 2:
        expected = int(shape[0])
        if expected != req:
            env_cfg["observation"]["vehicles_count"] = expected
            out["effective_vehicle_count"] = expected
            out["adjusted_for_model_observation"] = True
    return out


def evaluate_policy(model, env_id: str, seed: int, episodes: int, env_cfg: Dict = None) -> Dict:
    if episodes <= 0:
        return {
            "episodes": 0,
            "avg_reward": 0.0,
            "avg_episode_length": 0.0,
            "crash_rate": 0.0,
            "success_rate": 0.0,
            "avg_speed_mps": 0.0,
            "avg_progress_m": 0.0,
            "first5_brake_rate": 0.0,
            "action_histogram": {str(i): 0 for i in range(5)},
        }

    rewards = []
    steps = []
    crashes = 0
    speed_sum = 0.0
    speed_count = 0
    progress_sum = 0.0
    early_steps = 0
    early_brakes = 0
    hist = {str(i): 0 for i in range(5)}

    for i in range(episodes):
        env = gym.make(env_id)
        if env_cfg is not None:
            env.unwrapped.configure(env_cfg)
        obs, _ = env.reset(seed=seed + i)
        done = truncated = False
        ep_reward = 0.0
        ep_steps = 0
        start_speed, start_x = _extract_speed_and_x(env)
        _ = start_speed
        crashed = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_id = int(np.asarray(action).item())
            hist[str(action_id)] = hist.get(str(action_id), 0) + 1
            if ep_steps < 5:
                early_steps += 1
                if action_id == 4:
                    early_brakes += 1
            obs, reward, done, truncated, info = env.step(action_id)
            ep_reward += float(reward)
            ep_steps += 1
            speed, _ = _extract_speed_and_x(env)
            speed_sum += float(speed)
            speed_count += 1
            if info.get("crashed", False):
                crashed = True

        _, end_x = _extract_speed_and_x(env)
        progress_sum += float(end_x - start_x)
        if crashed:
            crashes += 1
        rewards.append(ep_reward)
        steps.append(ep_steps)
        env.close()

    n = max(episodes, 1)
    return {
        "episodes": episodes,
        "avg_reward": float(sum(rewards) / max(len(rewards), 1)),
        "avg_episode_length": float(sum(steps) / max(len(steps), 1)),
        "crash_rate": float(crashes / n),
        "success_rate": float(1.0 - (crashes / n)),
        "avg_speed_mps": float(speed_sum / max(speed_count, 1)),
        "avg_progress_m": float(progress_sum / n),
        "first5_brake_rate": float(early_brakes / max(early_steps, 1)),
        "action_histogram": hist,
    }


def summarize_rewards(rewards: List[float]) -> Dict:
    if not rewards:
        return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0, "last10_mean": 0.0}
    tail = rewards[-10:]
    return {
        "count": len(rewards),
        "mean": float(sum(rewards) / len(rewards)),
        "max": float(max(rewards)),
        "min": float(min(rewards)),
        "last10_mean": float(sum(tail) / len(tail)),
    }

def main():
    args = parse_args()
    ignored_legacy_flags = _warn_ignored_legacy_flags(args)

    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import BaseCallback
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required. Install with `pip install stable-baselines3`.") from exc

    total_timesteps = int(args.timesteps)
    budget_warning = _budget_warning(total_timesteps)
    if budget_warning:
        style = "red" if total_timesteps < 10_000 else "yellow"
        console.print(f"[{style}]{budget_warning}[/{style}]")

    requested_device = str(args.device).strip() or "auto"
    effective_device = requested_device
    if requested_device.lower().startswith("cuda") and not torch.cuda.is_available():
        console.print(f"[yellow]Requested device '{requested_device}' but CUDA is unavailable. Falling back to CPU.[/yellow]")
        effective_device = "cpu"

    progress_state = {"ratio": 0.0}

    class TrainCallback(BaseCallback):
        def __init__(self, total_steps: int, show_progress: bool):
            super().__init__()
            self.total_steps = int(total_steps)
            self.show_progress = bool(show_progress)
            self._progress = None
            self._task_id = None
            self._last_num = 0
            self.episode_rewards = []

        def _on_training_start(self):
            if not self.show_progress:
                return
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task("[cyan]DQN training[/cyan]", total=self.total_steps)

        def _on_step(self):
            progress_state["ratio"] = min(1.0, float(self.num_timesteps) / max(float(self.total_steps), 1.0))
            infos = self.locals.get("infos", [])
            for info in infos:
                ep = info.get("episode")
                if ep and "r" in ep:
                    self.episode_rewards.append(float(ep["r"]))
            if self.show_progress and self._progress is not None and self._task_id is not None:
                delta = int(self.num_timesteps - self._last_num)
                if delta > 0:
                    self._progress.advance(self._task_id, advance=delta)
                    self._last_num = int(self.num_timesteps)
            return True

        def _on_training_end(self):
            if self.show_progress and self._progress is not None:
                self._progress.stop()

    config_data = build_runtime_config(args.config)
    env_id = str(args.rl_env_id or config_data.get("rl_env_id", "highway-fast-v0")).strip() or "highway-fast-v0"
    train_env_cfg = build_native_env_cfg(env_id, config_data)

    requested_vectorization = {"vec_env": "dummy", "n_envs": 1}
    effective_vectorization = {"vec_env": "dummy", "n_envs": 1}

    tb_log = str(args.rl_tensorboard_log).strip() or "highway_dqn/"
    if tb_log and not _tensorboard_available():
        console.print("[yellow]TensorBoard is not installed; disabling tensorboard_log for this run.[/yellow]")
        tb_log = None

    train_env = gym.make(env_id, render_mode="rgb_array")
    train_env.unwrapped.configure(train_env_cfg)
    model_kwargs = {
        "seed": int(args.seed),
        "verbose": 0,
        "device": effective_device,
        "policy_kwargs": dict(net_arch=[256, 256]),
        "learning_rate": float(args.rl_learning_rate),
        "buffer_size": int(max(1, args.rl_dqn_buffer_size)),
        "learning_starts": int(max(0, args.rl_dqn_learning_starts)),
        "batch_size": int(max(1, args.rl_batch_size)),
        "gamma": float(args.rl_gamma),
        "train_freq": int(max(1, args.rl_dqn_train_freq)),
        "gradient_steps": int(max(1, args.rl_dqn_gradient_steps)),
        "target_update_interval": int(max(1, args.rl_dqn_target_update_interval)),
        "tensorboard_log": tb_log,
    }
    algorithm_name = "dqn"

    enable_progress = bool(sys.stdout.isatty())
    train_env_id = env_id
    console.print(
        f"[cyan]Starting {algorithm_name.upper()} training | profile={args.rl_train_profile}, env={train_env_id}, timesteps={total_timesteps}, "
        f"seed={int(args.seed)}, device={effective_device}, vec={effective_vectorization['vec_env']} x{effective_vectorization['n_envs']}, "
        f"progress_bar={'on' if enable_progress else 'off'}[/cyan]"
    )

    callback = TrainCallback(total_steps=total_timesteps, show_progress=enable_progress)
    model = DQN("MlpPolicy", train_env, **model_kwargs)
    console.print(f"[cyan]{algorithm_name.upper()} runtime device: {model.device}[/cyan]")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model_base = args.save_path[:-4] if args.save_path.endswith(".zip") else args.save_path
    ensure_parent_dir(model_base)
    model.save(model_base)
    saved_model_path = f"{model_base}.zip"

    eval_cfg = copy.deepcopy(train_env_cfg)
    eval_alignment = _align_vehicle_count_to_model(eval_cfg, model)
    evaluation = evaluate_policy(
        model,
        env_id=train_env_id,
        seed=int(args.seed) + 1000,
        episodes=int(args.eval_episodes),
        env_cfg=eval_cfg,
    )

    reward_curve_summary = summarize_rewards(callback.episode_rewards)

    metadata = {
        "model_path": saved_model_path,
        "config_path": args.config,
        "timesteps": total_timesteps,
        "seed": int(args.seed),
        "algorithm": algorithm_name,
        "train_profile": str(args.rl_train_profile),
        "train_env_id": train_env_id,
        "device_requested": requested_device,
        "device_effective": str(model.device),
        "eval_episodes": int(args.eval_episodes),
        "vectorization": {"requested": requested_vectorization, "effective": effective_vectorization},
        "subproc_fallback_used": False,
        "training_budget_warning": budget_warning or None,
        "is_short_budget_run": bool(total_timesteps < 100_000),
        "ignored_legacy_reward_flags": ignored_legacy_flags,
        "reward_curve_summary": reward_curve_summary,
        "evaluation_train_env": evaluation,
        "evaluation_observation_alignment": eval_alignment,
        "evaluation": evaluation,
    }

    metadata_path = f"{model_base}.meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    train_env.close()
    console.print(f"[green]Saved RL model: {saved_model_path}[/green]")
    console.print(f"[green]Saved metadata: {metadata_path}[/green]")
    console.print(
        f"[cyan]Eval({train_env_id}) success={evaluation['success_rate']:.2%}, "
        f"crash={evaluation['crash_rate']:.2%}, first5_brake={evaluation['first5_brake_rate']:.2%}[/cyan]"
    )


if __name__ == "__main__":
    main()
