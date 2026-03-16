import argparse
import collections
import copy
import json
import math
import os
import random
import shutil
import sys
from typing import Callable, Dict, List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gymnasium as gym
import yaml
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich import print

from dilu.runtime import DEFAULT_DILU_SEEDS, build_highway_env_config, ensure_dir, ensure_parent_dir
from dilu.scenario.envScenario import EnvScenario
from fine_tuning.pipeline import expert_decision_v2_left_pass_preferred, expert_decision_v3_balanced, init_expert_state, write_jsonl

console = Console()
ACTION_LABELS = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}
CURRICULUM_STAGES = [
    ("A", 120, 20, 1.8),
    ("B", 200, 30, 2.6),
    ("C", 300, 40, 3.5),
]


def setup_env(config):
    return build_highway_env_config(
        config,
        show_trajectories=True,
        render_agent=False,
    )


def _to_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


def load_rl_model(model_path: str, algorithm: str):
    from stable_baselines3 import DQN, PPO

    mode = str(algorithm).strip().lower()
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


def build_native_env_cfg(env_name: str, config_data: dict, stage_cfg: dict) -> dict:
    probe = gym.make(env_name)
    env_cfg = dict(probe.unwrapped.config)
    probe.close()

    env_cfg["duration"] = int(stage_cfg["simulation_duration"])
    env_cfg["vehicles_count"] = int(stage_cfg["vehicle_count"])
    env_cfg["vehicles_density"] = float(stage_cfg["vehicles_density"])
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


def build_collect_config(
    config: dict,
    collect_simulation_duration: int,
    collect_vehicle_count: int,
    collect_vehicles_density: float,
) -> dict:
    cfg = copy.deepcopy(config)
    cfg["simulation_duration"] = int(collect_simulation_duration)
    cfg["vehicle_count"] = int(collect_vehicle_count)
    cfg["vehicles_density"] = float(collect_vehicles_density)
    return cfg


def collect_episode(
    env,
    env_name: str,
    seed: int,
    temp_db_path: str,
    max_steps: int,
    initial_obs,
    action_selector: Callable[[object, object], Tuple[int, str]],
    rule_state: Optional[Dict[str, int]] = None,
    on_step: Optional[Callable[[int, bool], None]] = None,
):
    sce = EnvScenario(env, env_name, seed, temp_db_path)
    buffer = []
    actions_taken: List[int] = []
    done = False
    truncated = False
    step = 0
    info = {}
    crashed = False
    obs = initial_obs
    speed_sum = 0.0
    speed_count = 0
    _, start_x = _get_vehicle_state(env)
    end_x = start_x

    while not (done or truncated) and step < max_steps:
        description = sce.describe(step)
        available_actions = sce.availableActionsDescription()
        action_id, reasoning = action_selector(sce, obs, rule_state)
        actions_taken.append(int(action_id))
        obs, _, done, truncated, info = env.step(action_id)
        speed_mps, pos_x = _get_vehicle_state(env)
        speed_sum += float(speed_mps)
        speed_count += 1
        end_x = float(pos_x)

        full_input = f"Driving scenario description:\n{description}\nAvailable actions:\n{available_actions}"
        buffer.append(
            {
                "instruction": "You are an expert autonomous driving agent. Analyze the scenario and output a JSON decision.",
                "input": full_input,
                "output": json.dumps(
                    {
                        "analysis": "Scenario analyzed via internal sensor data.",
                        "reasoning": reasoning,
                        "action_id": action_id,
                    }
                ),
            }
        )
        step += 1
        if on_step is not None:
            on_step(step, bool(done or truncated))

        if info.get("crashed", False):
            crashed = True
            break

    early_window = min(5, len(actions_taken))
    early_brakes = sum(1 for a in actions_taken[:early_window] if int(a) == 4)
    metrics = {
        "avg_speed_mps": float(speed_sum / max(speed_count, 1)),
        "progress_m": float(end_x - start_x),
        "early_brake_ratio": float(early_brakes / max(early_window, 1)),
        "early_window_steps": int(early_window),
        "actions_total": int(len(actions_taken)),
    }

    return {
        "rows": [] if crashed else buffer,
        "crashed": crashed,
        "steps": step,
        "terminated": bool(done),
        "truncated": bool(truncated),
        "metrics": metrics,
    }


def _safe_ttc(distance_m: float, closing_speed_mps: float) -> float:
    if closing_speed_mps <= 0:
        return float("inf")
    return distance_m / max(closing_speed_mps, 1e-6)


def _get_vehicle_state(env) -> Tuple[float, float]:
    try:
        vehicle = env.unwrapped.vehicle
        return float(vehicle.speed), float(vehicle.position[0])
    except Exception:
        return 0.0, 0.0


def evaluate_rl_episode_quality(episode: Dict, max_steps: int, args) -> Tuple[bool, List[str], float]:
    metrics = episode.get("metrics", {}) or {}
    reasons: List[str] = []
    required_progress = max(float(args.rl_min_progress), float(2.5 * max_steps))
    if bool(episode.get("crashed", False)):
        reasons.append("crashed")
    if float(metrics.get("early_brake_ratio", 0.0)) > float(args.rl_max_early_brake_ratio):
        reasons.append("early_brake_ratio")
    if float(metrics.get("avg_speed_mps", 0.0)) < float(args.rl_min_avg_speed):
        reasons.append("avg_speed")
    if float(metrics.get("progress_m", 0.0)) < required_progress:
        reasons.append("progress")
    return (len(reasons) == 0), reasons, required_progress


def build_rl_reasoning(sce, action_id: int, predicted_action: int, fallback_used: bool, available_actions):
    vehicle = sce.env.unwrapped.vehicle
    road = sce.env.unwrapped.road
    front_vehicle, _ = road.neighbour_vehicles(vehicle, vehicle.lane_index)

    front_gap = float("inf")
    rel_speed = 0.0
    ttc_front = float("inf")
    if front_vehicle is not None:
        front_gap = float(np.linalg.norm(vehicle.position - front_vehicle.position))
        rel_speed = float(vehicle.speed - front_vehicle.speed)
        ttc_front = _safe_ttc(front_gap, rel_speed)

    cfg = getattr(sce.env.unwrapped, "config", {}) or {}
    lane_count = int(cfg.get("lanes_count", 4))
    lane_id = int(vehicle.lane_index[2])

    front_gap_txt = "inf" if math.isinf(front_gap) else f"{front_gap:.1f}"
    ttc_txt = "inf" if math.isinf(ttc_front) else f"{ttc_front:.2f}"
    reasoning = (
        "RL policy features: "
        f"lane={lane_id + 1}/{lane_count}, speed={vehicle.speed:.1f}m/s, "
        f"front_gap={front_gap_txt}m, rel_speed={rel_speed:.1f}m/s, "
        f"ttc_front={ttc_txt}s, left_available={0 in available_actions}, right_available={2 in available_actions}. "
        f"Selected {ACTION_LABELS.get(action_id, action_id)} (action_id={action_id}) for safe progress."
    )
    if fallback_used:
        reasoning += (
            f" Model predicted action_id={predicted_action}, which was unavailable; "
            f"fallback action_id={action_id} was applied."
        )
    return reasoning


def resolve_curriculum_stage(success_count: int, target_episodes: int) -> Dict[str, float]:
    cut_a = max(1, int(math.ceil(target_episodes * 0.40)))
    cut_b = max(cut_a + 1, int(math.ceil(target_episodes * 0.75)))
    if success_count < cut_a:
        stage = CURRICULUM_STAGES[0]
    elif success_count < cut_b:
        stage = CURRICULUM_STAGES[1]
    else:
        stage = CURRICULUM_STAGES[2]
    return {
        "name": stage[0],
        "simulation_duration": int(stage[1]),
        "vehicle_count": int(stage[2]),
        "vehicles_density": float(stage[3]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Collect expert trajectories for fine-tuning (rule-based or RL).")
    parser.add_argument("--config", default="config.yaml", help="Runtime config path.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of successful episodes to collect.")
    parser.add_argument("--output", default="data/gold_standard_data.jsonl", help="Output JSONL path.")
    parser.add_argument("--save-every", type=int, default=5, help="Incremental save cadence.")
    parser.add_argument(
        "--collect-simulation-duration",
        type=int,
        default=300,
        help="Collection-only simulation horizon per episode.",
    )
    parser.add_argument(
        "--collect-vehicle-count",
        type=int,
        default=40,
        help="Collection-only number of vehicles in scenario observations.",
    )
    parser.add_argument(
        "--collect-vehicles-density",
        type=float,
        default=3.5,
        help="Collection-only traffic density.",
    )
    parser.add_argument(
        "--collect-controller",
        default="rule",
        choices=["rule", "rl"],
        help="Controller used to generate collection actions.",
    )
    parser.add_argument(
        "--rl-model-path",
        default="",
        help="Path to trained SB3 model zip used when --collect-controller rl.",
    )
    parser.add_argument(
        "--rl-algorithm",
        choices=["auto", "ppo", "dqn"],
        default="auto",
        help="RL model algorithm (auto -> infer from metadata, then try PPO/DQN load).",
    )
    parser.add_argument(
        "--rl-env-id",
        default="",
        help="Gym env id for RL collection (default: config rl_env_id or highway-v0).",
    )
    native_group = parser.add_mutually_exclusive_group()
    native_group.add_argument(
        "--rl-native-env-defaults",
        dest="rl_native_env_defaults",
        action="store_true",
        help="Use native env defaults + top-level scenario overrides for RL collection.",
    )
    native_group.add_argument(
        "--no-rl-native-env-defaults",
        dest="rl_native_env_defaults",
        action="store_false",
        help="Use DiLu custom env builder for RL collection.",
    )
    parser.set_defaults(rl_native_env_defaults=None)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum attempts before stopping (0 -> episodes * 20).",
    )
    parser.add_argument(
        "--no-collect-curriculum",
        action="store_true",
        help="Disable collection curriculum and use fixed collection overrides.",
    )
    quality_gate_group = parser.add_mutually_exclusive_group()
    quality_gate_group.add_argument(
        "--rl-quality-gate",
        dest="rl_quality_gate",
        action="store_true",
        help="Enable RL quality gate for accepted episodes.",
    )
    quality_gate_group.add_argument(
        "--no-rl-quality-gate",
        dest="rl_quality_gate",
        action="store_false",
        help="Disable RL quality gate and accept non-crash RL episodes.",
    )
    parser.set_defaults(rl_quality_gate=True)
    parser.add_argument(
        "--rl-max-early-brake-ratio",
        type=float,
        default=0.6,
        help="Maximum allowed early brake ratio in first 5 decisions.",
    )
    parser.add_argument(
        "--rl-min-avg-speed",
        type=float,
        default=18.0,
        help="Minimum average speed (m/s) for RL episode acceptance.",
    )
    parser.add_argument(
        "--rl-min-progress",
        type=float,
        default=40.0,
        help="Minimum progress floor in meters; final threshold is max(this, 2.5*simulation_duration).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[cyan]Loading configuration from {args.config}...[/cyan]")
    config_data = yaml.load(open(args.config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    collect_cfg = build_collect_config(
        config=config_data,
        collect_simulation_duration=max(1, int(args.collect_simulation_duration)),
        collect_vehicle_count=max(1, int(args.collect_vehicle_count)),
        collect_vehicles_density=max(0.1, float(args.collect_vehicles_density)),
    )

    env_name = "highway-v0"
    ensure_parent_dir(args.output)
    temp_dir = ensure_dir("temp_dbs")
    samples = []
    success_count = 0
    crash_count = 0
    attempt_count = 0
    rejection_counts = collections.Counter()
    rejection_attempt_count = 0
    curriculum_enabled = not bool(args.no_collect_curriculum)
    max_attempts = int(args.max_attempts) if int(args.max_attempts) > 0 else max(1, args.episodes * 20)

    rl_model = None
    rl_algorithm_resolved = None
    rl_expected_obs_shape = None
    rl_expected_vehicle_count = None
    rl_native_env_defaults = False
    if args.collect_controller == "rl":
        if not args.rl_model_path.strip():
            raise ValueError("--rl-model-path is required when --collect-controller rl")
        try:
            from stable_baselines3 import DQN, PPO  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "stable-baselines3 is required for RL collection. Install it with `pip install stable-baselines3`."
            ) from exc
        rl_model, rl_algorithm_resolved = load_rl_model(args.rl_model_path, args.rl_algorithm)
        env_name = str(args.rl_env_id or config_data.get("rl_env_id", "highway-v0")).strip() or "highway-v0"
        if args.rl_native_env_defaults is None:
            rl_native_env_defaults = _to_bool(config_data.get("rl_use_native_env_defaults", False), default=False)
        else:
            rl_native_env_defaults = bool(args.rl_native_env_defaults)
        rl_expected_obs_shape = tuple(int(x) for x in rl_model.observation_space.shape)
        if len(rl_expected_obs_shape) >= 2:
            rl_expected_vehicle_count = int(rl_expected_obs_shape[0])
        if rl_expected_vehicle_count is not None and not rl_native_env_defaults:
            requested_count = int(collect_cfg["vehicle_count"])
            if requested_count != rl_expected_vehicle_count:
                print(
                    f"[yellow]RL model observation shape is {rl_expected_obs_shape}; "
                    f"overriding vehicle_count from {requested_count} to {rl_expected_vehicle_count} "
                    f"for compatibility.[/yellow]"
                )

    def select_action_rule(sce, _obs, rule_state: Optional[Dict[str, int]] = None):
        if rule_state is None:
            return expert_decision_v2_left_pass_preferred(sce)
        action_id, reasoning, new_state = expert_decision_v3_balanced(sce, rule_state)
        rule_state.update(new_state)
        return action_id, reasoning

    def select_action_rl(sce, obs, _rule_state: Optional[Dict[str, int]] = None):
        predicted, _ = rl_model.predict(obs, deterministic=True)
        predicted_action = int(np.asarray(predicted).item())
        available_actions = list(sce.env.unwrapped.get_available_actions())
        action_id = predicted_action
        fallback_used = False
        if action_id not in available_actions:
            fallback_used = True
            if 4 in available_actions:
                action_id = 4
            elif 1 in available_actions:
                action_id = 1
            else:
                action_id = int(available_actions[0])
        reasoning = build_rl_reasoning(
            sce=sce,
            action_id=action_id,
            predicted_action=predicted_action,
            fallback_used=fallback_used,
            available_actions=available_actions,
        )
        return action_id, reasoning

    action_selector = select_action_rl if args.collect_controller == "rl" else select_action_rule

    print(f"[cyan]Collecting {args.episodes} successful episodes[/cyan]")
    print(f"[dim]Output file: {args.output}[/dim]")
    print(f"[dim]Controller: {args.collect_controller}[/dim]")
    print(
        "[dim]Collection overrides: duration={duration}, vehicle_count={count}, vehicles_density={density}[/dim]".format(
            duration=int(collect_cfg["simulation_duration"]),
            count=int(collect_cfg["vehicle_count"]),
            density=float(collect_cfg["vehicles_density"]),
        )
    )
    print(f"[dim]Collection env: {env_name}[/dim]")
    if curriculum_enabled:
        print("[dim]Curriculum: enabled (A:120/20/1.8 -> B:200/30/2.6 -> C:300/40/3.5)[/dim]")
    else:
        print("[dim]Curriculum: disabled (fixed collection overrides)[/dim]")
    print(f"[dim]Max attempts: {max_attempts}[/dim]")
    if args.collect_controller == "rl":
        print(
            f"[dim]RL model: algo={rl_algorithm_resolved or args.rl_algorithm}, native_env_defaults={rl_native_env_defaults}[/dim]"
        )
        print(
            "[dim]RL quality gate: {enabled} | max_early_brake_ratio={ebr:.2f}, min_avg_speed={spd:.1f}, "
            "min_progress=max({prog:.1f}, 2.5*duration)[/dim]".format(
                enabled="on" if bool(args.rl_quality_gate) else "off",
                ebr=float(args.rl_max_early_brake_ratio),
                spd=float(args.rl_min_avg_speed),
                prog=float(args.rl_min_progress),
            )
        )

    use_rich_progress = bool(sys.stdout.isatty())

    if use_rich_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            success_task = progress.add_task("[cyan]Successful episodes", total=args.episodes)
            step_task = progress.add_task("[magenta]Current attempt steps", total=1)

            while success_count < args.episodes and attempt_count < max_attempts:
                attempt_count += 1
                stage_meta = resolve_curriculum_stage(success_count, args.episodes) if curriculum_enabled else {
                    "name": "fixed",
                    "simulation_duration": int(collect_cfg["simulation_duration"]),
                    "vehicle_count": int(collect_cfg["vehicle_count"]),
                    "vehicles_density": float(collect_cfg["vehicles_density"]),
                }
                stage_cfg = build_collect_config(
                    config=config_data,
                    collect_simulation_duration=int(stage_meta["simulation_duration"]),
                    collect_vehicle_count=int(stage_meta["vehicle_count"]),
                    collect_vehicles_density=float(stage_meta["vehicles_density"]),
                )
                if args.collect_controller == "rl" and rl_expected_vehicle_count is not None and not rl_native_env_defaults:
                    stage_cfg["vehicle_count"] = int(rl_expected_vehicle_count)
                max_steps = int(stage_cfg["simulation_duration"])
                if args.collect_controller == "rl" and rl_native_env_defaults:
                    env_cfg = build_native_env_cfg(env_name, config_data, stage_cfg)
                    if rl_expected_vehicle_count is not None and "observation" in env_cfg:
                        env_cfg["observation"]["vehicles_count"] = int(rl_expected_vehicle_count)
                else:
                    env_cfg_map = setup_env(stage_cfg)
                    if env_name not in env_cfg_map:
                        raise ValueError(
                            f"Environment '{env_name}' is not supported by DiLu builder mode. "
                            "Use --rl-native-env-defaults for non-highway-v0 RL envs."
                        )
                    env_cfg = env_cfg_map[env_name]
                seed = random.choice(DEFAULT_DILU_SEEDS)
                temp_db_path = os.path.join(temp_dir, f"temp_collect_{success_count}_{random.randint(1000, 9999)}.db")
                env = None
                progress.reset(step_task, total=max_steps, completed=0)
                progress.update(step_task, description=f"[magenta]Attempt {attempt_count} steps (stage {stage_meta['name']})")

                try:
                    env = gym.make(env_name, render_mode="rgb_array")
                    env.unwrapped.configure(env_cfg)
                    obs, _ = env.reset(seed=seed)
                    rule_state = init_expert_state() if args.collect_controller == "rule" else None
                    episode = collect_episode(
                        env=env,
                        env_name=env_name,
                        seed=seed,
                        temp_db_path=temp_db_path,
                        max_steps=max_steps,
                        initial_obs=obs,
                        action_selector=action_selector,
                        rule_state=rule_state,
                        on_step=lambda step, _done: progress.update(step_task, completed=min(step, max_steps)),
                    )
                    accepted = bool(episode["rows"])
                    gate_reasons: List[str] = []
                    required_progress = max(float(args.rl_min_progress), float(2.5 * max_steps))
                    if args.collect_controller == "rl" and args.rl_quality_gate:
                        accepted, gate_reasons, required_progress = evaluate_rl_episode_quality(episode, max_steps, args)
                        if not accepted:
                            rejection_attempt_count += 1
                            for reason in gate_reasons:
                                rejection_counts[reason] += 1

                    if accepted and episode["rows"]:
                        samples.extend(episode["rows"])
                        success_count += 1
                        progress.advance(success_task, 1)
                        if success_count % args.save_every == 0:
                            write_jsonl(args.output, samples)
                    else:
                        if episode["crashed"]:
                            crash_count += 1
                            progress.console.print(
                                f"[yellow]Attempt {attempt_count} discarded (crash at step {episode['steps']}, "
                                f"success_rate={success_count / max(attempt_count, 1):.2%}).[/yellow]"
                            )
                        elif gate_reasons:
                            metrics = episode.get("metrics", {})
                            progress.console.print(
                                f"[yellow]Attempt {attempt_count} rejected by RL quality gate "
                                f"(reasons={','.join(gate_reasons)} | early_brake_ratio={metrics.get('early_brake_ratio', 0.0):.2f}, "
                                f"avg_speed={metrics.get('avg_speed_mps', 0.0):.2f}, "
                                f"progress={metrics.get('progress_m', 0.0):.1f}/{required_progress:.1f}m).[/yellow]"
                            )
                        else:
                            progress.console.print(
                                f"[yellow]Attempt {attempt_count} produced no usable rows "
                                f"(success_rate={success_count / max(attempt_count, 1):.2%}).[/yellow]"
                            )
                except Exception as exc:
                    progress.console.print(f"[red]Attempt {attempt_count} failed: {exc}[/red]")
                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    if os.path.exists(temp_db_path):
                        try:
                            os.remove(temp_db_path)
                        except PermissionError:
                            pass
    else:
        while success_count < args.episodes and attempt_count < max_attempts:
            attempt_count += 1
            stage_meta = resolve_curriculum_stage(success_count, args.episodes) if curriculum_enabled else {
                "name": "fixed",
                "simulation_duration": int(collect_cfg["simulation_duration"]),
                "vehicle_count": int(collect_cfg["vehicle_count"]),
                "vehicles_density": float(collect_cfg["vehicles_density"]),
            }
            stage_cfg = build_collect_config(
                config=config_data,
                collect_simulation_duration=int(stage_meta["simulation_duration"]),
                collect_vehicle_count=int(stage_meta["vehicle_count"]),
                collect_vehicles_density=float(stage_meta["vehicles_density"]),
            )
            if args.collect_controller == "rl" and rl_expected_vehicle_count is not None and not rl_native_env_defaults:
                stage_cfg["vehicle_count"] = int(rl_expected_vehicle_count)
            max_steps = int(stage_cfg["simulation_duration"])
            if args.collect_controller == "rl" and rl_native_env_defaults:
                env_cfg = build_native_env_cfg(env_name, config_data, stage_cfg)
                if rl_expected_vehicle_count is not None and "observation" in env_cfg:
                    env_cfg["observation"]["vehicles_count"] = int(rl_expected_vehicle_count)
            else:
                env_cfg_map = setup_env(stage_cfg)
                if env_name not in env_cfg_map:
                    raise ValueError(
                        f"Environment '{env_name}' is not supported by DiLu builder mode. "
                        "Use --rl-native-env-defaults for non-highway-v0 RL envs."
                    )
                env_cfg = env_cfg_map[env_name]
            seed = random.choice(DEFAULT_DILU_SEEDS)
            temp_db_path = os.path.join(temp_dir, f"temp_collect_{success_count}_{random.randint(1000, 9999)}.db")
            env = None
            try:
                env = gym.make(env_name, render_mode="rgb_array")
                env.unwrapped.configure(env_cfg)
                obs, _ = env.reset(seed=seed)
                rule_state = init_expert_state() if args.collect_controller == "rule" else None
                episode = collect_episode(
                    env=env,
                    env_name=env_name,
                    seed=seed,
                    temp_db_path=temp_db_path,
                    max_steps=max_steps,
                    initial_obs=obs,
                    action_selector=action_selector,
                    rule_state=rule_state,
                )
                accepted = bool(episode["rows"])
                gate_reasons: List[str] = []
                required_progress = max(float(args.rl_min_progress), float(2.5 * max_steps))
                if args.collect_controller == "rl" and args.rl_quality_gate:
                    accepted, gate_reasons, required_progress = evaluate_rl_episode_quality(episode, max_steps, args)
                    if not accepted:
                        rejection_attempt_count += 1
                        for reason in gate_reasons:
                            rejection_counts[reason] += 1

                if accepted and episode["rows"]:
                    samples.extend(episode["rows"])
                    success_count += 1
                    print(
                        f"[cyan]Progress: {success_count}/{args.episodes} successful episodes "
                        f"(attempts={attempt_count}, success_rate={success_count / max(attempt_count, 1):.2%})[/cyan]"
                    )
                    if success_count % args.save_every == 0:
                        write_jsonl(args.output, samples)
                else:
                    if episode["crashed"]:
                        crash_count += 1
                        print(
                            f"[yellow]Attempt {attempt_count} discarded (crash at step {episode['steps']}, "
                            f"success_rate={success_count / max(attempt_count, 1):.2%}).[/yellow]"
                        )
                    elif gate_reasons:
                        metrics = episode.get("metrics", {})
                        print(
                            f"[yellow]Attempt {attempt_count} rejected by RL quality gate "
                            f"(reasons={','.join(gate_reasons)} | early_brake_ratio={metrics.get('early_brake_ratio', 0.0):.2f}, "
                            f"avg_speed={metrics.get('avg_speed_mps', 0.0):.2f}, "
                            f"progress={metrics.get('progress_m', 0.0):.1f}/{required_progress:.1f}m).[/yellow]"
                        )
                    else:
                        print(
                            f"[yellow]Attempt {attempt_count} produced no usable rows "
                            f"(success_rate={success_count / max(attempt_count, 1):.2%}).[/yellow]"
                        )
            except Exception as exc:
                print(f"[red]Attempt {attempt_count} failed: {exc}[/red]")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                if os.path.exists(temp_db_path):
                    try:
                        os.remove(temp_db_path)
                    except PermissionError:
                        pass

    if success_count < args.episodes:
        print(
            f"[yellow]Stopped early: reached max attempts ({attempt_count}/{max_attempts}) "
            f"with {success_count}/{args.episodes} successful episodes.[/yellow]"
        )

    write_jsonl(args.output, samples)

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    success_rate = success_count / max(attempt_count, 1)
    rejected_total = int(rejection_attempt_count)
    rejection_dist = {str(k): int(v) for k, v in sorted(rejection_counts.items())}
    print(
        f"[green]SUCCESS: Collected {len(samples)} samples in {args.output} | "
        f"attempts={attempt_count}, successes={success_count}, crashes={crash_count}, "
        f"success_rate={success_rate:.2%}[/green]"
    )
    if args.collect_controller == "rl" and args.rl_quality_gate:
        print(
            f"[cyan]RL gate summary: rejected_attempts={rejected_total}, "
            f"rejection_reasons={rejection_dist}[/cyan]"
        )

    summary_path = f"{os.path.splitext(args.output)[0]}_collection_summary.json"
    summary = {
        "output_path": args.output,
        "summary_path": summary_path,
        "controller": args.collect_controller,
        "env_name": env_name,
        "rl_algorithm_requested": str(args.rl_algorithm),
        "rl_algorithm_resolved": rl_algorithm_resolved,
        "rl_native_env_defaults": bool(rl_native_env_defaults),
        "episodes_target": int(args.episodes),
        "attempts": int(attempt_count),
        "successes": int(success_count),
        "crashes": int(crash_count),
        "rows_collected": int(len(samples)),
        "success_rate": float(success_rate),
        "curriculum_enabled": bool(curriculum_enabled),
        "max_attempts": int(max_attempts),
        "rl_quality_gate_enabled": bool(args.rl_quality_gate),
        "rl_quality_gate_thresholds": {
            "max_early_brake_ratio": float(args.rl_max_early_brake_ratio),
            "min_avg_speed": float(args.rl_min_avg_speed),
            "min_progress_floor": float(args.rl_min_progress),
            "dynamic_progress_multiplier": 2.5,
        },
        "rl_quality_gate_rejection_counts": rejection_dist,
        "rl_quality_gate_rejected_attempts": int(rejected_total),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[green]Saved collection summary: {summary_path}[/green]")


if __name__ == "__main__":
    main()
