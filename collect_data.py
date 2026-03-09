import argparse
import json
import os
import random
import shutil

import gymnasium as gym
import yaml
from rich import print
from tqdm import tqdm

from dilu.runtime import DEFAULT_DILU_SEEDS, build_highway_env_config, ensure_dir, ensure_parent_dir
from dilu.scenario.envScenario import EnvScenario
from fine_tuning.pipeline import expert_decision_v2_left_pass_preferred, write_jsonl


def setup_env(config):
    return build_highway_env_config(
        config,
        show_trajectories=True,
        render_agent=False,
    )


def collect_episode(env, env_name: str, seed: int, temp_db_path: str, max_steps: int):
    sce = EnvScenario(env, env_name, seed, temp_db_path)
    buffer = []
    done = False
    truncated = False
    step = 0
    info = {}

    while not (done or truncated) and step < max_steps:
        description = sce.describe(step)
        available_actions = sce.availableActionsDescription()
        action_id, reasoning = expert_decision_v2_left_pass_preferred(sce)
        _, _, done, truncated, info = env.step(action_id)

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

        if info.get("crashed", False):
            return []

    return [] if info.get("crashed", False) else buffer


def parse_args():
    parser = argparse.ArgumentParser(description="Collect rule-based expert trajectories for fine-tuning.")
    parser.add_argument("--config", default="config.yaml", help="Runtime config path.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of successful episodes to collect.")
    parser.add_argument("--output", default="data/gold_standard_data.jsonl", help="Output JSONL path.")
    parser.add_argument("--save-every", type=int, default=5, help="Incremental save cadence.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[cyan]Loading configuration from {args.config}...[/cyan]")
    config_data = yaml.load(open(args.config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    env_config = setup_env(config_data)

    env_name = "highway-v0"
    env = gym.make(env_name, render_mode="rgb_array")
    env.configure(env_config[env_name])

    ensure_parent_dir(args.output)
    temp_dir = ensure_dir("temp_dbs")
    samples = []
    success_count = 0

    print(f"[cyan]Collecting {args.episodes} successful episodes[/cyan]")
    print(f"[dim]Output file: {args.output}[/dim]")

    pbar = tqdm(total=args.episodes)
    while success_count < args.episodes:
        seed = random.choice(DEFAULT_DILU_SEEDS)
        env.reset(seed=seed)
        temp_db_path = os.path.join(temp_dir, f"temp_collect_{success_count}_{random.randint(1000, 9999)}.db")
        try:
            episode_rows = collect_episode(
                env=env,
                env_name=env_name,
                seed=seed,
                temp_db_path=temp_db_path,
                max_steps=int(config_data["simulation_duration"]),
            )
            if episode_rows:
                samples.extend(episode_rows)
                success_count += 1
                pbar.update(1)
                if success_count % args.save_every == 0:
                    write_jsonl(args.output, samples)
        except Exception as exc:
            print(f"[red]Episode failed: {exc}[/red]")
        finally:
            if os.path.exists(temp_db_path):
                try:
                    os.remove(temp_db_path)
                except PermissionError:
                    pass

    env.close()
    write_jsonl(args.output, samples)

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    print(f"[green]SUCCESS: Collected {len(samples)} samples in {args.output}[/green]")


if __name__ == "__main__":
    main()

