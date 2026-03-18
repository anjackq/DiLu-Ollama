import argparse
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gymnasium as gym
import yaml
from rich import print

from dilu.runtime import DEFAULT_DILU_SEEDS, build_highway_env_config, ensure_dir
from fine_tuning.pipeline import expert_decision_v3_balanced, init_expert_state


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark stateful rule expert with fixed seed replay.")
    parser.add_argument("--config", default="config.yaml", help="Runtime config path.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of default seeds to replay.")
    parser.add_argument(
        "--output-dir",
        default="results/diagnostics",
        help="Directory to write benchmark JSON summary.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["easy", "A", "B"],
        choices=["easy", "A", "B", "C"],
        help="Benchmark case set.",
    )
    return parser.parse_args()


def case_specs(case_names: List[str]) -> List[Tuple[str, int, int, float]]:
    mapping = {
        "easy": ("easy", 20, 20, 2.0),
        "A": ("A", 120, 20, 1.8),
        "B": ("B", 200, 30, 2.6),
        "C": ("C", 300, 40, 3.5),
    }
    return [mapping[name] for name in case_names]


def run_case(base_cfg: Dict, seeds: List[int], name: str, duration: int, vehicle_count: int, density: float) -> Dict:
    cfg = dict(base_cfg)
    cfg["simulation_duration"] = int(duration)
    cfg["vehicle_count"] = int(vehicle_count)
    cfg["vehicles_density"] = float(density)
    env_cfg = build_highway_env_config(cfg, show_trajectories=False, render_agent=False)["highway-v0"]

    success_count = 0
    crash_count = 0
    steps = []
    crash_steps = []
    action_hist = Counter()
    episode_summaries = []

    for seed in seeds:
        env = gym.make("highway-v0", render_mode="rgb_array")
        env.unwrapped.configure(env_cfg)
        env.reset(seed=seed)
        sce = SimpleNamespace(env=env)
        state = init_expert_state()

        done = False
        truncated = False
        step = 0
        crashed = False
        episode_actions = []
        while not (done or truncated) and step < duration:
            action, _, state = expert_decision_v3_balanced(sce, state)
            episode_actions.append(int(action))
            action_hist[int(action)] += 1
            _, _, done, truncated, info = env.step(int(action))
            step += 1
            if info.get("crashed", False):
                crashed = True
                break

        steps.append(step)
        if crashed:
            crash_count += 1
            crash_steps.append(step)
        else:
            success_count += 1
        episode_summaries.append(
            {
                "seed": int(seed),
                "steps": int(step),
                "crashed": bool(crashed),
                "actions_preview": episode_actions[:20],
            }
        )
        env.close()

    total = len(seeds)
    action_total = sum(action_hist.values()) or 1
    action_share = {str(k): float(v / action_total) for k, v in sorted(action_hist.items())}
    return {
        "case": name,
        "duration": int(duration),
        "vehicle_count": int(vehicle_count),
        "vehicles_density": float(density),
        "episodes": int(total),
        "success_count": int(success_count),
        "crash_count": int(crash_count),
        "success_rate": float(success_count / max(total, 1)),
        "avg_steps_completed": float(sum(steps) / max(len(steps), 1)),
        "avg_crash_step": float(sum(crash_steps) / len(crash_steps)) if crash_steps else None,
        "action_hist": {str(k): int(v) for k, v in sorted(action_hist.items())},
        "action_share": action_share,
        "episode_summaries": episode_summaries,
    }


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    seeds = DEFAULT_DILU_SEEDS[: max(1, int(args.seeds))]
    specs = case_specs(args.cases)
    results = []
    for name, duration, vehicle_count, density in specs:
        print(
            f"[cyan]Running case {name}: duration={duration}, vehicles={vehicle_count}, density={density}, "
            f"seeds={len(seeds)}[/cyan]"
        )
        results.append(run_case(base_cfg, seeds, name, duration, vehicle_count, density))

    agg_A_B = [r for r in results if r["case"] in {"A", "B"}]
    action_hist_ab = Counter()
    for r in agg_A_B:
        for action, count in r["action_hist"].items():
            action_hist_ab[int(action)] += int(count)
    total_ab = sum(action_hist_ab.values()) or 1
    diversity = {
        "lane_change_share": float((action_hist_ab.get(0, 0) + action_hist_ab.get(2, 0)) / total_ab),
        "accelerate_share": float(action_hist_ab.get(3, 0) / total_ab),
        "max_single_action_share": float(max(action_hist_ab.values()) / total_ab) if action_hist_ab else 0.0,
    }

    now_utc = datetime.now(UTC)
    summary = {
        "timestamp": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": args.config,
        "seed_count": len(seeds),
        "cases": results,
        "diversity_A_B": diversity,
        "acceptance": {
            "A_success_ge_0_90": next((r["success_rate"] >= 0.90 for r in results if r["case"] == "A"), None),
            "B_success_ge_0_80": next((r["success_rate"] >= 0.80 for r in results if r["case"] == "B"), None),
            "easy_success_ge_0_95": next((r["success_rate"] >= 0.95 for r in results if r["case"] == "easy"), None),
            "lane_change_share_ge_0_05": diversity["lane_change_share"] >= 0.05,
            "accelerate_share_ge_0_20": diversity["accelerate_share"] >= 0.20,
            "max_single_action_le_0_65": diversity["max_single_action_share"] <= 0.65,
        },
    }

    out_dir = ensure_dir(args.output_dir)
    out_path = os.path.join(out_dir, f"rule_expert_benchmark_{now_utc.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[green]Benchmark summary saved:[/green] {out_path}")
    for case in summary["cases"]:
        print(
            f"[cyan]{case['case']}[/cyan] success_rate={case['success_rate']:.2%}, "
            f"crash_count={case['crash_count']}/{case['episodes']}"
        )
    print(
        f"[cyan]A+B diversity[/cyan] lane_change={diversity['lane_change_share']:.2%}, "
        f"accelerate={diversity['accelerate_share']:.2%}, max_action={diversity['max_single_action_share']:.2%}"
    )


if __name__ == "__main__":
    main()
