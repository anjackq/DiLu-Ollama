import argparse
import copy
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import yaml
from rich import print

from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.runtime import (
    configure_runtime_env,
    build_highway_env_config,
    DEFAULT_DILU_SEEDS,
    ensure_dir,
    ensure_parent_dir,
    timestamped_results_path,
    current_timestamp,
    slugify_model_name,
    build_experiment_root,
    build_model_root,
    ensure_experiment_layout,
    write_json_atomic,
    read_json,
)
from dilu.scenario.envScenario import EnvScenario


STRICT_RESPONSE_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*([0-4])\s*$", re.IGNORECASE)


def build_env_config(config: Dict) -> Dict:
    return build_highway_env_config(
        config,
        show_trajectories=False,
        render_agent=False,
    )


def parse_seeds(raw: Optional[str]) -> List[int]:
    if not raw:
        return DEFAULT_DILU_SEEDS
    seeds = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def _safe_int_action(action) -> int:
    if isinstance(action, str):
        action = action.strip()
    action = int(action)
    if action < 0 or action > 4:
        raise ValueError(f"Invalid action id: {action}")
    return action


def _response_format_metrics(response_content: str) -> Dict:
    response_content = (response_content or "").strip()
    has_delimiter = "####" in response_content
    strict_match = STRICT_RESPONSE_PATTERN.search(response_content)
    direct_action_parseable = False
    parsed_action = None

    if has_delimiter:
        tail = response_content.split("####")[-1].strip()
        try:
            parsed_action = int(tail)
            if 0 <= parsed_action <= 4:
                direct_action_parseable = True
        except Exception:
            direct_action_parseable = False

    return {
        "has_delimiter": has_delimiter,
        "strict_format_match": bool(strict_match),
        "direct_action_parseable": direct_action_parseable,
        "strict_action": int(strict_match.group(1)) if strict_match else None,
        "direct_parsed_action": parsed_action,
    }


def extract_step_traffic_metrics(env, ttc_threshold_sec: float, headway_threshold_m: float) -> Dict:
    ego_speed_mps = None
    front_gap_m = None
    relative_speed_mps = None
    ttc_sec = None
    ttc_danger = False
    headway_violation = False

    try:
        uenv = env.unwrapped
        ego = getattr(uenv, "vehicle", None)
        road = getattr(uenv, "road", None)
        if ego is not None:
            ego_speed_mps = float(getattr(ego, "speed", 0.0))
        if ego is not None and road is not None:
            front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
            if front_vehicle is not None:
                front_gap_m = float(np.linalg.norm(ego.position - front_vehicle.position))
                relative_speed_mps = float(ego.speed - front_vehicle.speed)
                if relative_speed_mps > 0:
                    ttc_sec = front_gap_m / max(relative_speed_mps, 1e-6)
                    ttc_danger = bool(ttc_sec < ttc_threshold_sec)
                headway_violation = bool(front_gap_m < headway_threshold_m)
    except Exception:
        pass

    return {
        "ego_speed_mps": ego_speed_mps,
        "front_gap_m": front_gap_m,
        "relative_speed_mps": relative_speed_mps,
        "ttc_sec": ttc_sec,
        "ttc_danger": ttc_danger,
        "headway_violation": headway_violation,
    }


def run_episode(
    config: Dict,
    env_config: Dict,
    agent_memory: DrivingMemory,
    seed: int,
    few_shot_num: int,
    temp_dir: str,
    ttc_threshold_sec: float,
    headway_threshold_m: float,
    alignment_sample_rate: float,
    alignment_max_samples: int,
) -> Dict:
    env_type = "highway-v0"
    env = None
    temp_db_path = os.path.join(temp_dir, f"eval_{seed}_{int(time.time() * 1000)}.db")
    started = time.time()
    error = None
    crashed = False
    truncated = False
    terminated = False
    steps = 0
    final_info = {}
    decisions_made = 0
    responses_with_delimiter = 0
    responses_strict_format = 0
    responses_direct_parseable = 0
    format_failure_count = 0
    episode_reward_sum = 0.0
    ego_speed_sum = 0.0
    ego_speed_count = 0
    ttc_danger_steps = 0
    headway_violation_steps = 0
    lane_change_count = 0
    flap_accel_decel_count = 0
    prev_action_id = None
    alignment_samples = []

    try:
        env = gym.make(env_type, render_mode="rgb_array")
        env.configure(env_config[env_type])
        obs, info = env.reset(seed=seed)
        final_info = info

        sce = EnvScenario(env, env_type, seed, temp_db_path)
        agent = DriverAgent(sce, verbose=False)

        prev_action = "Not available"
        for frame_id in range(config["simulation_duration"]):
            _ = np.array(obs, dtype=float)

            fewshot_results = (
                agent_memory.retriveMemory(sce, frame_id, few_shot_num)
                if few_shot_num > 0 else []
            )
            fewshot_messages = [x["human_question"] for x in fewshot_results]
            fewshot_answers = [x["LLM_response"] for x in fewshot_results]

            sce_descrip = sce.describe(frame_id)
            avail_action = sce.availableActionsDescription()
            action, response, human_question, fewshot_answer = agent.few_shot_decision(
                scenario_description=sce_descrip,
                available_actions=avail_action,
                previous_decisions=prev_action,
                fewshot_messages=fewshot_messages,
                driving_intensions="Drive safely and avoid collisons",
                fewshot_answers=fewshot_answers,
            )
            prev_action = action
            decisions_made += 1
            fmt = _response_format_metrics(response)
            responses_with_delimiter += int(fmt["has_delimiter"])
            responses_strict_format += int(fmt["strict_format_match"])
            responses_direct_parseable += int(fmt["direct_action_parseable"])
            format_failure_count += int(not fmt["strict_format_match"])

            action = _safe_int_action(action)
            lane_change_count += int(action in (0, 2))
            if prev_action_id is not None and ((prev_action_id == 3 and action == 4) or (prev_action_id == 4 and action == 3)):
                flap_accel_decel_count += 1
            prev_action_id = action

            if alignment_sample_rate > 0 and len(alignment_samples) < alignment_max_samples and random.random() < alignment_sample_rate:
                alignment_samples.append({
                    "scenario_summary": (sce_descrip or "")[:800],
                    "model_response": response,
                    "action_id": int(action),
                    "step_idx": int(frame_id),
                    "seed": int(seed),
                })

            obs, reward, terminated, truncated, info = env.step(action)
            final_info = info
            crashed = bool(info.get("crashed", False))
            done = terminated or truncated
            steps += 1
            episode_reward_sum += float(reward)

            step_metrics = extract_step_traffic_metrics(env, ttc_threshold_sec, headway_threshold_m)
            if step_metrics["ego_speed_mps"] is not None:
                ego_speed_sum += float(step_metrics["ego_speed_mps"])
                ego_speed_count += 1
            ttc_danger_steps += int(step_metrics["ttc_danger"])
            headway_violation_steps += int(step_metrics["headway_violation"])

            # Keep DB prompt logs for replay/debugging if needed.
            try:
                sce.promptsCommit(frame_id, None, done, human_question, fewshot_answer, response)
            except Exception:
                pass

            if done:
                break

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except Exception:
                pass

    duration_sec = time.time() - started
    episode_reward_avg = episode_reward_sum / max(steps, 1)
    avg_ego_speed_mps = ego_speed_sum / max(ego_speed_count, 1)
    ttc_danger_rate = ttc_danger_steps / max(steps, 1)
    headway_violation_rate = headway_violation_steps / max(steps, 1)
    lane_change_rate = lane_change_count / max(steps, 1)
    flap_accel_decel_rate = flap_accel_decel_count / max(steps, 1)
    decision_latency_ms_avg = (duration_sec / max(steps, 1)) * 1000.0
    format_failure_rate = format_failure_count / max(decisions_made, 1)

    return {
        "seed": seed,
        "steps": steps,
        "max_steps": int(config["simulation_duration"]),
        "crashed": crashed,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success_no_collision": (error is None and not crashed),
        "episode_runtime_sec": round(duration_sec, 3),
        "avg_step_runtime_sec": round(duration_sec / max(steps, 1), 3),
        "decisions_made": decisions_made,
        "responses_with_delimiter": responses_with_delimiter,
        "responses_strict_format": responses_strict_format,
        "responses_direct_parseable": responses_direct_parseable,
        "format_failure_count": format_failure_count,
        "format_failure_rate": round(format_failure_rate, 4),
        "episode_reward_sum": round(episode_reward_sum, 4),
        "episode_reward_avg": round(episode_reward_avg, 4),
        "avg_ego_speed_mps": round(avg_ego_speed_mps, 4),
        "ttc_danger_steps": ttc_danger_steps,
        "ttc_danger_rate": round(ttc_danger_rate, 4),
        "headway_violation_steps": headway_violation_steps,
        "headway_violation_rate": round(headway_violation_rate, 4),
        "lane_change_count": lane_change_count,
        "lane_change_rate": round(lane_change_rate, 4),
        "flap_accel_decel_count": flap_accel_decel_count,
        "flap_accel_decel_rate": round(flap_accel_decel_rate, 4),
        "decision_latency_ms_avg": round(decision_latency_ms_avg, 3),
        "alignment_samples": alignment_samples,
        "error": error,
        "final_info": copy.deepcopy(final_info),
    }


def aggregate_results(model_name: str, episodes: List[Dict]) -> Dict:
    total = len(episodes)
    crashes = sum(1 for e in episodes if e["crashed"])
    errors = sum(1 for e in episodes if e["error"])
    no_collision = sum(1 for e in episodes if e["success_no_collision"])
    truncations = sum(1 for e in episodes if e["truncated"])
    terminations = sum(1 for e in episodes if e["terminated"])
    total_steps = sum(e["steps"] for e in episodes)
    total_runtime = sum(e["episode_runtime_sec"] for e in episodes)
    total_decisions = sum(e.get("decisions_made", 0) for e in episodes)
    total_delimiters = sum(e.get("responses_with_delimiter", 0) for e in episodes)
    total_strict = sum(e.get("responses_strict_format", 0) for e in episodes)
    total_direct = sum(e.get("responses_direct_parseable", 0) for e in episodes)
    total_format_failures = sum(e.get("format_failure_count", 0) for e in episodes)
    total_reward_sum = sum(float(e.get("episode_reward_sum", 0.0)) for e in episodes)
    total_speed = sum(float(e.get("avg_ego_speed_mps", 0.0)) for e in episodes)
    total_ttc_danger_rate = sum(float(e.get("ttc_danger_rate", 0.0)) for e in episodes)
    total_headway_rate = sum(float(e.get("headway_violation_rate", 0.0)) for e in episodes)
    total_lane_change_rate = sum(float(e.get("lane_change_rate", 0.0)) for e in episodes)
    total_flap_rate = sum(float(e.get("flap_accel_decel_rate", 0.0)) for e in episodes)
    total_decision_latency_ms = sum(float(e.get("decision_latency_ms_avg", 0.0)) for e in episodes)

    return {
        "model": model_name,
        "episodes": total,
        "crashes": crashes,
        "errors": errors,
        "no_collision_episodes": no_collision,
        "crash_rate": round(crashes / total, 4) if total else None,
        "no_collision_rate": round(no_collision / total, 4) if total else None,
        "error_rate": round(errors / total, 4) if total else None,
        "truncation_count": truncations,
        "termination_count": terminations,
        "avg_steps": round(total_steps / total, 2) if total else None,
        "avg_episode_runtime_sec": round(total_runtime / total, 3) if total else None,
        "avg_step_runtime_sec": round(total_runtime / max(total_steps, 1), 3),
        "decisions_total": total_decisions,
        "response_delimiter_rate": round(total_delimiters / total_decisions, 4) if total_decisions else None,
        "response_strict_format_rate": round(total_strict / total_decisions, 4) if total_decisions else None,
        "response_direct_parseable_rate": round(total_direct / total_decisions, 4) if total_decisions else None,
        "avg_reward_sum": round(total_reward_sum / total, 4) if total else None,
        "avg_reward_per_step": round(total_reward_sum / max(total_steps, 1), 4),
        "avg_ego_speed_mps": round(total_speed / total, 4) if total else None,
        "ttc_danger_rate_mean": round(total_ttc_danger_rate / total, 4) if total else None,
        "headway_violation_rate_mean": round(total_headway_rate / total, 4) if total else None,
        "lane_change_rate_mean": round(total_lane_change_rate / total, 4) if total else None,
        "flap_accel_decel_rate_mean": round(total_flap_rate / total, 4) if total else None,
        "format_failure_rate_mean": round(total_format_failures / max(total_decisions, 1), 4),
        "decision_latency_ms_avg": round(total_decision_latency_ms / total, 3) if total else None,
    }


def _build_model_extract(
    model_name: str,
    experiment_id: str,
    source_compare_report: str,
    aggregate: Dict,
    episodes: List[Dict],
    metrics_config: Dict,
) -> Dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_schema_version": "2.0",
        "source": "evaluate_models_ollama:model_extract",
        "experiment_id": experiment_id,
        "model": model_name,
        "source_compare_report": source_compare_report,
        "aggregate": aggregate,
        "episodes": episodes,
        "metrics_config": metrics_config,
    }


def _update_experiment_manifest_for_eval(
    experiment_root: str,
    experiment_id: str,
    config_path: str,
    memory_path: str,
    few_shot_num: int,
    simulation_duration: int,
    compare_report_path: str,
    model_summaries: Dict[str, Dict[str, str]],
) -> None:
    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})

    manifest.setdefault("experiment_id", experiment_id)
    manifest.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["config_path"] = config_path
    manifest["memory_path"] = memory_path
    manifest["few_shot_num"] = int(few_shot_num)
    manifest["simulation_duration"] = int(simulation_duration)

    compare_meta = manifest.setdefault("compare", {})
    compare_meta["latest_report"] = compare_report_path
    compare_meta.setdefault("history", [])
    if compare_report_path not in compare_meta["history"]:
        compare_meta["history"].append(compare_report_path)

    models = manifest.setdefault("models", {})
    for model_name, paths in model_summaries.items():
        model_item = models.setdefault(model_name, {})
        model_item.setdefault("slug", slugify_model_name(model_name))
        model_item.setdefault("root", os.path.join(experiment_root, "models", slugify_model_name(model_name)))
        model_item["latest_eval_summary"] = paths["summary"]
        model_item["latest_eval_episodes"] = paths["episodes"]

    write_json_atomic(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DiLu agent behavior across Ollama models on fixed seeds.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to compare (e.g. deepseek-r1:14b dilu-llama3_1-8b-v1)")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to DiLu fixed seed list.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of seeds after parsing.")
    parser.add_argument("--few-shot-num", type=int, default=None, help="Override config few_shot_num.")
    parser.add_argument("--memory-path", default=None, help="Override config memory_path.")
    parser.add_argument("--output", default=None, help="Write JSON report to this file (default: results/eval_compare_<timestamp>.json)")
    parser.add_argument("--experiment-id", default=None, help="Experiment id. Defaults to config or timestamp.")
    parser.add_argument("--results-root", default=None, help="Structured results root. Defaults to config or results/experiments.")
    parser.add_argument("--output-root", default=None, help="Optional compare-output folder override.")
    parser.add_argument("--no-structured-output", action="store_true", help="Disable structured experiment/model outputs.")
    parser.add_argument("--alignment-sample-rate", type=float, default=0.0, help="Sampling probability [0,1] for reasoning-alignment sample collection.")
    parser.add_argument("--alignment-max-samples", type=int, default=0, help="Max alignment samples per model.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seeds = parse_seeds(args.seeds)
    if args.limit is not None:
        seeds = seeds[:args.limit]
    if not seeds:
        raise ValueError("No seeds to evaluate.")

    few_shot_num = config["few_shot_num"] if args.few_shot_num is None else args.few_shot_num
    if args.memory_path:
        config["memory_path"] = args.memory_path
    ttc_threshold_sec = float(config.get("metrics_ttc_threshold_sec", 2.0))
    headway_threshold_m = float(config.get("metrics_headway_threshold_m", 15.0))
    alignment_sample_rate = max(0.0, min(1.0, float(args.alignment_sample_rate)))
    alignment_max_samples = max(0, int(args.alignment_max_samples))
    structured_output = not args.no_structured_output

    results_root = (
        args.results_root
        or config.get("results_root")
        or os.path.join("results", "experiments")
    )
    experiment_id = (
        args.experiment_id
        or config.get("experiment_id")
        or current_timestamp()
    )

    experiment_root = None
    model_roots: Dict[str, str] = {}
    compare_dir = None
    if structured_output:
        experiment_root = build_experiment_root(results_root, experiment_id)
        experiment_id = os.path.basename(experiment_root)
        model_roots = ensure_experiment_layout(experiment_root, args.models)
        compare_dir = ensure_dir(args.output_root) if args.output_root else ensure_dir(os.path.join(experiment_root, "compare"))
    else:
        compare_dir = ensure_dir(args.output_root) if args.output_root else ensure_dir("results")

    env_config = build_env_config(config)
    temp_dir = ensure_dir(os.path.join("temp", "eval_compare"))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_schema_version": "2.0",
        "config_path": args.config,
        "source": "evaluate_models_ollama",
        "experiment_id": experiment_id,
        "experiment_root": experiment_root,
        "compare_dir": compare_dir,
        "structured_output": structured_output,
        "openai_api_type": config["OPENAI_API_TYPE"],
        "models": args.models,
        "model_roots": model_roots,
        "seeds": seeds,
        "few_shot_num": few_shot_num,
        "memory_path": config["memory_path"],
        "simulation_duration": int(config["simulation_duration"]),
        "metrics_config": {
            "ttc_threshold_sec": ttc_threshold_sec,
            "headway_threshold_m": headway_threshold_m,
            "flapping_mode": "accel_decel",
            "alignment_sample_rate": alignment_sample_rate,
            "alignment_max_samples": alignment_max_samples,
        },
        "per_model": {},
        "aggregates": [],
        "alignment_samples": [],
        "model_eval_outputs": {},
    }

    aggregate_by_model: Dict[str, Dict] = {}
    for model_name in args.models:
        print(f"\n[bold cyan]Evaluating model[/bold cyan]: {model_name}")
        configure_runtime_env(config, chat_model_override=model_name)
        agent_memory = DrivingMemory(db_path=config["memory_path"])

        episodes = []
        model_alignment_samples = []
        for idx, seed in enumerate(seeds, start=1):
            print(f"[dim]  Seed {idx}/{len(seeds)}: {seed}[/dim]")
            episode_result = run_episode(
                config=config,
                env_config=env_config,
                agent_memory=agent_memory,
                seed=seed,
                few_shot_num=few_shot_num,
                temp_dir=temp_dir,
                ttc_threshold_sec=ttc_threshold_sec,
                headway_threshold_m=headway_threshold_m,
                alignment_sample_rate=alignment_sample_rate,
                alignment_max_samples=alignment_max_samples,
            )
            episode_alignment_samples = episode_result.pop("alignment_samples", [])
            for sample in episode_alignment_samples:
                sample["model"] = model_name
                model_alignment_samples.append(sample)
            episodes.append(episode_result)
            status = "CRASH" if episode_result["crashed"] else ("ERROR" if episode_result["error"] else "OK")
            print(
                f"    -> {status} | steps={episode_result['steps']}/{episode_result['max_steps']} "
                f"| t={episode_result['episode_runtime_sec']}s"
            )
            if episode_result["error"]:
                print(f"    -> [red]{episode_result['error']}[/red]")

        report["per_model"][model_name] = episodes
        agg = aggregate_results(model_name, episodes)
        report["aggregates"].append(agg)
        aggregate_by_model[model_name] = agg
        report["alignment_samples"].extend(model_alignment_samples[:alignment_max_samples] if alignment_max_samples > 0 else [])

    user_out_path = None
    if args.output:
        user_out_path = ensure_parent_dir(args.output)
        write_json_atomic(user_out_path, report)

    if structured_output:
        out_path = timestamped_results_path("eval_compare", ext=".json", results_dir=compare_dir)
        write_json_atomic(out_path, report)
    else:
        if user_out_path:
            out_path = user_out_path
        else:
            out_path = timestamped_results_path("eval_compare", ext=".json", results_dir=compare_dir)
            write_json_atomic(out_path, report)

    model_summary_paths: Dict[str, Dict[str, str]] = {}
    compare_base = os.path.basename(out_path)
    compare_name, _ = os.path.splitext(compare_base)
    compare_ts = compare_name.replace("eval_compare_", "")

    if structured_output and experiment_root:
        for model_name in args.models:
            model_root = build_model_root(experiment_root, model_name)
            eval_dir = ensure_dir(os.path.join(model_root, "eval"))
            summary_path = os.path.join(eval_dir, f"eval_summary_{compare_ts}.json")
            episodes_path = os.path.join(eval_dir, f"eval_episodes_{compare_ts}.json")

            model_extract = _build_model_extract(
                model_name=model_name,
                experiment_id=experiment_id,
                source_compare_report=out_path,
                aggregate=aggregate_by_model[model_name],
                episodes=report["per_model"][model_name],
                metrics_config=report["metrics_config"],
            )
            write_json_atomic(summary_path, model_extract)
            write_json_atomic(episodes_path, {
                "model": model_name,
                "experiment_id": experiment_id,
                "source_compare_report": out_path,
                "episodes": report["per_model"][model_name],
            })
            model_summary_paths[model_name] = {
                "summary": summary_path,
                "episodes": episodes_path,
            }

        report["model_eval_outputs"] = model_summary_paths
        write_json_atomic(out_path, report)

        _update_experiment_manifest_for_eval(
            experiment_root=experiment_root,
            experiment_id=experiment_id,
            config_path=args.config,
            memory_path=config["memory_path"],
            few_shot_num=int(few_shot_num),
            simulation_duration=int(config["simulation_duration"]),
            compare_report_path=out_path,
            model_summaries=model_summary_paths,
        )

    print("\n[bold green]Aggregate Summary[/bold green]")
    for row in report["aggregates"]:
        print(
            f"- {row['model']}: crashes={row['crashes']}/{row['episodes']} "
            f"(rate={row['crash_rate']}), no_collision_rate={row['no_collision_rate']}, "
            f"avg_steps={row['avg_steps']}, strict_format_rate={row['response_strict_format_rate']}, "
            f"ttc_danger_rate={row['ttc_danger_rate_mean']}, headway_violation_rate={row['headway_violation_rate_mean']}, "
            f"avg_episode_runtime_sec={row['avg_episode_runtime_sec']}"
        )
    print(f"\nSaved report: [bold]{out_path}[/bold]")
    if user_out_path and user_out_path != out_path:
        print(f"Saved user-requested output copy: [bold]{user_out_path}[/bold]")


if __name__ == "__main__":
    main()
