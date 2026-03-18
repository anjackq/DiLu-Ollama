import argparse
import copy
from contextlib import nullcontext
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.wrappers import RecordVideo
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.markup import escape
from rich import print

from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.runtime import (
    configure_runtime_env,
    resolve_model_policy,
    apply_model_policy_to_env,
    build_decision_timeout_penalty_state,
    update_decision_timeout_penalty_state,
    decision_timeout_penalty_snapshot,
    resolve_simulation_env_bundle,
    DEFAULT_DILU_SEEDS,
    ensure_dir,
    ensure_parent_dir,
    timestamped_results_path,
    current_timestamp,
    slugify_model_name,
    build_experiment_root,
    build_model_root,
    build_model_run_dir,
    ensure_experiment_layout,
    write_json_atomic,
    read_json,
)
from dilu.scenario.envScenario import EnvScenario


STRICT_RESPONSE_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*([0-4])\s*$", re.IGNORECASE)


def build_env_bundle(
    config: Dict,
    env_id_override: Optional[str] = None,
    native_env_defaults_override: Optional[bool] = None,
) -> Dict:
    return resolve_simulation_env_bundle(
        config,
        show_trajectories=False,
        render_agent=False,
        env_id_override=env_id_override,
        native_env_defaults_override=native_env_defaults_override,
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


def _resolve_quiet_mode(config: Dict, cli_override: Optional[bool], mode: str = "eval") -> bool:
    def _as_bool(value, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    if cli_override is not None:
        return bool(cli_override)
    global_default = _as_bool(config.get("quiet_mode", False), default=False)
    mode_key = "eval_quiet_mode" if str(mode).strip().lower() == "eval" else "runtime_quiet_mode"
    mode_value = config.get(mode_key)
    if mode_value is None:
        return global_default
    return _as_bool(mode_value, default=global_default)


def _resolve_progress_mode(config: Dict, cli_override: Optional[bool], mode: str = "eval") -> bool:
    def _as_bool(value, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    if cli_override is not None:
        return bool(cli_override)
    global_default = _as_bool(config.get("progress_bar", True), default=True)
    mode_key = "eval_progress_bar" if str(mode).strip().lower() == "eval" else "runtime_progress_bar"
    mode_value = config.get(mode_key)
    if mode_value is None:
        return global_default
    return _as_bool(mode_value, default=global_default)


def _normalize_progress_reply_mode(value: Optional[str]) -> str:
    mode = str(value or "off").strip().lower()
    if mode in {"off", "compact", "full"}:
        return mode
    return "off"


def _resolve_progress_reply_mode(config: Dict, cli_override: Optional[str], mode: str = "eval") -> str:
    if cli_override is not None:
        return _normalize_progress_reply_mode(cli_override)
    global_default = _normalize_progress_reply_mode(config.get("progress_reply_mode", "off"))
    mode_key = "eval_progress_reply_mode" if str(mode).strip().lower() == "eval" else "runtime_progress_reply_mode"
    mode_value = config.get(mode_key)
    if mode_value is None:
        return global_default
    return _normalize_progress_reply_mode(mode_value)


def _normalize_reply_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _compact_reply_preview(step_idx: int, action_id: int, response_text: str, max_len: int = 180) -> str:
    normalized = _normalize_reply_text(response_text)
    if not normalized:
        normalized = "<empty>"
    if len(normalized) > max_len:
        normalized = normalized[: max_len - 3] + "..."
    return f"[dim]      step={step_idx:02d} action={action_id} | {escape(normalized)}[/dim]"


def _full_reply_preview(step_idx: int, action_id: int, response_text: str) -> str:
    body = (response_text or "").strip() or "<empty>"
    return f"[dim]      step={step_idx:02d} action={action_id}[/dim]\n{escape(body)}"


def _is_interactive_output() -> bool:
    try:
        return bool(getattr(sys.stdout, "isatty", lambda: False)())
    except Exception:
        return False


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


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


def extract_step_traffic_metrics(
    env,
    ttc_threshold_sec: float,
    headway_threshold_m: float,
    rear_ttc_threshold_sec: float,
    rear_headway_threshold_m: float,
    low_speed_blocking_threshold_mps: float,
    blocking_front_gap_safe_m: float,
    blocking_front_ttc_safe_sec: float,
) -> Dict:
    ego_speed_mps = None
    front_gap_m = None
    relative_speed_mps = None
    ttc_sec = None
    ttc_danger = False
    headway_violation = False
    rear_gap_m = None
    rear_closing_speed_mps = None
    rear_ttc_sec = None
    rear_ttc_danger = False
    rear_headway_violation = False
    low_speed_blocking = False

    try:
        uenv = env.unwrapped
        ego = getattr(uenv, "vehicle", None)
        road = getattr(uenv, "road", None)
        if ego is not None:
            ego_speed_mps = float(getattr(ego, "speed", 0.0))
        if ego is not None and road is not None:
            front_vehicle, rear_vehicle = road.neighbour_vehicles(ego, ego.lane_index)
            if front_vehicle is not None:
                front_gap_m = float(np.linalg.norm(ego.position - front_vehicle.position))
                relative_speed_mps = float(ego.speed - front_vehicle.speed)
                if relative_speed_mps > 0:
                    ttc_sec = front_gap_m / max(relative_speed_mps, 1e-6)
                    ttc_danger = bool(ttc_sec < ttc_threshold_sec)
                headway_violation = bool(front_gap_m < headway_threshold_m)
            if rear_vehicle is not None:
                rear_gap_m = float(np.linalg.norm(ego.position - rear_vehicle.position))
                rear_closing_speed_mps = float(rear_vehicle.speed - ego.speed)
                if rear_closing_speed_mps > 0:
                    rear_ttc_sec = rear_gap_m / max(rear_closing_speed_mps, 1e-6)
                    rear_ttc_danger = bool(rear_ttc_sec < rear_ttc_threshold_sec)
                rear_headway_violation = bool(rear_gap_m < rear_headway_threshold_m)

            front_gap_safe = bool(front_gap_m is None or front_gap_m >= blocking_front_gap_safe_m)
            front_ttc_safe = bool(ttc_sec is None or ttc_sec >= blocking_front_ttc_safe_sec)
            low_speed_blocking = bool(
                ego_speed_mps is not None
                and ego_speed_mps < low_speed_blocking_threshold_mps
                and front_gap_safe
                and front_ttc_safe
            )
    except Exception:
        pass

    return {
        "ego_speed_mps": ego_speed_mps,
        "front_gap_m": front_gap_m,
        "relative_speed_mps": relative_speed_mps,
        "ttc_sec": ttc_sec,
        "ttc_danger": ttc_danger,
        "headway_violation": headway_violation,
        "rear_gap_m": rear_gap_m,
        "rear_closing_speed_mps": rear_closing_speed_mps,
        "rear_ttc_sec": rear_ttc_sec,
        "rear_ttc_danger": rear_ttc_danger,
        "rear_headway_violation": rear_headway_violation,
        "low_speed_blocking": low_speed_blocking,
    }


def run_episode(
    config: Dict,
    env_config: Dict,
    env_type: str,
    agent_memory: DrivingMemory,
    seed: int,
    few_shot_num: int,
    temp_dir: str,
    ttc_threshold_sec: float,
    headway_threshold_m: float,
    rear_ttc_threshold_sec: float,
    rear_headway_threshold_m: float,
    low_speed_blocking_threshold_mps: float,
    blocking_front_gap_safe_m: float,
    blocking_front_ttc_safe_sec: float,
    alignment_sample_rate: float,
    alignment_max_samples: int,
    slow_decision_threshold_sec: float,
    timeout_penalty_state: Optional[Dict] = None,
    save_artifacts: bool = False,
    run_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    quiet_mode: bool = False,
    on_step: Optional[Callable[[int, bool], None]] = None,
    on_decision: Optional[Callable[[int, int, str, Dict], None]] = None,
) -> Dict:
    env = None
    result_prefix = f"highway_seed_{seed}"
    if save_artifacts:
        if not run_dir:
            raise ValueError("run_dir is required when save_artifacts is enabled.")
        ensure_dir(run_dir)
        database_path = os.path.join(run_dir, f"{result_prefix}.db")
    else:
        database_path = os.path.join(temp_dir, f"eval_{seed}_{int(time.time() * 1000)}.db")
    started = time.time()
    error = None
    crashed = False
    truncated = False
    terminated = False
    steps = 0
    final_info = {}
    episode_stop_reason = "completed"
    decisions_made = 0
    decision_calls_total = 0
    decision_timeout_count = 0
    fallback_action_count = 0
    first_timeout_step = None
    ollama_requested_think_mode = None
    ollama_effective_think_modes_seen = set()
    ollama_native_retry_count = 0
    ollama_openai_fallback_count = 0
    ollama_native_decision_count = 0
    ollama_native_timeout_count = 0
    ollama_native_timeout_short_circuit_count = 0
    responses_with_delimiter = 0
    responses_strict_format = 0
    responses_direct_parseable = 0
    format_failure_count = 0
    episode_reward_sum = 0.0
    ego_speed_sum = 0.0
    ego_speed_count = 0
    ttc_danger_steps = 0
    headway_violation_steps = 0
    rear_ttc_danger_steps = 0
    rear_headway_violation_steps = 0
    low_speed_blocking_steps = 0
    lane_change_count = 0
    flap_accel_decel_count = 0
    prev_action_id = None
    alignment_samples = []
    decision_latencies_sec = []
    slow_decision_count = 0
    penalty_start_events = (
        int(timeout_penalty_state.get("penalty_events", 0))
        if isinstance(timeout_penalty_state, dict)
        else 0
    )
    penalty_start_timeout_triggers = (
        int(timeout_penalty_state.get("timeout_triggers", 0))
        if isinstance(timeout_penalty_state, dict)
        else 0
    )
    penalty_start_slow_triggers = (
        int(timeout_penalty_state.get("slow_triggers", 0))
        if isinstance(timeout_penalty_state, dict)
        else 0
    )
    timeout_penalty_stage_max = (
        int(timeout_penalty_state.get("stage", 0))
        if isinstance(timeout_penalty_state, dict)
        else 0
    )

    try:
        env = gym.make(env_type, render_mode="rgb_array")
        env.unwrapped.configure(env_config[env_type])
        if save_artifacts:
            env = RecordVideo(
                env,
                run_dir,
                episode_trigger=lambda episode_id: True,
                name_prefix=result_prefix,
            )
            try:
                env.unwrapped.set_record_video_wrapper(env)
            except Exception:
                pass
        obs, info = env.reset(seed=seed)
        final_info = info

        sce = EnvScenario(env, env_type, seed, database_path)
        agent = DriverAgent(sce, verbose=True)
        initial_penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
        if initial_penalty_snapshot.get("enabled") and initial_penalty_snapshot.get("effective_decision_timeout_sec") is not None:
            try:
                agent.set_decision_timeout_sec(
                    float(initial_penalty_snapshot["effective_decision_timeout_sec"])
                )
            except Exception:
                pass

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
            decision_calls_total += 1
            decisions_made += 1
            decision_meta = getattr(agent, "last_decision_meta", {}) or {}
            timed_out = bool(decision_meta.get("timed_out", False))
            used_fallback = bool(decision_meta.get("used_fallback", False))
            ollama_requested_think_mode = decision_meta.get("ollama_requested_think_mode") or ollama_requested_think_mode
            ollama_effective_mode = decision_meta.get("ollama_effective_think_mode")
            ollama_transport = decision_meta.get("ollama_transport")
            ollama_native_retry_used = bool(decision_meta.get("ollama_native_retry_used", False))
            ollama_native_timeout = bool(decision_meta.get("ollama_native_timeout", False))
            ollama_native_timeout_short_circuit = bool(
                decision_meta.get("ollama_native_timeout_short_circuit", False)
            )
            decision_elapsed_sec = float(decision_meta.get("decision_elapsed_sec", 0.0) or 0.0)
            decision_timeout_count += int(timed_out)
            fallback_action_count += int(used_fallback)
            ollama_native_retry_count += int(ollama_native_retry_used)
            ollama_openai_fallback_count += int(ollama_transport == "openai_compat_fallback")
            ollama_native_decision_count += int(ollama_transport == "native")
            ollama_native_timeout_count += int(ollama_native_timeout)
            ollama_native_timeout_short_circuit_count += int(ollama_native_timeout_short_circuit)
            decision_latencies_sec.append(decision_elapsed_sec)
            slow_decision_count += int(decision_elapsed_sec >= max(0.001, slow_decision_threshold_sec))
            penalty_update = update_decision_timeout_penalty_state(
                timeout_penalty_state,
                timed_out=timed_out,
                decision_elapsed_sec=decision_elapsed_sec,
                slow_threshold_sec=slow_decision_threshold_sec,
            )
            timeout_penalty_stage_max = max(timeout_penalty_stage_max, int(penalty_update.get("stage", 0)))
            if penalty_update.get("escalated"):
                effective_decision_timeout_sec = penalty_update.get("effective_decision_timeout_sec")
                if effective_decision_timeout_sec is not None:
                    try:
                        agent.set_decision_timeout_sec(float(effective_decision_timeout_sec))
                    except Exception:
                        pass
                if not quiet_mode:
                    print(
                        "[yellow]Adaptive timeout penalty escalated[/yellow] "
                        f"(reason={penalty_update.get('reason')}, stage={penalty_update.get('stage')}, "
                        f"decision_timeout={round(float(effective_decision_timeout_sec), 3) if effective_decision_timeout_sec is not None else 'n/a'}s)"
                    )
            if ollama_effective_mode:
                ollama_effective_think_modes_seen.add(str(ollama_effective_mode))
            if timed_out and first_timeout_step is None:
                first_timeout_step = int(frame_id)

            fmt = _response_format_metrics(response)
            responses_with_delimiter += int(fmt["has_delimiter"])
            responses_strict_format += int(fmt["strict_format_match"])
            responses_direct_parseable += int(fmt["direct_action_parseable"])
            format_failure_count += int(not fmt["strict_format_match"])

            action = _safe_int_action(action)
            if on_decision is not None:
                try:
                    on_decision(int(frame_id + 1), int(action), response, dict(decision_meta))
                except Exception:
                    pass
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
            if on_step is not None:
                try:
                    on_step(int(steps), bool(done))
                except Exception:
                    pass
            episode_reward_sum += float(reward)

            step_metrics = extract_step_traffic_metrics(
                env,
                ttc_threshold_sec,
                headway_threshold_m,
                rear_ttc_threshold_sec,
                rear_headway_threshold_m,
                low_speed_blocking_threshold_mps,
                blocking_front_gap_safe_m,
                blocking_front_ttc_safe_sec,
            )
            if step_metrics["ego_speed_mps"] is not None:
                ego_speed_sum += float(step_metrics["ego_speed_mps"])
                ego_speed_count += 1
            ttc_danger_steps += int(step_metrics["ttc_danger"])
            headway_violation_steps += int(step_metrics["headway_violation"])
            rear_ttc_danger_steps += int(step_metrics["rear_ttc_danger"])
            rear_headway_violation_steps += int(step_metrics["rear_headway_violation"])
            low_speed_blocking_steps += int(step_metrics["low_speed_blocking"])

            # Keep DB prompt logs for replay/debugging if needed.
            try:
                sce.promptsCommit(frame_id, None, done, human_question, fewshot_answer, response)
            except Exception:
                pass

            if done:
                break

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        episode_stop_reason = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if (not save_artifacts) and os.path.exists(database_path):
            try:
                os.remove(database_path)
            except Exception:
                pass

    duration_sec = time.time() - started
    episode_reward_avg = episode_reward_sum / max(steps, 1)
    avg_ego_speed_mps = ego_speed_sum / max(ego_speed_count, 1)
    ttc_danger_rate = ttc_danger_steps / max(steps, 1)
    headway_violation_rate = headway_violation_steps / max(steps, 1)
    rear_ttc_danger_rate = rear_ttc_danger_steps / max(steps, 1)
    rear_headway_violation_rate = rear_headway_violation_steps / max(steps, 1)
    low_speed_blocking_rate = low_speed_blocking_steps / max(steps, 1)
    lane_change_rate = lane_change_count / max(steps, 1)
    flap_accel_decel_rate = flap_accel_decel_count / max(steps, 1)
    decision_latency_ms_avg = (duration_sec / max(steps, 1)) * 1000.0
    format_failure_rate = format_failure_count / max(decisions_made, 1)
    decision_timeout_rate = decision_timeout_count / max(decision_calls_total, 1)
    fallback_action_rate = fallback_action_count / max(decision_calls_total, 1)
    ollama_native_retry_rate = ollama_native_retry_count / max(decision_calls_total, 1)
    ollama_openai_fallback_rate = ollama_openai_fallback_count / max(decision_calls_total, 1)
    p95_decision_latency_sec = float(np.percentile(decision_latencies_sec, 95)) if decision_latencies_sec else 0.0
    timeout_triggered = decision_timeout_count > 0
    penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
    timeout_penalty_events = (
        int(penalty_snapshot.get("penalty_events", 0)) - penalty_start_events
        if penalty_snapshot.get("enabled")
        else 0
    )
    timeout_penalty_timeout_triggers = (
        int(penalty_snapshot.get("timeout_triggers", 0)) - penalty_start_timeout_triggers
        if penalty_snapshot.get("enabled")
        else 0
    )
    timeout_penalty_slow_triggers = (
        int(penalty_snapshot.get("slow_triggers", 0)) - penalty_start_slow_triggers
        if penalty_snapshot.get("enabled")
        else 0
    )

    if episode_stop_reason != "error":
        if crashed:
            episode_stop_reason = "crash"
        elif truncated:
            episode_stop_reason = "truncated"
        elif terminated:
            episode_stop_reason = "terminated"
        else:
            episode_stop_reason = "completed"

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
        "episode_stop_reason": episode_stop_reason,
        "timeout_triggered": bool(timeout_triggered),
        "first_timeout_step": first_timeout_step,
        "decision_calls_total": decision_calls_total,
        "decision_timeout_count": decision_timeout_count,
        "decision_timeout_rate": round(decision_timeout_rate, 4),
        "fallback_action_count": fallback_action_count,
        "fallback_action_rate": round(fallback_action_rate, 4),
        "ollama_requested_think_mode": ollama_requested_think_mode,
        "ollama_effective_think_modes_seen": sorted(ollama_effective_think_modes_seen),
        "ollama_native_retry_count": ollama_native_retry_count,
        "ollama_native_retry_rate": round(ollama_native_retry_rate, 4),
        "ollama_openai_fallback_count": ollama_openai_fallback_count,
        "ollama_openai_fallback_rate": round(ollama_openai_fallback_rate, 4),
        "ollama_native_decision_count": ollama_native_decision_count,
        "ollama_native_timeout_count": ollama_native_timeout_count,
        "ollama_native_timeout_short_circuit_count": ollama_native_timeout_short_circuit_count,
        "ollama_downgrade_triggered": bool(ollama_native_retry_count > 0 or ("auto" in ollama_effective_think_modes_seen and ollama_requested_think_mode == "think")),
        "slow_decision_count": int(slow_decision_count),
        "p95_decision_latency_sec": round(p95_decision_latency_sec, 4),
        "timeout_penalty_stage_max": int(timeout_penalty_stage_max),
        "timeout_penalty_events": int(max(0, timeout_penalty_events)),
        "timeout_penalty_timeout_triggers": int(max(0, timeout_penalty_timeout_triggers)),
        "timeout_penalty_slow_triggers": int(max(0, timeout_penalty_slow_triggers)),
        "timeout_penalty_final_decision_timeout_sec": (
            round(float(penalty_snapshot.get("effective_decision_timeout_sec")), 4)
            if penalty_snapshot.get("effective_decision_timeout_sec") is not None
            else None
        ),
        # Deprecated alias for one transition cycle.
        "timeout_penalty_final_native_timeout_sec": (
            round(float(penalty_snapshot.get("effective_decision_timeout_sec")), 4)
            if penalty_snapshot.get("effective_decision_timeout_sec") is not None
            else None
        ),
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
        "rear_ttc_danger_steps": rear_ttc_danger_steps,
        "rear_ttc_danger_rate": round(rear_ttc_danger_rate, 4),
        "rear_headway_violation_steps": rear_headway_violation_steps,
        "rear_headway_violation_rate": round(rear_headway_violation_rate, 4),
        "low_speed_blocking_steps": low_speed_blocking_steps,
        "low_speed_blocking_rate": round(low_speed_blocking_rate, 4),
        "lane_change_count": lane_change_count,
        "lane_change_rate": round(lane_change_rate, 4),
        "flap_accel_decel_count": flap_accel_decel_count,
        "flap_accel_decel_rate": round(flap_accel_decel_rate, 4),
        "decision_latency_ms_avg": round(decision_latency_ms_avg, 3),
        "alignment_samples": alignment_samples,
        "model": model_name,
        "database_path": database_path if save_artifacts else None,
        "video_prefix": result_prefix if save_artifacts else None,
        "run_id": run_id if save_artifacts else None,
        "run_dir": run_dir if save_artifacts else None,
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
    total_decision_calls = sum(e.get("decision_calls_total", e.get("decisions_made", 0)) for e in episodes)
    total_decision_timeouts = sum(e.get("decision_timeout_count", 0) for e in episodes)
    timeout_episode_count = sum(1 for e in episodes if e.get("timeout_triggered", False))
    total_fallback_actions = sum(e.get("fallback_action_count", 0) for e in episodes)
    total_ollama_native_retries = sum(e.get("ollama_native_retry_count", 0) for e in episodes)
    total_ollama_openai_fallbacks = sum(e.get("ollama_openai_fallback_count", 0) for e in episodes)
    total_ollama_native_decisions = sum(e.get("ollama_native_decision_count", 0) for e in episodes)
    total_ollama_native_timeouts = sum(e.get("ollama_native_timeout_count", 0) for e in episodes)
    total_ollama_native_timeout_short_circuits = sum(
        e.get("ollama_native_timeout_short_circuit_count", 0) for e in episodes
    )
    ollama_native_timeout_episode_count = sum(1 for e in episodes if e.get("ollama_native_timeout_count", 0) > 0)
    ollama_downgrade_episode_count = sum(1 for e in episodes if e.get("ollama_downgrade_triggered", False))
    timeout_cap_stops = sum(1 for e in episodes if e.get("episode_stop_reason") == "episode_timeout_cap")
    total_delimiters = sum(e.get("responses_with_delimiter", 0) for e in episodes)
    total_strict = sum(e.get("responses_strict_format", 0) for e in episodes)
    total_direct = sum(e.get("responses_direct_parseable", 0) for e in episodes)
    total_format_failures = sum(e.get("format_failure_count", 0) for e in episodes)
    total_reward_sum = sum(float(e.get("episode_reward_sum", 0.0)) for e in episodes)
    total_speed = sum(float(e.get("avg_ego_speed_mps", 0.0)) for e in episodes)
    total_ttc_danger_rate = sum(float(e.get("ttc_danger_rate", 0.0)) for e in episodes)
    total_headway_rate = sum(float(e.get("headway_violation_rate", 0.0)) for e in episodes)
    total_rear_ttc_danger_rate = sum(float(e.get("rear_ttc_danger_rate", 0.0)) for e in episodes)
    total_rear_headway_rate = sum(float(e.get("rear_headway_violation_rate", 0.0)) for e in episodes)
    total_low_speed_blocking_rate = sum(float(e.get("low_speed_blocking_rate", 0.0)) for e in episodes)
    total_lane_change_rate = sum(float(e.get("lane_change_rate", 0.0)) for e in episodes)
    total_flap_rate = sum(float(e.get("flap_accel_decel_rate", 0.0)) for e in episodes)
    total_decision_latency_ms = sum(float(e.get("decision_latency_ms_avg", 0.0)) for e in episodes)
    total_timeout_penalty_events = sum(int(e.get("timeout_penalty_events", 0)) for e in episodes)
    total_timeout_penalty_timeout_triggers = sum(
        int(e.get("timeout_penalty_timeout_triggers", 0)) for e in episodes
    )
    total_timeout_penalty_slow_triggers = sum(
        int(e.get("timeout_penalty_slow_triggers", 0)) for e in episodes
    )
    timeout_penalty_stage_max_values = [int(e.get("timeout_penalty_stage_max", 0)) for e in episodes]
    timeout_penalty_final_values = [
        float(e.get("timeout_penalty_final_decision_timeout_sec"))
        for e in episodes
        if e.get("timeout_penalty_final_decision_timeout_sec") is not None
    ]

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
        "decision_calls_total": total_decision_calls,
        "decision_timeouts_total": total_decision_timeouts,
        "decision_timeout_rate_mean": round(total_decision_timeouts / max(total_decision_calls, 1), 4),
        "timeout_episode_count": timeout_episode_count,
        "timeout_episode_rate": round(timeout_episode_count / total, 4) if total else None,
        "fallback_actions_total": total_fallback_actions,
        "fallback_action_rate_mean": round(total_fallback_actions / max(total_decision_calls, 1), 4),
        "ollama_native_retries_total": total_ollama_native_retries,
        "ollama_native_retry_rate_mean": round(total_ollama_native_retries / max(total_decision_calls, 1), 4),
        "ollama_openai_fallbacks_total": total_ollama_openai_fallbacks,
        "ollama_openai_fallback_rate_mean": round(total_ollama_openai_fallbacks / max(total_decision_calls, 1), 4),
        "ollama_native_decisions_total": total_ollama_native_decisions,
        "ollama_native_decision_rate_mean": round(total_ollama_native_decisions / max(total_decision_calls, 1), 4),
        "ollama_native_timeouts_total": total_ollama_native_timeouts,
        "ollama_native_timeout_rate_mean": round(total_ollama_native_timeouts / max(total_decision_calls, 1), 4),
        "ollama_native_timeout_short_circuits_total": total_ollama_native_timeout_short_circuits,
        "ollama_native_timeout_episode_count": ollama_native_timeout_episode_count,
        "ollama_native_timeout_episode_rate": round(ollama_native_timeout_episode_count / total, 4) if total else None,
        "ollama_downgrade_episode_count": ollama_downgrade_episode_count,
        "ollama_downgrade_episode_rate": round(ollama_downgrade_episode_count / total, 4) if total else None,
        "episodes_stopped_by_timeout_cap": timeout_cap_stops,
        "response_delimiter_rate": round(total_delimiters / total_decisions, 4) if total_decisions else None,
        "response_strict_format_rate": round(total_strict / total_decisions, 4) if total_decisions else None,
        "response_direct_parseable_rate": round(total_direct / total_decisions, 4) if total_decisions else None,
        "avg_reward_sum": round(total_reward_sum / total, 4) if total else None,
        "avg_reward_per_step": round(total_reward_sum / max(total_steps, 1), 4),
        "avg_ego_speed_mps": round(total_speed / total, 4) if total else None,
        "ttc_danger_rate_mean": round(total_ttc_danger_rate / total, 4) if total else None,
        "headway_violation_rate_mean": round(total_headway_rate / total, 4) if total else None,
        "rear_ttc_danger_rate_mean": round(total_rear_ttc_danger_rate / total, 4) if total else None,
        "rear_headway_violation_rate_mean": round(total_rear_headway_rate / total, 4) if total else None,
        "low_speed_blocking_rate_mean": round(total_low_speed_blocking_rate / total, 4) if total else None,
        "lane_change_rate_mean": round(total_lane_change_rate / total, 4) if total else None,
        "flap_accel_decel_rate_mean": round(total_flap_rate / total, 4) if total else None,
        "format_failure_rate_mean": round(total_format_failures / max(total_decisions, 1), 4),
        "decision_latency_ms_avg": round(total_decision_latency_ms / total, 3) if total else None,
        "timeout_penalty_events_total": int(total_timeout_penalty_events),
        "timeout_penalty_timeout_triggers_total": int(total_timeout_penalty_timeout_triggers),
        "timeout_penalty_slow_triggers_total": int(total_timeout_penalty_slow_triggers),
        "timeout_penalty_events_rate_mean": round(total_timeout_penalty_events / max(total_decision_calls, 1), 4),
        "timeout_penalty_stage_max_mean": (
            round(sum(timeout_penalty_stage_max_values) / max(total, 1), 4)
            if total else None
        ),
        "timeout_penalty_stage_max_global": max(timeout_penalty_stage_max_values) if timeout_penalty_stage_max_values else 0,
        "timeout_penalty_final_decision_timeout_sec_mean": (
            round(sum(timeout_penalty_final_values) / len(timeout_penalty_final_values), 4)
            if timeout_penalty_final_values else None
        ),
        # Deprecated alias for one transition cycle.
        "timeout_penalty_final_native_timeout_sec_mean": (
            round(sum(timeout_penalty_final_values) / len(timeout_penalty_final_values), 4)
            if timeout_penalty_final_values else None
        ),
    }


def _append_eval_run_log(log_path: str, model_name: str, episode: Dict) -> None:
    ensure_parent_dir(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            "Model {model} | Seed {seed} | Stop {stop} | Steps {steps}/{max_steps} | "
            "Crash {crashed} | Error {error} | Runtime {runtime}s | Penalty(stage={stage},events={events},decision_timeout={timeout}s) | DB {db} | Video {video}\n".format(
                model=model_name,
                seed=episode.get("seed"),
                stop=episode.get("episode_stop_reason"),
                steps=episode.get("steps"),
                max_steps=episode.get("max_steps"),
                crashed=episode.get("crashed"),
                error=episode.get("error"),
                runtime=episode.get("episode_runtime_sec"),
                stage=episode.get("timeout_penalty_stage_max"),
                events=episode.get("timeout_penalty_events"),
                timeout=episode.get("timeout_penalty_final_decision_timeout_sec"),
                db=episode.get("database_path"),
                video=episode.get("video_prefix"),
            )
        )


def _build_eval_run_metrics_report(
    model_name: str,
    experiment_id: str,
    experiment_root: Optional[str],
    run_id: str,
    run_dir: str,
    config_path: str,
    openai_api_type: str,
    few_shot_num: int,
    memory_path: str,
    simulation_duration: int,
    metrics_config: Dict,
    episodes: List[Dict],
    aggregate: Dict,
) -> Dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_schema_version": "2.0",
        "source": "evaluate_models_ollama:run_artifacts",
        "config_path": config_path,
        "openai_api_type": openai_api_type,
        "model": model_name,
        "experiment_id": experiment_id,
        "experiment_root": experiment_root,
        "run_id": run_id,
        "run_dir": run_dir,
        "few_shot_num": int(few_shot_num),
        "memory_path": memory_path,
        "simulation_duration": int(simulation_duration),
        "metrics_config": metrics_config,
        "episodes": episodes,
        "aggregate": aggregate,
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
    model_run_outputs: Optional[Dict[str, Dict[str, str]]] = None,
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
        if model_run_outputs and model_name in model_run_outputs:
            run_paths = model_run_outputs[model_name]
            model_item["latest_eval_run_id"] = run_paths.get("run_id")
            model_item["latest_eval_run_dir"] = run_paths.get("run_dir")
            model_item["latest_eval_run_metrics"] = run_paths.get("run_metrics")

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
    parser.add_argument("--env-id", default=None, help="Simulation env id override (default: config sim_env_id -> rl_env_id alias -> highway-fast-v0).")
    env_native_group = parser.add_mutually_exclusive_group()
    env_native_group.add_argument(
        "--native-env-defaults",
        dest="native_env_defaults",
        action="store_true",
        help="Use native env defaults with top-level config overrides (default).",
    )
    env_native_group.add_argument(
        "--no-native-env-defaults",
        dest="native_env_defaults",
        action="store_false",
        help="Use legacy DiLu env builder behavior.",
    )
    parser.set_defaults(native_env_defaults=None)
    parser.add_argument("--save-run-artifacts", action="store_true", help="Save run-style artifacts (video/db/log/run_metrics) per model during evaluation.")
    parser.add_argument("--eval-run-id", default=None, help="Run id used under models/<slug>/runs/<eval_run_id> when --save-run-artifacts is enabled.")
    parser.add_argument("--quiet", action="store_true", help="Suppress high-frequency step/decision logs.")
    parser.add_argument("--no-quiet", action="store_true", help="Force step/decision logs on even if config quiet mode is enabled.")
    parser.add_argument("--progress", action="store_true", help="Show CLI progress bars.")
    parser.add_argument("--no-progress", action="store_true", help="Disable CLI progress bars.")
    parser.add_argument(
        "--progress-replies",
        choices=["off", "compact", "full"],
        help="Show LLM replies while progress bars are active.",
    )
    parser.add_argument("--decision-timeout-sec", type=float, default=None, help="Hard timeout per model decision call. Default: config eval_decision_timeout_sec or 60.")
    parser.add_argument("--decision-max-output-tokens", type=int, default=None, help="Deprecated in policy mode: ignored for timeout-only policy.")
    parser.add_argument("--disable-streaming", action="store_true", help="Deprecated in policy mode: ignored for timeout-only policy.")
    parser.add_argument("--disable-checker-llm", action="store_true", help="Deprecated in policy mode: ignored for timeout-only policy.")
    parser.add_argument("--ollama-think-mode", choices=["auto", "think", "no_think"], default=None, help="Deprecated in policy mode: ignored for timeout-only policy.")
    parser.add_argument("--ollama-use-native-chat", action="store_true", help="Deprecated in policy mode: ignored for timeout-only policy.")
    parser.add_argument("--ollama-disable-native-chat", action="store_true", help="Deprecated in policy mode: ignored for timeout-only policy.")
    parser.add_argument("--alignment-sample-rate", type=float, default=0.0, help="Sampling probability [0,1] for reasoning-alignment sample collection.")
    parser.add_argument("--alignment-max-samples", type=int, default=0, help="Max alignment samples per model.")
    args = parser.parse_args()
    if args.quiet and args.no_quiet:
        raise ValueError("Use only one of --quiet or --no-quiet.")
    if args.progress and args.no_progress:
        raise ValueError("Use only one of --progress or --no-progress.")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cli_quiet_override = True if args.quiet else (False if args.no_quiet else None)
    resolved_eval_quiet_mode = _resolve_quiet_mode(config, cli_quiet_override, mode="eval")
    cli_progress_override = True if args.progress else (False if args.no_progress else None)
    resolved_eval_progress_mode = _resolve_progress_mode(config, cli_progress_override, mode="eval")
    progress_enabled = bool(resolved_eval_progress_mode and _is_interactive_output())
    resolved_eval_progress_reply_mode = _resolve_progress_reply_mode(
        config,
        args.progress_replies,
        mode="eval",
    )
    effective_eval_progress_reply_mode = (
        resolved_eval_progress_reply_mode
        if (progress_enabled and (not resolved_eval_quiet_mode))
        else "off"
    )
    step_log_quiet_mode = bool(resolved_eval_quiet_mode or progress_enabled)

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
    rear_ttc_threshold_sec = float(config.get("metrics_rear_ttc_threshold_sec", 2.5))
    rear_headway_threshold_m = float(config.get("metrics_rear_headway_threshold_m", 12.0))
    low_speed_blocking_threshold_mps = float(config.get("metrics_low_speed_blocking_threshold_mps", 8.5))
    blocking_front_gap_safe_m = float(config.get("metrics_blocking_front_gap_safe_m", 25.0))
    blocking_front_ttc_safe_sec = float(config.get("metrics_blocking_front_ttc_safe_sec", 4.0))
    alignment_sample_rate = max(0.0, min(1.0, float(args.alignment_sample_rate)))
    alignment_max_samples = max(0, int(args.alignment_max_samples))
    structured_output = not args.no_structured_output
    save_run_artifacts = bool(config.get("eval_save_run_artifacts", False)) or bool(args.save_run_artifacts)
    eval_run_id = str(
        args.eval_run_id
        or config.get("eval_run_id")
        or f"eval_run_{current_timestamp()}"
    ).strip()
    if not eval_run_id:
        eval_run_id = f"eval_run_{current_timestamp()}"
    if save_run_artifacts and not structured_output:
        raise ValueError("--save-run-artifacts requires structured output. Remove --no-structured-output.")
    default_decision_timeout_sec = float(config.get("eval_decision_timeout_sec", 60.0))
    slow_decision_threshold_sec = float(config.get("eval_slow_decision_threshold_sec", 5.0))
    adaptive_timeout_penalty_enabled = bool(config.get("adaptive_timeout_penalty_enabled", True))
    adaptive_timeout_halving_factor = float(config.get("adaptive_timeout_halving_factor", 0.5))
    if adaptive_timeout_halving_factor <= 0.0 or adaptive_timeout_halving_factor >= 1.0:
        adaptive_timeout_halving_factor = 0.5
    adaptive_timeout_min_sec = float(config.get("adaptive_timeout_min_sec", 4.0))
    adaptive_timeout_min_sec = max(1.0, adaptive_timeout_min_sec)
    adaptive_timeout_trigger_consecutive_slow = int(
        config.get("adaptive_timeout_trigger_consecutive_slow", 2)
    )
    adaptive_timeout_trigger_consecutive_slow = max(1, adaptive_timeout_trigger_consecutive_slow)
    provider = str(config.get("OPENAI_API_TYPE", "")).strip().lower()
    shared_policy_overrides = config.get("model_policy_overrides", {})
    if not isinstance(shared_policy_overrides, dict):
        shared_policy_overrides = {}
    legacy_eval_overrides = config.get("eval_model_overrides", {})
    if not isinstance(legacy_eval_overrides, dict):
        legacy_eval_overrides = {}
    deprecated_override_fields_declared = sorted(
        {
            key
            for override_map in (shared_policy_overrides, legacy_eval_overrides)
            for value in override_map.values()
            if isinstance(value, dict)
            for key in value.keys()
            if key
            in {
                "decision_max_output_tokens",
                "disable_streaming",
                "disable_checker_llm",
                "ollama_think_mode",
                "ollama_use_native_chat",
                "ollama_native_chat_timeout_sec",
            }
        }
    )
    if deprecated_override_fields_declared:
        print(
            "[yellow]Deprecated output-affecting policy fields were found in config overrides and will be ignored "
            f"(timeout-only policy): {', '.join(deprecated_override_fields_declared)}[/yellow]"
        )

    cli_decision_timeout_sec = (
        float(args.decision_timeout_sec) if args.decision_timeout_sec is not None else None
    )
    cli_decision_max_output_tokens = int(args.decision_max_output_tokens) if args.decision_max_output_tokens is not None else None
    cli_disable_streaming = bool(args.disable_streaming)
    cli_disable_checker_llm = bool(args.disable_checker_llm)
    cli_ollama_think_mode = str(args.ollama_think_mode).strip().lower() if args.ollama_think_mode else None
    cli_ollama_use_native_chat = bool(args.ollama_use_native_chat or args.ollama_disable_native_chat)
    cli_policy_overrides = {}
    if cli_decision_timeout_sec is not None:
        cli_policy_overrides["decision_timeout_sec"] = float(cli_decision_timeout_sec)
    if cli_decision_max_output_tokens is not None:
        cli_policy_overrides["decision_max_output_tokens"] = int(cli_decision_max_output_tokens)
    if cli_disable_streaming:
        cli_policy_overrides["disable_streaming"] = True
    if cli_disable_checker_llm:
        cli_policy_overrides["disable_checker_llm"] = True
    if cli_ollama_think_mode:
        cli_policy_overrides["ollama_think_mode"] = cli_ollama_think_mode
    if cli_ollama_use_native_chat:
        cli_policy_overrides["ollama_use_native_chat"] = True
    deprecated_cli_policy_flags = sorted(
        [
            name
            for name, enabled in {
                "decision_max_output_tokens": cli_decision_max_output_tokens is not None,
                "disable_streaming": cli_disable_streaming,
                "disable_checker_llm": cli_disable_checker_llm,
                "ollama_think_mode": bool(cli_ollama_think_mode),
                "ollama_use_native_chat": bool(cli_ollama_use_native_chat),
            }.items()
            if enabled
        ]
    )
    if deprecated_cli_policy_flags:
        print(
            "[yellow]Deprecated policy CLI flags were provided and will be ignored "
            f"(timeout-only policy): {', '.join(deprecated_cli_policy_flags)}[/yellow]"
        )

    config["eval_save_run_artifacts"] = bool(save_run_artifacts)
    config["eval_run_id"] = eval_run_id

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

    env_bundle = build_env_bundle(
        config,
        env_id_override=args.env_id,
        native_env_defaults_override=args.native_env_defaults,
    )
    for warning_msg in env_bundle.get("warnings", []):
        print(f"[yellow]{warning_msg}[/yellow]")
    env_config = env_bundle["env_config_map"]
    env_type = str(env_bundle["env_id"])
    env_config_snapshot = env_bundle["env_config_snapshot"]
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
        "save_run_artifacts": bool(save_run_artifacts),
        "eval_run_id": eval_run_id if save_run_artifacts else None,
        "openai_api_type": config["OPENAI_API_TYPE"],
        "models": args.models,
        "model_roots": model_roots,
        "seeds": seeds,
        "few_shot_num": few_shot_num,
        "memory_path": config["memory_path"],
        "simulation_duration": int(config["simulation_duration"]),
        "metrics_config": {
            "env_id": str(env_type),
            "native_env_defaults": bool(env_bundle.get("use_native_env_defaults")),
            "requested_env_id": str(env_bundle.get("requested_env_id")),
            "env_resolution_sources": {
                "env_id": env_bundle.get("env_source"),
                "native_env_defaults": env_bundle.get("native_source"),
            },
            "env_resolution_warnings": list(env_bundle.get("warnings", [])),
            "env_config_snapshot": _json_safe(copy.deepcopy(env_config_snapshot)),
            "ttc_threshold_sec": ttc_threshold_sec,
            "headway_threshold_m": headway_threshold_m,
            "rear_ttc_threshold_sec": rear_ttc_threshold_sec,
            "rear_headway_threshold_m": rear_headway_threshold_m,
            "low_speed_blocking_threshold_mps": low_speed_blocking_threshold_mps,
            "blocking_front_gap_safe_m": blocking_front_gap_safe_m,
            "blocking_front_ttc_safe_sec": blocking_front_ttc_safe_sec,
            "flapping_mode": "accel_decel",
            "decision_timeout_sec": round(max(1.0, default_decision_timeout_sec), 3),
            "slow_decision_threshold_sec": round(max(0.001, slow_decision_threshold_sec), 3),
            "model_overrides_enabled": bool(legacy_eval_overrides),
            "model_overrides_keys": sorted(list(legacy_eval_overrides.keys())),
            "model_policy_overrides_enabled": bool(shared_policy_overrides),
            "model_policy_overrides_keys": sorted(list(shared_policy_overrides.keys())),
            "adaptive_timeout_penalty_enabled": bool(adaptive_timeout_penalty_enabled),
            "adaptive_timeout_halving_factor": float(adaptive_timeout_halving_factor),
            "adaptive_timeout_min_sec": float(adaptive_timeout_min_sec),
            "adaptive_timeout_trigger_consecutive_slow": int(adaptive_timeout_trigger_consecutive_slow),
            "quiet_mode": bool(resolved_eval_quiet_mode),
            "progress_bar": bool(progress_enabled),
            "progress_bar_requested": bool(resolved_eval_progress_mode),
            "progress_reply_mode_requested": str(resolved_eval_progress_reply_mode),
            "progress_reply_mode_effective": str(effective_eval_progress_reply_mode),
            "policy_mode": "timeout_only",
            "deprecated_policy_cli_fields_ignored": deprecated_cli_policy_flags,
            "deprecated_policy_override_fields_ignored": deprecated_override_fields_declared,
            "deprecated_metric_aliases": {
                "timeout_penalty_final_native_timeout_sec": "timeout_penalty_final_decision_timeout_sec",
                "timeout_penalty_final_native_timeout_sec_mean": "timeout_penalty_final_decision_timeout_sec_mean",
            },
            "save_run_artifacts": bool(save_run_artifacts),
            "eval_run_id": eval_run_id if save_run_artifacts else None,
            "alignment_sample_rate": alignment_sample_rate,
            "alignment_max_samples": alignment_max_samples,
        },
        "per_model": {},
        "aggregates": [],
        "alignment_samples": [],
        "model_eval_outputs": {},
        "model_run_outputs": {},
        "model_runtime_policies": {},
    }

    aggregate_by_model: Dict[str, Dict] = {}
    model_run_outputs: Dict[str, Dict[str, str]] = {}
    model_metrics_configs: Dict[str, Dict] = {}
    deprecated_policy_fields_ignored_union = set(deprecated_cli_policy_flags)
    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        )
        if progress_enabled
        else nullcontext(None)
    )
    with progress_cm as progress:
        emit = progress.console.print if progress is not None else print
        model_task = (
            progress.add_task("Models", total=len(args.models))
            if progress is not None
            else None
        )

        for model_name in args.models:
            resolved_policy = resolve_model_policy(
                config=config,
                model_name=model_name,
                provider=provider,
                mode="eval",
                cli_overrides=cli_policy_overrides,
            )
            policy_meta = dict(resolved_policy.get("policy_meta", {}))
            deprecated_policy_fields_ignored_union.update(
                policy_meta.get("deprecated_policy_fields_ignored", []) or []
            )
            resolved_decision_timeout_sec = float(resolved_policy["decision_timeout_sec"])
            timeout_penalty_state = build_decision_timeout_penalty_state(
                config=config,
                provider=provider,
                mode="eval",
                baseline_decision_timeout_sec=resolved_decision_timeout_sec,
            )

            configure_runtime_env(
                config,
                chat_model_override=model_name,
                mode="eval",
                quiet_override=step_log_quiet_mode,
                progress_override=progress_enabled,
            )
            apply_model_policy_to_env(resolved_policy, provider=provider)

            report["model_runtime_policies"][model_name] = {
                "decision_timeout_sec": round(resolved_decision_timeout_sec, 3),
                "policy_meta": policy_meta,
                "matched_override": {
                    "model_policy": policy_meta.get("matched_model_policy_override_key"),
                    "legacy_eval_model": policy_meta.get("matched_eval_model_override_key"),
                },
                "deprecated_policy_fields_ignored": policy_meta.get("deprecated_policy_fields_ignored", []),
                "decision_timeout_penalty": decision_timeout_penalty_snapshot(timeout_penalty_state),
                # Deprecated alias key for one transition cycle.
                "native_timeout_penalty": decision_timeout_penalty_snapshot(timeout_penalty_state),
            }
            model_metrics_configs[model_name] = {
                **dict(report["metrics_config"]),
                "resolved_model_policy": dict(report["model_runtime_policies"][model_name]),
            }
            emit(f"\n[bold cyan]Evaluating model[/bold cyan]: {model_name}")
            source_parts = []
            if policy_meta.get("matched_model_policy_override_key"):
                source_parts.append(f"model_override={policy_meta['matched_model_policy_override_key']}")
            if policy_meta.get("matched_eval_model_override_key"):
                source_parts.append(f"legacy_eval_override={policy_meta['matched_eval_model_override_key']}")
            if policy_meta.get("cli_override_keys"):
                source_parts.append(f"cli={','.join(policy_meta['cli_override_keys'])}")
            source_label = " | ".join(source_parts) if source_parts else "base_defaults"
            emit(
                "[dim]  Policy (timeout-only): decision_timeout={timeout}s | source={source}[/dim]".format(
                    timeout=round(resolved_decision_timeout_sec, 3),
                    source=source_label,
                )
            )
            if policy_meta.get("deprecated_policy_fields_ignored"):
                emit(
                    "[yellow]  Deprecated policy fields ignored:[/yellow] "
                    f"{', '.join(policy_meta['deprecated_policy_fields_ignored'])}"
                )
            penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
            emit(
                "[dim]  Adaptive decision-timeout penalty: enabled={enabled}, baseline={baseline}s, floor={floor}s, "
                "factor={factor}, trigger_consecutive_slow={trigger}[/dim]".format(
                    enabled=bool(penalty_snapshot.get("enabled")),
                    baseline=round(float(penalty_snapshot.get("baseline_decision_timeout_sec") or 0.0), 3),
                    floor=round(float(penalty_snapshot.get("min_timeout_sec") or 0.0), 3),
                    factor=round(float(penalty_snapshot.get("halving_factor") or 0.0), 3),
                    trigger=int(penalty_snapshot.get("trigger_consecutive_slow") or 0),
                )
            )
            agent_memory = DrivingMemory(db_path=config["memory_path"])
            model_run_dir = None
            model_log_path = None
            if save_run_artifacts:
                model_run_dir = build_model_run_dir(experiment_root, model_name, eval_run_id)
                model_log_path = os.path.join(model_run_dir, "log.txt")
                with open(model_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        "=== Eval Run {run_id} | Model {model} | Created {created} ===\n".format(
                            run_id=eval_run_id,
                            model=model_name,
                            created=datetime.now().isoformat(timespec="seconds"),
                        )
                    )

            seed_task = (
                progress.add_task(f"{model_name} seeds", total=len(seeds))
                if progress is not None
                else None
            )
            step_task = (
                progress.add_task(f"{model_name} steps", total=int(config["simulation_duration"]))
                if progress is not None
                else None
            )
            episodes = []
            model_alignment_samples = []
            for idx, seed in enumerate(seeds, start=1):
                emit(f"[dim]  Seed {idx}/{len(seeds)}: {seed}[/dim]")
                if progress is not None and step_task is not None:
                    progress.update(
                        step_task,
                        description=f"{model_name} | seed {idx}/{len(seeds)}",
                        total=int(config["simulation_duration"]),
                        completed=0,
                    )

                def _on_step(step_completed: int, done: bool) -> None:
                    if progress is not None and step_task is not None:
                        progress.update(
                            step_task,
                            completed=min(int(step_completed), int(config["simulation_duration"])),
                        )

                def _on_decision(step_idx: int, action_id: int, response_text: str, _decision_meta: Dict) -> None:
                    if effective_eval_progress_reply_mode == "compact":
                        emit(_compact_reply_preview(step_idx, action_id, response_text))
                    elif effective_eval_progress_reply_mode == "full":
                        emit(_full_reply_preview(step_idx, action_id, response_text))

                episode_result = run_episode(
                    config=config,
                    env_config=env_config,
                    env_type=env_type,
                    agent_memory=agent_memory,
                    seed=seed,
                    few_shot_num=few_shot_num,
                    temp_dir=temp_dir,
                    ttc_threshold_sec=ttc_threshold_sec,
                    headway_threshold_m=headway_threshold_m,
                    rear_ttc_threshold_sec=rear_ttc_threshold_sec,
                    rear_headway_threshold_m=rear_headway_threshold_m,
                    low_speed_blocking_threshold_mps=low_speed_blocking_threshold_mps,
                    blocking_front_gap_safe_m=blocking_front_gap_safe_m,
                    blocking_front_ttc_safe_sec=blocking_front_ttc_safe_sec,
                    alignment_sample_rate=alignment_sample_rate,
                    alignment_max_samples=alignment_max_samples,
                    slow_decision_threshold_sec=slow_decision_threshold_sec,
                    timeout_penalty_state=timeout_penalty_state,
                    save_artifacts=save_run_artifacts,
                    run_dir=model_run_dir,
                    run_id=eval_run_id if save_run_artifacts else None,
                    model_name=model_name,
                    quiet_mode=step_log_quiet_mode,
                    on_step=_on_step if progress is not None else None,
                    on_decision=_on_decision if progress is not None else None,
                )
                episode_alignment_samples = episode_result.pop("alignment_samples", [])
                for sample in episode_alignment_samples:
                    sample["model"] = model_name
                    model_alignment_samples.append(sample)
                episodes.append(episode_result)
                if save_run_artifacts and model_log_path:
                    _append_eval_run_log(model_log_path, model_name, episode_result)
                if progress is not None and seed_task is not None:
                    progress.update(seed_task, advance=1)
                status = "CRASH" if episode_result["crashed"] else ("ERROR" if episode_result["error"] else ("TIMEOUT" if episode_result.get("timeout_triggered") else "OK"))
                emit(
                    f"    -> {status} | steps={episode_result['steps']}/{episode_result['max_steps']} "
                    f"| t={episode_result['episode_runtime_sec']}s | timeout_steps={episode_result.get('decision_timeout_count', 0)}"
                )
                if episode_result["error"]:
                    emit(f"    -> [red]{episode_result['error']}[/red]")

            report["model_runtime_policies"][model_name]["decision_timeout_penalty"] = (
                decision_timeout_penalty_snapshot(timeout_penalty_state)
            )
            report["model_runtime_policies"][model_name]["native_timeout_penalty"] = (
                decision_timeout_penalty_snapshot(timeout_penalty_state)
            )
            model_metrics_configs[model_name]["resolved_model_policy"] = dict(
                report["model_runtime_policies"][model_name]
            )
            report["per_model"][model_name] = episodes
            agg = aggregate_results(model_name, episodes)
            report["aggregates"].append(agg)
            aggregate_by_model[model_name] = agg
            report["alignment_samples"].extend(model_alignment_samples[:alignment_max_samples] if alignment_max_samples > 0 else [])
            if save_run_artifacts and model_run_dir:
                run_metrics_report = _build_eval_run_metrics_report(
                    model_name=model_name,
                    experiment_id=experiment_id,
                    experiment_root=experiment_root,
                    run_id=eval_run_id,
                    run_dir=model_run_dir,
                    config_path=args.config,
                    openai_api_type=str(config.get("OPENAI_API_TYPE", "")),
                    few_shot_num=int(few_shot_num),
                    memory_path=str(config.get("memory_path", "")),
                    simulation_duration=int(config["simulation_duration"]),
                    metrics_config=model_metrics_configs.get(model_name, report["metrics_config"]),
                    episodes=episodes,
                    aggregate=agg,
                )
                run_metrics_path = timestamped_results_path("run_metrics", ext=".json", results_dir=model_run_dir)
                write_json_atomic(run_metrics_path, run_metrics_report)
                model_run_outputs[model_name] = {
                    "run_id": eval_run_id,
                    "run_dir": model_run_dir,
                    "log_path": model_log_path,
                    "run_metrics": run_metrics_path,
                }

            if progress is not None and model_task is not None:
                progress.update(model_task, advance=1)
            if progress is not None and seed_task is not None:
                progress.remove_task(seed_task)
            if progress is not None and step_task is not None:
                progress.remove_task(step_task)

    report["metrics_config"]["deprecated_policy_fields_ignored"] = sorted(
        deprecated_policy_fields_ignored_union
    )
    report["model_run_outputs"] = model_run_outputs

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
                metrics_config=model_metrics_configs.get(model_name, report["metrics_config"]),
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
            model_run_outputs=model_run_outputs if save_run_artifacts else None,
        )

    print("\n[bold green]Aggregate Summary[/bold green]")
    for row in report["aggregates"]:
        print(
            f"- {row['model']}: crashes={row['crashes']}/{row['episodes']} "
            f"(rate={row['crash_rate']}), no_collision_rate={row['no_collision_rate']}, "
                f"avg_steps={row['avg_steps']}, strict_format_rate={row['response_strict_format_rate']}, "
                f"ttc_danger_rate={row['ttc_danger_rate_mean']}, headway_violation_rate={row['headway_violation_rate_mean']}, "
                f"rear_ttc_danger_rate={row.get('rear_ttc_danger_rate_mean')}, "
                f"low_speed_blocking_rate={row.get('low_speed_blocking_rate_mean')}, "
                f"decision_timeout_rate={row.get('decision_timeout_rate_mean')}, "
                f"native_timeout_rate={row.get('ollama_native_timeout_rate_mean')}, "
                f"fallback_action_rate={row.get('fallback_action_rate_mean')}, "
                f"avg_episode_runtime_sec={row['avg_episode_runtime_sec']}"
            )
    print(f"\nSaved report: [bold]{out_path}[/bold]")
    if user_out_path and user_out_path != out_path:
        print(f"Saved user-requested output copy: [bold]{user_out_path}[/bold]")


if __name__ == "__main__":
    main()
