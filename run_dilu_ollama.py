import argparse
import atexit
import copy
from contextlib import nullcontext
import random
import numpy as np
import yaml
import os
import json
import re
import sys
import time
from rich import print
from rich.table import Table
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.markup import escape
from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from dilu.scenario.envScenario import EnvScenario
from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.driver_agent.reflectionAgent import ReflectionAgent
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

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_progress_mode(config, cli_override: Optional[bool], mode: str = "runtime") -> bool:
    if cli_override is not None:
        return bool(cli_override)
    global_default = _to_bool(config.get("progress_bar", True), default=True)
    mode_key = "eval_progress_bar" if str(mode).strip().lower() == "eval" else "runtime_progress_bar"
    mode_value = config.get(mode_key)
    if mode_value is None:
        return global_default
    return _to_bool(mode_value, default=global_default)


def _normalize_progress_reply_mode(value: Optional[str]) -> str:
    mode = str(value or "off").strip().lower()
    if mode in {"off", "compact", "full"}:
        return mode
    return "off"


def _resolve_progress_reply_mode(config, cli_override: Optional[str], mode: str = "runtime") -> str:
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


STRICT_RESPONSE_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*([0-4])\s*$", re.IGNORECASE)


def _safe_int_action(action) -> int:
    if isinstance(action, str):
        action = action.strip()
    action = int(action)
    if action < 0 or action > 4:
        raise ValueError(f"Invalid action id: {action}")
    return action


def _response_format_metrics(response_content: str) -> dict:
    response_content = (response_content or "").strip()
    has_delimiter = "####" in response_content
    strict_match = STRICT_RESPONSE_PATTERN.search(response_content)
    direct_action_parseable = False

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
    }


def extract_step_traffic_metrics(env, ttc_threshold_sec: float, headway_threshold_m: float) -> dict:
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


def aggregate_run_results(episodes: list) -> dict:
    total = len(episodes)
    crashes = sum(1 for e in episodes if e.get("crashed"))
    errors = sum(1 for e in episodes if e.get("error"))
    no_collision = sum(1 for e in episodes if e.get("success_no_collision"))
    truncations = sum(1 for e in episodes if e.get("truncated"))
    terminations = sum(1 for e in episodes if e.get("terminated"))
    total_steps = sum(int(e.get("steps", 0)) for e in episodes)
    total_runtime = sum(float(e.get("episode_runtime_sec", 0.0)) for e in episodes)
    total_decisions = sum(int(e.get("decisions_made", 0)) for e in episodes)
    total_delimiters = sum(int(e.get("responses_with_delimiter", 0)) for e in episodes)
    total_strict = sum(int(e.get("responses_strict_format", 0)) for e in episodes)
    total_direct = sum(int(e.get("responses_direct_parseable", 0)) for e in episodes)
    total_format_failures = sum(int(e.get("format_failure_count", 0)) for e in episodes)
    total_reward_sum = sum(float(e.get("episode_reward_sum", 0.0)) for e in episodes)
    total_speed = sum(float(e.get("avg_ego_speed_mps", 0.0)) for e in episodes)
    total_ttc_danger_rate = sum(float(e.get("ttc_danger_rate", 0.0)) for e in episodes)
    total_headway_rate = sum(float(e.get("headway_violation_rate", 0.0)) for e in episodes)
    total_lane_change_rate = sum(float(e.get("lane_change_rate", 0.0)) for e in episodes)
    total_flap_rate = sum(float(e.get("flap_accel_decel_rate", 0.0)) for e in episodes)
    total_decision_latency_ms = sum(float(e.get("decision_latency_ms_avg", 0.0)) for e in episodes)
    total_timeout_penalty_events = sum(int(e.get("timeout_penalty_events", 0)) for e in episodes)
    timeout_penalty_stage_max_values = [int(e.get("timeout_penalty_stage_max", 0)) for e in episodes]
    timeout_penalty_final_values = [
        float(e.get("timeout_penalty_final_decision_timeout_sec"))
        for e in episodes
        if e.get("timeout_penalty_final_decision_timeout_sec") is not None
    ]

    return {
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
        "timeout_penalty_events_total": int(total_timeout_penalty_events),
        "timeout_penalty_events_rate_mean": round(total_timeout_penalty_events / max(total_decisions, 1), 4),
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


def _update_experiment_manifest(
    experiment_root: str,
    experiment_id: str,
    model_name: str,
    model_slug: str,
    run_id: str,
    run_dir: str,
    metrics_report_path: str,
    config_path: str,
    memory_path: str,
    few_shot_num: int,
    simulation_duration: int,
) -> None:
    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})
    model_key = model_name

    manifest.setdefault("experiment_id", experiment_id)
    manifest.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["config_path"] = config_path
    manifest["memory_path"] = memory_path
    manifest["few_shot_num"] = int(few_shot_num)
    manifest["simulation_duration"] = int(simulation_duration)

    model_meta = manifest.setdefault("models", {})
    model_record = model_meta.setdefault(model_key, {})
    model_record["slug"] = model_slug
    model_record["root"] = os.path.join(experiment_root, "models", model_slug)
    model_record["latest_run_id"] = run_id
    model_record["latest_run_dir"] = run_dir
    model_record["latest_run_metrics"] = metrics_report_path

    run_records = manifest.setdefault("runs", {})
    run_records[f"{model_slug}:{run_id}"] = {
        "model": model_name,
        "model_slug": model_slug,
        "run_id": run_id,
        "run_dir": run_dir,
        "run_metrics": metrics_report_path,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    write_json_atomic(manifest_path, manifest)


class TelemetryHUDWrapper(gym.Wrapper):
    """Overlay live ego telemetry on rgb_array frames for video/render output."""

    def __init__(self, env):
        super().__init__(env)
        self.step_idx = 0
        self._warned_pil = False
        self._warned_pygame = False
        self._font = None
        self._pygame_font = None

    def __getstate__(self):
        # Prevent deepcopy failures when upstream code snapshots scenario/env objects.
        state = self.__dict__.copy()
        state["_font"] = None
        return state

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_idx = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_idx += 1
        return obs, reward, terminated, truncated, info

    def _ego_telemetry_from_env(self):
        uenv = self.env.unwrapped
        ego = getattr(uenv, "vehicle", None)
        road = getattr(uenv, "road", None)
        if ego is None or road is None:
            return None

        lane_id = ego.lane_index[2] if isinstance(ego.lane_index, tuple) and len(ego.lane_index) >= 3 else "N/A"

        lane_pos = None
        try:
            lane = road.network.get_lane(ego.lane_index)
            lane_pos = float(np.linalg.norm(ego.position - lane.start))
        except Exception:
            lane_pos = None

        front_vehicle = None
        try:
            front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
        except Exception:
            pass

        front_gap = None
        front_rel_speed = None
        if front_vehicle is not None:
            front_gap = float(np.linalg.norm(ego.position - front_vehicle.position))
            front_rel_speed = float(ego.speed - front_vehicle.speed)

        min_gap = None
        try:
            perception_distance = getattr(uenv, "PERCEPTION_DISTANCE", 80)
            nearby = road.close_vehicles_to(
                ego, perception_distance, count=9, see_behind=True, sort="sorted"
            )
            if nearby:
                min_gap = min(float(np.linalg.norm(ego.position - v.position)) for v in nearby)
        except Exception:
            pass

        danger = "SAFE"
        if front_gap is not None:
            if front_gap < 10 or (front_gap < 25 and (front_rel_speed or 0) > 0):
                danger = "DANGER"
            elif front_gap < 35:
                danger = "CAUTION"
        if min_gap is not None and min_gap < 8:
            danger = "DANGER"

        return {
            "step": self.step_idx,
            "speed": float(getattr(ego, "speed", 0.0)),
            "accel": float(getattr(ego, "action", {}).get("acceleration", 0.0)),
            "lane": lane_id,
            "lane_pos": lane_pos,
            "front_gap": front_gap,
            "front_rel_speed": front_rel_speed,
            "min_gap": min_gap,
            "danger": danger,
        }

    def _draw_hud(self, frame: np.ndarray, t: dict) -> np.ndarray:
        if not PIL_AVAILABLE:
            if not self._warned_pil:
                print("[yellow]Pillow not installed. Telemetry HUD overlay disabled.[/yellow]")
                self._warned_pil = True
            return frame

        h, w = frame.shape[:2]
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, "RGBA")
        if self._font is None:
            try:
                self._font = ImageFont.load_default()
            except Exception:
                self._font = None

        speed_kmh = t["speed"] * 3.6
        # Compact in-frame bottom overlay panel (same frame size, no window resize).
        hud_h = 50
        panel_y0 = h - hud_h
        panel_x0, panel_x1 = 6, w - 6
        panel_y1 = h - 4
        draw.rounded_rectangle((panel_x0, panel_y0 + 2, panel_x1, panel_y1), radius=8, fill=(20, 20, 20, 190))

        danger_color = {"SAFE": (80, 220, 120), "CAUTION": (255, 210, 80), "DANGER": (255, 90, 90)}.get(
            t["danger"], (255, 255, 255)
        )

        cells = [
            ("Step", str(t["step"])),
            ("Lane", str(t["lane"])),
            ("Danger", str(t["danger"])),
            ("Speed", f"{speed_kmh:.1f} km/h"),
            ("Accel", f"{t['accel']:.1f} m/s²"),
            ("LanePos", "N/A" if t["lane_pos"] is None else f"{t['lane_pos']:.1f} m"),
            ("FrontGap", "N/A" if t["front_gap"] is None else f"{t['front_gap']:.1f} m"),
            ("RelV", "N/A" if t["front_rel_speed"] is None else f"{t['front_rel_speed']:.1f} m/s"),
            ("Nearest", "N/A" if t["min_gap"] is None else f"{t['min_gap']:.1f} m"),
        ]

        cols = 3
        rows = 3
        inner_pad = 6
        grid_x0 = panel_x0 + 6
        grid_y0 = panel_y0 + 6
        grid_x1 = panel_x1 - 6
        grid_y1 = panel_y1 - 4
        cell_w = (grid_x1 - grid_x0) // cols
        cell_h = (grid_y1 - grid_y0) // rows

        for idx, (label, value) in enumerate(cells):
            r = idx // cols
            c = idx % cols
            x0 = grid_x0 + c * cell_w
            y0 = grid_y0 + r * cell_h
            x1 = grid_x0 + (c + 1) * cell_w - 2
            y1 = grid_y0 + (r + 1) * cell_h - 2
            draw.rounded_rectangle((x0, y0, x1, y1), radius=4, fill=(35, 35, 35, 200), outline=(90, 90, 90, 220))

            text = f"{label}: {value}"
            text_color = danger_color if label == "Danger" else (255, 255, 255)
            text_y = y0 + max(1, (cell_h - 10) // 2)
            draw.text((x0 + inner_pad, text_y), text, fill=text_color, font=self._font)

        return np.array(img)

    def _draw_pygame_hud(self, frame_with_hud: np.ndarray) -> None:
        if not PYGAME_AVAILABLE:
            if not self._warned_pygame:
                print("[yellow]pygame not available for live HUD overlay.[/yellow]")
                self._warned_pygame = True
            return

        try:
            viewer = getattr(self.env.unwrapped, "viewer", None)
            screen = getattr(viewer, "screen", None) if viewer is not None else None
            if viewer is None:
                return
            h, w = frame_with_hud.shape[:2]
            if screen is None or screen.get_width() != w or screen.get_height() != h:
                # Keep the live window exactly the same size as the rendered frame.
                viewer.screen = pygame.display.set_mode((w, h))
                screen = viewer.screen

            # pygame.surfarray expects (width, height, channels)
            surface = pygame.surfarray.make_surface(np.transpose(frame_with_hud[:, :, :3], (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
        except Exception:
            # Do not break the simulation if live HUD drawing fails on a platform/viewer variant.
            return

    def render(self):
        frame = self.env.render()
        if isinstance(frame, np.ndarray):
            telemetry = self._ego_telemetry_from_env()
            if telemetry is not None:
                # Render telemetry as a compact in-frame overlay and mirror it to the live pygame window.
                frame_with_hud = self._draw_hud(frame, telemetry)
                self._draw_pygame_hud(frame_with_hud)
                return frame_with_hud
        return frame


def get_ego_telemetry(sce: EnvScenario):
    """Collect live ego-vehicle telemetry and a simple danger indicator."""
    ego = sce.ego
    road = sce.road

    lane_id = ego.lane_index[2] if isinstance(ego.lane_index, tuple) and len(ego.lane_index) >= 3 else "N/A"
    lane_pos = None
    try:
        lane_pos = sce.getLanePosition(ego)
    except Exception:
        lane_pos = None

    front_vehicle = None
    rear_vehicle = None
    try:
        front_vehicle, rear_vehicle = road.neighbour_vehicles(ego, ego.lane_index)
    except Exception:
        pass

    front_gap = None
    front_rel_speed = None
    if front_vehicle is not None:
        front_gap = float(np.linalg.norm(ego.position - front_vehicle.position))
        front_rel_speed = float(ego.speed - front_vehicle.speed)

    min_surrounding_gap = None
    try:
        nearby = sce.getSurrendVehicles(10)
        if nearby:
            min_surrounding_gap = min(float(np.linalg.norm(ego.position - v.position)) for v in nearby)
    except Exception:
        pass

    danger = "SAFE"
    if front_gap is not None:
        if front_gap < 10 or (front_gap < 25 and (front_rel_speed or 0) > 0):
            danger = "DANGER"
        elif front_gap < 35:
            danger = "CAUTION"
    if min_surrounding_gap is not None and min_surrounding_gap < 8:
        danger = "DANGER"

    return {
        "speed": float(ego.speed),
        "accel": float(ego.action.get("acceleration", 0.0)),
        "lane_id": lane_id,
        "lane_pos": lane_pos,
        "front_gap": front_gap,
        "front_rel_speed": front_rel_speed,
        "min_surrounding_gap": min_surrounding_gap,
        "danger": danger,
    }


def print_ego_telemetry(step_idx: int, telemetry: dict):
    color = {"SAFE": "green", "CAUTION": "yellow", "DANGER": "red"}.get(telemetry["danger"], "white")
    table = Table(title=f"Ego Telemetry | Step {step_idx}", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Speed (m/s)", f"{telemetry['speed']:.2f}")
    table.add_row("Acceleration (m/s^2)", f"{telemetry['accel']:.2f}")
    table.add_row("Lane", str(telemetry["lane_id"]))
    table.add_row("Lane Position (m)", "N/A" if telemetry["lane_pos"] is None else f"{telemetry['lane_pos']:.2f}")
    table.add_row("Front Gap (m)", "N/A" if telemetry["front_gap"] is None else f"{telemetry['front_gap']:.2f}")
    table.add_row(
        "Front Relative Speed (m/s)",
        "N/A" if telemetry["front_rel_speed"] is None else f"{telemetry['front_rel_speed']:.2f}"
    )
    table.add_row(
        "Nearest Vehicle Gap (m)",
        "N/A" if telemetry["min_surrounding_gap"] is None else f"{telemetry['min_surrounding_gap']:.2f}"
    )
    table.add_row("Danger Indicator", f"[{color}]{telemetry['danger']}[/{color}]")
    print(table)


def setup_env(
    config,
    quiet_override: Optional[bool] = None,
    progress_override: Optional[bool] = None,
    env_id_override: Optional[str] = None,
    native_env_defaults_override: Optional[bool] = None,
):
    selected_model = configure_runtime_env(
        config,
        mode="runtime",
        quiet_override=quiet_override,
        progress_override=progress_override,
    )
    provider = str(config.get("OPENAI_API_TYPE", "")).strip().lower()
    resolved_policy = resolve_model_policy(
        config=config,
        model_name=selected_model or "",
        provider=provider,
        mode="runtime",
        cli_overrides=None,
    )
    apply_model_policy_to_env(resolved_policy, provider=provider)

    policy_meta = dict(resolved_policy.get("policy_meta", {}))
    source_parts = []
    if policy_meta.get("matched_model_policy_override_key"):
        source_parts.append(f"model_override={policy_meta['matched_model_policy_override_key']}")
    if policy_meta.get("matched_eval_model_override_key"):
        source_parts.append(f"legacy_eval_override={policy_meta['matched_eval_model_override_key']}")
    source_label = " | ".join(source_parts) if source_parts else "base_defaults"
    if provider == 'ollama':
        print(f"[bold yellow]Configured for Local Ollama: {selected_model}[/bold yellow]")
    elif provider == "gemini":
        print(f"[bold yellow]Configured for Gemini API: {selected_model}[/bold yellow]")
    print(
        "[dim]Runtime policy (timeout-only): decision_timeout={timeout}s | source={source}[/dim]".format(
            timeout=round(float(resolved_policy["decision_timeout_sec"]), 3),
            source=source_label,
        )
    )
    if policy_meta.get("deprecated_policy_fields_ignored"):
        print(
            "[yellow]Deprecated policy fields ignored (timeout-only policy):[/yellow] "
            f"{', '.join(policy_meta['deprecated_policy_fields_ignored'])}"
        )

    env_bundle = resolve_simulation_env_bundle(
        config,
        show_trajectories=True,
        render_agent=True,
        env_id_override=env_id_override,
        native_env_defaults_override=native_env_defaults_override,
    )
    for warning_msg in env_bundle.get("warnings", []):
        print(f"[yellow]{warning_msg}[/yellow]")
    return env_bundle, selected_model, resolved_policy


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Run DiLu autonomous-driving simulation for a single configured model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--quiet", action="store_true", help="Suppress high-frequency step/decision logs.")
    parser.add_argument("--no-quiet", action="store_true", help="Force step/decision logs on even if config quiet mode is enabled.")
    parser.add_argument("--progress", action="store_true", help="Show CLI progress bars.")
    parser.add_argument("--no-progress", action="store_true", help="Disable CLI progress bars.")
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
    parser.add_argument(
        "--progress-replies",
        choices=["off", "compact", "full"],
        help="Show LLM replies while progress bars are active.",
    )
    args = parser.parse_args()
    if args.quiet and args.no_quiet:
        raise ValueError("Use only one of --quiet or --no-quiet.")
    if args.progress and args.no_progress:
        raise ValueError("Use only one of --progress or --no-progress.")

    quiet_override = True if args.quiet else (False if args.no_quiet else None)
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    progress_override = True if args.progress else (False if args.no_progress else None)
    resolved_runtime_progress_mode = _resolve_progress_mode(config, progress_override, mode="runtime")
    runtime_progress_enabled = bool(resolved_runtime_progress_mode and _is_interactive_output())
    env_bundle, selected_model, resolved_model_policy = setup_env(
        config,
        quiet_override=quiet_override,
        progress_override=runtime_progress_enabled,
        env_id_override=args.env_id,
        native_env_defaults_override=args.native_env_defaults,
    )
    env_config = env_bundle["env_config_map"]
    envType = env_bundle["env_id"]
    resolved_env_snapshot = env_bundle["env_config_snapshot"]
    runtime_quiet_mode = _to_bool(os.getenv("DILU_QUIET_MODE"), default=False)
    resolved_runtime_progress_reply_mode = _resolve_progress_reply_mode(
        config,
        args.progress_replies,
        mode="runtime",
    )
    effective_runtime_progress_reply_mode = (
        resolved_runtime_progress_reply_mode
        if (runtime_progress_enabled and (not runtime_quiet_mode))
        else "off"
    )
    step_log_quiet_mode = bool(runtime_quiet_mode or runtime_progress_enabled)

    REFLECTION = config["reflection_module"]
    reflection_interactive = _to_bool(config.get("reflection_interactive", True), default=True)
    reflection_auto_add = _to_bool(config.get("reflection_auto_add", False), default=False)
    reflection_add_every_n = int(config.get("reflection_add_every_n", 5))
    if reflection_add_every_n <= 0:
        reflection_add_every_n = 1
    memory_path = config["memory_path"]
    few_shot_num = config["few_shot_num"]

    results_root_cfg = str(config.get("results_root", "") or "").strip()
    experiment_id_cfg = str(config.get("experiment_id", "") or "").strip()
    run_id_cfg = str(config.get("run_id", "") or "").strip()
    result_folder_override = str(config.get("result_folder_override", "") or "").strip()
    legacy_result_folder = str(config.get("result_folder", "") or "").strip()

    model_name_for_paths = (
        selected_model
        or config.get("OPENAI_CHAT_MODEL")
        or config.get("AZURE_CHAT_DEPLOY_NAME")
        or "unknown_model"
    )
    model_name_for_paths = str(model_name_for_paths)
    model_slug = slugify_model_name(model_name_for_paths)

    structured_mode = False
    experiment_root = None
    experiment_id = None
    model_root = None
    run_id = None

    if result_folder_override:
        result_folder = ensure_dir(result_folder_override)
    elif results_root_cfg:
        structured_mode = True
        experiment_root = build_experiment_root(results_root_cfg, experiment_id_cfg or None)
        experiment_id = os.path.basename(experiment_root)
        ensure_experiment_layout(experiment_root, [model_name_for_paths])
        model_root = build_model_root(experiment_root, model_name_for_paths)
        run_dir = build_model_run_dir(experiment_root, model_name_for_paths, run_id_cfg or None)
        run_id = os.path.basename(run_dir)
        result_folder = run_dir
    elif legacy_result_folder:
        # Backward-compatible fallback for older config files without structured-results keys.
        result_folder = ensure_dir(legacy_result_folder)
    else:
        structured_mode = True
        experiment_root = build_experiment_root(os.path.join("results", "experiments"), experiment_id_cfg or None)
        experiment_id = os.path.basename(experiment_root)
        ensure_experiment_layout(experiment_root, [model_name_for_paths])
        model_root = build_model_root(experiment_root, model_name_for_paths)
        run_dir = build_model_run_dir(experiment_root, model_name_for_paths, run_id_cfg or None)
        run_id = os.path.basename(run_dir)
        result_folder = run_dir

    ensure_dir(result_folder)
    ttc_threshold_sec = float(config.get("metrics_ttc_threshold_sec", 2.0))
    headway_threshold_m = float(config.get("metrics_headway_threshold_m", 15.0))
    runtime_slow_decision_threshold_sec = float(
        config.get(
            "runtime_slow_decision_threshold_sec",
            config.get("eval_slow_decision_threshold_sec", 5.0),
        )
    )
    runtime_slow_decision_threshold_sec = max(0.001, runtime_slow_decision_threshold_sec)
    provider = str(config.get("OPENAI_API_TYPE", "")).strip().lower()
    timeout_penalty_state = build_decision_timeout_penalty_state(
        config=config,
        provider=provider,
        mode="runtime",
        baseline_decision_timeout_sec=float(resolved_model_policy["decision_timeout_sec"]),
    )
    initial_penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
    print(
        "[dim]Adaptive decision-timeout penalty: enabled={enabled}, baseline={baseline}s, floor={floor}s, "
        "factor={factor}, trigger_consecutive_slow={trigger}, slow_threshold={threshold}s[/dim]".format(
            enabled=bool(initial_penalty_snapshot.get("enabled")),
            baseline=round(float(initial_penalty_snapshot.get("baseline_decision_timeout_sec") or 0.0), 3),
            floor=round(float(initial_penalty_snapshot.get("min_timeout_sec") or 0.0), 3),
            factor=round(float(initial_penalty_snapshot.get("halving_factor") or 0.0), 3),
            trigger=int(initial_penalty_snapshot.get("trigger_consecutive_slow") or 0),
            threshold=round(runtime_slow_decision_threshold_sec, 3),
        )
    )

    metrics_report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report_schema_version": "2.0",
        "source": "run_dilu_ollama",
        "config_path": config_path,
        "openai_api_type": config.get("OPENAI_API_TYPE"),
        "chat_model": selected_model,
        "memory_path": memory_path,
        "few_shot_num": int(few_shot_num),
        "episodes_num": int(config["episodes_num"]),
        "simulation_duration": int(config["simulation_duration"]),
        "result_folder": result_folder,
        "structured_results_mode": structured_mode,
        "experiment_id": experiment_id,
        "experiment_root": experiment_root,
        "model_root": model_root,
        "model_slug": model_slug,
        "run_id": run_id,
        "run_dir": result_folder,
        "model_runtime_policy": copy.deepcopy(resolved_model_policy),
        "metrics_config": {
            "env_id": str(envType),
            "native_env_defaults": bool(env_bundle.get("use_native_env_defaults")),
            "requested_env_id": str(env_bundle.get("requested_env_id")),
            "env_resolution_sources": {
                "env_id": env_bundle.get("env_source"),
                "native_env_defaults": env_bundle.get("native_source"),
            },
            "env_resolution_warnings": list(env_bundle.get("warnings", [])),
            "env_config_snapshot": _json_safe(copy.deepcopy(resolved_env_snapshot)),
            "ttc_threshold_sec": ttc_threshold_sec,
            "headway_threshold_m": headway_threshold_m,
            "flapping_mode": "accel_decel",
            "runtime_slow_decision_threshold_sec": round(runtime_slow_decision_threshold_sec, 3),
            "decision_timeout_sec": round(float(resolved_model_policy["decision_timeout_sec"]), 3),
            "adaptive_timeout_penalty_enabled": bool(config.get("adaptive_timeout_penalty_enabled", True)),
            "adaptive_timeout_halving_factor": float(config.get("adaptive_timeout_halving_factor", 0.5)),
            "adaptive_timeout_min_sec": float(config.get("adaptive_timeout_min_sec", 4.0)),
            "adaptive_timeout_trigger_consecutive_slow": int(config.get("adaptive_timeout_trigger_consecutive_slow", 2)),
            "quiet_mode": bool(runtime_quiet_mode),
            "progress_bar": bool(runtime_progress_enabled),
            "progress_bar_requested": bool(resolved_runtime_progress_mode),
            "progress_reply_mode_requested": str(resolved_runtime_progress_reply_mode),
            "progress_reply_mode_effective": str(effective_runtime_progress_reply_mode),
            "policy_mode": "timeout_only",
            "deprecated_policy_fields_ignored": resolved_model_policy.get("policy_meta", {}).get("deprecated_policy_fields_ignored", []),
            "deprecated_metric_aliases": {
                "timeout_penalty_final_native_timeout_sec": "timeout_penalty_final_decision_timeout_sec",
                "timeout_penalty_final_native_timeout_sec_mean": "timeout_penalty_final_decision_timeout_sec_mean",
            },
            "decision_timeout_penalty": decision_timeout_penalty_snapshot(timeout_penalty_state),
            # Deprecated alias key for one transition cycle.
            "native_timeout_penalty": decision_timeout_penalty_snapshot(timeout_penalty_state),
        },
        "episodes": [],
        "aggregate": None,
    }
    log_path = os.path.join(result_folder, "log.txt")
    with open(log_path, 'w') as f:
        f.write("memory_path {} | result_folder {} | few_shot_num: {} | lanes_count: {} \n".format(
            memory_path, result_folder, few_shot_num, env_config[envType]['lanes_count']))
        f.write(
            "env_id {} | native_env_defaults {} | requested_env_id {} | env_source {} | native_source {} \n".format(
                envType,
                bool(env_bundle.get("use_native_env_defaults")),
                env_bundle.get("requested_env_id"),
                env_bundle.get("env_source"),
                env_bundle.get("native_source"),
            )
        )
        if env_bundle.get("warnings"):
            f.write("env_resolution_warnings {} \n".format("; ".join(env_bundle["warnings"])))
        f.write("reflection_module {} | reflection_interactive {} | reflection_auto_add {} | reflection_add_every_n {} \n".format(
            REFLECTION, reflection_interactive, reflection_auto_add, reflection_add_every_n
        ))
        f.write("structured_results_mode {} | experiment_id {} | model_slug {} | run_id {} \n".format(
            structured_mode, experiment_id, model_slug, run_id
        ))
        f.write(
            "runtime_policy timeout_only decision_timeout={} \n".format(
                round(float(resolved_model_policy["decision_timeout_sec"]), 3),
            )
        )
        penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
        f.write(
            "adaptive_decision_timeout_penalty enabled={} | baseline={} | floor={} | factor={} | trigger_consecutive_slow={} | slow_threshold={} \n".format(
                bool(penalty_snapshot.get("enabled")),
                round(float(penalty_snapshot.get("baseline_decision_timeout_sec") or 0.0), 3),
                round(float(penalty_snapshot.get("min_timeout_sec") or 0.0), 3),
                round(float(penalty_snapshot.get("halving_factor") or 0.0), 3),
                int(penalty_snapshot.get("trigger_consecutive_slow") or 0),
                round(runtime_slow_decision_threshold_sec, 3),
            )
        )

    agent_memory = DrivingMemory(db_path=memory_path)
    if REFLECTION:
        updated_memory = DrivingMemory(db_path=memory_path + "_updated")
        updated_memory.combineMemory(agent_memory)

    progress = None
    progress_stop_registered = False
    episode_task = None
    step_task = None
    if runtime_progress_enabled:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        )
        progress.start()
        atexit.register(progress.stop)
        progress_stop_registered = True
        episode_task = progress.add_task("Episodes", total=int(config["episodes_num"]))
        step_task = progress.add_task("Steps", total=int(config["simulation_duration"]))
    emit = progress.console.print if progress is not None else print

    episode = 0
    while episode < config["episodes_num"]:
        if progress is not None and step_task is not None:
            progress.update(
                step_task,
                description=f"Episode {episode + 1}/{int(config['episodes_num'])} steps",
                total=int(config["simulation_duration"]),
                completed=0,
            )
        # setup highway-env
        env = gym.make(envType, render_mode="rgb_array")
        env.unwrapped.configure(env_config[envType])
        env = TelemetryHUDWrapper(env)
        result_prefix = f"highway_{episode}"
        env = RecordVideo(env, result_folder, name_prefix=result_prefix)
        env.unwrapped.set_record_video_wrapper(env)
        seed = random.choice(DEFAULT_DILU_SEEDS)
        obs, info = env.reset(seed=seed)
        env.render()

        # scenario and driver agent setting
        database_path = os.path.join(result_folder, f"{result_prefix}.db")
        sce = EnvScenario(env, envType, seed, database_path)
        DA = DriverAgent(sce, verbose=not step_log_quiet_mode)
        initial_penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
        if initial_penalty_snapshot.get("enabled") and initial_penalty_snapshot.get("effective_decision_timeout_sec") is not None:
            try:
                DA.set_decision_timeout_sec(float(initial_penalty_snapshot["effective_decision_timeout_sec"]))
            except Exception:
                pass
        if REFLECTION:
            RA = ReflectionAgent(verbose=not step_log_quiet_mode)

        response = "Not available"
        action = "Not available"
        docs = []
        collision_frame = -1
        crashed = False
        terminated = False
        truncated = False
        episode_error = None
        episode_started = time.time()
        already_decision_steps = 0

        decisions_made = 0
        penalty_start_events = int(timeout_penalty_state.get("penalty_events", 0))
        timeout_penalty_stage_max = int(timeout_penalty_state.get("stage", 0))
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

        try:
            for i in range(0, config["simulation_duration"]):
                obs = np.array(obs, dtype=float)

                if not step_log_quiet_mode:
                    print("[cyan]Retreive similar memories...[/cyan]")
                fewshot_results = agent_memory.retriveMemory(
                    sce, i, few_shot_num) if few_shot_num > 0 else []
                fewshot_messages = []
                fewshot_answers = []
                fewshot_actions = []
                for fewshot_result in fewshot_results:
                    fewshot_messages.append(
                        fewshot_result["human_question"])
                    fewshot_answers.append(fewshot_result["LLM_response"])
                    fewshot_actions.append(fewshot_result["action"])
                if few_shot_num == 0:
                    if not step_log_quiet_mode:
                        print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
                else:
                    if not step_log_quiet_mode:
                        print("[green4]Successfully find[/green4]", len(
                            fewshot_actions), "[green4]similar memories![/green4]")

                sce_descrip = sce.describe(i)
                avail_action = sce.availableActionsDescription()
                if not step_log_quiet_mode:
                    print_ego_telemetry(i, get_ego_telemetry(sce))
                    print('[cyan]Scenario description: [/cyan]\n', sce_descrip)
                # print('[cyan]Available actions: [/cyan]\n',avail_action)
                action, response, human_question, fewshot_answer = DA.few_shot_decision(
                    scenario_description=sce_descrip, available_actions=avail_action,
                    previous_decisions=action,
                    fewshot_messages=fewshot_messages,
                    driving_intensions="Drive safely and avoid collisons",
                    fewshot_answers=fewshot_answers,
                )
                decisions_made += 1
                decision_meta = getattr(DA, "last_decision_meta", {}) or {}
                decision_timed_out = bool(decision_meta.get("timed_out", False))
                decision_elapsed_sec = float(decision_meta.get("decision_elapsed_sec", 0.0) or 0.0)
                penalty_update = update_decision_timeout_penalty_state(
                    timeout_penalty_state,
                    timed_out=decision_timed_out,
                    decision_elapsed_sec=decision_elapsed_sec,
                    slow_threshold_sec=runtime_slow_decision_threshold_sec,
                )
                timeout_penalty_stage_max = max(
                    timeout_penalty_stage_max,
                    int(penalty_update.get("stage", 0)),
                )
                if penalty_update.get("escalated"):
                    effective_timeout = penalty_update.get("effective_decision_timeout_sec")
                    if effective_timeout is not None:
                        try:
                            DA.set_decision_timeout_sec(float(effective_timeout))
                        except Exception:
                            pass
                    if not step_log_quiet_mode:
                        print(
                            "[yellow]Adaptive timeout penalty escalated[/yellow] "
                            f"(reason={penalty_update.get('reason')}, stage={penalty_update.get('stage')}, "
                            f"decision_timeout={round(float(effective_timeout), 3) if effective_timeout is not None else 'n/a'}s)"
                        )
                fmt = _response_format_metrics(response)
                responses_with_delimiter += int(fmt["has_delimiter"])
                responses_strict_format += int(fmt["strict_format_match"])
                responses_direct_parseable += int(fmt["direct_action_parseable"])
                format_failure_count += int(not fmt["strict_format_match"])

                docs.append({
                    "sce_descrip": sce_descrip,
                    "human_question": human_question,
                    "response": response,
                    "action": action,
                    # Deep-copying the scenario is unnecessary here and can fail due to wrapper internals (e.g., PIL font objects).
                    "sce": None
                })

                #obs, reward, done, info, _ = env.step(action)
                action = _safe_int_action(action)
                if effective_runtime_progress_reply_mode == "compact":
                    emit(_compact_reply_preview(i + 1, action, response))
                elif effective_runtime_progress_reply_mode == "full":
                    emit(_full_reply_preview(i + 1, action, response))
                lane_change_count += int(action in (0, 2))
                if prev_action_id is not None and ((prev_action_id == 3 and action == 4) or (prev_action_id == 4 and action == 3)):
                    flap_accel_decel_count += 1
                prev_action_id = action

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                crashed = bool(info.get("crashed", False))

                already_decision_steps += 1
                episode_reward_sum += float(reward)
                if progress is not None and step_task is not None:
                    progress.update(
                        step_task,
                        completed=min(already_decision_steps, int(config["simulation_duration"])),
                    )

                step_metrics = extract_step_traffic_metrics(env, ttc_threshold_sec, headway_threshold_m)
                if step_metrics["ego_speed_mps"] is not None:
                    ego_speed_sum += float(step_metrics["ego_speed_mps"])
                    ego_speed_count += 1
                ttc_danger_steps += int(step_metrics["ttc_danger"])
                headway_violation_steps += int(step_metrics["headway_violation"])

                env.render()
                sce.promptsCommit(i, None, done, human_question,
                                  fewshot_answer, response)
                #env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()

                if not step_log_quiet_mode:
                    print("--------------------")

                if done:
                    emit(f"[red]Episode ended at step {i}. terminated={terminated}, truncated={truncated}, info={info}[/red]")
                    if crashed:
                        emit(f"[red]Simulation crash after running steps:[/red] {i}")
                        collision_frame = i
                    else:
                        emit("[yellow]Episode ended without collision (e.g., timeout/truncation).[/yellow]")
                    break
        except Exception as exc:
            episode_error = f"{type(exc).__name__}: {exc}"
            emit(f"[red]Episode {episode} error: {episode_error}[/red]")
        finally:

            with open(log_path, 'a') as f:
                current_penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
                f.write(
                    "Simulation {} | Seed {} | Steps: {} | File prefix: {} | penalty_stage={} | penalty_events={} | decision_timeout={} \n".format(
                        episode,
                        seed,
                        already_decision_steps,
                        result_prefix,
                        int(current_penalty_snapshot.get("stage", 0) or 0),
                        int(current_penalty_snapshot.get("penalty_events", 0) or 0),
                        round(float(current_penalty_snapshot.get("effective_decision_timeout_sec") or 0.0), 3),
                    )
                )
                
            if REFLECTION:
                emit("[yellow]Now running reflection agent...[/yellow]")
                if collision_frame != -1: # End with collision
                    for i in range(collision_frame, -1, -1):
                        if docs[i]["action"] != 4:  # not decelearate
                            corrected_response = RA.reflection(
                                docs[i]["human_question"], docs[i]["response"])

                            if reflection_auto_add:
                                choice = 'Y'
                            elif reflection_interactive:
                                choice = input("[yellow]Do you want to add this new memory item to update memory module? (Y/N): ").strip().upper()
                            else:
                                choice = 'N'

                            if choice == 'Y':
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    corrected_response,
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="mistake-correction"
                                )
                                emit("[green] Successfully add a new memory item to update memory module.[/green]. Now the database has " + str(len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings'])) + " items.")
                            else:
                                emit("[blue]Ignore this new memory item[/blue]")
                            break
                else:
                    planned_additions = len(docs) // reflection_add_every_n
                    print("[yellow]Do you want to add[/yellow]", planned_additions, "[yellow]new memory item to update memory module?[/yellow]", end="")
                    if reflection_auto_add:
                        choice = 'Y'
                    elif reflection_interactive:
                        choice = input("(Y/N): ").strip().upper()
                    else:
                        choice = 'N'
                    if choice == 'Y':
                        cnt = 0
                        for i in range(0, len(docs)):
                            if i % reflection_add_every_n == 1:
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    docs[i]["response"],
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="no-mistake-direct"
                                )
                                cnt +=1
                        emit("[green] Successfully add[/green] " + str(cnt) + " [green]new memory item to update memory module.[/green]. Now the database has " + str(len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings'])) + " items.")
                    else:
                        emit("[blue]Ignore these new memory items[/blue]")

            duration_sec = time.time() - episode_started
            episode_reward_avg = episode_reward_sum / max(already_decision_steps, 1)
            avg_ego_speed_mps = ego_speed_sum / max(ego_speed_count, 1)
            ttc_danger_rate = ttc_danger_steps / max(already_decision_steps, 1)
            headway_violation_rate = headway_violation_steps / max(already_decision_steps, 1)
            lane_change_rate = lane_change_count / max(already_decision_steps, 1)
            flap_accel_decel_rate = flap_accel_decel_count / max(already_decision_steps, 1)
            decision_latency_ms_avg = (duration_sec / max(already_decision_steps, 1)) * 1000.0
            format_failure_rate = format_failure_count / max(decisions_made, 1)
            penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
            timeout_penalty_events = max(
                0,
                int(penalty_snapshot.get("penalty_events", 0)) - penalty_start_events,
            )

            metrics_report["episodes"].append({
                "episode_index": int(episode),
                "seed": int(seed),
                "result_prefix": result_prefix,
                "video_prefix": result_prefix,
                "database_path": database_path,
                "steps": int(already_decision_steps),
                "max_steps": int(config["simulation_duration"]),
                "crashed": bool(crashed),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success_no_collision": (episode_error is None and not crashed),
                "episode_runtime_sec": round(duration_sec, 3),
                "avg_step_runtime_sec": round(duration_sec / max(already_decision_steps, 1), 3),
                "decisions_made": int(decisions_made),
                "responses_with_delimiter": int(responses_with_delimiter),
                "responses_strict_format": int(responses_strict_format),
                "responses_direct_parseable": int(responses_direct_parseable),
                "format_failure_count": int(format_failure_count),
                "format_failure_rate": round(format_failure_rate, 4),
                "episode_reward_sum": round(episode_reward_sum, 4),
                "episode_reward_avg": round(episode_reward_avg, 4),
                "avg_ego_speed_mps": round(avg_ego_speed_mps, 4),
                "ttc_danger_steps": int(ttc_danger_steps),
                "ttc_danger_rate": round(ttc_danger_rate, 4),
                "headway_violation_steps": int(headway_violation_steps),
                "headway_violation_rate": round(headway_violation_rate, 4),
                "lane_change_count": int(lane_change_count),
                "lane_change_rate": round(lane_change_rate, 4),
                "flap_accel_decel_count": int(flap_accel_decel_count),
                "flap_accel_decel_rate": round(flap_accel_decel_rate, 4),
                "decision_latency_ms_avg": round(decision_latency_ms_avg, 3),
                "timeout_penalty_stage_max": int(timeout_penalty_stage_max),
                "timeout_penalty_events": int(timeout_penalty_events),
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
                "error": episode_error,
            })

            emit("==========Simulation {} Done==========".format(episode))
            if progress is not None and episode_task is not None:
                progress.update(episode_task, advance=1)
            episode += 1

            try:
                env.close()
            except Exception:
                pass

    if progress is not None:
        try:
            if step_task is not None:
                progress.remove_task(step_task)
            if episode_task is not None:
                progress.remove_task(episode_task)
        finally:
            progress.stop()
            if progress_stop_registered:
                try:
                    atexit.unregister(progress.stop)
                except Exception:
                    pass

    final_penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
    metrics_report["model_runtime_policy"]["decision_timeout_penalty"] = copy.deepcopy(final_penalty_snapshot)
    metrics_report["model_runtime_policy"]["native_timeout_penalty"] = copy.deepcopy(final_penalty_snapshot)
    metrics_report["metrics_config"]["decision_timeout_penalty"] = copy.deepcopy(final_penalty_snapshot)
    metrics_report["metrics_config"]["native_timeout_penalty"] = copy.deepcopy(final_penalty_snapshot)
    metrics_report["aggregate"] = aggregate_run_results(metrics_report["episodes"])
    metrics_report_path = timestamped_results_path("run_metrics", ext=".json", results_dir=result_folder)
    write_json_atomic(metrics_report_path, metrics_report)

    if structured_mode and experiment_root and experiment_id and run_id:
        _update_experiment_manifest(
            experiment_root=experiment_root,
            experiment_id=experiment_id,
            model_name=model_name_for_paths,
            model_slug=model_slug,
            run_id=run_id,
            run_dir=result_folder,
            metrics_report_path=metrics_report_path,
            config_path=config_path,
            memory_path=memory_path,
            few_shot_num=int(few_shot_num),
            simulation_duration=int(config["simulation_duration"]),
        )
    print(f"[bold green]Saved run metrics report:[/bold green] {metrics_report_path}")


