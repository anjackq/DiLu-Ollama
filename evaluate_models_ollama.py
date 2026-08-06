import argparse
import copy
import functools
import gc
import inspect
from collections import Counter
from contextlib import nullcontext
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import gymnasium as gym
import numpy as np
import requests
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
from dilu.driver_agent.prompt_modules import PromptArtifact
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.runtime import (
    configure_runtime_env,
    load_runtime_config,
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
    DEFAULT_BENCHMARK_CASE_SET,
    load_benchmark_case_set,
    build_benchmark_case_set_fingerprint,
    build_case_env_config,
    benchmark_max_steps,
    build_benchmark_instruction,
    benchmark_metric_config,
    validate_benchmark_case_set,
    summarize_benchmark_episodes,
    benchmark_result_validity,
    BenchmarkEpisodeEvaluator,
    augment_behavior_aware_benchmark_episode,
    TOKEN_COUNT_METHOD,
    load_idle_calibration,
    save_idle_calibration,
    enrich_episode_energy_metrics,
    summarize_energy_latency_episodes,
    create_energy_monitor,
    system_hardware_snapshot,
    build_energy_tradeoff_summary,
    aggregate_episode_token_usage,
    annotate_aggregate_with_scientific_reporting,
    build_primary_metric_spec,
    write_scientific_analysis_artifacts,
    SPLIT_SCORING_POLICY_VERSION,
    SPLIT_SCORE_FIELDS,
    compute_split_scores_for_episode,
    normalize_ollama_think_mode,
    resolve_ollama_native_chat_mode,
)
from dilu.runtime.highway_scenario_spec import (
    apply_highway_scenario_events,
    apply_highway_scenario_spec,
)
from dilu.runtime.scientific_reporting import bootstrap_ci95
from dilu.runtime.harness_config import ExecutionMode, ShieldConfig
from dilu.runtime.action_resolution import ActionResolutionResult
from dilu.runtime.ollama_scientific_client import ScientificGenerationAbort
from dilu.runtime.scientific_runtime import ScientificEpisodeRuntime
from dilu.runtime.ollama_transport import inspect_ollama_model_identity
from dilu.runtime.runtime_failures import (
    ProtocolInvariantCode,
    ProtocolInvariantViolation,
    RuntimeProtocolError,
)
from dilu.runtime.scientific_trace import (
    DecisionTraceRecord,
    ScientificSimulatorAbort,
    ScientificTraceWriteError,
    ScientificTraceWriter,
    TraceDisposition,
    append_trace_before_step,
)
from dilu.runtime.scientific_transport_records import GenerationResult
from dilu.runtime.shield_stack import ShieldStackResult, execute_shield_stack
from dilu.scenario.envScenario import EnvScenario


STRICT_RESPONSE_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*([0-4])\s*$", re.IGNORECASE)
STRICT_RESPONSE_FIRST_LINE_PATTERN = re.compile(
    r"^Response to user:\s*\#{4}\s*([0-4])\s*$",
    re.IGNORECASE,
)
STOP_THRESHOLD_MPS_DEFAULT = 0.5
NEAR_STOP_THRESHOLD_MPS_DEFAULT = 2.0
LEGACY_BENCHMARK_VARIANT = "legacy_direct_action"
LEGACY_EXECUTION_MODE = "direct_action_loop"
_PROPOSED_ACTION_UNSET = object()


def build_env_bundle(
    config: Dict,
    env_id_override: Optional[str] = None,
    native_env_defaults_override: Optional[bool] = None,
    action_target_speeds_override: Optional[str] = None,
    env_config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict:
    return resolve_simulation_env_bundle(
        config,
        show_trajectories=False,
        render_agent=False,
        env_id_override=env_id_override,
        native_env_defaults_override=native_env_defaults_override,
        action_target_speeds_override=action_target_speeds_override,
        env_config_overrides=env_config_overrides,
        require_discrete_meta_action=True,
    )


def _resolve_benchmark_env_id_override(cli_env_id: Optional[str], benchmark_case_set: Optional[Dict]) -> Optional[str]:
    if cli_env_id and str(cli_env_id).strip():
        return str(cli_env_id).strip()
    if benchmark_case_set is None:
        return None
    target_env_id = str(benchmark_case_set.get("target_env_id") or "").strip()
    return target_env_id or None


def _benchmark_default_env_overrides(benchmark_case_set: Optional[Dict]) -> Optional[Dict[str, Any]]:
    if benchmark_case_set is None:
        return None
    defaults = benchmark_case_set.get("defaults") or {}
    overrides = defaults.get("env_overrides") or {}
    return copy.deepcopy(overrides) if isinstance(overrides, dict) and overrides else None


def parse_seeds(raw: Optional[Any]) -> List[int]:
    if raw is None or raw == "":
        return list(DEFAULT_DILU_SEEDS)
    if isinstance(raw, str):
        tokens = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        tokens = raw
    else:
        tokens = [raw]
    seeds = []
    for token in tokens:
        token = str(token).strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def _parse_benchmark_category_filter(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        raise ValueError("--benchmark-categories must not be empty.")
    categories: List[str] = []
    for token in str(raw).split(","):
        category = token.strip()
        if not category:
            raise ValueError("--benchmark-categories contains an empty category token.")
        if category not in categories:
            categories.append(category)
    return categories


def _filter_benchmark_cases_by_category(
    case_set: Dict,
    requested_categories: List[str],
) -> Tuple[Dict, List[Dict]]:
    filtered_case_set = dict(case_set)
    cases = list(case_set.get("cases") or [])
    if not requested_categories:
        filtered_case_set["cases"] = list(cases)
        filtered_case_set["categories"] = sorted({str(case["category"]) for case in cases})
        return filtered_case_set, list(cases)

    available_categories = sorted({str(case.get("category") or "") for case in cases if case.get("category")})
    requested_set = set(requested_categories)
    missing_categories = [category for category in requested_categories if category not in available_categories]
    if missing_categories:
        raise ValueError(
            "Unknown benchmark category filter value(s): {missing}. Available categories: {available}.".format(
                missing=", ".join(missing_categories),
                available=", ".join(available_categories),
            )
        )

    filtered_cases = [
        case
        for case in cases
        if str(case.get("category") or "") in requested_set
    ]
    if not filtered_cases:
        raise ValueError(
            "No benchmark cases matched --benchmark-categories={categories}.".format(
                categories=", ".join(requested_categories)
            )
        )
    filtered_case_set["cases"] = list(filtered_cases)
    filtered_case_set["categories"] = sorted({str(case["category"]) for case in filtered_cases})
    return filtered_case_set, filtered_cases


def resolve_eval_seeds(
    config: Optional[Dict[str, Any]],
    cli_seeds: Optional[str],
    seed_count: Optional[int] = None,
) -> List[int]:
    if cli_seeds is not None:
        seeds = parse_seeds(cli_seeds)
    else:
        config = config or {}
        seeds = parse_seeds(None)
        for key in ("eval_seed_bank", "eval_seeds"):
            if config.get(key) is not None:
                seeds = parse_seeds(config.get(key))
                break
    if seed_count is None:
        return seeds
    count = int(seed_count)
    if count <= 0:
        raise ValueError("--seed-count must be greater than zero.")
    if count > len(seeds):
        raise ValueError(f"--seed-count={count} exceeds available seed bank size {len(seeds)}.")
    return seeds[:count]


def _format_default_seed_count() -> int:
    return len(DEFAULT_DILU_SEEDS)


def parse_action_target_speeds(raw: Optional[str]) -> Optional[str]:
    text = str(raw or "").strip()
    return text or None


def _clamp_int(value: Any, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return max(minimum, int(default))


def _clamp_float(value: Any, default: float, minimum: float = 0.0, maximum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def _normalize_on_off(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "on":
        return True
    if text == "off":
        return False
    raise ValueError(f"Unsupported on/off value: {value}")


def _resolve_eval_timeout_early_stop_policy(config: Dict, args) -> Dict[str, Any]:
    enabled_override = _normalize_on_off(getattr(args, "timeout_early_stop", None))
    require_max_level_override = _normalize_on_off(getattr(args, "timeout_early_stop_require_max_level", None))
    enabled = (
        enabled_override
        if enabled_override is not None
        else _config_as_bool(config.get("eval_timeout_early_stop_enabled", True), default=True)
    )
    return {
        "enabled": bool(enabled),
        "min_decisions": _clamp_int(
            getattr(args, "timeout_early_stop_min_decisions", None)
            if getattr(args, "timeout_early_stop_min_decisions", None) is not None
            else config.get("eval_timeout_early_stop_min_decisions", 5),
            default=5,
            minimum=1,
        ),
        "consecutive_timeout_fallbacks": _clamp_int(
            getattr(args, "timeout_early_stop_consecutive", None)
            if getattr(args, "timeout_early_stop_consecutive", None) is not None
            else config.get("eval_timeout_early_stop_consecutive_timeout_fallbacks", 3),
            default=3,
            minimum=1,
        ),
        "collapse_rate_threshold": _clamp_float(
            getattr(args, "timeout_early_stop_rate", None)
            if getattr(args, "timeout_early_stop_rate", None) is not None
            else config.get("eval_timeout_early_stop_collapse_rate_threshold", 0.8),
            default=0.8,
            minimum=0.0,
            maximum=1.0,
        ),
        "require_max_timeout_level": (
            require_max_level_override
            if require_max_level_override is not None
            else _config_as_bool(config.get("eval_timeout_early_stop_require_max_timeout_level", True), default=True)
        ),
        "quarantine_after_collapses": _clamp_int(
            getattr(args, "timeout_collapse_quarantine_after", None)
            if getattr(args, "timeout_collapse_quarantine_after", None) is not None
            else config.get("eval_timeout_model_quarantine_after_collapses", 2),
            default=2,
            minimum=1,
        ),
    }


def _should_early_stop_timeout_episode(
    *,
    decision_calls_total: int,
    decision_timeout_count: int,
    fallback_action_count: int,
    consecutive_timeout_fallbacks: int,
    timeout_policy_mode: Optional[str],
    active_timeout_level: Optional[float],
    max_timeout_level: Optional[float],
    policy: Optional[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    policy = dict(policy or {})
    if not policy.get("enabled", False):
        return False, None
    if int(decision_calls_total) < int(policy.get("min_decisions", 1) or 1):
        return False, None
    if (
        str(timeout_policy_mode or "").strip().lower() == "laddered"
        and bool(policy.get("require_max_timeout_level", False))
    ):
        if active_timeout_level is None or max_timeout_level is None:
            return False, None
        if float(active_timeout_level) + 1e-9 < float(max_timeout_level):
            return False, None
    if int(consecutive_timeout_fallbacks) >= int(policy.get("consecutive_timeout_fallbacks", 1) or 1):
        return True, "consecutive_timeout_fallback_collapse"
    timeout_rate = float(decision_timeout_count) / max(int(decision_calls_total), 1)
    fallback_rate = float(fallback_action_count) / max(int(decision_calls_total), 1)
    threshold = float(policy.get("collapse_rate_threshold", 1.0) or 1.0)
    if timeout_rate >= threshold and fallback_rate >= threshold:
        return True, "rate_timeout_fallback_collapse"
    return False, None


def _summarize_decision_latency_samples(decision_latencies_sec: List[float]) -> Dict[str, Optional[float]]:
    samples = [max(0.0, float(value)) for value in decision_latencies_sec if value is not None]
    if not samples:
        return {
            "decision_latency_ms_avg": None,
            "p95_decision_latency_sec": None,
        }
    return {
        "decision_latency_ms_avg": round((sum(samples) / len(samples)) * 1000.0, 3),
        "p95_decision_latency_sec": round(float(np.percentile(np.array(samples, dtype=float), 95)), 4),
    }


def _index_ollama_preflight_results(preflight_results: List[Dict]) -> Dict[str, Dict]:
    indexed: Dict[str, Dict] = {}
    for item in preflight_results or []:
        if not isinstance(item, dict):
            continue
        model_name = str(item.get("model") or "").strip()
        if model_name:
            indexed[model_name] = copy.deepcopy(item)
    return indexed


def _annotate_aggregate_with_ollama_preflight_status(
    aggregate: Dict,
    preflight_results_by_model: Dict[str, Dict],
) -> Dict:
    annotated = dict(aggregate)
    probe = preflight_results_by_model.get(str(aggregate.get("model") or "").strip())
    annotated["ollama_preflight_ok"] = None if probe is None else bool(probe.get("ok"))
    annotated["ollama_preflight_transport"] = None if probe is None else probe.get("transport")
    annotated["ollama_preflight_native_chat_configured"] = (
        None if probe is None else probe.get("ollama_native_chat_configured")
    )
    annotated["ollama_preflight_native_chat_effective"] = (
        None if probe is None else probe.get("ollama_native_chat_effective")
    )
    annotated["ollama_preflight_native_chat_resolution_reason"] = (
        None if probe is None else probe.get("ollama_native_chat_resolution_reason")
    )
    annotated["ollama_preflight_elapsed_sec"] = None if probe is None else probe.get("elapsed_sec")
    annotated["ollama_preflight_error"] = None if probe is None else probe.get("error")
    return annotated


def _extend_invalid_reason(existing: Optional[str], *extras: Optional[str]) -> Optional[str]:
    parts = [str(existing).strip()] if existing else []
    for item in extras:
        text = str(item or "").strip()
        if text and text not in parts:
            parts.append(text)
    return "; ".join(parts) if parts else None


def _build_measurement_integrity_summary(
    preflight_results: List[Dict],
    preflight_warning: Optional[str],
    skipped_models_due_to_preflight: Optional[List[Dict[str, Any]]] = None,
    quarantined_models_due_to_timeout_collapse: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List]:
    failed_models = []
    for item in preflight_results or []:
        if not isinstance(item, dict) or bool(item.get("ok")):
            continue
        failed_models.append(
            {
                "model": item.get("model"),
                "transport": item.get("transport"),
                "ollama_native_chat_configured": item.get("ollama_native_chat_configured"),
                "ollama_native_chat_effective": item.get("ollama_native_chat_effective"),
                "ollama_native_chat_resolution_reason": item.get(
                    "ollama_native_chat_resolution_reason"
                ),
                "elapsed_sec": item.get("elapsed_sec"),
                "timeout_sec": item.get("timeout_sec"),
                "error": item.get("error"),
            }
        )
    warnings = [str(preflight_warning)] if preflight_warning else []
    return {
        "measurement_integrity_warnings": warnings,
        "ollama_preflight_failed_models": failed_models,
        "skipped_models_due_to_preflight": list(skipped_models_due_to_preflight or []),
        "quarantined_models_due_to_timeout_collapse": list(quarantined_models_due_to_timeout_collapse or []),
    }


def _build_skipped_model_aggregate(
    *,
    model_name: str,
    planned_episode_count: int,
    reason: str,
    preflight_probe: Optional[Dict[str, Any]] = None,
    benchmark_mode: bool = False,
) -> Dict[str, Any]:
    aggregate = {
        "model": model_name,
        "episodes": 0,
        "planned_episode_count": int(max(0, planned_episode_count)),
        "executed_episode_count": 0,
        "skipped_episode_count": int(max(0, planned_episode_count)),
        "episode_execution_complete": False,
        "model_skipped_due_to_preflight": True,
        "model_skipped_reason": str(reason),
        "model_quarantined_due_to_timeout_collapse": False,
        "model_quarantine_reason": None,
        "episodes_stopped_by_timeout_cap": 0,
        "crash_rate": None,
        "no_collision_rate": None,
        "error_rate": None,
        "avg_steps": None,
        "avg_episode_runtime_sec": None,
        "avg_step_runtime_sec": None,
        "decision_timeout_rate_mean": None,
        "timeout_episode_rate": None,
        "fallback_action_rate_mean": None,
        "decision_latency_ms_avg": None,
        "timeout_collapse_detected": False,
        "timeout_collapse_reason": None,
        "split_scoring_policy_version": SPLIT_SCORING_POLICY_VERSION,
        "ollama_preflight_ok": None if preflight_probe is None else bool(preflight_probe.get("ok")),
        "ollama_preflight_transport": None if preflight_probe is None else preflight_probe.get("transport"),
        "ollama_preflight_native_chat_configured": (
            None if preflight_probe is None else preflight_probe.get("ollama_native_chat_configured")
        ),
        "ollama_preflight_native_chat_effective": (
            None if preflight_probe is None else preflight_probe.get("ollama_native_chat_effective")
        ),
        "ollama_preflight_native_chat_resolution_reason": (
            None
            if preflight_probe is None
            else preflight_probe.get("ollama_native_chat_resolution_reason")
        ),
        "ollama_preflight_elapsed_sec": None if preflight_probe is None else preflight_probe.get("elapsed_sec"),
        "ollama_preflight_error": None if preflight_probe is None else preflight_probe.get("error"),
    }
    for split_field in SPLIT_SCORE_FIELDS:
        aggregate[split_field] = None
        aggregate[f"{split_field}_ci95"] = None
    if benchmark_mode:
        aggregate.update(
            {
                "task_completion_rate": None,
                "overall_score_mean": None,
                "driving_score": None,
                "benchmark_result_valid": False,
                "benchmark_result_invalid_reason": _extend_invalid_reason(
                    None,
                    "incomplete_episode_set",
                    str(reason),
                ),
            }
        )
    return aggregate


def _normalize_energy_mode(raw: Optional[str]) -> str:
    mode = str(raw or "none").strip().lower()
    if mode == "none":
        return "none"
    if mode in {"latency_only", "joulescope_hw"}:
        return mode
    return "none"


def _apply_measurement_runtime_overrides(
    config: Dict,
    args,
    *,
    energy_mode: str,
) -> Dict:
    runtime_config = copy.deepcopy(config)
    if _normalize_energy_mode(energy_mode) == "none":
        return runtime_config
    if str(runtime_config.get("OPENAI_API_TYPE", "")).strip().lower() != "ollama":
        return runtime_config

    think_override = getattr(args, "ollama_think_mode", None)
    use_native_flag = bool(getattr(args, "ollama_use_native_chat", False))
    disable_native_flag = bool(getattr(args, "ollama_disable_native_chat", False))
    if use_native_flag and disable_native_flag:
        raise ValueError(
            "Use only one of --ollama-use-native-chat or --ollama-disable-native-chat/--no-ollama-use-native-chat."
        )

    native_chat_override = None
    if use_native_flag:
        native_chat_override = True
    elif disable_native_flag:
        native_chat_override = False

    auto_forced_native = False
    if think_override is not None:
        runtime_config["OLLAMA_THINK_MODE"] = str(think_override)
        if native_chat_override is None:
            runtime_config["OLLAMA_USE_NATIVE_CHAT"] = True
            auto_forced_native = True
        else:
            runtime_config["OLLAMA_USE_NATIVE_CHAT"] = bool(native_chat_override)
    elif native_chat_override is not None:
        runtime_config["OLLAMA_USE_NATIVE_CHAT"] = bool(native_chat_override)

    runtime_config["_benchmark_ollama_runtime_overrides"] = {
        "ollama_think_mode": runtime_config.get("OLLAMA_THINK_MODE"),
        "ollama_use_native_chat": runtime_config.get("OLLAMA_USE_NATIVE_CHAT"),
        "auto_forced_native_chat": bool(auto_forced_native),
    }
    return runtime_config


def _inspect_ollama_model(model_name: str) -> Dict[str, Any]:
    identity_fields: Dict[str, Any] = {
        "model_digest": None,
        "model_identity_verified": False,
        "model_identity_source": "ollama_native_api_tags",
        "model_identity_error": None,
    }
    try:
        identity = inspect_ollama_model_identity(
            os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
            model_name,
        )
        identity_fields["model_digest"] = identity.model_digest
        identity_fields["model_identity_verified"] = True
    except Exception as exc:
        identity_fields["model_identity_error"] = type(exc).__name__

    try:
        proc = subprocess.run(
            ["ollama", "show", model_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return {
            "model_tag": model_name,
            "available": None,
            "family": None,
            "quantization": None,
            "parameters": None,
            **identity_fields,
        }

    text = (proc.stdout or proc.stderr or "").strip()
    result = {
        "model_tag": model_name,
        "available": proc.returncode == 0,
        "family": None,
        "quantization": None,
        "parameters": None,
        **identity_fields,
    }
    for line in text.splitlines():
        lower = line.lower()
        if "family" in lower and result["family"] is None:
            result["family"] = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
        elif "quantization" in lower and result["quantization"] is None:
            result["quantization"] = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
        elif "parameter" in lower and result["parameters"] is None:
            result["parameters"] = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
    return result


def _measurement_record(episode: Dict) -> Dict:
    return {
        "episode_id": str(episode.get("case_id") or f"seed_{episode.get('seed')}"),
        "seed": episode.get("seed"),
        "case_id": episode.get("case_id"),
        "category": episode.get("category"),
        "episode_wall_time_start": episode.get("episode_wall_time_start"),
        "episode_wall_time_end": episode.get("episode_wall_time_end"),
        "episode_runtime_sec": episode.get("episode_runtime_sec"),
        "energy_mode": episode.get("energy_mode"),
        "raw_energy_j": episode.get("raw_energy_j"),
        "idle_baseline_energy_j": episode.get("idle_baseline_energy_j"),
        "net_energy_j": episode.get("net_energy_j"),
        "avg_power_w": episode.get("avg_power_w"),
        "peak_power_w": episode.get("peak_power_w"),
        "energy_per_decision_j": episode.get("energy_per_decision_j"),
        "energy_per_token_j": episode.get("energy_per_token_j"),
        "prompt_tokens_total": episode.get("prompt_tokens_total"),
        "completion_tokens_total": episode.get("completion_tokens_total"),
        "total_tokens": episode.get("total_tokens"),
        "tokens_generated_total": episode.get("tokens_generated_total"),
        "tokens_per_second": episode.get("tokens_per_second"),
        "latency_to_first_action_sec": episode.get("latency_to_first_action_sec"),
        "token_count_method": episode.get("token_count_method"),
        "token_usage_source": episode.get("token_usage_source"),
        "energy_measurement_meta": episode.get("energy_measurement_meta"),
    }


def _action_histogram(action_trace: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in action_trace:
        key = str(int(item.get("action_id", -1)))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _bump_counter(counter: Counter, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        counter[text] += 1


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _merge_count_dicts(episodes: List[Dict], field_name: str) -> Dict[str, int]:
    merged: Counter = Counter()
    for episode in episodes:
        raw_counts = episode.get(field_name, {})
        if not isinstance(raw_counts, dict):
            continue
        for key, value in raw_counts.items():
            try:
                increment = int(value)
            except Exception:
                continue
            if increment > 0:
                merged[str(key)] += increment
    return _counter_dict(merged)


def _summarize_ollama_case_unloads(episodes: List[Dict]) -> Dict[str, Any]:
    attempts = 0
    successes = 0
    failures = 0
    for episode in episodes:
        unload_result = episode.get("ollama_case_unload")
        if not isinstance(unload_result, dict) or not unload_result.get("attempted"):
            continue
        attempts += 1
        if bool(unload_result.get("ok")):
            successes += 1
        else:
            failures += 1
    return {
        "ollama_case_unload_attempts": int(attempts),
        "ollama_case_unload_successes": int(successes),
        "ollama_case_unload_failures": int(failures),
        "ollama_case_unload_success_rate": round(successes / attempts, 4) if attempts else None,
    }


def _scenario_ego_speed_mps(sce: EnvScenario) -> Optional[float]:
    try:
        return round(float(getattr(sce.env.unwrapped.vehicle, "speed")), 4)
    except Exception:
        return None


_TRACE_METADATA_KEYS = (
    "runtime_parse_path",
    "fallback_reason",
    "semantic_recovery_used",
    "semantic_recovery_label",
    "semantic_recovery_reason",
    "intent_resolver_used",
    "intent_resolver_model",
    "intent_resolver_action_id",
    "intent_resolver_abstained",
    "intent_resolver_reason",
    "final_action_source",
    "response_contract_satisfied",
    "response_contract_recovered",
    "response_unparseable",
    "response_truncated_before_contract",
    "response_action_line",
    "response_first_line",
    "shield_execution_mode",
    "shield_proposed_action_id",
    "shield_fallback_modified_action_id",
    "shield_unshielded_action_id",
    "shield_shielded_action_id",
    "shield_executed_action_id",
    "shield_final_action_id",
    "shield_stage_order",
    "lane_change_stage_input_action_id",
    "lane_change_stage_output_action_id",
    "lane_change_stage_applied",
    "lane_change_stage_bypassed",
    "lane_change_stage_reason",
    "longitudinal_safety_stage_input_action_id",
    "longitudinal_safety_stage_output_action_id",
    "longitudinal_safety_stage_applied",
    "longitudinal_safety_stage_bypassed",
    "longitudinal_safety_stage_reason",
    "low_speed_recovery_stage_input_action_id",
    "low_speed_recovery_stage_output_action_id",
    "low_speed_recovery_stage_applied",
    "low_speed_recovery_stage_bypassed",
    "low_speed_recovery_stage_reason",
    "lane_change_shield_reason",
    "lane_change_original_action_id",
    "lane_change_final_action_id",
    "lane_change_target_lane_rank",
    "lane_change_front_gap_m",
    "lane_change_front_ttc_sec",
    "lane_change_rear_gap_m",
    "lane_change_rear_ttc_sec",
    "lane_change_target_front_gap_m",
    "lane_change_target_front_ttc_sec",
    "lane_change_target_rear_gap_m",
    "lane_change_target_rear_ttc_sec",
    "lane_change_required_front_gap_m",
    "lane_change_required_rear_gap_m",
    "lane_change_required_front_ttc_sec",
    "lane_change_required_rear_ttc_sec",
    "longitudinal_safety_shield_reason",
    "longitudinal_safety_original_action_id",
    "longitudinal_safety_final_action_id",
    "longitudinal_safety_front_gap_m",
    "longitudinal_safety_front_ttc_sec",
    "longitudinal_safety_current_front_gap_m",
    "longitudinal_safety_current_front_ttc_sec",
    "longitudinal_safety_required_front_gap_m",
    "longitudinal_safety_required_front_ttc_sec",
    "longitudinal_safety_projected_front_gap_m",
    "longitudinal_safety_projected_front_ttc_sec",
    "longitudinal_safety_projected_ego_speed_mps",
    "longitudinal_safety_projection_horizon_sec",
    "flow_recovery_shield_reason",
    "flow_recovery_original_action_id",
    "flow_recovery_final_action_id",
    "flow_recovery_reason",
    "flow_recovery_front_gap_m",
    "flow_recovery_front_ttc_sec",
    "reactive_safety_original_action_id",
    "reactive_safety_final_action_id",
    "timeout_phase",
    "warmup_active",
    "warmup_success_observed",
    "warmup_timeout_failures",
    "warmup_timeout_sec",
    "benchmark_events_applied",
    "benchmark_event_ids",
    "benchmark_event_types",
    "benchmark_event_step",
    "benchmark_events",
)


def _response_first_nonempty_line(response_text: str) -> str:
    for line in str(response_text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _decision_trace_item(
    *,
    step_idx: int,
    action_id: int,
    response_text: str,
    decision_meta: Dict[str, Any],
    include_response_text: bool = False,
) -> Dict[str, Any]:
    model_action_value = decision_meta.get("original_selected_action", action_id)
    model_action = (
        int(model_action_value) if model_action_value is not None else None
    )
    final_action = int(decision_meta.get("selected_action", action_id))
    trace_item: Dict[str, Any] = {
        "step_idx": int(step_idx),
        "action_id": final_action,
        "original_action_id": model_action,
        "model_action_id": model_action,
        "final_action_id": final_action,
        "reactive_safety_shield_applied": bool(
            decision_meta.get("reactive_safety_shield_applied", False)
        ),
        "lane_change_shield_applied": bool(decision_meta.get("lane_change_shield_applied", False)),
        "longitudinal_safety_shield_applied": bool(
            decision_meta.get("longitudinal_safety_shield_applied", False)
        ),
        "flow_recovery_shield_applied": bool(decision_meta.get("flow_recovery_shield_applied", False)),
        "ego_speed_mps": decision_meta.get("ego_speed_mps"),
        "decision_elapsed_sec": (
            round(float(decision_meta.get("decision_elapsed_sec")), 6)
            if decision_meta.get("decision_elapsed_sec") is not None
            else None
        ),
        "timed_out": bool(decision_meta.get("timed_out", False)),
        "used_fallback": bool(decision_meta.get("used_fallback", False)),
        "ollama_transport": decision_meta.get("ollama_transport"),
        "ollama_native_chat_configured": decision_meta.get("ollama_native_chat_configured"),
        "ollama_native_chat_effective": decision_meta.get("ollama_native_chat_effective"),
        "ollama_native_chat_resolution_reason": decision_meta.get(
            "ollama_native_chat_resolution_reason"
        ),
        "ollama_requested_think_mode": decision_meta.get("ollama_requested_think_mode"),
        "ollama_effective_think_mode": decision_meta.get("ollama_effective_think_mode"),
        "ollama_native_timeout": bool(decision_meta.get("ollama_native_timeout", False)),
        "ollama_native_timeout_short_circuit": bool(
            decision_meta.get("ollama_native_timeout_short_circuit", False)
        ),
        "prompt_tokens": int(decision_meta.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(decision_meta.get("completion_tokens", 0) or 0),
        "total_tokens": int(decision_meta.get("total_tokens", 0) or 0),
        "token_count_method": decision_meta.get("token_count_method"),
        "token_usage_source": decision_meta.get("token_usage_source"),
        "response_first_line": _response_first_nonempty_line(response_text),
    }
    for key in _TRACE_METADATA_KEYS:
        if key in decision_meta:
            trace_item[key] = decision_meta.get(key)
    if include_response_text:
        trace_item["response_text"] = response_text
    return trace_item


def _apply_reactive_safety_shields(
    action: int,
    sce: EnvScenario,
    decision_meta: Dict,
    *,
    execution_mode: ExecutionMode = ExecutionMode.SHIELDED,
    proposed_action_id: Any = _PROPOSED_ACTION_UNSET,
    shield_config: Optional[ShieldConfig] = None,
) -> Tuple[int, Dict[str, Any]]:
    result = _execute_reactive_safety_shield_stack(
        action,
        sce,
        execution_mode=execution_mode,
        proposed_action_id=proposed_action_id,
        shield_config=shield_config,
    )
    shield_meta = result.to_metadata()
    _update_shield_decision_metadata(decision_meta, result, shield_meta)
    return result.executed_action_id, shield_meta


def _execute_reactive_safety_shield_stack(
    action: int,
    sce: EnvScenario,
    *,
    execution_mode: ExecutionMode = ExecutionMode.SHIELDED,
    proposed_action_id: Any = _PROPOSED_ACTION_UNSET,
    shield_config: Optional[ShieldConfig] = None,
) -> ShieldStackResult:
    fallback_modified_action = _safe_int_action(action)
    proposed_action = (
        fallback_modified_action
        if proposed_action_id is _PROPOSED_ACTION_UNSET
        else (
            None
            if proposed_action_id is None
            else _safe_int_action(proposed_action_id)
        )
    )
    return execute_shield_stack(
        scenario=sce,
        proposed_action_id=proposed_action,
        fallback_modified_action_id=fallback_modified_action,
        execution_mode=execution_mode,
        shield_config=shield_config or ShieldConfig.implementation_defaults(),
    )


def _update_shield_decision_metadata(
    decision_meta: Dict[str, Any],
    result: ShieldStackResult,
    shield_meta: Dict[str, Any],
) -> None:
    override_classes = {
        "lane_change": "safety_shield",
        "longitudinal_safety": "longitudinal_safety_shield",
        "low_speed_recovery": "flow_recovery_shield",
    }
    for stage in result.stages:
        if stage.applied:
            decision_meta["runtime_override_reason_class"] = override_classes[
                stage.stage_name
            ]
            decision_meta["runtime_override_reason"] = stage.reason
    decision_meta.update(shield_meta)
    decision_meta["original_selected_action"] = result.proposed_action_id
    decision_meta["selected_action"] = result.executed_action_id


def _compose_scientific_trace_record(
    factory: Callable[..., DecisionTraceRecord],
    *,
    decision_index: int,
    env_step_index: int,
    prompt_artifact: PromptArtifact,
    generation: GenerationResult,
    resolution: Optional[ActionResolutionResult],
    shield_stack: Optional[ShieldStackResult],
    disposition: TraceDisposition,
    decision_latency_ms: float,
    benchmark_event_meta: Dict[str, Any],
) -> DecisionTraceRecord:
    try:
        record = factory(
            decision_index=decision_index,
            env_step_index=env_step_index,
            prompt_artifact=prompt_artifact,
            generation=generation,
            resolution=resolution,
            shield_stack=shield_stack,
            disposition=disposition,
            decision_latency_ms=decision_latency_ms,
            benchmark_event_meta=dict(benchmark_event_meta),
        )
        if not isinstance(record, DecisionTraceRecord):
            raise ValueError("Scientific trace factory returned an invalid record.")
        if (
            record.prompt_artifact is not prompt_artifact
            or record.generation is not generation
            or record.resolution is not resolution
            or record.shield_stack is not shield_stack
            or record.disposition is not disposition
            or record.context.key.decision_index != decision_index
            or record.context.key.env_step_index != env_step_index
        ):
            raise ValueError("Scientific trace factory did not preserve typed evidence.")
        return record
    except ScientificTraceWriteError:
        raise
    except Exception as exc:
        raise ScientificTraceWriteError(
            "Scientific decision trace composition failed."
        ) from exc


def _resolve_measurement_output_root(
    *,
    args,
    experiment_id: str,
) -> str:
    if args.output_root:
        return ensure_dir(args.output_root)
    results_root = args.results_root or os.path.join("results", "energy_benchmarks")
    return ensure_dir(os.path.join(results_root, experiment_id))


def _calibration_output_path(
    *,
    args,
    output_root: str,
) -> str:
    if args.calibration_output:
        return args.calibration_output
    return os.path.join(output_root, "calibration", f"idle_power_{current_timestamp()}.json")


def _normalize_performance_mode(value: Optional[str]) -> str:
    mode = str(value or "default").strip().lower()
    if mode in {"default", "fast"}:
        return mode
    return "default"


def _resolve_eval_performance_mode(config: Dict, cli_override: Optional[str]) -> str:
    if cli_override is not None:
        return _normalize_performance_mode(cli_override)
    return _normalize_performance_mode(config.get("eval_performance_mode", "default"))


def _resolve_eval_artifact_modes(config: Dict, args, *, measurement_mode: bool) -> Tuple[bool, bool]:
    save_run_artifacts = _config_as_bool(config.get("eval_save_run_artifacts", False), default=False) or bool(
        getattr(args, "save_run_artifacts", False)
    )
    if bool(getattr(args, "no_save_run_artifacts", False)):
        save_run_artifacts = False

    record_video = _config_as_bool(config.get("eval_record_video", False), default=False) or bool(
        getattr(args, "record_video", False)
    )
    if bool(getattr(args, "no_record_video", False)) or bool(getattr(args, "no_save_run_artifacts", False)):
        record_video = False

    if record_video:
        save_run_artifacts = True
    if measurement_mode:
        return False, False
    return bool(save_run_artifacts), bool(record_video)


def _config_as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _sanitize_process_output(text: Any, *, limit: int = 500) -> str:
    if text is None:
        return ""
    cleaned = ANSI_ESCAPE_PATTERN.sub("", str(text))
    cleaned = "".join(
        char
        for char in cleaned
        if char in {"\n", "\r", "\t"} or (ord(char) >= 32 and ord(char) != 127)
    )
    return cleaned.strip()[: max(0, int(limit))]


def _ollama_case_unload_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    enabled = _config_as_bool(config.get("ollama_unload_after_case", False), default=False)
    try:
        timeout_sec = float(config.get("ollama_unload_after_case_timeout_sec", 10) or 10)
    except Exception:
        timeout_sec = 10.0
    return {
        "enabled": bool(enabled),
        "method": "ollama_stop_model" if enabled else None,
        "timeout_sec": max(1.0, timeout_sec),
    }


def _stop_ollama_model_after_case(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    policy = _ollama_case_unload_policy(config)
    if not policy["enabled"]:
        return {"attempted": False, "enabled": False}
    started = time.time()
    try:
        proc = subprocess.run(
            ["ollama", "stop", str(model_name)],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(policy["timeout_sec"]),
        )
        return {
            "attempted": True,
            "enabled": True,
            "ok": proc.returncode == 0,
            "method": policy["method"],
            "model": str(model_name),
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - started, 3),
            "stdout": _sanitize_process_output(proc.stdout),
            "stderr": _sanitize_process_output(proc.stderr),
        }
    except Exception as exc:
        return {
            "attempted": True,
            "enabled": True,
            "ok": False,
            "method": policy["method"],
            "model": str(model_name),
            "elapsed_sec": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _maybe_stop_ollama_after_preflight(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    result = _stop_ollama_model_after_case(config, model_name)
    if result.get("attempted"):
        result["phase"] = "preflight"
    return result


def _resolve_simulation_duration(
    config: Dict,
    env_config_snapshot: Optional[Dict[str, Any]] = None,
) -> int:
    if config.get("simulation_duration") is not None:
        return int(config["simulation_duration"])
    if env_config_snapshot is not None and env_config_snapshot.get("duration") is not None:
        return int(env_config_snapshot["duration"])
    return 30


def _resolve_eval_ollama_preflight_enabled(config: Dict, cli_skip: bool) -> bool:
    if cli_skip:
        return False
    return _config_as_bool(config.get("eval_ollama_preflight_enabled", True), default=True)


def _resolve_eval_ollama_preflight_timeout_sec(config: Dict, cli_override: Optional[float]) -> float:
    if cli_override is not None:
        timeout_sec = float(cli_override)
    else:
        timeout_sec = float(config.get("eval_ollama_preflight_timeout_sec", 15.0))
    return max(1.0, timeout_sec)


def _normalize_measurement_hard_preflight_policy(value: Optional[str]) -> str:
    text = str(value or "skip_model").strip().lower()
    if text in {"warn_run", "skip_model", "abort_run"}:
        return text
    return "skip_model"


def _resolve_measurement_hard_preflight_policy(config: Dict, args) -> str:
    if getattr(args, "strict_ollama_preflight", False):
        return "abort_run"
    cli_value = getattr(args, "measurement_hard_preflight_policy", None)
    if cli_value is not None:
        return _normalize_measurement_hard_preflight_policy(cli_value)
    return _normalize_measurement_hard_preflight_policy(
        config.get("eval_measurement_hard_preflight_policy", "skip_model")
    )


def _classify_ollama_preflight_failure(result: Dict[str, Any]) -> str:
    status_code = result.get("status_code")
    if isinstance(status_code, int):
        if 400 <= status_code < 500:
            return "hard"
        return "soft"
    error_text = str(result.get("error") or "").strip().lower()
    hard_markers = [
        "404",
        "400",
        "401",
        "403",
        "422",
        "not found",
        "unsupported",
        "invalid request",
        "bad request",
        "model not found",
    ]
    return "hard" if any(marker in error_text for marker in hard_markers) else "soft"


def _format_ollama_preflight_failures(failures: List[Dict[str, Any]]) -> str:
    if not failures:
        return ""
    lines = [
        "Ollama preflight failed before evaluation.",
        "The backend did not respond to a tiny test completion, so the eval could not be trusted as-is.",
    ]
    for item in failures:
        lines.append(
            f"- model={item['model']} transport={item['transport']} "
            f"timeout={item['timeout_sec']}s error={item['error']}"
        )
    lines.extend(
        [
            "Recommended checks:",
            "1. Restart Ollama and ensure no models are stuck in 'Stopping...'.",
            "2. Run `ollama run <model> \"Reply with exactly 4\"` manually.",
            "3. Reduce GPU contention or unload stale Ollama model processes.",
            "4. Use `--skip-ollama-preflight` only if you intentionally want to wait through long model calls.",
        ]
    )
    return "\n".join(lines)


def _ollama_native_chat_url(api_base: str) -> str:
    base = str(api_base or "http://localhost:11434/v1").strip()
    if base.endswith("/"):
        base = base[:-1]
    parsed = urlparse(base)
    normalized_path = parsed.path.rstrip("/")
    if normalized_path.endswith("/v1"):
        root_path = normalized_path[:-3]
    elif normalized_path == "/v1":
        root_path = ""
    else:
        root_path = normalized_path
    if not root_path.endswith("/"):
        root_path += "/"
    return f"{parsed.scheme}://{parsed.netloc}{root_path}api/chat"


def _ollama_v1_chat_completions_url(api_base: str) -> str:
    base = str(api_base or "http://localhost:11434/v1").strip().rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


OLLAMA_RUNTIME_CONTROL_CONFIG_KEYS = {
    "ollama_runtime_num_ctx": "DILU_OLLAMA_NUM_CTX",
    "ollama_runtime_keep_alive": "DILU_OLLAMA_KEEP_ALIVE",
    "ollama_runtime_max_loaded_models": "OLLAMA_MAX_LOADED_MODELS",
    "ollama_runtime_num_parallel": "OLLAMA_NUM_PARALLEL",
    "ollama_runtime_max_queue": "OLLAMA_MAX_QUEUE",
}


def _ollama_runtime_controls_report(config: Dict[str, Any]) -> Dict[str, Any]:
    exported_env = {
        env_key: os.environ.get(env_key)
        for env_key in OLLAMA_RUNTIME_CONTROL_CONFIG_KEYS.values()
        if os.environ.get(env_key) is not None
    }
    return {
        "configured": {
            key: config.get(key)
            for key in OLLAMA_RUNTIME_CONTROL_CONFIG_KEYS
            if config.get(key) is not None
        },
        "exported_env": exported_env,
        # Deprecated alias for one transition cycle.
        "effective_env": dict(exported_env),
        "server_env_notice": (
            "OLLAMA_* server environment variables affect the Ollama server only if "
            "the server was started from this environment. Native per-request "
            "num_ctx, num_predict, and keep_alive controls are sent in request payloads."
        ),
    }


def _ollama_native_probe_options(config: Dict[str, Any]) -> Dict[str, Any]:
    options: Dict[str, Any] = {"num_predict": 8, "temperature": 0}
    num_ctx = config.get("ollama_runtime_num_ctx")
    if num_ctx is not None:
        try:
            parsed = int(num_ctx)
        except Exception:
            parsed = 0
        if parsed > 0:
            options["num_ctx"] = parsed
    return options


def _ollama_preflight_probe(config: Dict, model_name: str, timeout_sec: float) -> Dict:
    api_base = str(config.get("OLLAMA_API_BASE", "http://localhost:11434/v1"))
    api_key = str(config.get("OLLAMA_API_KEY", "ollama"))
    think_mode = normalize_ollama_think_mode(config.get("OLLAMA_THINK_MODE", "auto"))
    native_resolution = resolve_ollama_native_chat_mode(
        model_name,
        config.get("OLLAMA_USE_NATIVE_CHAT", "auto"),
        think_mode,
    )
    use_native_chat = native_resolution.effective_native_chat
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = "Reply with exactly: 4"
    transport = "native_api_chat" if use_native_chat else "openai_compat_v1"
    if use_native_chat:
        url = _ollama_native_chat_url(api_base)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": _ollama_native_probe_options(config),
        }
        if config.get("ollama_runtime_keep_alive") is not None:
            payload["keep_alive"] = str(config.get("ollama_runtime_keep_alive"))
        if think_mode == "think":
            payload["think"] = True
        elif think_mode == "no_think":
            payload["think"] = False
    else:
        url = _ollama_v1_chat_completions_url(api_base)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 8,
            "temperature": 0,
        }
    started = time.time()
    response = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
    response.raise_for_status()
    elapsed_sec = round(time.time() - started, 3)
    text_preview = ""
    try:
        data = response.json()
        if use_native_chat:
            msg = data.get("message", {}) or {}
            text_preview = str(msg.get("content", "")).strip()
        else:
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message", {}) or {}
                text_preview = str(message.get("content", "")).strip()
    except Exception:
        text_preview = ""
    return {
        "model": model_name,
        "ok": True,
        "transport": transport,
        "ollama_native_chat_configured": native_resolution.configured_mode,
        "ollama_native_chat_effective": bool(native_resolution.effective_native_chat),
        "ollama_native_chat_resolution_reason": native_resolution.reason,
        "elapsed_sec": elapsed_sec,
        "timeout_sec": float(timeout_sec),
        "response_preview": text_preview[:80],
    }


def _run_ollama_preflight(
    config: Dict,
    model_names: List[str],
    timeout_sec: float,
    quiet_mode: bool = False,
) -> List[Dict]:
    results: List[Dict] = []
    failures: List[Dict] = []
    if not quiet_mode:
        print(
            f"[cyan]Ollama preflight:[/cyan] probing {len(model_names)} model(s) "
            f"with timeout={timeout_sec:.1f}s"
    )
    for model_name in model_names:
        try:
            probe = _ollama_preflight_probe(config, model_name, timeout_sec)
            preflight_unload = _maybe_stop_ollama_after_preflight(config, model_name)
            if preflight_unload.get("attempted"):
                probe["ollama_preflight_unload"] = preflight_unload
            results.append(probe)
            if not quiet_mode:
                preview = probe.get("response_preview") or "<empty>"
                print(
                    f"  Preflight OK | model={model_name} | transport={probe['transport']} "
                    f"| elapsed={probe['elapsed_sec']}s | preview={escape(preview)}"
                )
        except Exception as exc:
            status_code = None
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                status_code = exc.response.status_code
            think_mode = normalize_ollama_think_mode(config.get("OLLAMA_THINK_MODE", "auto"))
            native_resolution = resolve_ollama_native_chat_mode(
                model_name,
                config.get("OLLAMA_USE_NATIVE_CHAT", "auto"),
                think_mode,
            )
            failure = {
                "model": model_name,
                "ok": False,
                "transport": "native_api_chat"
                if native_resolution.effective_native_chat
                else "openai_compat_v1",
                "ollama_native_chat_configured": native_resolution.configured_mode,
                "ollama_native_chat_effective": bool(native_resolution.effective_native_chat),
                "ollama_native_chat_resolution_reason": native_resolution.reason,
                "timeout_sec": float(timeout_sec),
                "error": f"{type(exc).__name__}: {exc}",
                "status_code": status_code,
            }
            preflight_unload = _maybe_stop_ollama_after_preflight(config, model_name)
            if preflight_unload.get("attempted"):
                failure["ollama_preflight_unload"] = preflight_unload
            results.append(failure)
            failures.append(failure)
    return results


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
    nonempty_lines = [line.strip() for line in response_content.splitlines() if line.strip()]
    first_line_match = (
        STRICT_RESPONSE_FIRST_LINE_PATTERN.fullmatch(nonempty_lines[0])
        if nonempty_lines
        else None
    )
    strict_match = first_line_match or STRICT_RESPONSE_PATTERN.search(response_content)
    direct_action_parseable = False
    parsed_action = None

    if first_line_match:
        parsed_action = int(first_line_match.group(1))
        direct_action_parseable = True
    elif has_delimiter:
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
    stop_threshold_mps: float,
    near_stop_threshold_mps: float,
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
    stopped = False
    near_stop = False

    try:
        uenv = env.unwrapped
        ego = getattr(uenv, "vehicle", None)
        road = getattr(uenv, "road", None)
        if ego is not None:
            ego_speed_mps = float(getattr(ego, "speed", 0.0))
            stopped = bool(ego_speed_mps <= stop_threshold_mps)
            near_stop = bool(ego_speed_mps <= near_stop_threshold_mps)
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
        "stopped": stopped,
        "near_stop": near_stop,
    }


def _scientific_attempt_guard(
    function: Callable[..., Dict],
) -> Callable[..., Dict]:
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Dict:
        bound = signature.bind_partial(*args, **kwargs)
        runtime = bound.arguments.get("scientific_runtime")
        if not isinstance(runtime, ScientificEpisodeRuntime):
            return function(*args, **kwargs)
        runtime.begin_attempt()
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            _abort_scientific_attempt(runtime, exc)
            raise
        try:
            references = runtime.current_trace_references()
            runtime.complete_attempt(references, result=result)
        except Exception as exc:
            _abort_scientific_attempt(runtime, exc)
            raise
        return result

    return wrapped


def _abort_scientific_attempt(
    runtime: ScientificEpisodeRuntime,
    error: Exception,
) -> None:
    references = runtime.cached_trace_references()
    try:
        runtime.abort_attempt(error, references)
    except Exception as terminal_error:
        raise terminal_error from error


def _scientific_protocol_error(
    code: ProtocolInvariantCode,
    message: str,
) -> RuntimeProtocolError:
    return RuntimeProtocolError(ProtocolInvariantViolation.from_mapping(code, message))


@_scientific_attempt_guard
def run_episode(
    config: Dict,
    env_config: Dict,
    env_type: str,
    agent_memory: Optional[DrivingMemory],
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
    stop_threshold_mps: float,
    near_stop_threshold_mps: float,
    alignment_sample_rate: float,
    alignment_max_samples: int,
    slow_decision_threshold_sec: float,
    timeout_penalty_state: Optional[Dict] = None,
    save_artifacts: bool = False,
    record_video: bool = False,
    run_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    quiet_mode: bool = False,
    enable_db_logging: bool = True,
    on_step: Optional[Callable[[int, bool], None]] = None,
    on_decision: Optional[Callable[[int, int, str, Dict], None]] = None,
    benchmark_case: Optional[Dict] = None,
    driving_instruction: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    timeout_early_stop_policy: Optional[Dict[str, Any]] = None,
    execution_mode: ExecutionMode = ExecutionMode.SHIELDED,
    shield_config: Optional[ShieldConfig] = None,
    scientific_trace_writer: Optional[ScientificTraceWriter] = None,
    scientific_trace_record_factory: Optional[
        Callable[..., DecisionTraceRecord]
    ] = None,
    scientific_runtime: Optional[ScientificEpisodeRuntime] = None,
) -> Dict:
    env = None
    if scientific_runtime is not None:
        if not isinstance(scientific_runtime, ScientificEpisodeRuntime):
            raise ValueError("scientific_runtime must be ScientificEpisodeRuntime.")
        if (
            scientific_trace_writer is not None
            or scientific_trace_record_factory is not None
        ):
            raise ValueError(
                "Scientific runtime owns its trace writer and record factory."
            )
        scientific_runtime.validate_binding()
        if seed != scientific_runtime.identity.simulator_seed:
            raise ValueError("Simulator seed drifted from the scientific runtime.")
        if few_shot_num != 0:
            raise ValueError("Scientific runtime requires zero few-shot examples.")
        scientific_trace_writer = scientific_runtime.trace_writer
        timeout_penalty_state = None
        execution_mode = scientific_runtime.harness_config.condition.execution_mode
        shield_config = scientific_runtime.harness_config.shield
        model_name = scientific_runtime.model_tag
    if scientific_runtime is None and (scientific_trace_writer is None) != (
        scientific_trace_record_factory is None
    ):
        raise ValueError(
            "Scientific trace writer and record factory must be provided together."
        )
    if benchmark_case is not None:
        case_slug = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(benchmark_case.get("case_id") or seed)
        ).strip("_")
        result_prefix = f"{case_slug}_seed_{seed}"
    else:
        result_prefix = f"highway_seed_{seed}"
    env_snapshot = None
    if isinstance(env_config, dict):
        env_snapshot = (
            env_config.get(env_type)
            if isinstance(env_config.get(env_type), dict)
            else None
        )
    episode_max_steps = int(
        max_steps_override or _resolve_simulation_duration(config, env_snapshot)
    )
    if save_artifacts or record_video:
        if not run_dir:
            raise ValueError(
                "run_dir is required when save_artifacts or record_video is enabled."
            )
        ensure_dir(run_dir)
    if save_artifacts:
        database_path = os.path.join(run_dir, f"{result_prefix}.db")
    elif enable_db_logging:
        database_path = os.path.join(temp_dir, f"eval_{seed}_{int(time.time() * 1000)}.db")
    else:
        database_path = ""
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
    ollama_native_chat_configured = None
    ollama_native_chat_effective_seen = set()
    ollama_native_chat_resolution_reasons: Counter = Counter()
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
    min_ego_speed_mps = None
    stop_steps = 0
    near_stop_steps = 0
    lane_change_count = 0
    lane_change_shield_count = 0
    unsafe_lane_change_attempt_count = 0
    longitudinal_safety_shield_count = 0
    unsafe_longitudinal_action_attempt_count = 0
    flow_recovery_shield_count = 0
    semantic_recovery_count = 0
    intent_resolver_used_count = 0
    intent_resolver_recovery_count = 0
    intent_resolver_abstain_count = 0
    flap_accel_decel_count = 0
    prev_action_id = None
    alignment_samples = []
    run_action_trace = []
    scientific_trace_references = []
    fallback_reason_counts: Counter = Counter()
    runtime_parse_path_counts: Counter = Counter()
    lane_change_shield_reason_counts: Counter = Counter()
    longitudinal_safety_shield_reason_counts: Counter = Counter()
    flow_recovery_reason_counts: Counter = Counter()
    semantic_recovery_label_counts: Counter = Counter()
    timeout_phase_counts: Counter = Counter()
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
    timeout_policy_mode = None
    timeout_level_initial_sec = None
    timeout_level_final_sec = None
    timeout_level_max_sec = None
    timeout_escalation_count = 0
    timeout_recovery_count = 0
    timeout_level_counts = {15.0: 0, 20.0: 0, 30.0: 0}
    timeout_early_stop_triggered = False
    timeout_early_stop_reason = None
    timeout_early_stop_step = None
    consecutive_timeout_fallbacks = 0
    benchmark_evaluator: Optional[BenchmarkEpisodeEvaluator] = None
    scenario_spec_metadata: Dict[str, Any] = {
        "benchmark_scenario_spec_applied": False,
        "benchmark_scenario_spec": {},
        "benchmark_scenario_vehicle_count": None,
    }
    benchmark_applied_event_ids = set()

    try:
        env = gym.make(env_type, render_mode="rgb_array")
        env.unwrapped.configure(env_config[env_type])
        if record_video:
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
        if benchmark_case is not None:
            scenario_spec_metadata = apply_highway_scenario_spec(env, benchmark_case)
        final_info = info

        sce = EnvScenario(
            env,
            env_type,
            seed,
            database_path or None,
            enable_db=bool(enable_db_logging or save_artifacts),
        )
        agent = DriverAgent(
            sce,
            verbose=True,
            scientific_runtime=scientific_runtime,
        )
        if scientific_runtime is not None:

            def _runtime_trace_record_factory(**kwargs):
                available = getattr(
                    agent,
                    "last_scientific_available_action_ids",
                    None,
                )
                if not isinstance(available, tuple):
                    raise _scientific_protocol_error(
                        ProtocolInvariantCode.TRACE_EVIDENCE_MISSING,
                        "Scientific action availability was not retained.",
                    )
                return scientific_runtime.build_trace_record(
                    available_action_ids=available,
                    **kwargs,
                )

            scientific_trace_record_factory = _runtime_trace_record_factory
        if benchmark_case is not None:
            benchmark_evaluator = BenchmarkEpisodeEvaluator(
                benchmark_case,
                env,
                scenario_spec_metadata=scenario_spec_metadata,
            )
            episode_max_steps = int(max_steps_override or benchmark_evaluator.max_steps)
        initial_penalty_snapshot = decision_timeout_penalty_snapshot(
            timeout_penalty_state
        )
        timeout_policy_mode = initial_penalty_snapshot.get("policy_mode")
        timeout_level_initial_sec = initial_penalty_snapshot.get(
            "effective_decision_timeout_sec"
        )
        timeout_level_final_sec = timeout_level_initial_sec
        timeout_level_max_sec = timeout_level_initial_sec
        timeout_level_cap_sec = None
        ladder_levels = initial_penalty_snapshot.get("timeout_ladder_sec") or []
        if ladder_levels:
            timeout_level_cap_sec = max(float(value) for value in ladder_levels)
        if (
            initial_penalty_snapshot.get("enabled")
            and initial_penalty_snapshot.get("effective_decision_timeout_sec")
            is not None
        ):
            try:
                agent.set_decision_timeout_sec(
                    float(initial_penalty_snapshot["effective_decision_timeout_sec"])
                )
            except Exception:
                pass

        prev_action = "Not available"
        effective_driving_instruction = (
            driving_instruction or "Drive safely and avoid collisons"
        )
        for frame_id in range(episode_max_steps):
            benchmark_event_meta: Dict[str, Any] = {
                "benchmark_events_applied": False,
                "benchmark_event_ids": [],
                "benchmark_event_types": [],
                "benchmark_event_step": int(frame_id + 1),
                "benchmark_events": [],
            }
            if benchmark_case is not None:
                benchmark_event_meta = apply_highway_scenario_events(
                    env,
                    benchmark_case,
                    step_idx=int(frame_id + 1),
                    applied_event_ids=benchmark_applied_event_ids,
                )
            _ = np.array(obs, dtype=float)

            fewshot_results = (
                agent_memory.retriveMemory(sce, frame_id, few_shot_num)
                if (few_shot_num > 0 and agent_memory is not None)
                else []
            )
            fewshot_messages = [x["human_question"] for x in fewshot_results]
            fewshot_answers = [x["LLM_response"] for x in fewshot_results]

            sce_descrip = sce.describe(frame_id)
            avail_action = sce.availableActionsDescription()
            if scientific_runtime is not None:
                agent.scientific_generation_context = (
                    scientific_runtime.generation_context(frame_id)
                )
            try:
                action, response, human_question, fewshot_answer = (
                    agent.few_shot_decision(
                        scenario_description=sce_descrip,
                        available_actions=avail_action,
                        previous_decisions=prev_action,
                        fewshot_messages=fewshot_messages,
                        driving_intensions=effective_driving_instruction,
                        fewshot_answers=fewshot_answers,
                    )
                )
            except ScientificGenerationAbort as exc:
                if scientific_trace_writer is not None:
                    prompt_artifact = getattr(agent, "last_prompt_artifact", None)
                    if not isinstance(prompt_artifact, PromptArtifact):
                        raise _scientific_protocol_error(
                            ProtocolInvariantCode.TRACE_EVIDENCE_MISSING,
                            "Blocked generation is missing its prompt artifact.",
                        )
                    blocked_record = _compose_scientific_trace_record(
                        scientific_trace_record_factory,
                        decision_index=int(frame_id),
                        env_step_index=int(frame_id),
                        prompt_artifact=prompt_artifact,
                        generation=exc.result,
                        resolution=None,
                        shield_stack=None,
                        disposition=TraceDisposition.BLOCKED_BEFORE_EXECUTION,
                        decision_latency_ms=exc.result.latency_ms,
                        benchmark_event_meta=benchmark_event_meta,
                    )
                    trace_reference = scientific_trace_writer.append(blocked_record)
                    scientific_trace_references.append(trace_reference.to_dict())
                    raise ScientificGenerationAbort(
                        exc.result,
                        trace_reference=trace_reference,
                    ) from exc
                raise
            prev_action = action
            level_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
            active_timeout_level = level_snapshot.get("effective_decision_timeout_sec")
            if active_timeout_level is not None:
                active_timeout_level = float(active_timeout_level)
                if active_timeout_level in timeout_level_counts:
                    timeout_level_counts[active_timeout_level] += 1
                timeout_level_max_sec = (
                    active_timeout_level
                    if timeout_level_max_sec is None
                    else max(float(timeout_level_max_sec), active_timeout_level)
                )
            decision_calls_total += 1
            decisions_made += 1
            decision_meta = getattr(agent, "last_decision_meta", {}) or {}
            timed_out = bool(decision_meta.get("timed_out", False))
            used_fallback = bool(decision_meta.get("used_fallback", False))
            ollama_requested_think_mode = decision_meta.get("ollama_requested_think_mode") or ollama_requested_think_mode
            ollama_native_chat_configured = (
                decision_meta.get("ollama_native_chat_configured")
                or ollama_native_chat_configured
            )
            if decision_meta.get("ollama_native_chat_effective") is not None:
                ollama_native_chat_effective_seen.add(
                    bool(decision_meta.get("ollama_native_chat_effective"))
                )
            _bump_counter(
                ollama_native_chat_resolution_reasons,
                decision_meta.get("ollama_native_chat_resolution_reason"),
            )
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
            _bump_counter(runtime_parse_path_counts, decision_meta.get("runtime_parse_path"))
            if used_fallback:
                _bump_counter(
                    fallback_reason_counts,
                    decision_meta.get("fallback_reason") or decision_meta.get("runtime_parse_path"),
                )
            semantic_recovery_used = bool(decision_meta.get("semantic_recovery_used", False))
            intent_resolver_used = bool(decision_meta.get("intent_resolver_used", False))
            intent_resolver_abstained = bool(decision_meta.get("intent_resolver_abstained", False))
            runtime_parse_path = str(decision_meta.get("runtime_parse_path") or "")
            semantic_recovery_count += int(semantic_recovery_used)
            intent_resolver_used_count += int(intent_resolver_used)
            intent_resolver_recovery_count += int(intent_resolver_used and runtime_parse_path == "intent_resolver_direct")
            intent_resolver_abstain_count += int(intent_resolver_used and intent_resolver_abstained)
            if semantic_recovery_used:
                _bump_counter(semantic_recovery_label_counts, decision_meta.get("semantic_recovery_label"))
            consecutive_timeout_fallbacks = (
                consecutive_timeout_fallbacks + 1
                if (timed_out and used_fallback)
                else 0
            )
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
            post_penalty_snapshot = decision_timeout_penalty_snapshot(timeout_penalty_state)
            decision_meta.update(
                {
                    "timeout_phase": post_penalty_snapshot.get("timeout_phase"),
                    "warmup_active": bool(post_penalty_snapshot.get("warmup_active", False)),
                    "warmup_success_observed": bool(
                        post_penalty_snapshot.get("warmup_success_observed", False)
                    ),
                    "warmup_timeout_failures": int(
                        post_penalty_snapshot.get("warmup_timeout_failures", 0) or 0
                    ),
                    "warmup_timeout_sec": post_penalty_snapshot.get("warmup_timeout_sec"),
                    "ego_speed_mps": _scenario_ego_speed_mps(sce),
                }
            )
            decision_meta.update(benchmark_event_meta)
            _bump_counter(timeout_phase_counts, decision_meta.get("timeout_phase"))
            timeout_penalty_stage_max = max(timeout_penalty_stage_max, int(penalty_update.get("stage", 0)))
            if penalty_update.get("escalated") or penalty_update.get("recovered") or penalty_update.get("warmup_completed"):
                effective_decision_timeout_sec = penalty_update.get("effective_decision_timeout_sec")
                if effective_decision_timeout_sec is not None:
                    try:
                        agent.set_decision_timeout_sec(float(effective_decision_timeout_sec))
                    except Exception:
                        pass
            if penalty_update.get("warmup_completed") and not quiet_mode:
                effective_decision_timeout_sec = penalty_update.get("effective_decision_timeout_sec")
                print(
                    "[green]Eval timeout warmup completed[/green] "
                    f"(phase=active, decision_timeout="
                    f"{round(float(effective_decision_timeout_sec), 3) if effective_decision_timeout_sec is not None else 'n/a'}s)"
                )
            if penalty_update.get("escalated"):
                timeout_escalation_count += 1
                if not quiet_mode:
                    print(
                        "[yellow]Eval timeout policy escalated[/yellow] "
                        f"(reason={penalty_update.get('reason')}, stage={penalty_update.get('stage')}, "
                        f"decision_timeout={round(float(effective_decision_timeout_sec), 3) if effective_decision_timeout_sec is not None else 'n/a'}s)"
                    )
            if penalty_update.get("recovered"):
                timeout_recovery_count += 1
                if not quiet_mode:
                    effective_decision_timeout_sec = penalty_update.get("effective_decision_timeout_sec")
                    print(
                        "[green]Eval timeout policy recovered[/green] "
                        f"(stage={penalty_update.get('stage')}, "
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
            proposed_action = (
                decision_meta["original_selected_action"]
                if "original_selected_action" in decision_meta
                else action
            )
            shield_stack = _execute_reactive_safety_shield_stack(
                action,
                sce,
                execution_mode=execution_mode,
                proposed_action_id=proposed_action,
                shield_config=shield_config,
            )
            action = shield_stack.executed_action_id
            shield_meta = shield_stack.to_metadata()
            _update_shield_decision_metadata(decision_meta, shield_stack, shield_meta)
            benchmark_action_context = dict(shield_meta)
            benchmark_action_context.update(benchmark_event_meta)
            benchmark_action_context.update(
                {
                    "action_id": int(action),
                    "final_action_id": int(action),
                    "runtime_parse_path": decision_meta.get("runtime_parse_path"),
                    "fallback_reason": decision_meta.get("fallback_reason"),
                }
            )
            lane_change_shield_count += int(bool(shield_meta.get("lane_change_shield_applied", False)))
            unsafe_lane_change_attempt_count += int(bool(shield_meta.get("lane_change_shield_applied", False)))
            longitudinal_safety_shield_count += int(
                bool(shield_meta.get("longitudinal_safety_shield_applied", False))
            )
            unsafe_longitudinal_action_attempt_count += int(
                bool(shield_meta.get("longitudinal_safety_shield_applied", False))
            )
            flow_recovery_shield_count += int(bool(shield_meta.get("flow_recovery_shield_applied", False)))
            if shield_meta.get("lane_change_shield_applied"):
                _bump_counter(lane_change_shield_reason_counts, shield_meta.get("lane_change_shield_reason"))
            if shield_meta.get("longitudinal_safety_shield_applied"):
                _bump_counter(
                    longitudinal_safety_shield_reason_counts,
                    shield_meta.get("longitudinal_safety_shield_reason"),
                )
            if shield_meta.get("flow_recovery_shield_applied"):
                _bump_counter(
                    flow_recovery_reason_counts, shield_meta.get("flow_recovery_reason")
                )
            if save_artifacts:
                try:
                    pre_step_metrics = extract_step_traffic_metrics(
                        env,
                        ttc_threshold_sec,
                        headway_threshold_m,
                        rear_ttc_threshold_sec,
                        rear_headway_threshold_m,
                        low_speed_blocking_threshold_mps,
                        blocking_front_gap_safe_m,
                        blocking_front_ttc_safe_sec,
                        stop_threshold_mps,
                        near_stop_threshold_mps,
                    )
                except Exception:
                    pre_step_metrics = {}
                if pre_step_metrics.get("ego_speed_mps") is not None:
                    decision_meta["ego_speed_mps"] = pre_step_metrics.get(
                        "ego_speed_mps"
                    )
                run_action_trace.append(
                    _decision_trace_item(
                        step_idx=int(frame_id + 1),
                        action_id=int(action),
                        response_text=response,
                        decision_meta=decision_meta,
                    )
                )
            if on_decision is not None:
                try:
                    on_decision(
                        int(frame_id + 1), int(action), response, dict(decision_meta)
                    )
                except Exception:
                    pass
            lane_change_count += int(action in (0, 2))
            if prev_action_id is not None and (
                (prev_action_id == 3 and action == 4)
                or (prev_action_id == 4 and action == 3)
            ):
                flap_accel_decel_count += 1
            prev_action_id = action

            if (
                alignment_sample_rate > 0
                and len(alignment_samples) < alignment_max_samples
                and random.random() < alignment_sample_rate
            ):
                alignment_samples.append(
                    {
                        "scenario_summary": (sce_descrip or "")[:800],
                        "model_response": response,
                        "action_id": int(action),
                        "step_idx": int(frame_id),
                        "seed": int(seed),
                    }
                )

            if scientific_trace_writer is not None:
                prompt_artifact = getattr(agent, "last_prompt_artifact", None)
                generation = getattr(agent, "last_generation_result", None)
                resolution = getattr(agent, "last_action_resolution", None)
                if (
                    not isinstance(prompt_artifact, PromptArtifact)
                    or not isinstance(generation, GenerationResult)
                    or not isinstance(resolution, ActionResolutionResult)
                ):
                    raise _scientific_protocol_error(
                        ProtocolInvariantCode.TRACE_EVIDENCE_MISSING,
                        "Scientific decision is missing typed runtime evidence.",
                    )
                trace_record = _compose_scientific_trace_record(
                    scientific_trace_record_factory,
                    decision_index=int(frame_id),
                    env_step_index=int(frame_id),
                    prompt_artifact=prompt_artifact,
                    generation=generation,
                    resolution=resolution,
                    shield_stack=shield_stack,
                    disposition=TraceDisposition.READY_FOR_ENV_STEP,
                    decision_latency_ms=max(
                        generation.latency_ms,
                        decision_elapsed_sec * 1000.0,
                    ),
                    benchmark_event_meta=benchmark_event_meta,
                )
                trace_reference, step_result = append_trace_before_step(
                    scientific_trace_writer,
                    trace_record,
                    env.step,
                )
                scientific_trace_references.append(trace_reference.to_dict())
                obs, reward, terminated, truncated, info = step_result
            else:
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
                stop_threshold_mps,
                near_stop_threshold_mps,
            )
            if step_metrics["ego_speed_mps"] is not None:
                ego_speed_sum += float(step_metrics["ego_speed_mps"])
                ego_speed_count += 1
                if min_ego_speed_mps is None:
                    min_ego_speed_mps = float(step_metrics["ego_speed_mps"])
                else:
                    min_ego_speed_mps = min(min_ego_speed_mps, float(step_metrics["ego_speed_mps"]))
            ttc_danger_steps += int(step_metrics["ttc_danger"])
            headway_violation_steps += int(step_metrics["headway_violation"])
            rear_ttc_danger_steps += int(step_metrics["rear_ttc_danger"])
            rear_headway_violation_steps += int(step_metrics["rear_headway_violation"])
            low_speed_blocking_steps += int(step_metrics["low_speed_blocking"])
            stop_steps += int(step_metrics["stopped"])
            near_stop_steps += int(step_metrics["near_stop"])
            if benchmark_evaluator is not None:
                benchmark_evaluator.update(
                    env,
                    step_idx=int(steps),
                    step_metrics=step_metrics,
                    crashed=crashed,
                    info=info,
                    action_context=benchmark_action_context,
                )

            if enable_db_logging or save_artifacts:
                try:
                    sce.promptsCommit(frame_id, None, done, human_question, fewshot_answer, response)
                except Exception:
                    pass

            if done:
                break

            should_early_stop, early_stop_reason = _should_early_stop_timeout_episode(
                decision_calls_total=decision_calls_total,
                decision_timeout_count=decision_timeout_count,
                fallback_action_count=fallback_action_count,
                consecutive_timeout_fallbacks=consecutive_timeout_fallbacks,
                timeout_policy_mode=timeout_policy_mode,
                active_timeout_level=penalty_update.get("effective_decision_timeout_sec"),
                max_timeout_level=timeout_level_cap_sec,
                policy=timeout_early_stop_policy,
            )
            if should_early_stop and not crashed:
                timeout_early_stop_triggered = True
                timeout_early_stop_reason = str(early_stop_reason)
                timeout_early_stop_step = int(steps)
                truncated = True
                episode_stop_reason = "episode_timeout_cap"
                if not quiet_mode:
                    print(
                        "[yellow]Timeout early-stop triggered[/yellow] "
                        f"(reason={timeout_early_stop_reason}, step={timeout_early_stop_step}, "
                        f"decision_timeouts={decision_timeout_count}/{decision_calls_total}, "
                        f"fallbacks={fallback_action_count}/{decision_calls_total})"
                    )
                break

    except (
        RuntimeProtocolError,
        ScientificGenerationAbort,
        ScientificSimulatorAbort,
        ScientificTraceWriteError,
    ) as exc:
        raise
    except Exception as exc:
        if scientific_runtime is not None:
            raise
        error = f"{type(exc).__name__}: {exc}"
        episode_stop_reason = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if (not save_artifacts) and database_path and os.path.exists(database_path):
            try:
                os.remove(database_path)
            except Exception:
                pass
        if save_artifacts or record_video:
            gc.collect()

    duration_sec = time.time() - started
    episode_reward_avg = episode_reward_sum / max(steps, 1)
    avg_ego_speed_mps = ego_speed_sum / max(ego_speed_count, 1)
    ttc_danger_rate = ttc_danger_steps / max(steps, 1)
    headway_violation_rate = headway_violation_steps / max(steps, 1)
    rear_ttc_danger_rate = rear_ttc_danger_steps / max(steps, 1)
    rear_headway_violation_rate = rear_headway_violation_steps / max(steps, 1)
    low_speed_blocking_rate = low_speed_blocking_steps / max(steps, 1)
    stop_rate = stop_steps / max(steps, 1)
    near_stop_rate = near_stop_steps / max(steps, 1)
    lane_change_rate = lane_change_count / max(steps, 1)
    lane_change_shield_rate = lane_change_shield_count / max(decision_calls_total, 1)
    longitudinal_safety_shield_rate = longitudinal_safety_shield_count / max(decision_calls_total, 1)
    flow_recovery_shield_rate = flow_recovery_shield_count / max(decision_calls_total, 1)
    semantic_recovery_rate = semantic_recovery_count / max(decision_calls_total, 1)
    intent_resolver_used_rate = intent_resolver_used_count / max(decision_calls_total, 1)
    intent_resolver_recovery_rate = intent_resolver_recovery_count / max(decision_calls_total, 1)
    intent_resolver_abstain_rate = intent_resolver_abstain_count / max(decision_calls_total, 1)
    flap_accel_decel_rate = flap_accel_decel_count / max(steps, 1)
    format_failure_rate = format_failure_count / max(decisions_made, 1)
    decision_timeout_rate = decision_timeout_count / max(decision_calls_total, 1)
    fallback_action_rate = fallback_action_count / max(decision_calls_total, 1)
    ollama_native_retry_rate = ollama_native_retry_count / max(decision_calls_total, 1)
    ollama_openai_fallback_rate = ollama_openai_fallback_count / max(decision_calls_total, 1)
    latency_stats = _summarize_decision_latency_samples(decision_latencies_sec)
    decision_latency_ms_avg = latency_stats["decision_latency_ms_avg"]
    p95_decision_latency_sec = latency_stats["p95_decision_latency_sec"]
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
    timeout_level_final_sec = penalty_snapshot.get("effective_decision_timeout_sec")
    if timeout_level_final_sec is not None:
        timeout_level_final_sec = float(timeout_level_final_sec)
    if timeout_level_initial_sec is not None:
        timeout_level_initial_sec = float(timeout_level_initial_sec)
    if timeout_level_max_sec is not None:
        timeout_level_max_sec = float(timeout_level_max_sec)
    timeout_level_15_rate = timeout_level_counts[15.0] / max(decision_calls_total, 1)
    timeout_level_20_rate = timeout_level_counts[20.0] / max(decision_calls_total, 1)
    timeout_level_30_rate = timeout_level_counts[30.0] / max(decision_calls_total, 1)

    if episode_stop_reason != "error":
        if episode_stop_reason == "episode_timeout_cap":
            pass
        elif crashed:
            episode_stop_reason = "crash"
        elif truncated:
            episode_stop_reason = "truncated"
        elif terminated:
            episode_stop_reason = "terminated"
        else:
            episode_stop_reason = "completed"

    if benchmark_case is not None:
        if benchmark_evaluator is not None:
            benchmark_metrics = benchmark_evaluator.finalize(
                crashed=crashed,
                episode_stop_reason=episode_stop_reason,
            )
        else:
            benchmark_metrics = {
                "case_id": str(benchmark_case.get("case_id")),
                "instruction": str(benchmark_case.get("instruction", "")),
                "category": str(benchmark_case.get("category", "")),
                "tags": list(benchmark_case.get("tags") or []),
                "time_limit_sec": round(
                    float(benchmark_case.get("time_limit_sec") or 0.0), 3
                ),
                "benchmark_case_env_overrides": copy.deepcopy(
                    benchmark_case.get("env_overrides") or {}
                ),
                "benchmark_success_criteria": copy.deepcopy(
                    benchmark_case.get("success_criteria") or {}
                ),
                "benchmark_scenario_spec": copy.deepcopy(
                    scenario_spec_metadata.get("benchmark_scenario_spec") or {}
                ),
                "benchmark_scenario_spec_applied": bool(
                    scenario_spec_metadata.get("benchmark_scenario_spec_applied", False)
                ),
                "benchmark_initial_lane_rank": None,
                "benchmark_initial_front_gap_m": None,
                "benchmark_completion_step": None,
                "benchmark_completion_time_sec": None,
                "benchmark_overtake_latency_steps": None,
                "benchmark_recovery_clear_step": None,
                "benchmark_recovery_time_steps": None,
                "benchmark_unsafe_lane_change_attempts": 0,
                "benchmark_lane_change_count": 0,
                "benchmark_flap_accel_decel_count": 0,
                "benchmark_low_speed_blocking_steps": 0,
                "task_completed": False,
                "completion_rate": 0.0,
                "ttc_score": 0.0,
                "speed_variance_score": 0.0,
                "time_efficiency_score": 0.0,
                "overall_score": 0.0,
                "driving_score": 0.0,
                "benchmark_failure_reason": (
                    episode_stop_reason if error is None else "episode_error"
                ),
                "benchmark_speed_std_mps": None,
                "benchmark_min_positive_ttc_sec": None,
                "benchmark_max_progress_m": 0.0,
            }
    else:
        benchmark_metrics = {}

    if scientific_runtime is not None:
        condition = scientific_runtime.harness_config.condition
        identity = scientific_runtime.identity
        prompt_profile = condition.policy_content.value
        scientific_identity = {
            "campaign_id": identity.campaign_id,
            "episode_attempt_id": identity.episode_attempt_id,
            "condition_id": scientific_runtime.harness_config.condition_id(),
            "config_sha256": scientific_runtime.runtime_lock.config_sha256,
            "model_digest": scientific_runtime.model_digest,
            "policy_content": condition.policy_content.value,
            "output_enforcement": condition.output_enforcement.value,
            "execution_mode": condition.execution_mode.value,
            "runtime_lock_source_artifact_sha256": (
                scientific_runtime.runtime_lock.source_artifact_sha256
            ),
        }
    else:
        prompt_profile = (
            str(config.get("eval_prompt_profile", "harness_v2") or "harness_v2")
            .strip()
            .lower()
            .replace("-", "_")
        )
        scientific_identity = {}

    episode_result = {
        "seed": seed,
        "prompt_profile": prompt_profile,
        "steps": steps,
        "max_steps": int(episode_max_steps),
        "crashed": crashed,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success_no_collision": (error is None and not crashed),
        "episode_runtime_sec": round(duration_sec, 3),
        "avg_step_runtime_sec": round(duration_sec / max(steps, 1), 3),
        "episode_stop_reason": episode_stop_reason,
        "timeout_triggered": bool(timeout_triggered),
        "timeout_early_stop_triggered": bool(timeout_early_stop_triggered),
        "timeout_early_stop_reason": timeout_early_stop_reason,
        "timeout_early_stop_step": timeout_early_stop_step,
        "first_timeout_step": first_timeout_step,
        "decision_calls_total": decision_calls_total,
        "decision_timeout_count": decision_timeout_count,
        "decision_timeout_rate": round(decision_timeout_rate, 4),
        "fallback_action_count": fallback_action_count,
        "fallback_action_rate": round(fallback_action_rate, 4),
        "ollama_native_chat_configured": ollama_native_chat_configured,
        "ollama_native_chat_effective_seen": sorted(ollama_native_chat_effective_seen),
        "ollama_native_chat_resolution_reason_counts": _counter_dict(
            ollama_native_chat_resolution_reasons
        ),
        "ollama_requested_think_mode": ollama_requested_think_mode,
        "ollama_effective_think_modes_seen": sorted(ollama_effective_think_modes_seen),
        "ollama_native_retry_count": ollama_native_retry_count,
        "ollama_native_retry_rate": round(ollama_native_retry_rate, 4),
        "ollama_openai_fallback_count": ollama_openai_fallback_count,
        "ollama_openai_fallback_rate": round(ollama_openai_fallback_rate, 4),
        "ollama_native_decision_count": ollama_native_decision_count,
        "ollama_native_timeout_count": ollama_native_timeout_count,
        "ollama_native_timeout_short_circuit_count": ollama_native_timeout_short_circuit_count,
        "ollama_downgrade_triggered": bool(
            ollama_native_retry_count > 0
            or (
                "auto" in ollama_effective_think_modes_seen
                and ollama_requested_think_mode == "think"
            )
        ),
        "slow_decision_count": int(slow_decision_count),
        "p95_decision_latency_sec": p95_decision_latency_sec,
        "timeout_penalty_stage_max": int(timeout_penalty_stage_max),
        "timeout_penalty_events": int(max(0, timeout_penalty_events)),
        "timeout_penalty_timeout_triggers": int(
            max(0, timeout_penalty_timeout_triggers)
        ),
        "timeout_penalty_slow_triggers": int(max(0, timeout_penalty_slow_triggers)),
        "timeout_penalty_final_decision_timeout_sec": (
            round(float(penalty_snapshot.get("effective_decision_timeout_sec")), 4)
            if penalty_snapshot.get("effective_decision_timeout_sec") is not None
            else None
        ),
        "timeout_policy_mode": timeout_policy_mode,
        "timeout_level_initial_sec": (
            round(float(timeout_level_initial_sec), 4)
            if timeout_level_initial_sec is not None
            else None
        ),
        "timeout_level_final_sec": (
            round(float(timeout_level_final_sec), 4)
            if timeout_level_final_sec is not None
            else None
        ),
        "timeout_level_max_sec": (
            round(float(timeout_level_max_sec), 4)
            if timeout_level_max_sec is not None
            else None
        ),
        "timeout_escalation_count": int(timeout_escalation_count),
        "timeout_recovery_count": int(timeout_recovery_count),
        "timeout_level_15_rate": round(timeout_level_15_rate, 4),
        "timeout_level_20_rate": round(timeout_level_20_rate, 4),
        "timeout_level_30_rate": round(timeout_level_30_rate, 4),
        "timeout_phase": penalty_snapshot.get("timeout_phase"),
        "timeout_phase_counts": _counter_dict(timeout_phase_counts),
        "warmup_active": bool(penalty_snapshot.get("warmup_active", False)),
        "warmup_success_observed": bool(
            penalty_snapshot.get("warmup_success_observed", False)
        ),
        "warmup_timeout_failures": int(
            penalty_snapshot.get("warmup_timeout_failures", 0) or 0
        ),
        "warmup_timeout_sec": (
            round(float(penalty_snapshot.get("warmup_timeout_sec")), 4)
            if penalty_snapshot.get("warmup_timeout_sec") is not None
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
        "fallback_reason_counts": _counter_dict(fallback_reason_counts),
        "runtime_parse_path_counts": _counter_dict(runtime_parse_path_counts),
        "semantic_recovery_count": int(semantic_recovery_count),
        "semantic_recovery_rate": round(semantic_recovery_rate, 4),
        "semantic_recovery_label_counts": _counter_dict(semantic_recovery_label_counts),
        "intent_resolver_used_count": int(intent_resolver_used_count),
        "intent_resolver_used_rate": round(intent_resolver_used_rate, 4),
        "intent_resolver_recovery_count": int(intent_resolver_recovery_count),
        "intent_resolver_recovery_rate": round(intent_resolver_recovery_rate, 4),
        "intent_resolver_abstain_count": int(intent_resolver_abstain_count),
        "intent_resolver_abstain_rate": round(intent_resolver_abstain_rate, 4),
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
        "min_ego_speed_mps": (
            round(float(min_ego_speed_mps), 4)
            if min_ego_speed_mps is not None
            else None
        ),
        "stopped_ever": bool(stop_steps > 0),
        "stop_steps": int(stop_steps),
        "stop_rate": round(stop_rate, 4),
        "near_stop_steps": int(near_stop_steps),
        "near_stop_rate": round(near_stop_rate, 4),
        "lane_change_count": lane_change_count,
        "lane_change_rate": round(lane_change_rate, 4),
        "lane_change_shield_count": int(lane_change_shield_count),
        "lane_change_shield_rate": round(lane_change_shield_rate, 4),
        "lane_change_shield_reason_counts": _counter_dict(
            lane_change_shield_reason_counts
        ),
        "unsafe_lane_change_attempt_count": int(unsafe_lane_change_attempt_count),
        "longitudinal_safety_shield_count": int(longitudinal_safety_shield_count),
        "longitudinal_safety_shield_rate": round(longitudinal_safety_shield_rate, 4),
        "longitudinal_safety_shield_reason_counts": _counter_dict(
            longitudinal_safety_shield_reason_counts
        ),
        "unsafe_longitudinal_action_attempt_count": int(
            unsafe_longitudinal_action_attempt_count
        ),
        "flow_recovery_shield_count": int(flow_recovery_shield_count),
        "flow_recovery_shield_rate": round(flow_recovery_shield_rate, 4),
        "flow_recovery_reason_counts": _counter_dict(flow_recovery_reason_counts),
        "action_trace": run_action_trace if save_artifacts else None,
        "scientific_trace_references": (
            scientific_trace_references if scientific_trace_writer is not None else None
        ),
        "flap_accel_decel_count": flap_accel_decel_count,
        "flap_accel_decel_rate": round(flap_accel_decel_rate, 4),
        "decision_latency_ms_avg": decision_latency_ms_avg,
        "alignment_samples": alignment_samples,
        "model": model_name,
        "database_path": database_path if save_artifacts else None,
        "video_prefix": result_prefix if record_video else None,
        "run_id": run_id if (save_artifacts or record_video) else None,
        "run_dir": run_dir if (save_artifacts or record_video) else None,
        "error": error,
        "final_info": copy.deepcopy(final_info),
        **scientific_identity,
        **benchmark_metrics,
    }
    if benchmark_case is not None:
        episode_result = augment_behavior_aware_benchmark_episode(episode_result)
    try:
        scored_episode = compute_split_scores_for_episode(episode_result)
    except Exception:
        raise
    return scored_episode


def aggregate_results(
    model_name: str,
    episodes: List[Dict],
    *,
    planned_episode_count: Optional[int] = None,
    model_quarantined_due_to_timeout_collapse: bool = False,
    model_quarantine_reason: Optional[str] = None,
    model_skipped_due_to_preflight: bool = False,
    model_skipped_reason: Optional[str] = None,
    primary_metric_spec: Optional[Dict[str, Any]] = None,
) -> Dict:
    total = len(episodes)
    planned_total = max(total, int(planned_episode_count if planned_episode_count is not None else total))
    skipped_episode_count = max(0, planned_total - total)
    episode_execution_complete = skipped_episode_count == 0 and not bool(model_skipped_due_to_preflight)
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
    total_semantic_recoveries = sum(int(e.get("semantic_recovery_count", 0)) for e in episodes)
    semantic_recovery_label_counts = _merge_count_dicts(episodes, "semantic_recovery_label_counts")
    total_intent_resolver_used = sum(int(e.get("intent_resolver_used_count", 0)) for e in episodes)
    total_intent_resolver_recoveries = sum(
        int(e.get("intent_resolver_recovery_count", 0)) for e in episodes
    )
    total_intent_resolver_abstains = sum(
        int(e.get("intent_resolver_abstain_count", 0)) for e in episodes
    )
    total_ollama_native_retries = sum(e.get("ollama_native_retry_count", 0) for e in episodes)
    total_ollama_openai_fallbacks = sum(e.get("ollama_openai_fallback_count", 0) for e in episodes)
    total_ollama_native_decisions = sum(e.get("ollama_native_decision_count", 0) for e in episodes)
    total_ollama_native_timeouts = sum(e.get("ollama_native_timeout_count", 0) for e in episodes)
    ollama_native_chat_configured_values = sorted(
        {
            str(e.get("ollama_native_chat_configured"))
            for e in episodes
            if e.get("ollama_native_chat_configured") is not None
        }
    )
    ollama_native_chat_effective_values = sorted(
        {
            bool(value)
            for e in episodes
            for value in (e.get("ollama_native_chat_effective_seen") or [])
        }
    )
    ollama_native_chat_resolution_reason_counts = _merge_count_dicts(
        episodes,
        "ollama_native_chat_resolution_reason_counts",
    )
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
    fallback_reason_counts = _merge_count_dicts(episodes, "fallback_reason_counts")
    runtime_parse_path_counts = _merge_count_dicts(episodes, "runtime_parse_path_counts")
    total_reward_sum = sum(float(e.get("episode_reward_sum", 0.0)) for e in episodes)
    total_speed = sum(float(e.get("avg_ego_speed_mps", 0.0)) for e in episodes)
    total_ttc_danger_rate = sum(float(e.get("ttc_danger_rate", 0.0)) for e in episodes)
    total_headway_rate = sum(float(e.get("headway_violation_rate", 0.0)) for e in episodes)
    total_rear_ttc_danger_rate = sum(float(e.get("rear_ttc_danger_rate", 0.0)) for e in episodes)
    total_rear_headway_rate = sum(float(e.get("rear_headway_violation_rate", 0.0)) for e in episodes)
    total_low_speed_blocking_rate = sum(float(e.get("low_speed_blocking_rate", 0.0)) for e in episodes)
    total_stop_rate = sum(float(e.get("stop_rate", 0.0)) for e in episodes)
    total_near_stop_rate = sum(float(e.get("near_stop_rate", 0.0)) for e in episodes)
    total_lane_change_rate = sum(float(e.get("lane_change_rate", 0.0)) for e in episodes)
    total_lane_change_shields = sum(int(e.get("lane_change_shield_count", 0)) for e in episodes)
    lane_change_shield_reason_counts = _merge_count_dicts(episodes, "lane_change_shield_reason_counts")
    total_longitudinal_safety_shields = sum(
        int(e.get("longitudinal_safety_shield_count", 0)) for e in episodes
    )
    longitudinal_safety_shield_reason_counts = _merge_count_dicts(
        episodes,
        "longitudinal_safety_shield_reason_counts",
    )
    total_flow_recovery_shields = sum(int(e.get("flow_recovery_shield_count", 0)) for e in episodes)
    flow_recovery_reason_counts = _merge_count_dicts(episodes, "flow_recovery_reason_counts")
    total_unsafe_lane_change_attempts = sum(int(e.get("unsafe_lane_change_attempt_count", 0)) for e in episodes)
    total_unsafe_longitudinal_action_attempts = sum(
        int(e.get("unsafe_longitudinal_action_attempt_count", 0)) for e in episodes
    )
    total_flap_rate = sum(float(e.get("flap_accel_decel_rate", 0.0)) for e in episodes)
    decision_latency_ms_values = [
        float(e.get("decision_latency_ms_avg"))
        for e in episodes
        if e.get("decision_latency_ms_avg") is not None
    ]
    min_ego_speed_values = [
        float(e.get("min_ego_speed_mps"))
        for e in episodes
        if e.get("min_ego_speed_mps") is not None
    ]
    stop_episode_count = sum(1 for e in episodes if e.get("stopped_ever"))
    near_stop_episode_count = sum(1 for e in episodes if float(e.get("near_stop_rate", 0.0) or 0.0) > 0.0)
    total_timeout_penalty_events = sum(int(e.get("timeout_penalty_events", 0)) for e in episodes)
    total_timeout_penalty_timeout_triggers = sum(
        int(e.get("timeout_penalty_timeout_triggers", 0)) for e in episodes
    )
    total_timeout_penalty_slow_triggers = sum(
        int(e.get("timeout_penalty_slow_triggers", 0)) for e in episodes
    )
    total_timeout_escalation_count = sum(int(e.get("timeout_escalation_count", 0)) for e in episodes)
    total_timeout_recovery_count = sum(int(e.get("timeout_recovery_count", 0)) for e in episodes)
    total_timeout_level_15_rate = sum(float(e.get("timeout_level_15_rate", 0.0)) for e in episodes)
    total_timeout_level_20_rate = sum(float(e.get("timeout_level_20_rate", 0.0)) for e in episodes)
    total_timeout_level_30_rate = sum(float(e.get("timeout_level_30_rate", 0.0)) for e in episodes)
    timeout_phase_counts = _merge_count_dicts(episodes, "timeout_phase_counts")
    total_warmup_timeout_failures = sum(int(e.get("warmup_timeout_failures", 0) or 0) for e in episodes)
    warmup_active_episode_count = sum(1 for e in episodes if bool(e.get("warmup_active", False)))
    warmup_success_observed_episode_count = sum(
        1 for e in episodes if bool(e.get("warmup_success_observed", False))
    )
    timeout_penalty_stage_max_values = [int(e.get("timeout_penalty_stage_max", 0)) for e in episodes]
    timeout_penalty_final_values = [
        float(e.get("timeout_penalty_final_decision_timeout_sec"))
        for e in episodes
        if e.get("timeout_penalty_final_decision_timeout_sec") is not None
    ]
    timeout_level_max_values = [
        float(e.get("timeout_level_max_sec"))
        for e in episodes
        if e.get("timeout_level_max_sec") is not None
    ]
    decision_timeout_rate_mean = round(total_decision_timeouts / max(total_decision_calls, 1), 4)
    fallback_action_rate_mean = round(total_fallback_actions / max(total_decision_calls, 1), 4)
    ollama_native_timeout_rate_mean = round(total_ollama_native_timeouts / max(total_decision_calls, 1), 4)
    timeout_collapse_detected = bool(
        total_decision_calls > 0
        and decision_timeout_rate_mean >= 0.95
        and fallback_action_rate_mean >= 0.95
    )
    timeout_collapse_reason = None
    if timeout_collapse_detected:
        timeout_collapse_reason = (
            "ollama_native_timeout_collapse"
            if ollama_native_timeout_rate_mean >= 0.95
            else "decision_timeout_collapse"
        )
    ollama_case_unload_summary = _summarize_ollama_case_unloads(episodes)
    prompt_profiles = sorted(
        {
            str(e.get("prompt_profile"))
            for e in episodes
            if e.get("prompt_profile") is not None
        }
    )
    aggregate = {
        "model": model_name,
        "prompt_profile": prompt_profiles[0] if len(prompt_profiles) == 1 else ",".join(prompt_profiles),
        "episodes": total,
        "planned_episode_count": planned_total,
        "executed_episode_count": total,
        "skipped_episode_count": skipped_episode_count,
        "episode_execution_complete": bool(episode_execution_complete),
        "model_skipped_due_to_preflight": bool(model_skipped_due_to_preflight),
        "model_skipped_reason": model_skipped_reason,
        "model_quarantined_due_to_timeout_collapse": bool(model_quarantined_due_to_timeout_collapse),
        "model_quarantine_reason": model_quarantine_reason,
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
        "decision_timeout_rate_mean": decision_timeout_rate_mean,
        "timeout_episode_count": timeout_episode_count,
        "timeout_episode_rate": round(timeout_episode_count / total, 4) if total else None,
        "fallback_actions_total": total_fallback_actions,
        "fallback_action_rate_mean": fallback_action_rate_mean,
        "fallback_reason_counts": fallback_reason_counts,
        "semantic_recoveries_total": int(total_semantic_recoveries),
        "semantic_recovery_rate_mean": round(total_semantic_recoveries / max(total_decision_calls, 1), 4),
        "semantic_recovery_label_counts": semantic_recovery_label_counts,
        "intent_resolver_used_total": int(total_intent_resolver_used),
        "intent_resolver_used_rate_mean": round(total_intent_resolver_used / max(total_decision_calls, 1), 4),
        "intent_resolver_recoveries_total": int(total_intent_resolver_recoveries),
        "intent_resolver_recovery_rate_mean": round(
            total_intent_resolver_recoveries / max(total_decision_calls, 1),
            4,
        ),
        "intent_resolver_abstains_total": int(total_intent_resolver_abstains),
        "intent_resolver_abstain_rate_mean": round(
            total_intent_resolver_abstains / max(total_decision_calls, 1),
            4,
        ),
        "ollama_native_retries_total": total_ollama_native_retries,
        "ollama_native_retry_rate_mean": round(total_ollama_native_retries / max(total_decision_calls, 1), 4),
        "ollama_openai_fallbacks_total": total_ollama_openai_fallbacks,
        "ollama_openai_fallback_rate_mean": round(total_ollama_openai_fallbacks / max(total_decision_calls, 1), 4),
        "ollama_native_chat_configured_values": ollama_native_chat_configured_values,
        "ollama_native_chat_effective_values": ollama_native_chat_effective_values,
        "ollama_native_chat_resolution_reason_counts": ollama_native_chat_resolution_reason_counts,
        "ollama_native_decisions_total": total_ollama_native_decisions,
        "ollama_native_decision_rate_mean": round(total_ollama_native_decisions / max(total_decision_calls, 1), 4),
        "ollama_native_timeouts_total": total_ollama_native_timeouts,
        "ollama_native_timeout_rate_mean": ollama_native_timeout_rate_mean,
        "ollama_native_timeout_short_circuits_total": total_ollama_native_timeout_short_circuits,
        "ollama_native_timeout_episode_count": ollama_native_timeout_episode_count,
        "ollama_native_timeout_episode_rate": round(ollama_native_timeout_episode_count / total, 4) if total else None,
        "ollama_downgrade_episode_count": ollama_downgrade_episode_count,
        "ollama_downgrade_episode_rate": round(ollama_downgrade_episode_count / total, 4) if total else None,
        **ollama_case_unload_summary,
        "episodes_stopped_by_timeout_cap": timeout_cap_stops,
        "response_delimiter_rate": round(total_delimiters / total_decisions, 4) if total_decisions else None,
        "response_strict_format_rate": round(total_strict / total_decisions, 4) if total_decisions else None,
        "response_direct_parseable_rate": round(total_direct / total_decisions, 4) if total_decisions else None,
        "runtime_parse_path_counts": runtime_parse_path_counts,
        "avg_reward_sum": round(total_reward_sum / total, 4) if total else None,
        "avg_reward_per_step": round(total_reward_sum / max(total_steps, 1), 4),
        "avg_ego_speed_mps": round(total_speed / total, 4) if total else None,
        "ttc_danger_rate_mean": round(total_ttc_danger_rate / total, 4) if total else None,
        "headway_violation_rate_mean": round(total_headway_rate / total, 4) if total else None,
        "rear_ttc_danger_rate_mean": round(total_rear_ttc_danger_rate / total, 4) if total else None,
        "rear_headway_violation_rate_mean": round(total_rear_headway_rate / total, 4) if total else None,
        "low_speed_blocking_rate_mean": round(total_low_speed_blocking_rate / total, 4) if total else None,
        "min_ego_speed_mps_mean": round(sum(min_ego_speed_values) / len(min_ego_speed_values), 4) if min_ego_speed_values else None,
        "stop_episode_rate": round(stop_episode_count / total, 4) if total else None,
        "stop_rate_mean": round(total_stop_rate / total, 4) if total else None,
        "near_stop_episode_rate": round(near_stop_episode_count / total, 4) if total else None,
        "near_stop_rate_mean": round(total_near_stop_rate / total, 4) if total else None,
        "lane_change_rate_mean": round(total_lane_change_rate / total, 4) if total else None,
        "lane_change_shields_total": int(total_lane_change_shields),
        "lane_change_shield_rate_mean": round(total_lane_change_shields / max(total_decision_calls, 1), 4),
        "lane_change_shield_reason_counts": lane_change_shield_reason_counts,
        "unsafe_lane_change_attempts_total": int(total_unsafe_lane_change_attempts),
        "longitudinal_safety_shields_total": int(total_longitudinal_safety_shields),
        "longitudinal_safety_shield_rate_mean": round(
            total_longitudinal_safety_shields / max(total_decision_calls, 1),
            4,
        ),
        "longitudinal_safety_shield_reason_counts": longitudinal_safety_shield_reason_counts,
        "unsafe_longitudinal_action_attempts_total": int(total_unsafe_longitudinal_action_attempts),
        "flow_recovery_shields_total": int(total_flow_recovery_shields),
        "flow_recovery_shield_rate_mean": round(total_flow_recovery_shields / max(total_decision_calls, 1), 4),
        "flow_recovery_reason_counts": flow_recovery_reason_counts,
        "flap_accel_decel_rate_mean": round(total_flap_rate / total, 4) if total else None,
        "format_failure_rate_mean": round(total_format_failures / max(total_decisions, 1), 4),
        "decision_latency_ms_avg": (
            round(sum(decision_latency_ms_values) / len(decision_latency_ms_values), 3)
            if decision_latency_ms_values else None
        ),
        "timeout_penalty_events_total": int(total_timeout_penalty_events),
        "timeout_penalty_timeout_triggers_total": int(total_timeout_penalty_timeout_triggers),
        "timeout_penalty_slow_triggers_total": int(total_timeout_penalty_slow_triggers),
        "timeout_penalty_events_rate_mean": round(total_timeout_penalty_events / max(total_decision_calls, 1), 4),
        "timeout_penalty_stage_max_mean": (
            round(sum(timeout_penalty_stage_max_values) / max(total, 1), 4)
            if total else None
        ),
        "timeout_penalty_stage_max_global": max(timeout_penalty_stage_max_values) if timeout_penalty_stage_max_values else 0,
        "timeout_level_max_sec_mean": (
            round(sum(timeout_level_max_values) / len(timeout_level_max_values), 4)
            if timeout_level_max_values else None
        ),
        "timeout_escalation_count_mean": round(total_timeout_escalation_count / total, 4) if total else None,
        "timeout_recovery_count_mean": round(total_timeout_recovery_count / total, 4) if total else None,
        "timeout_level_15_rate_mean": round(total_timeout_level_15_rate / total, 4) if total else None,
        "timeout_level_20_rate_mean": round(total_timeout_level_20_rate / total, 4) if total else None,
        "timeout_level_30_rate_mean": round(total_timeout_level_30_rate / total, 4) if total else None,
        "timeout_phase_counts": timeout_phase_counts,
        "warmup_timeout_failures_total": int(total_warmup_timeout_failures),
        "warmup_active_episode_count": int(warmup_active_episode_count),
        "warmup_success_observed_episode_count": int(warmup_success_observed_episode_count),
        "timeout_penalty_final_decision_timeout_sec_mean": (
            round(sum(timeout_penalty_final_values) / len(timeout_penalty_final_values), 4)
            if timeout_penalty_final_values else None
        ),
        # Deprecated alias for one transition cycle.
        "timeout_penalty_final_native_timeout_sec_mean": (
            round(sum(timeout_penalty_final_values) / len(timeout_penalty_final_values), 4)
            if timeout_penalty_final_values else None
        ),
        "timeout_collapse_detected": timeout_collapse_detected,
        "timeout_collapse_reason": timeout_collapse_reason,
        "split_scoring_policy_version": SPLIT_SCORING_POLICY_VERSION,
    }
    for idx, split_field in enumerate(SPLIT_SCORE_FIELDS):
        split_values = [
            float(episode.get(split_field))
            for episode in episodes
            if episode.get(split_field) is not None
        ]
        if split_values:
            aggregate[split_field] = round(sum(split_values) / len(split_values), 4)
            aggregate[f"{split_field}_ci95"] = bootstrap_ci95(
                split_values,
                seed=20260327 + idx,
            )
        else:
            aggregate[split_field] = None
            aggregate[f"{split_field}_ci95"] = None

    benchmark_episodes = [episode for episode in episodes if "task_completed" in episode]
    if benchmark_episodes:
        aggregate.update(summarize_benchmark_episodes(benchmark_episodes))
        benchmark_result_valid, benchmark_result_invalid_reason = benchmark_result_validity(
            decision_timeout_rate_mean=aggregate.get("decision_timeout_rate_mean"),
            fallback_action_rate_mean=aggregate.get("fallback_action_rate_mean"),
            timeout_episode_rate=aggregate.get("timeout_episode_rate"),
        )
        aggregate.update(
            {
                "benchmark_result_valid": bool(benchmark_result_valid),
                "benchmark_result_invalid_reason": benchmark_result_invalid_reason,
            }
        )
        if not episode_execution_complete:
            aggregate["benchmark_result_valid"] = False
            aggregate["benchmark_result_invalid_reason"] = _extend_invalid_reason(
                aggregate.get("benchmark_result_invalid_reason"),
                "incomplete_episode_set",
            )
        if model_quarantined_due_to_timeout_collapse:
            aggregate["benchmark_result_valid"] = False
            aggregate["benchmark_result_invalid_reason"] = _extend_invalid_reason(
                aggregate.get("benchmark_result_invalid_reason"),
                "timeout_collapse_quarantine",
                model_quarantine_reason,
            )

    return annotate_aggregate_with_scientific_reporting(
        aggregate,
        episodes,
        primary_metric_spec,
    )


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
    benchmark_mode = bool(metrics_config.get("benchmark_mode"))
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
        "benchmark_mode": benchmark_mode,
        "benchmark_case_set": metrics_config.get("benchmark_case_set") if benchmark_mode else None,
        "benchmark_variant": metrics_config.get("benchmark_variant") if benchmark_mode else None,
        "execution_mode": metrics_config.get("execution_mode") if benchmark_mode else None,
        "benchmark_fingerprint": metrics_config.get("benchmark_fingerprint") if benchmark_mode else None,
        "headline_task_metric": (
            "driving_score_balanced_v1" if benchmark_mode else None
        ),
        "headline_llm_metric": "llm_driver_score_v1" if benchmark_mode else None,
        "headline_joint_metric": "dilu_joint_score_v1" if benchmark_mode else None,
        "legacy_headline_task_metric": (
            metrics_config.get("benchmark_metric_config", {}) or {}
        ).get("recommended_headline_metric") if benchmark_mode else None,
        "efficiency_metrics_reported": bool(aggregate.get("decision_latency_ms_avg_mean") is not None),
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


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare DiLu agent behavior across Ollama models on fixed seeds.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", required=False, help="Model names to compare (e.g. deepseek-r1:14b dilu-llama3_1-8b-v1)")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to the configured seed bank.")
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help=f"Run the first N seeds from the selected seed bank/list; default bank has {_format_default_seed_count()} seeds.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of seeds/cases after parsing.")
    parser.add_argument(
        "--benchmark-case-set",
        default=None,
        help=(
            "Optional LaMPilot-style task case set. Use a JSON path or a built-in name such as "
            f"`{DEFAULT_BENCHMARK_CASE_SET}`."
        ),
    )
    parser.add_argument(
        "--benchmark-categories",
        default=None,
        help=(
            "Optional comma-separated benchmark category filter. Requires --benchmark-case-set; "
            "applied before --limit."
        ),
    )
    parser.add_argument("--few-shot-num", type=int, default=None, help="Override config few_shot_num.")
    parser.add_argument("--memory-path", default=None, help="Override config memory_path.")
    parser.add_argument(
        "--prompt-profile",
        choices=["harness_v2", "legacy_dilu_like"],
        default=None,
        help="Runtime prompt profile for ablations. Default comes from config eval_prompt_profile or harness_v2.",
    )
    parser.add_argument("--output", default=None, help="Write JSON report to this file (default: results/eval_compare_<timestamp>.json)")
    parser.add_argument("--experiment-id", default=None, help="Experiment id. Defaults to config or timestamp.")
    parser.add_argument("--results-root", default=None, help="Structured results root. Defaults to config or results/experiments.")
    parser.add_argument("--output-root", default=None, help="Optional compare-output folder override.")
    parser.add_argument(
        "--energy-mode",
        choices=["none", "latency_only", "joulescope_hw"],
        default="none",
        help="Optional measurement mode. `none` keeps standard evaluation behavior.",
    )
    parser.add_argument("--idle-calibration", default=None, help="Path to idle calibration JSON artifact.")
    parser.add_argument("--calibrate-idle", action="store_true", help="Run idle power calibration and exit.")
    parser.add_argument("--idle-duration-sec", type=float, default=60.0, help="Idle calibration duration in seconds.")
    parser.add_argument("--calibration-output", default=None, help="Explicit output path for idle calibration JSON.")
    parser.add_argument(
        "--measurement-hard-preflight-policy",
        choices=["warn_run", "skip_model", "abort_run"],
        default=None,
        help="Measurement-mode handling for hard Ollama preflight failures. Default comes from config.",
    )
    parser.add_argument(
        "--action-target-speeds",
        default=None,
        help="Optional comma-separated DiscreteMetaAction target speeds, e.g. 0,5,10,15,20,25,30.",
    )
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
    parser.add_argument(
        "--save-run-artifacts",
        action="store_true",
        help="Save run-style artifacts (db/log/run_metrics/action_traces) per model during evaluation.",
    )
    parser.add_argument(
        "--no-save-run-artifacts",
        action="store_true",
        help="Disable run-style artifacts even when eval_save_run_artifacts is enabled in config.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record per-episode videos. Implies run artifacts and should be used only for small debug runs.",
    )
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Disable video recording even when eval_record_video is enabled in config.",
    )
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
    parser.add_argument(
        "--performance-mode",
        choices=["default", "fast"],
        default=None,
        help="Eval performance mode. fast disables expensive extras for long runs.",
    )
    parser.add_argument(
        "--skip-ollama-preflight",
        action="store_true",
        help="Skip the Ollama backend responsiveness probe before evaluation.",
    )
    parser.add_argument(
        "--strict-ollama-preflight",
        action="store_true",
        help="In measurement mode, abort if Ollama preflight fails. Standard eval remains fail-fast by default.",
    )
    parser.add_argument(
        "--ollama-preflight-timeout-sec",
        type=float,
        default=None,
        help="Timeout for the Ollama preflight probe. Default: config eval_ollama_preflight_timeout_sec or 15s.",
    )
    parser.add_argument(
        "--timeout-early-stop",
        choices=["on", "off"],
        default=None,
        help="Enable or disable early-stop for timeout-collapsing episodes.",
    )
    parser.add_argument(
        "--timeout-early-stop-min-decisions",
        type=int,
        default=None,
        help="Minimum number of decisions before timeout early-stop can trigger.",
    )
    parser.add_argument(
        "--timeout-early-stop-consecutive",
        type=int,
        default=None,
        help="Consecutive timeout+fallback decisions required for early-stop.",
    )
    parser.add_argument(
        "--timeout-early-stop-rate",
        type=float,
        default=None,
        help="Decision-timeout and fallback-rate threshold for early-stop.",
    )
    parser.add_argument(
        "--timeout-early-stop-require-max-level",
        choices=["on", "off"],
        default=None,
        help="Require laddered eval to reach the max timeout rung before early-stop can trigger.",
    )
    parser.add_argument(
        "--timeout-collapse-quarantine-after",
        type=int,
        default=None,
        help="Quarantine a model after this many timeout-cap episodes in the same run.",
    )
    parser.add_argument("--decision-timeout-sec", type=float, default=None, help="Hard timeout per model decision call. Default: config eval_decision_timeout_sec or 60.")
    parser.add_argument("--decision-max-output-tokens", type=int, default=None, help="Deprecated in policy mode: ignored for timeout-only policy.")
    parser.add_argument("--disable-streaming", action="store_true", help="Deprecated in policy mode: ignored for timeout-only policy.")
    parser.add_argument("--disable-checker-llm", action="store_true", help="Deprecated in policy mode: ignored for timeout-only policy.")
    parser.add_argument("--ollama-think-mode", choices=["auto", "think", "no_think"], default=None, help="Measurement mode override for Ollama think mode. Ignored in standard eval mode.")
    parser.add_argument("--ollama-use-native-chat", action="store_true", help="Measurement mode override to force Ollama native /api/chat. Ignored in standard eval mode.")
    parser.add_argument("--ollama-disable-native-chat", action="store_true", help="Measurement mode override to force Ollama /v1 transport. Ignored in standard eval mode.")
    parser.add_argument("--no-ollama-use-native-chat", dest="ollama_disable_native_chat", action="store_true", help="Alias for --ollama-disable-native-chat.")
    parser.add_argument("--alignment-sample-rate", type=float, default=0.0, help="Sampling probability [0,1] for reasoning-alignment sample collection.")
    parser.add_argument("--alignment-max-samples", type=int, default=0, help="Max alignment samples per model.")
    args = parser.parse_args(argv)
    if args.quiet and args.no_quiet:
        raise ValueError("Use only one of --quiet or --no-quiet.")
    if args.progress and args.no_progress:
        raise ValueError("Use only one of --progress or --no-progress.")
    if args.save_run_artifacts and args.no_save_run_artifacts:
        raise ValueError("Use only one of --save-run-artifacts or --no-save-run-artifacts.")
    if args.record_video and args.no_record_video:
        raise ValueError("Use only one of --record-video or --no-record-video.")
    if args.record_video and args.no_save_run_artifacts:
        raise ValueError("--record-video requires run artifacts; remove --no-save-run-artifacts.")
    if args.ollama_use_native_chat and args.ollama_disable_native_chat:
        raise ValueError("Use only one of --ollama-use-native-chat or --ollama-disable-native-chat/--no-ollama-use-native-chat.")
    benchmark_category_filter_requested = _parse_benchmark_category_filter(args.benchmark_categories)
    if benchmark_category_filter_requested and not args.benchmark_case_set:
        raise ValueError("--benchmark-categories requires --benchmark-case-set.")

    config = load_runtime_config(args.config)
    if args.prompt_profile is not None:
        config["eval_prompt_profile"] = args.prompt_profile
    prompt_profile = str(config.get("eval_prompt_profile", "harness_v2") or "harness_v2").strip().lower().replace("-", "_")
    if prompt_profile not in {"harness_v2", "legacy_dilu_like"}:
        raise ValueError("eval_prompt_profile must be one of: harness_v2, legacy_dilu_like.")
    config["eval_prompt_profile"] = prompt_profile
    energy_mode = _normalize_energy_mode(args.energy_mode)
    measurement_mode = energy_mode != "none"
    config = _apply_measurement_runtime_overrides(config, args, energy_mode=energy_mode)
    timeout_early_stop_policy = _resolve_eval_timeout_early_stop_policy(config, args)
    measurement_hard_preflight_policy = _resolve_measurement_hard_preflight_policy(config, args)
    if args.calibrate_idle:
        if energy_mode != "joulescope_hw":
            raise ValueError("--calibrate-idle requires --energy-mode joulescope_hw.")
        calibration_experiment_id = args.experiment_id or "energy_latency_benchmark"
        calibration_root = _resolve_measurement_output_root(
            args=args,
            experiment_id=str(calibration_experiment_id),
        )
        calibration_path = _calibration_output_path(args=args, output_root=calibration_root)
        monitor = create_energy_monitor("joulescope_hw")
        try:
            written_path = monitor.calibrate_idle(args.idle_duration_sec, calibration_path)
        finally:
            monitor.close()
        print(f"Saved idle calibration: [bold]{written_path}[/bold]")
        return
    if not args.models:
        raise ValueError("--models is required unless --calibrate-idle is used.")
    idle_calibration = (
        load_idle_calibration(args.idle_calibration)
        if (measurement_mode and args.idle_calibration)
        else None
    )
    monitor = (
        create_energy_monitor(energy_mode, idle_calibration=idle_calibration)
        if measurement_mode
        else None
    )

    requested_performance_mode = _resolve_eval_performance_mode(config, args.performance_mode)
    performance_mode_effective = requested_performance_mode
    performance_optimizations_applied: List[str] = []
    if requested_performance_mode == "fast" and (args.save_run_artifacts or args.record_video):
        raise ValueError("--performance-mode fast cannot be combined with --save-run-artifacts or --record-video.")
    if requested_performance_mode == "fast":
        if bool(config.get("eval_save_run_artifacts", False)):
            print("[yellow]Fast performance mode overrides config eval_save_run_artifacts=true.[/yellow]")
        if bool(config.get("eval_record_video", False)):
            print("[yellow]Fast performance mode overrides config eval_record_video=true.[/yellow]")
        config["eval_save_run_artifacts"] = False
        config["eval_record_video"] = False
        config["eval_disable_checker_llm"] = True
        performance_optimizations_applied.extend(
            [
                "save_run_artifacts_disabled",
                "video_recording_disabled",
                "checker_llm_disabled",
            ]
        )

    cli_quiet_override = True if args.quiet else (False if args.no_quiet else None)
    resolved_eval_quiet_mode = _resolve_quiet_mode(config, cli_quiet_override, mode="eval")
    if requested_performance_mode == "fast" and not resolved_eval_quiet_mode:
        resolved_eval_quiet_mode = True
        performance_optimizations_applied.append("step_logs_suppressed")
    cli_progress_override = True if args.progress else (False if args.no_progress else None)
    resolved_eval_progress_mode = _resolve_progress_mode(config, cli_progress_override, mode="eval")
    progress_enabled = bool(resolved_eval_progress_mode and _is_interactive_output())
    resolved_eval_progress_reply_mode = _resolve_progress_reply_mode(
        config,
        args.progress_replies,
        mode="eval",
    )
    if requested_performance_mode == "fast" and resolved_eval_progress_reply_mode != "off":
        print("[yellow]Fast performance mode disables progress reply previews.[/yellow]")
        performance_optimizations_applied.append("progress_reply_previews_disabled")
    effective_eval_progress_reply_mode = (
        resolved_eval_progress_reply_mode
        if (progress_enabled and (not resolved_eval_quiet_mode))
        else "off"
    )
    if requested_performance_mode == "fast":
        effective_eval_progress_reply_mode = "off"
    step_log_quiet_mode = bool(resolved_eval_quiet_mode or progress_enabled)

    benchmark_case_set = None
    benchmark_cases: List[Dict] = []
    benchmark_mode = bool(args.benchmark_case_set)
    benchmark_fingerprint = None
    benchmark_category_filter_effective: List[str] = []
    if args.limit is not None and args.seed_count is not None:
        raise ValueError("--seed-count cannot be combined with --limit.")
    if benchmark_mode:
        if args.seeds:
            raise ValueError("--seeds cannot be combined with --benchmark-case-set. Cases define their own seeds.")
        if args.seed_count is not None:
            raise ValueError("--seed-count cannot be combined with --benchmark-case-set. Cases define their own seeds.")
        benchmark_case_set = load_benchmark_case_set(args.benchmark_case_set)
        benchmark_case_set, benchmark_cases = _filter_benchmark_cases_by_category(
            benchmark_case_set,
            benchmark_category_filter_requested,
        )
        if args.limit is not None:
            benchmark_cases = benchmark_cases[:args.limit]
        if not benchmark_cases:
            raise ValueError("No benchmark cases to evaluate.")
        benchmark_case_set = dict(benchmark_case_set)
        benchmark_case_set["cases"] = list(benchmark_cases)
        benchmark_case_set["categories"] = sorted({case["category"] for case in benchmark_cases})
        benchmark_category_filter_effective = (
            list(benchmark_case_set["categories"])
            if benchmark_category_filter_requested
            else []
        )
        benchmark_fingerprint = build_benchmark_case_set_fingerprint(benchmark_case_set)
        seeds = [int(case["seed"]) for case in benchmark_cases]
    else:
        seeds = resolve_eval_seeds(config, args.seeds, args.seed_count)
        if args.limit is not None:
            seeds = seeds[:args.limit]
        if not seeds:
            raise ValueError("No seeds to evaluate.")

    few_shot_num = config["few_shot_num"] if args.few_shot_num is None else args.few_shot_num
    if args.memory_path:
        config["memory_path"] = args.memory_path
    if int(few_shot_num) > 0:
        performance_optimizations_applied.append("shared_driving_memory_reuse")
    ttc_threshold_sec = float(config.get("metrics_ttc_threshold_sec", 2.0))
    headway_threshold_m = float(config.get("metrics_headway_threshold_m", 15.0))
    rear_ttc_threshold_sec = float(config.get("metrics_rear_ttc_threshold_sec", 2.5))
    rear_headway_threshold_m = float(config.get("metrics_rear_headway_threshold_m", 12.0))
    low_speed_blocking_threshold_mps = float(config.get("metrics_low_speed_blocking_threshold_mps", 8.5))
    blocking_front_gap_safe_m = float(config.get("metrics_blocking_front_gap_safe_m", 25.0))
    blocking_front_ttc_safe_sec = float(config.get("metrics_blocking_front_ttc_safe_sec", 4.0))
    stop_threshold_mps = float(config.get("metrics_stop_threshold_mps", STOP_THRESHOLD_MPS_DEFAULT))
    near_stop_threshold_mps = float(config.get("metrics_near_stop_threshold_mps", NEAR_STOP_THRESHOLD_MPS_DEFAULT))
    alignment_sample_rate = max(0.0, min(1.0, float(args.alignment_sample_rate)))
    alignment_max_samples = max(0, int(args.alignment_max_samples))
    structured_output = not args.no_structured_output
    artifact_modes_requested = (
        _config_as_bool(config.get("eval_save_run_artifacts", False), default=False)
        or bool(args.save_run_artifacts)
        or _config_as_bool(config.get("eval_record_video", False), default=False)
        or bool(args.record_video)
    )
    save_run_artifacts, record_video = _resolve_eval_artifact_modes(
        config,
        args,
        measurement_mode=measurement_mode,
    )
    eval_run_id = str(
        args.eval_run_id
        or config.get("eval_run_id")
        or f"eval_run_{current_timestamp()}"
    ).strip()
    if not eval_run_id:
        eval_run_id = f"eval_run_{current_timestamp()}"
    if measurement_mode and not structured_output:
        raise ValueError("--energy-mode requires structured output. Remove --no-structured-output.")
    if measurement_mode and artifact_modes_requested:
        print("[yellow]Measurement mode disables eval run artifacts to preserve benchmark-compatible outputs.[/yellow]")
    if save_run_artifacts and not structured_output:
        raise ValueError("--save-run-artifacts requires structured output. Remove --no-structured-output.")
    if not save_run_artifacts:
        performance_optimizations_applied.append("sqlite_db_logging_disabled_without_artifacts")
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
    eval_timeout_policy_preview = decision_timeout_penalty_snapshot(
        build_decision_timeout_penalty_state(
            config=config,
            provider=provider,
            mode="eval",
            baseline_decision_timeout_sec=default_decision_timeout_sec,
        )
    )
    eval_timeout_legacy_fields_ignored = []
    if eval_timeout_policy_preview.get("policy_mode") == "laddered":
        eval_timeout_legacy_fields_ignored = [
            "adaptive_timeout_halving_factor",
            "adaptive_timeout_min_sec",
            "adaptive_timeout_trigger_consecutive_slow",
        ]
    ollama_preflight_enabled = _resolve_eval_ollama_preflight_enabled(
        config,
        cli_skip=bool(args.skip_ollama_preflight),
    )
    ollama_preflight_timeout_sec = _resolve_eval_ollama_preflight_timeout_sec(
        config,
        args.ollama_preflight_timeout_sec,
    )
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
    cli_ollama_think_mode = (
        str(args.ollama_think_mode).strip().lower()
        if (args.ollama_think_mode and not measurement_mode)
        else None
    )
    cli_ollama_use_native_chat = (
        bool(args.ollama_use_native_chat or args.ollama_disable_native_chat)
        if not measurement_mode
        else False
    )
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
    config["eval_record_video"] = bool(record_video)
    config["eval_run_id"] = eval_run_id

    if measurement_mode:
        experiment_id = str(
            args.experiment_id
            or config.get("experiment_id")
            or current_timestamp()
        )
        experiment_root = _resolve_measurement_output_root(
            args=args,
            experiment_id=experiment_id,
        )
        experiment_id = os.path.basename(experiment_root.rstrip("\\/"))
        model_roots = {model_name: build_model_root(experiment_root, model_name) for model_name in args.models}
        compare_dir = ensure_dir(os.path.join(experiment_root, "compare"))
    else:
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

    effective_env_id_override = _resolve_benchmark_env_id_override(args.env_id, benchmark_case_set)
    benchmark_env_overrides = _benchmark_default_env_overrides(benchmark_case_set)
    env_bundle = build_env_bundle(
        config,
        env_id_override=effective_env_id_override,
        native_env_defaults_override=args.native_env_defaults,
        action_target_speeds_override=parse_action_target_speeds(args.action_target_speeds),
        env_config_overrides=benchmark_env_overrides,
    )
    for warning_msg in env_bundle.get("warnings", []):
        print(f"[yellow]{warning_msg}[/yellow]")
    env_config = env_bundle["env_config_map"]
    env_type = str(env_bundle["env_id"])
    env_config_snapshot = env_bundle["env_config_snapshot"]
    resolved_simulation_duration = _resolve_simulation_duration(config, env_config_snapshot)
    resolved_action_target_speeds = list(env_bundle.get("resolved_action_target_speeds") or [])
    env_profile_label = str(env_bundle.get("env_profile_label") or "default")
    benchmark_validation = None
    if benchmark_mode and benchmark_case_set is not None:
        benchmark_validation = validate_benchmark_case_set(
            benchmark_case_set,
            env_config,
            env_type,
        )
        if not benchmark_validation.get("passed"):
            invalid_cases = benchmark_validation.get("invalid_cases", [])
            invalid_preview = "; ".join(
                f"{item.get('case_id')} ({', '.join(item.get('reasons') or [])})"
                for item in invalid_cases[:5]
            )
            if len(invalid_cases) > 5:
                invalid_preview += f"; ... +{len(invalid_cases) - 5} more"
            raise ValueError(
                "Benchmark case-set validation failed before model evaluation. "
                f"invalid_case_count={benchmark_validation['summary'].get('invalid_case_count', 0)}. "
                f"{invalid_preview}"
            )
    temp_dir = ensure_dir(os.path.join("temp", "eval_compare"))
    primary_metric_spec = build_primary_metric_spec(config)
    ollama_preflight_results: List[Dict] = []
    ollama_preflight_warning = None
    preflight_skip_models: Dict[str, Dict[str, Any]] = {}
    if provider == "ollama" and ollama_preflight_enabled:
        configure_runtime_env(
            config,
            chat_model_override=args.models[0],
            mode="eval",
            quiet_override=step_log_quiet_mode,
            progress_override=progress_enabled,
        )
        ollama_preflight_results = _run_ollama_preflight(
            config,
            args.models,
            timeout_sec=ollama_preflight_timeout_sec,
            quiet_mode=step_log_quiet_mode,
        )
        preflight_failures = [
            item for item in ollama_preflight_results
            if isinstance(item, dict) and not bool(item.get("ok"))
        ]
        hard_preflight_failures = [
            item for item in preflight_failures
            if _classify_ollama_preflight_failure(item) == "hard"
        ]
        soft_preflight_failures = [
            item for item in preflight_failures
            if _classify_ollama_preflight_failure(item) != "hard"
        ]
        if measurement_mode:
            ollama_overrides = config.get("_benchmark_ollama_runtime_overrides", {})
            if ollama_overrides:
                print(
                    "[bold cyan]Measurement-mode Ollama overrides[/bold cyan]: "
                    f"think_mode={ollama_overrides.get('ollama_think_mode')} | "
                    f"use_native_chat={ollama_overrides.get('ollama_use_native_chat')}"
                )
                if ollama_overrides.get("auto_forced_native_chat"):
                    print(
                        "[yellow]  Native chat was enabled automatically[/yellow] "
                        "because a measurement-mode think override was requested."
                    )
            if args.strict_ollama_preflight and preflight_failures:
                raise RuntimeError(_format_ollama_preflight_failures(preflight_failures))
            if hard_preflight_failures and measurement_hard_preflight_policy == "abort_run":
                raise RuntimeError(_format_ollama_preflight_failures(hard_preflight_failures))
            if hard_preflight_failures and measurement_hard_preflight_policy == "skip_model":
                preflight_skip_models = {
                    str(item.get("model")): dict(item)
                    for item in hard_preflight_failures
                    if item.get("model")
                }
                hard_message = _format_ollama_preflight_failures(hard_preflight_failures)
                ollama_preflight_warning = (
                    "Measurement-mode preflight skipped hard failures.\n"
                    f"{hard_message}"
                )
                print(
                    "[yellow]Measurement-mode preflight warning[/yellow]: skipping hard-failed models.\n"
                    f"{hard_message}"
                )
            if soft_preflight_failures:
                soft_message = _format_ollama_preflight_failures(soft_preflight_failures)
                ollama_preflight_warning = _extend_invalid_reason(
                    ollama_preflight_warning,
                    soft_message,
                )
                print(
                    "[yellow]Measurement-mode preflight warning[/yellow]: continuing despite soft failures.\n"
                    "For latency/energy benchmarking, a slow first response is itself part of the measurement.\n"
                    f"{soft_message}"
                )
            if (
                hard_preflight_failures
                and measurement_hard_preflight_policy == "warn_run"
                and not args.strict_ollama_preflight
            ):
                hard_message = _format_ollama_preflight_failures(hard_preflight_failures)
                ollama_preflight_warning = _extend_invalid_reason(
                    ollama_preflight_warning,
                    hard_message,
                )
                print(
                    "[yellow]Measurement-mode preflight warning[/yellow]: continuing despite failure.\n"
                    "For latency/energy benchmarking, a slow first response is itself part of the measurement.\n"
                    f"{hard_message}"
                )
        else:
            if preflight_failures:
                raise RuntimeError(_format_ollama_preflight_failures(preflight_failures))
    ollama_preflight_results_by_model = _index_ollama_preflight_results(ollama_preflight_results)
    measurement_integrity = _build_measurement_integrity_summary(
        ollama_preflight_results,
        ollama_preflight_warning,
        skipped_models_due_to_preflight=[
            {
                "model": model_name,
                "reason": "hard_ollama_preflight_failure",
                "transport": probe.get("transport"),
                "error": probe.get("error"),
            }
            for model_name, probe in preflight_skip_models.items()
        ],
    )
    shared_agent_memory: Optional[DrivingMemory] = None
    if int(few_shot_num) > 0:
        configure_runtime_env(
            config,
            chat_model_override=args.models[0],
            mode="eval",
            quiet_override=step_log_quiet_mode,
            progress_override=progress_enabled,
        )
        shared_agent_memory = DrivingMemory(db_path=config["memory_path"])

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
        "record_video": bool(record_video),
        "eval_run_id": eval_run_id if save_run_artifacts else None,
        "performance_mode_requested": requested_performance_mode,
        "performance_mode_effective": performance_mode_effective,
        "performance_optimizations_applied": sorted(set(performance_optimizations_applied)),
        "ollama_runtime_controls": _ollama_runtime_controls_report(config) if provider == "ollama" else {},
        "ollama_case_unload_policy": _ollama_case_unload_policy(config) if provider == "ollama" else {},
        "ollama_preflight_enabled": bool(provider == "ollama" and ollama_preflight_enabled),
        "ollama_preflight_timeout_sec": ollama_preflight_timeout_sec if provider == "ollama" else None,
        "measurement_hard_preflight_policy": (
            measurement_hard_preflight_policy
            if provider == "ollama" and measurement_mode
            else None
        ),
        "ollama_preflight_results": ollama_preflight_results if provider == "ollama" else [],
        "ollama_preflight_warning": ollama_preflight_warning if measurement_mode else None,
        "measurement_integrity_warnings": list(measurement_integrity.get("measurement_integrity_warnings", [])),
        "ollama_preflight_failed_models": list(measurement_integrity.get("ollama_preflight_failed_models", [])),
        "skipped_models_due_to_preflight": list(measurement_integrity.get("skipped_models_due_to_preflight", [])),
        "quarantined_models_due_to_timeout_collapse": list(
            measurement_integrity.get("quarantined_models_due_to_timeout_collapse", [])
        ),
        "openai_api_type": config["OPENAI_API_TYPE"],
        "prompt_profile": prompt_profile,
        "benchmark_mode": bool(benchmark_mode),
        "headline_task_metric": (
            "driving_score_balanced_v1"
            if benchmark_mode
            else None
        ),
        "headline_llm_metric": "llm_driver_score_v1" if benchmark_mode else None,
        "headline_joint_metric": "dilu_joint_score_v1" if benchmark_mode else None,
        "legacy_headline_task_metric": (
            benchmark_metric_config(
                benchmark_case_set.get("scenario_family") if benchmark_case_set is not None else "highway"
            ).get("recommended_headline_metric")
            if benchmark_mode
            else None
        ),
        "efficiency_metrics_reported": bool(measurement_mode),
        "benchmark_case_set": (
            benchmark_case_set["benchmark_name"] if benchmark_case_set is not None else None
        ),
        "benchmark_variant": (
            LEGACY_BENCHMARK_VARIANT if benchmark_case_set is not None else None
        ),
        "execution_mode": (
            LEGACY_EXECUTION_MODE if benchmark_case_set is not None else None
        ),
        "benchmark_fingerprint": benchmark_fingerprint if benchmark_case_set is not None else None,
        "benchmark_case_set_path": (
            benchmark_case_set["case_set_path"] if benchmark_case_set is not None else None
        ),
        "benchmark_categories": (
            list(benchmark_case_set["categories"]) if benchmark_case_set is not None else []
        ),
        "benchmark_category_filter_requested": list(benchmark_category_filter_requested),
        "benchmark_category_filter_effective": list(benchmark_category_filter_effective),
        "resolved_action_target_speeds": list(resolved_action_target_speeds),
        "env_profile_label": env_profile_label,
        "benchmark_validation_passed": (
            bool(benchmark_validation.get("passed")) if benchmark_validation is not None else None
        ),
        "benchmark_invalid_cases": (
            benchmark_validation.get("invalid_cases", []) if benchmark_validation is not None else []
        ),
        "benchmark_validation_summary": (
            benchmark_validation.get("summary") if benchmark_validation is not None else None
        ),
        "models": args.models,
        "model_roots": model_roots,
        "seeds": seeds,
        "few_shot_num": few_shot_num,
        "memory_path": config["memory_path"],
        "simulation_duration": int(resolved_simulation_duration),
        "metrics_config": {
            "env_id": str(env_type),
            "native_env_defaults": bool(env_bundle.get("use_native_env_defaults")),
            "requested_env_id": str(env_bundle.get("requested_env_id")),
            "env_resolution_sources": {
                "env_id": env_bundle.get("env_source"),
                "native_env_defaults": env_bundle.get("native_source"),
            },
            "env_resolution_warnings": list(env_bundle.get("warnings", [])),
            "resolved_action_target_speeds": list(resolved_action_target_speeds),
            "env_profile_label": env_profile_label,
            "env_config_snapshot": _json_safe(copy.deepcopy(env_config_snapshot)),
            "ttc_threshold_sec": ttc_threshold_sec,
            "headway_threshold_m": headway_threshold_m,
            "rear_ttc_threshold_sec": rear_ttc_threshold_sec,
            "rear_headway_threshold_m": rear_headway_threshold_m,
            "low_speed_blocking_threshold_mps": low_speed_blocking_threshold_mps,
            "blocking_front_gap_safe_m": blocking_front_gap_safe_m,
            "blocking_front_ttc_safe_sec": blocking_front_ttc_safe_sec,
            "stop_threshold_mps": stop_threshold_mps,
            "near_stop_threshold_mps": near_stop_threshold_mps,
            "flapping_mode": "accel_decel",
            "eval_timeout_policy_mode": eval_timeout_policy_preview.get("policy_mode"),
            "eval_timeout_ladder_sec": eval_timeout_policy_preview.get("timeout_ladder_sec"),
            "eval_timeout_recovery_successes": eval_timeout_policy_preview.get("recovery_successes_required"),
            "eval_timeout_legacy_fields_ignored": list(eval_timeout_legacy_fields_ignored),
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
            "prompt_profile": prompt_profile,
            "policy_mode": "timeout_only",
            "deprecated_policy_cli_fields_ignored": deprecated_cli_policy_flags,
            "deprecated_policy_override_fields_ignored": deprecated_override_fields_declared,
            "deprecated_metric_aliases": {
                "timeout_penalty_final_native_timeout_sec": "timeout_penalty_final_decision_timeout_sec",
                "timeout_penalty_final_native_timeout_sec_mean": "timeout_penalty_final_decision_timeout_sec_mean",
            },
            "save_run_artifacts": bool(save_run_artifacts),
            "record_video": bool(record_video),
            "eval_run_id": eval_run_id if save_run_artifacts else None,
            "performance_mode_requested": requested_performance_mode,
            "performance_mode_effective": performance_mode_effective,
            "performance_optimizations_applied": sorted(set(performance_optimizations_applied)),
            "ollama_runtime_controls": _ollama_runtime_controls_report(config) if provider == "ollama" else {},
            "ollama_case_unload_policy": _ollama_case_unload_policy(config) if provider == "ollama" else {},
            "ollama_preflight_enabled": bool(provider == "ollama" and ollama_preflight_enabled),
            "ollama_preflight_timeout_sec": ollama_preflight_timeout_sec if provider == "ollama" else None,
            "measurement_hard_preflight_policy": measurement_hard_preflight_policy if provider == "ollama" else None,
            "timeout_early_stop_policy": dict(timeout_early_stop_policy),
            "alignment_sample_rate": alignment_sample_rate,
            "alignment_max_samples": alignment_max_samples,
            "benchmark_mode": bool(benchmark_mode),
            "benchmark_case_set": (
                benchmark_case_set["benchmark_name"] if benchmark_case_set is not None else None
            ),
            "benchmark_variant": (
                LEGACY_BENCHMARK_VARIANT if benchmark_case_set is not None else None
            ),
            "execution_mode": (
                LEGACY_EXECUTION_MODE if benchmark_case_set is not None else None
            ),
            "benchmark_fingerprint": benchmark_fingerprint if benchmark_case_set is not None else None,
            "benchmark_case_set_path": (
                benchmark_case_set["case_set_path"] if benchmark_case_set is not None else None
            ),
            "benchmark_categories": (
                list(benchmark_case_set["categories"]) if benchmark_case_set is not None else []
            ),
            "benchmark_category_filter_requested": list(benchmark_category_filter_requested),
            "benchmark_category_filter_effective": list(benchmark_category_filter_effective),
            "benchmark_metric_config": (
                benchmark_metric_config(
                    benchmark_case_set.get("scenario_family") if benchmark_case_set is not None else "highway"
                )
                if benchmark_mode
                else None
            ),
            "split_scoring_policy": {
                "version": SPLIT_SCORING_POLICY_VERSION,
                "headline_policy": "split_primary",
                "primary_driving_metric": "driving_score_balanced_v1",
                "driving_behavior_component": "driving_score_behavior_v1",
                "driving_task_component": "driving_score_v2",
                "balanced_driving_score_formula": "sqrt(driving_score_behavior_v1 * driving_score_v2)",
                "primary_llm_metric": "llm_driver_score_v1",
                "secondary_joint_metric": "dilu_joint_score_v1",
                "legacy_compatibility_metrics": ["driving_score", "driving_score_v2"],
                "llm_score_revision_note": (
                    "llm_driver_score_v1 uses intervention-dependence scoring under "
                    "dilu_split_score_v1.1+; do not directly mix LLM-score comparisons "
                    "with dilu_split_score_v1 reports."
                ),
                "balanced_driving_score_revision_note": (
                    "dilu_split_score_v1.2 uses driving_score_balanced_v1 as the "
                    "benchmark driving headline; do not directly mix headline "
                    "driving-score comparisons with dilu_split_score_v1.1 reports."
                ),
            },
            "primary_metric_spec": primary_metric_spec,
            "benchmark_validation_passed": (
                bool(benchmark_validation.get("passed")) if benchmark_validation is not None else None
            ),
            "benchmark_invalid_cases": (
                benchmark_validation.get("invalid_cases", []) if benchmark_validation is not None else []
            ),
            "benchmark_validation_summary": (
                benchmark_validation.get("summary") if benchmark_validation is not None else None
            ),
        },
        "per_model": {},
        "aggregates": [],
        "alignment_samples": [],
        "model_eval_outputs": {},
        "model_run_outputs": {},
        "model_runtime_policies": {},
    }
    if measurement_mode:
        report.update(
            {
                "source": "evaluate_models_ollama:measurement_mode",
                "output_root": experiment_root,
                "energy_mode_requested": args.energy_mode,
                "energy_mode_effective": energy_mode,
                "idle_calibration_path": args.idle_calibration,
                "idle_calibration": idle_calibration,
                "hardware_info": system_hardware_snapshot(),
                "ollama_runtime_overrides": config.get("_benchmark_ollama_runtime_overrides", {}),
                "ollama_preflight_strict": bool(args.strict_ollama_preflight),
                "model_runtime_metadata": {},
                "energy_tradeoff_summary": {},
                "model_outputs": {},
            }
        )
        report["metrics_config"].update(
            {
                "few_shot_num": few_shot_num,
                "energy_mode": energy_mode,
                "idle_baseline_subtraction_enabled": bool(
                    idle_calibration is not None and idle_calibration.get("avg_idle_power_w") is not None
                ),
                "token_count_method": "ollama_usage_with_estimate_fallback",
                "ollama_runtime_overrides": config.get("_benchmark_ollama_runtime_overrides", {}),
            }
        )
    if benchmark_mode and benchmark_case_set is not None:
        filter_suffix = (
            f" | filter={', '.join(benchmark_category_filter_requested)}"
            if benchmark_category_filter_requested
            else ""
        )
        print(
            "[bold cyan]Benchmark mode[/bold cyan]: "
            f"{benchmark_case_set['benchmark_name']} | cases={len(benchmark_cases)} | "
            f"categories={', '.join(benchmark_case_set['categories'])}"
            f"{filter_suffix}"
        )
        if benchmark_validation is not None:
            summary = benchmark_validation.get("summary", {})
            print(
                "[bold cyan]Benchmark validation[/bold cyan]: "
                f"passed={benchmark_validation.get('passed')} | "
                f"valid_cases={summary.get('valid_case_count')} | "
                f"invalid_cases={summary.get('invalid_case_count')}"
            )

    aggregate_by_model: Dict[str, Dict] = {}
    model_run_outputs: Dict[str, Dict[str, str]] = {}
    model_metrics_configs: Dict[str, Dict] = {}
    deprecated_policy_fields_ignored_union = set(deprecated_cli_policy_flags)
    quarantined_models_due_to_timeout_collapse: List[Dict[str, Any]] = []
    episode_specs: List = benchmark_cases if benchmark_mode else list(seeds)
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
            baseline_timeout_penalty_state = build_decision_timeout_penalty_state(
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
            model_ollama_runtime_controls = (
                _ollama_runtime_controls_report(config) if provider == "ollama" else {}
            )
            model_ollama_case_unload_policy = (
                _ollama_case_unload_policy(config) if provider == "ollama" else {}
            )

            report["model_runtime_policies"][model_name] = {
                "decision_timeout_sec": round(resolved_decision_timeout_sec, 3),
                "policy_meta": policy_meta,
                "matched_override": {
                    "model_policy": policy_meta.get("matched_model_policy_override_key"),
                    "legacy_eval_model": policy_meta.get("matched_eval_model_override_key"),
                },
                "deprecated_policy_fields_ignored": policy_meta.get("deprecated_policy_fields_ignored", []),
                "decision_timeout_penalty": decision_timeout_penalty_snapshot(baseline_timeout_penalty_state),
                "eval_timeout_policy_mode": eval_timeout_policy_preview.get("policy_mode"),
                "eval_timeout_ladder_sec": eval_timeout_policy_preview.get("timeout_ladder_sec"),
                "eval_timeout_recovery_successes": eval_timeout_policy_preview.get("recovery_successes_required"),
                "eval_timeout_legacy_fields_ignored": list(eval_timeout_legacy_fields_ignored),
                "timeout_early_stop_policy": dict(timeout_early_stop_policy),
                "ollama_runtime_controls": model_ollama_runtime_controls,
                "ollama_case_unload_policy": model_ollama_case_unload_policy,
                # Deprecated alias key for one transition cycle.
                "native_timeout_penalty": decision_timeout_penalty_snapshot(baseline_timeout_penalty_state),
            }
            model_metrics_configs[model_name] = {
                **dict(report["metrics_config"]),
                "resolved_model_policy": dict(report["model_runtime_policies"][model_name]),
                "ollama_runtime_controls": model_ollama_runtime_controls,
                "ollama_case_unload_policy": model_ollama_case_unload_policy,
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
            penalty_snapshot = decision_timeout_penalty_snapshot(baseline_timeout_penalty_state)
            if penalty_snapshot.get("policy_mode") == "laddered":
                ladder_values = penalty_snapshot.get("timeout_ladder_sec") or []
                ladder_label = " -> ".join(
                    str(int(value)) if float(value).is_integer() else str(value)
                    for value in ladder_values
                )
                emit(
                    "[dim]  Eval timeout ladder: levels={levels}s | recovery_successes={recovery} | "
                    "active_timeout={active}s[/dim]".format(
                        levels=ladder_label,
                        recovery=int(penalty_snapshot.get("recovery_successes_required") or 0),
                        active=round(float(penalty_snapshot.get("effective_decision_timeout_sec") or 0.0), 3),
                    )
                )
                if eval_timeout_legacy_fields_ignored:
                    emit(
                        "[dim]  Legacy adaptive eval fields ignored in laddered mode: "
                        f"{', '.join(eval_timeout_legacy_fields_ignored)}[/dim]"
                    )
            else:
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
            model_run_dir = None
            model_log_path = None
            model_energy_dir = None
            measurements_path = None
            action_traces_path = None
            measurement_rows: List[Dict] = []
            action_trace_rows: List[Dict] = []
            if measurement_mode:
                report["model_runtime_metadata"][model_name] = (
                    _inspect_ollama_model(model_name)
                    if provider == "ollama"
                    else {"model_tag": model_name}
                )
                emit(
                    f"[dim]  Measurement mode={energy_mode} | env_id={env_type} | "
                    f"episodes={len(episode_specs)} | few_shot_num={few_shot_num}[/dim]"
                )
                model_energy_dir = ensure_dir(os.path.join(model_roots[model_name], "energy"))
                measurement_ts = current_timestamp()
                measurements_path = os.path.join(model_energy_dir, f"episode_measurements_{measurement_ts}.json")
                action_traces_path = os.path.join(model_energy_dir, f"action_traces_{measurement_ts}.json")
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

            planned_episode_count = len(episode_specs)
            if measurement_mode and model_name in preflight_skip_models:
                emit(
                    "[yellow]  Skipping model due to hard Ollama preflight failure[/yellow]: "
                    f"{preflight_skip_models[model_name].get('error')}"
                )
                report["per_model"][model_name] = []
                agg = _build_skipped_model_aggregate(
                    model_name=model_name,
                    planned_episode_count=planned_episode_count,
                    reason="hard_ollama_preflight_failure",
                    preflight_probe=preflight_skip_models.get(model_name),
                    benchmark_mode=benchmark_mode,
                )
                agg = annotate_aggregate_with_scientific_reporting(agg, [], primary_metric_spec)
                report["aggregates"].append(agg)
                aggregate_by_model[model_name] = agg
                if model_task is not None:
                    progress.update(model_task, advance=1)
                continue

            seed_task = (
                progress.add_task(
                    f"{model_name} {'cases' if benchmark_mode else 'seeds'}",
                    total=planned_episode_count,
                )
                if progress is not None
                else None
            )
            step_task = (
                progress.add_task(
                    f"{model_name} steps",
                    total=(
                        benchmark_max_steps(
                            benchmark_cases[0],
                            env_config_snapshot,
                            int(resolved_simulation_duration),
                        )
                        if benchmark_mode and benchmark_cases
                        else int(resolved_simulation_duration)
                    ),
                )
                if progress is not None
                else None
            )
            episodes = []
            model_alignment_samples = []
            timeout_cap_episode_count = 0
            model_quarantined_due_to_timeout_collapse = False
            model_quarantine_reason = None
            for idx, episode_spec in enumerate(episode_specs, start=1):
                benchmark_case = episode_spec if benchmark_mode else None
                seed = int(benchmark_case["seed"]) if benchmark_case is not None else int(episode_spec)
                case_env_config = env_config
                case_env_snapshot = env_config_snapshot
                case_instruction = None
                case_max_steps = int(resolved_simulation_duration)
                case_label = f"seed {idx}/{len(episode_specs)}"
                if benchmark_case is not None:
                    case_env_config, case_env_snapshot = build_case_env_config(
                        env_config,
                        env_type,
                        benchmark_case,
                    )
                    case_max_steps = benchmark_max_steps(
                        benchmark_case,
                        case_env_snapshot,
                        int(resolved_simulation_duration),
                    )
                    case_instruction = build_benchmark_instruction(benchmark_case)
                    case_label = f"case {idx}/{len(episode_specs)}"
                    emit(
                        "[dim]  Case {idx}/{total}: {case_id} | seed={seed} | category={category}[/dim]".format(
                            idx=idx,
                            total=len(episode_specs),
                            case_id=benchmark_case["case_id"],
                            seed=seed,
                            category=benchmark_case["category"],
                        )
                    )
                else:
                    emit(f"[dim]  Seed {idx}/{len(episode_specs)}: {seed}[/dim]")
                if progress is not None and step_task is not None:
                    progress.update(
                        step_task,
                        description=f"{model_name} | {case_label}",
                        total=int(case_max_steps),
                        completed=0,
                    )
                timeout_penalty_state = build_decision_timeout_penalty_state(
                    config=config,
                    provider=provider,
                    mode="eval",
                    baseline_decision_timeout_sec=resolved_decision_timeout_sec,
                )
                decision_token_usage_records: List[Dict] = []
                action_trace: List[Dict] = []
                first_action_latency_sec: Optional[float] = None

                def _on_step(step_completed: int, done: bool) -> None:
                    if progress is not None and step_task is not None:
                        progress.update(
                            step_task,
                            completed=min(int(step_completed), int(case_max_steps)),
                        )

                def _on_decision(step_idx: int, action_id: int, response_text: str, decision_meta: Dict) -> None:
                    nonlocal first_action_latency_sec
                    if measurement_mode:
                        token_usage_record = {
                            "prompt_tokens": int(decision_meta.get("prompt_tokens", 0) or 0),
                            "completion_tokens": int(decision_meta.get("completion_tokens", 0) or 0),
                            "total_tokens": int(decision_meta.get("total_tokens", 0) or 0),
                            "token_count_method": decision_meta.get("token_count_method"),
                            "token_usage_source": decision_meta.get("token_usage_source"),
                        }
                        decision_token_usage_records.append(token_usage_record)
                        if first_action_latency_sec is None and decision_meta.get("decision_elapsed_sec") is not None:
                            first_action_latency_sec = float(decision_meta.get("decision_elapsed_sec"))
                    if measurement_mode or save_run_artifacts:
                        trace_item = _decision_trace_item(
                            step_idx=int(step_idx),
                            action_id=int(action_id),
                            response_text=response_text,
                            decision_meta=decision_meta,
                            include_response_text=measurement_mode,
                        )
                        action_trace.append(trace_item)
                    if effective_eval_progress_reply_mode == "compact":
                        emit(_compact_reply_preview(step_idx, action_id, response_text))
                    elif effective_eval_progress_reply_mode == "full":
                        emit(_full_reply_preview(step_idx, action_id, response_text))

                episode_wall_time_start = (
                    datetime.now().isoformat(timespec="seconds") if measurement_mode else None
                )
                if measurement_mode and monitor is not None:
                    monitor.start_episode(
                        {
                            "model": model_name,
                            "seed": seed,
                            "case_id": benchmark_case.get("case_id") if benchmark_case is not None else None,
                        }
                    )
                ollama_case_unload_result: Dict[str, Any] = {}
                try:
                    episode_result = run_episode(
                        config=config,
                        env_config=case_env_config,
                        env_type=env_type,
                        agent_memory=shared_agent_memory,
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
                        stop_threshold_mps=stop_threshold_mps,
                        near_stop_threshold_mps=near_stop_threshold_mps,
                        alignment_sample_rate=alignment_sample_rate,
                        alignment_max_samples=alignment_max_samples,
                        slow_decision_threshold_sec=slow_decision_threshold_sec,
                        timeout_penalty_state=timeout_penalty_state,
                        save_artifacts=save_run_artifacts,
                        record_video=record_video,
                        run_dir=model_run_dir,
                        run_id=eval_run_id if save_run_artifacts else None,
                        model_name=model_name,
                        quiet_mode=step_log_quiet_mode,
                        enable_db_logging=bool(save_run_artifacts),
                        on_step=_on_step if progress is not None else None,
                        on_decision=_on_decision if progress is not None else None,
                        benchmark_case=benchmark_case,
                        driving_instruction=case_instruction,
                        max_steps_override=case_max_steps,
                        timeout_early_stop_policy=timeout_early_stop_policy,
                    )
                finally:
                    if provider == "ollama":
                        ollama_case_unload_result = _stop_ollama_model_after_case(config, model_name)
                if ollama_case_unload_result.get("attempted"):
                    episode_result["ollama_case_unload"] = ollama_case_unload_result
                    if not ollama_case_unload_result.get("ok", False):
                        emit(
                            "[yellow]  Ollama case-unload warning:[/yellow] "
                            f"{ollama_case_unload_result.get('error') or ollama_case_unload_result.get('stderr')}"
                        )
                measurement = monitor.stop_episode() if (measurement_mode and monitor is not None) else {}
                if measurement_mode:
                    episode_wall_time_end = datetime.now().isoformat(timespec="seconds")
                    episode_token_usage = aggregate_episode_token_usage(decision_token_usage_records)
                    episode_result = enrich_episode_energy_metrics(
                        episode_result,
                        energy_mode=energy_mode,
                        raw_energy_j=measurement.get("raw_energy_j"),
                        idle_power_w=monitor.idle_power_w if monitor is not None else None,
                        avg_power_w=measurement.get("avg_power_w"),
                        peak_power_w=measurement.get("peak_power_w"),
                        prompt_tokens_total=int(episode_token_usage.get("prompt_tokens_total", 0)),
                        completion_tokens_total=int(episode_token_usage.get("completion_tokens_total", 0)),
                        total_tokens=int(episode_token_usage.get("total_tokens", 0)),
                        token_count_method=str(episode_token_usage.get("token_count_method") or TOKEN_COUNT_METHOD),
                        token_usage_source=str(episode_token_usage.get("token_usage_source") or "estimate_fallback"),
                        latency_to_first_action_sec=first_action_latency_sec,
                    )
                    episode_result["episode_wall_time_start"] = episode_wall_time_start
                    episode_result["episode_wall_time_end"] = episode_wall_time_end
                    episode_result["energy_measurement_meta"] = measurement.get("measurement_meta", {})
                    episode_result["action_sequence"] = [int(item["action_id"]) for item in action_trace]
                    episode_result["action_histogram"] = _action_histogram(action_trace)
                    episode_result = compute_split_scores_for_episode(episode_result)
                episode_alignment_samples = episode_result.pop("alignment_samples", [])
                for sample in episode_alignment_samples:
                    sample["model"] = model_name
                    model_alignment_samples.append(sample)
                episodes.append(episode_result)
                if measurement_mode:
                    measurement_rows.append(_measurement_record(episode_result))
                    action_trace_rows.append(
                        {
                            "episode_id": str(episode_result.get("case_id") or f"seed_{episode_result.get('seed')}"),
                            "seed": episode_result.get("seed"),
                            "case_id": episode_result.get("case_id"),
                            "category": episode_result.get("category"),
                            "crashed": bool(episode_result.get("crashed", False)),
                            "task_completed": episode_result.get("task_completed"),
                            "action_sequence": list(episode_result.get("action_sequence") or []),
                            "action_histogram": dict(episode_result.get("action_histogram") or {}),
                            "trace": action_trace,
                        }
                    )
                if save_run_artifacts and model_log_path:
                    _append_eval_run_log(model_log_path, model_name, episode_result)
                if progress is not None and seed_task is not None:
                    progress.update(seed_task, advance=1)
                status = "CRASH" if episode_result["crashed"] else ("ERROR" if episode_result["error"] else ("TIMEOUT" if episode_result.get("timeout_triggered") else "OK"))
                benchmark_suffix = ""
                if "task_completed" in episode_result:
                    driving_score_value = episode_result.get("driving_score_balanced_v1")
                    driving_score_label = "driving_score_balanced_v1"
                    if driving_score_value is None:
                        driving_score_value = episode_result.get("driving_score_behavior_v1")
                        driving_score_label = "driving_score_behavior_v1"
                    if driving_score_value is None:
                        driving_score_value = episode_result.get("driving_score_v2")
                        driving_score_label = "driving_score_v2"
                    if driving_score_value is None:
                        driving_score_value = episode_result.get("driving_score")
                        driving_score_label = "driving_score"
                    benchmark_suffix = (
                        f" | task_completed={episode_result.get('task_completed')} "
                        f"| {driving_score_label}={driving_score_value} "
                        f"| llm_driver_score_v1={episode_result.get('llm_driver_score_v1')}"
                    )
                emit(
                    f"    -> {status} | steps={episode_result['steps']}/{episode_result['max_steps']} "
                    f"| t={episode_result['episode_runtime_sec']}s | timeout_steps={episode_result.get('decision_timeout_count', 0)}"
                    f"{benchmark_suffix}"
                )
                if episode_result["error"]:
                    emit(f"    -> [red]{episode_result['error']}[/red]")
                if episode_result.get("episode_stop_reason") == "episode_timeout_cap":
                    timeout_cap_episode_count += 1
                    if timeout_cap_episode_count >= int(timeout_early_stop_policy.get("quarantine_after_collapses", 2)):
                        remaining_episodes = planned_episode_count - len(episodes)
                        if remaining_episodes > 0:
                            model_quarantined_due_to_timeout_collapse = True
                            model_quarantine_reason = "timeout_collapse_quarantine"
                            quarantined_models_due_to_timeout_collapse.append(
                                {
                                    "model": model_name,
                                    "reason": model_quarantine_reason,
                                    "timeout_cap_episode_count": int(timeout_cap_episode_count),
                                    "remaining_episode_count": int(remaining_episodes),
                                }
                            )
                            emit(
                                "[yellow]  Quarantining remaining episodes for model[/yellow]: "
                                f"remaining={remaining_episodes} after {timeout_cap_episode_count} timeout-cap episode(s)"
                            )
                            if progress is not None and seed_task is not None:
                                progress.update(seed_task, completed=planned_episode_count)
                            break

            report["model_runtime_policies"][model_name]["decision_timeout_penalty"] = (
                decision_timeout_penalty_snapshot(
                    build_decision_timeout_penalty_state(
                        config=config,
                        provider=provider,
                        mode="eval",
                        baseline_decision_timeout_sec=resolved_decision_timeout_sec,
                    )
                )
            )
            report["model_runtime_policies"][model_name]["native_timeout_penalty"] = (
                decision_timeout_penalty_snapshot(
                    build_decision_timeout_penalty_state(
                        config=config,
                        provider=provider,
                        mode="eval",
                        baseline_decision_timeout_sec=resolved_decision_timeout_sec,
                    )
                )
            )
            model_metrics_configs[model_name]["resolved_model_policy"] = dict(
                report["model_runtime_policies"][model_name]
            )
            report["per_model"][model_name] = episodes
            agg = aggregate_results(
                model_name,
                episodes,
                planned_episode_count=planned_episode_count,
                model_quarantined_due_to_timeout_collapse=model_quarantined_due_to_timeout_collapse,
                model_quarantine_reason=model_quarantine_reason,
                primary_metric_spec=primary_metric_spec,
            )
            agg = _annotate_aggregate_with_ollama_preflight_status(
                agg,
                ollama_preflight_results_by_model,
            )
            if measurement_mode:
                agg.update(summarize_energy_latency_episodes(episodes))
                if "task_completion_rate" in agg:
                    benchmark_valid, benchmark_invalid_reason = benchmark_result_validity(
                        decision_timeout_rate_mean=agg.get("decision_timeout_rate_mean"),
                        fallback_action_rate_mean=agg.get("fallback_action_rate_mean"),
                        timeout_episode_rate=agg.get("timeout_episode_rate"),
                    )
                    agg["benchmark_result_valid"] = bool(benchmark_valid)
                    agg["benchmark_result_invalid_reason"] = benchmark_invalid_reason
                    if not agg.get("episode_execution_complete", True):
                        agg["benchmark_result_valid"] = False
                        agg["benchmark_result_invalid_reason"] = _extend_invalid_reason(
                            agg.get("benchmark_result_invalid_reason"),
                            "incomplete_episode_set",
                        )
                    if agg.get("model_quarantined_due_to_timeout_collapse"):
                        agg["benchmark_result_valid"] = False
                        agg["benchmark_result_invalid_reason"] = _extend_invalid_reason(
                            agg.get("benchmark_result_invalid_reason"),
                            "timeout_collapse_quarantine",
                            agg.get("model_quarantine_reason"),
                        )
                agg = annotate_aggregate_with_scientific_reporting(
                    agg,
                    episodes,
                    primary_metric_spec,
                )
            report["aggregates"].append(agg)
            aggregate_by_model[model_name] = agg
            report["alignment_samples"].extend(model_alignment_samples[:alignment_max_samples] if alignment_max_samples > 0 else [])
            if measurement_mode and model_energy_dir and measurements_path and action_traces_path:
                summary_ts = current_timestamp()
                summary_path = os.path.join(model_energy_dir, f"energy_summary_{summary_ts}.json")
                episodes_path = os.path.join(model_energy_dir, f"energy_episodes_{summary_ts}.json")
                write_json_atomic(
                    summary_path,
                    _build_model_extract(
                        model_name=model_name,
                        experiment_id=experiment_id,
                        source_compare_report="",
                        aggregate=agg,
                        episodes=episodes,
                        metrics_config=model_metrics_configs.get(model_name, report["metrics_config"]),
                    ),
                )
                write_json_atomic(
                    episodes_path,
                    {
                        "model": model_name,
                        "experiment_id": experiment_id,
                        "episodes": episodes,
                    },
                )
                write_json_atomic(
                    measurements_path,
                    {"model": model_name, "measurements": measurement_rows},
                )
                write_json_atomic(
                    action_traces_path,
                    {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "source": "evaluate_models_ollama:measurement_mode:action_traces",
                        "model": model_name,
                        "experiment_id": experiment_id,
                        "episodes": action_trace_rows,
                    },
                )
                report["model_outputs"][model_name] = {
                    "summary": summary_path,
                    "episodes": episodes_path,
                    "measurements": measurements_path,
                    "action_traces": action_traces_path,
                }
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
                    simulation_duration=int(resolved_simulation_duration),
                    metrics_config=model_metrics_configs.get(model_name, report["metrics_config"]),
                    episodes=episodes,
                    aggregate=agg,
                )
                run_metrics_path = timestamped_results_path("run_metrics", ext=".json", results_dir=model_run_dir)
                write_json_atomic(run_metrics_path, run_metrics_report)
                run_action_traces_path = os.path.join(model_run_dir, "action_traces.json")
                write_json_atomic(
                    run_action_traces_path,
                    {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "source": "evaluate_models_ollama:save_run_artifacts:action_traces",
                        "model": model_name,
                        "experiment_id": experiment_id,
                        "episodes": [
                            {
                                "episode_id": str(episode.get("case_id") or f"seed_{episode.get('seed')}"),
                                "seed": episode.get("seed"),
                                "case_id": episode.get("case_id"),
                                "category": episode.get("category"),
                                "crashed": bool(episode.get("crashed", False)),
                                "trace": list(episode.get("action_trace") or []),
                            }
                            for episode in episodes
                        ],
                    },
                )
                model_run_outputs[model_name] = {
                    "run_id": eval_run_id,
                    "run_dir": model_run_dir,
                    "log_path": model_log_path,
                    "run_metrics": run_metrics_path,
                    "action_traces": run_action_traces_path,
                }
                if record_video:
                    model_run_outputs[model_name]["videos_dir"] = model_run_dir

            if progress is not None and model_task is not None:
                progress.update(model_task, advance=1)
            if progress is not None and seed_task is not None:
                progress.remove_task(seed_task)
            if progress is not None and step_task is not None:
                progress.remove_task(step_task)

    report["metrics_config"]["deprecated_policy_fields_ignored"] = sorted(
        deprecated_policy_fields_ignored_union
    )
    measurement_integrity = _build_measurement_integrity_summary(
        ollama_preflight_results,
        ollama_preflight_warning,
        skipped_models_due_to_preflight=list(report.get("skipped_models_due_to_preflight", [])),
        quarantined_models_due_to_timeout_collapse=quarantined_models_due_to_timeout_collapse,
    )
    report["measurement_integrity_warnings"] = list(measurement_integrity.get("measurement_integrity_warnings", []))
    report["ollama_preflight_failed_models"] = list(measurement_integrity.get("ollama_preflight_failed_models", []))
    report["skipped_models_due_to_preflight"] = list(measurement_integrity.get("skipped_models_due_to_preflight", []))
    report["quarantined_models_due_to_timeout_collapse"] = list(
        measurement_integrity.get("quarantined_models_due_to_timeout_collapse", [])
    )
    report["model_run_outputs"] = model_run_outputs

    user_out_path = None
    if args.output:
        user_out_path = ensure_parent_dir(args.output)
    if measurement_mode:
        report["energy_tradeoff_summary"] = build_energy_tradeoff_summary(report["aggregates"])
        out_path = timestamped_results_path("energy_latency_compare", ext=".json", results_dir=compare_dir)
        report["compare_report_path"] = out_path
        if structured_output and experiment_root:
            report["scientific_outputs"] = write_scientific_analysis_artifacts(
                report,
                os.path.join(experiment_root, "analysis"),
            )
        write_json_atomic(out_path, report)
        for model_name, outputs in report.get("model_outputs", {}).items():
            if not outputs.get("summary"):
                continue
            write_json_atomic(
                outputs["summary"],
                _build_model_extract(
                    model_name=model_name,
                    experiment_id=experiment_id,
                    source_compare_report=out_path,
                    aggregate=aggregate_by_model[model_name],
                    episodes=report["per_model"][model_name],
                    metrics_config=model_metrics_configs.get(model_name, report["metrics_config"]),
                ),
            )
        write_json_atomic(out_path, report)
        if user_out_path:
            write_json_atomic(user_out_path, report)

        print("\n[bold green]Energy / Latency Summary[/bold green]")
        for row in report["aggregates"]:
            summary = (
                f"- {row['model']}: crash_rate={row.get('crash_rate')}, "
                f"stop_episode_rate={row.get('stop_episode_rate')}, "
                f"net_energy_j_mean={row.get('net_energy_j_mean')}, "
                f"energy_per_decision_j_mean={row.get('energy_per_decision_j_mean')}, "
                f"tokens_per_second_mean={row.get('tokens_per_second_mean')}, "
                f"decision_latency_ms_avg_mean={row.get('decision_latency_ms_avg_mean')}, "
                f"p95_decision_latency_sec_mean={row.get('p95_decision_latency_sec_mean')}"
            )
            if row.get("task_completion_rate") is not None:
                driving_score_value = row.get("driving_score_balanced_v1")
                driving_score_label = "driving_score_balanced_v1"
                if driving_score_value is None:
                    driving_score_value = row.get("driving_score_behavior_v1")
                    driving_score_label = "driving_score_behavior_v1"
                if driving_score_value is None:
                    driving_score_value = row.get("driving_score_v2")
                    driving_score_label = "driving_score_v2"
                if driving_score_value is None:
                    driving_score_value = row.get("driving_score")
                    driving_score_label = "driving_score"
                summary += (
                    f", task_completion_rate={row.get('task_completion_rate')}, "
                    f"{driving_score_label}={driving_score_value}, "
                    f"llm_driver_score_v1={row.get('llm_driver_score_v1')}, "
                    f"dilu_joint_score_v1={row.get('dilu_joint_score_v1')}, "
                    f"benchmark_result_valid={row.get('benchmark_result_valid')}"
                )
            if row.get("ollama_preflight_ok") is False:
                summary += ", ollama_preflight_ok=False"
            print(summary)
        if report.get("ollama_preflight_failed_models"):
            print("\n[yellow]Measurement integrity warnings[/yellow]")
            for item in report["ollama_preflight_failed_models"]:
                print(
                    f"- model={item.get('model')} transport={item.get('transport')} "
                    f"error={item.get('error')}"
                )
        print(f"\nSaved report: [bold]{out_path}[/bold]")
        if user_out_path and user_out_path != out_path:
            print(f"Saved user-requested output copy: [bold]{user_out_path}[/bold]")
    else:
        if structured_output:
            out_path = timestamped_results_path("eval_compare", ext=".json", results_dir=compare_dir)
            report["compare_report_path"] = out_path
            write_json_atomic(out_path, report)
        else:
            if user_out_path:
                out_path = user_out_path
            else:
                out_path = timestamped_results_path("eval_compare", ext=".json", results_dir=compare_dir)
                write_json_atomic(out_path, report)
            report["compare_report_path"] = out_path

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
            report["scientific_outputs"] = write_scientific_analysis_artifacts(
                report,
                os.path.join(experiment_root, "analysis"),
            )
            write_json_atomic(out_path, report)

            _update_experiment_manifest_for_eval(
                experiment_root=experiment_root,
                experiment_id=experiment_id,
                config_path=args.config,
                memory_path=config["memory_path"],
                few_shot_num=int(few_shot_num),
                simulation_duration=int(resolved_simulation_duration),
                compare_report_path=out_path,
                model_summaries=model_summary_paths,
                model_run_outputs=model_run_outputs if save_run_artifacts else None,
            )

        if user_out_path and user_out_path != out_path:
            write_json_atomic(user_out_path, report)

        print("\n[bold green]Aggregate Summary[/bold green]")
        for row in report["aggregates"]:
            summary = (
                f"- {row['model']}: crashes={row['crashes']}/{row['episodes']} "
                f"(rate={row['crash_rate']}), no_collision_rate={row['no_collision_rate']}, "
                f"avg_steps={row['avg_steps']}, strict_format_rate={row['response_strict_format_rate']}, "
                f"ttc_danger_rate={row['ttc_danger_rate_mean']}, headway_violation_rate={row['headway_violation_rate_mean']}, "
                f"rear_ttc_danger_rate={row.get('rear_ttc_danger_rate_mean')}, "
                f"low_speed_blocking_rate={row.get('low_speed_blocking_rate_mean')}, "
                f"stop_episode_rate={row.get('stop_episode_rate')}, "
                f"decision_timeout_rate={row.get('decision_timeout_rate_mean')}, "
                f"native_timeout_rate={row.get('ollama_native_timeout_rate_mean')}, "
                f"fallback_action_rate={row.get('fallback_action_rate_mean')}, "
                f"avg_episode_runtime_sec={row['avg_episode_runtime_sec']}"
            )
            if row.get("task_completion_rate") is not None:
                driving_score_value = row.get("driving_score_balanced_v1")
                driving_score_label = "driving_score_balanced_v1"
                if driving_score_value is None:
                    driving_score_value = row.get("driving_score_behavior_v1")
                    driving_score_label = "driving_score_behavior_v1"
                if driving_score_value is None:
                    driving_score_value = row.get("driving_score_v2")
                    driving_score_label = "driving_score_v2"
                if driving_score_value is None:
                    driving_score_value = row.get("driving_score")
                    driving_score_label = "driving_score"
                summary += (
                    f", task_completion_rate={row.get('task_completion_rate')}, "
                    f"ttc_score_mean={row.get('ttc_score_mean')}, "
                    f"time_efficiency_score_mean={row.get('time_efficiency_score_mean')}, "
                    f"{driving_score_label}={driving_score_value}, "
                    f"llm_driver_score_v1={row.get('llm_driver_score_v1')}, "
                    f"dilu_joint_score_v1={row.get('dilu_joint_score_v1')}"
                )
                if row.get("benchmark_result_valid") is not None:
                    summary += (
                        f", benchmark_result_valid={row.get('benchmark_result_valid')}"
                    )
            if row.get("ollama_preflight_ok") is False:
                summary += ", ollama_preflight_ok=False"
            print(summary)
            if row.get("timeout_collapse_detected"):
                print(
                    "[bold red]  ! Timeout-collapse detected[/bold red]: "
                    f"{row.get('timeout_collapse_reason')} | "
                    "comparison is dominated by fallback action 4. "
                    "Check OLLAMA_USE_NATIVE_CHAT, few_shot_num, and eval streaming settings."
                )
            if row.get("benchmark_result_valid") is False:
                print(
                    "[bold red]  ! Benchmark result invalid[/bold red]: "
                    f"{row.get('benchmark_result_invalid_reason')}. "
                    "Do not compare this benchmark run against valid benchmark results."
                )
        if report.get("ollama_preflight_failed_models"):
            print("\n[yellow]Preflight warnings[/yellow]")
            for item in report["ollama_preflight_failed_models"]:
                print(
                    f"- model={item.get('model')} transport={item.get('transport')} "
                    f"error={item.get('error')}"
                )
        print(f"\nSaved report: [bold]{out_path}[/bold]")
        if user_out_path and user_out_path != out_path:
            print(f"Saved user-requested output copy: [bold]{user_out_path}[/bold]")

    if monitor is not None:
        monitor.close()


if __name__ == "__main__":
    main()
