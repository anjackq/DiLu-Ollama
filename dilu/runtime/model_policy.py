import fnmatch
import os
from typing import Any, Dict, Optional, Tuple


POLICY_TIMEOUT_FIELDS = ("decision_timeout_sec",)
DEPRECATED_POLICY_FIELDS = (
    "decision_max_output_tokens",
    "disable_streaming",
    "disable_checker_llm",
    "ollama_think_mode",
    "ollama_use_native_chat",
    "ollama_native_chat_timeout_sec",
)


def _clamp_float(value: Any, default: float, minimum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    return max(float(minimum), parsed)


def _clamp_int(value: Any, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(int(minimum), parsed)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_provider(provider: Any) -> str:
    return str(provider or "").strip().lower()


def _match_override(model_name: str, overrides: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    if not isinstance(overrides, dict):
        return None, {}
    lower_name = str(model_name or "").strip().lower()
    if not lower_name:
        return None, {}

    for key, value in overrides.items():
        if str(key).strip().lower() == lower_name and isinstance(value, dict):
            return str(key), dict(value)

    best_key = None
    best_value = None
    best_len = -1
    for key, value in overrides.items():
        if not isinstance(value, dict):
            continue
        key_str = str(key).strip()
        if not key_str:
            continue
        key_lower = key_str.lower()
        if "*" in key_lower or "?" in key_lower:
            matched = fnmatch.fnmatchcase(lower_name, key_lower)
        else:
            matched = lower_name.startswith(key_lower)
        if matched and len(key_lower) > best_len:
            best_key = key_str
            best_value = value
            best_len = len(key_lower)

    if best_key is None:
        return None, {}
    return best_key, dict(best_value or {})


def _collect_deprecated_fields(payload: Any, source: str) -> list[str]:
    if not isinstance(payload, dict):
        return []
    ignored = []
    for key in DEPRECATED_POLICY_FIELDS:
        if key in payload and payload[key] is not None:
            ignored.append(f"{source}.{key}")
    return ignored


def _base_policy_defaults(config: Dict[str, Any], provider: str, mode: str) -> Dict[str, Any]:
    _ = provider  # kept for compatibility and future provider-specific timeout defaults
    if mode == "runtime":
        decision_timeout_sec = float(config.get("runtime_decision_timeout_sec", 60.0))
    else:
        decision_timeout_sec = float(config.get("eval_decision_timeout_sec", 60.0))

    return {
        "decision_timeout_sec": decision_timeout_sec,
    }


def resolve_model_policy(
    config: Dict[str, Any],
    model_name: str,
    provider: str,
    mode: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    norm_provider = _normalize_provider(provider)
    norm_mode = str(mode or "eval").strip().lower()
    if norm_mode not in {"eval", "runtime"}:
        norm_mode = "eval"

    resolved = _base_policy_defaults(config, norm_provider, norm_mode)
    shared_overrides = config.get("model_policy_overrides", {})
    legacy_eval_overrides = config.get("eval_model_overrides", {})
    deprecated_policy_fields_ignored: list[str] = []

    shared_key, shared_values = _match_override(model_name, shared_overrides)
    if shared_values:
        if "decision_timeout_sec" in shared_values and shared_values["decision_timeout_sec"] is not None:
            resolved["decision_timeout_sec"] = shared_values["decision_timeout_sec"]
        deprecated_policy_fields_ignored.extend(
            _collect_deprecated_fields(shared_values, "model_policy_overrides")
        )

    legacy_key = None
    legacy_values: Dict[str, Any] = {}
    if norm_mode == "eval" and not shared_values:
        legacy_key, legacy_values = _match_override(model_name, legacy_eval_overrides)
        if legacy_values:
            if "decision_timeout_sec" in legacy_values and legacy_values["decision_timeout_sec"] is not None:
                resolved["decision_timeout_sec"] = legacy_values["decision_timeout_sec"]
            deprecated_policy_fields_ignored.extend(
                _collect_deprecated_fields(legacy_values, "eval_model_overrides")
            )

    cli_values: Dict[str, Any] = {}
    if isinstance(cli_overrides, dict):
        for key in POLICY_TIMEOUT_FIELDS:
            if key in cli_overrides and cli_overrides[key] is not None:
                cli_values[key] = cli_overrides[key]
        if cli_values:
            resolved.update(cli_values)
        deprecated_policy_fields_ignored.extend(_collect_deprecated_fields(cli_overrides, "cli"))

    resolved["decision_timeout_sec"] = max(1.0, float(resolved["decision_timeout_sec"]))

    resolved["policy_meta"] = {
        "provider": norm_provider,
        "mode": norm_mode,
        "matched_model_policy_override_key": shared_key,
        "matched_eval_model_override_key": legacy_key,
        "cli_override_keys": sorted(cli_values.keys()),
        "deprecated_policy_fields_ignored": sorted(set(deprecated_policy_fields_ignored)),
    }
    return resolved


def apply_model_policy_to_env(policy: Dict[str, Any], provider: str) -> Dict[str, str]:
    _ = provider
    applied: Dict[str, str] = {}
    applied["DILU_DECISION_TIMEOUT_SEC"] = str(float(policy["decision_timeout_sec"]))
    for key, value in applied.items():
        os.environ[key] = value
    return applied


def build_decision_timeout_penalty_state(
    config: Dict[str, Any],
    provider: str,
    mode: str,
    baseline_decision_timeout_sec: float,
) -> Dict[str, Any]:
    norm_provider = _normalize_provider(provider)
    norm_mode = str(mode or "eval").strip().lower()
    enabled = _as_bool(config.get("adaptive_timeout_penalty_enabled"), default=True)

    halving_factor = _clamp_float(
        config.get("adaptive_timeout_halving_factor", 0.5),
        default=0.5,
        minimum=0.01,
    )
    if halving_factor >= 1.0:
        halving_factor = 0.5

    min_timeout_sec = _clamp_float(
        config.get("adaptive_timeout_min_sec", 4.0),
        default=4.0,
        minimum=1.0,
    )
    # Provider/API hard floors. Gemini rejects deadlines below 10 seconds.
    if norm_provider == "gemini":
        min_timeout_sec = max(10.0, min_timeout_sec)
    trigger_consecutive_slow = _clamp_int(
        config.get("adaptive_timeout_trigger_consecutive_slow", 2),
        default=2,
        minimum=1,
    )
    slow_key = "runtime_slow_decision_threshold_sec" if norm_mode == "runtime" else "eval_slow_decision_threshold_sec"
    slow_threshold_sec = _clamp_float(
        config.get(slow_key, config.get("eval_slow_decision_threshold_sec", 5.0)),
        default=5.0,
        minimum=0.001,
    )

    baseline_timeout_sec = _clamp_float(
        baseline_decision_timeout_sec,
        default=min_timeout_sec,
        minimum=min_timeout_sec,
    )
    return {
        "enabled": bool(enabled),
        "provider": norm_provider,
        "mode": norm_mode,
        "baseline_decision_timeout_sec": baseline_timeout_sec,
        "effective_decision_timeout_sec": baseline_timeout_sec,
        "min_timeout_sec": min_timeout_sec,
        "halving_factor": halving_factor,
        "trigger_consecutive_slow": trigger_consecutive_slow,
        "slow_threshold_sec": slow_threshold_sec,
        "stage": 0,
        "consecutive_slow_count": 0,
        "penalty_events": 0,
        "timeout_triggers": 0,
        "slow_triggers": 0,
    }


def _effective_timeout_from_stage(state: Dict[str, Any]) -> float:
    baseline = _clamp_float(state.get("baseline_decision_timeout_sec", 4.0), default=4.0, minimum=1.0)
    min_timeout = _clamp_float(state.get("min_timeout_sec", 4.0), default=4.0, minimum=1.0)
    halving_factor = _clamp_float(state.get("halving_factor", 0.5), default=0.5, minimum=0.01)
    if halving_factor >= 1.0:
        halving_factor = 0.5
    stage = _clamp_int(state.get("stage", 0), default=0, minimum=0)
    return max(min_timeout, baseline * (halving_factor ** stage))


def update_decision_timeout_penalty_state(
    state: Optional[Dict[str, Any]],
    *,
    timed_out: bool,
    decision_elapsed_sec: float,
    slow_threshold_sec: Optional[float] = None,
) -> Dict[str, Any]:
    if not isinstance(state, dict) or not state.get("enabled", False):
        effective = None
        if isinstance(state, dict):
            effective = _clamp_float(
                state.get("effective_decision_timeout_sec", state.get("baseline_decision_timeout_sec", 4.0)),
                default=4.0,
                minimum=1.0,
            )
        return {
            "escalated": False,
            "reason": None,
            "stage": 0 if not isinstance(state, dict) else int(state.get("stage", 0) or 0),
            "effective_decision_timeout_sec": effective,
            "consecutive_slow_count": 0 if not isinstance(state, dict) else int(state.get("consecutive_slow_count", 0) or 0),
        }

    threshold = _clamp_float(
        slow_threshold_sec if slow_threshold_sec is not None else state.get("slow_threshold_sec", 5.0),
        default=5.0,
        minimum=0.001,
    )
    trigger_slow = _clamp_int(state.get("trigger_consecutive_slow", 2), default=2, minimum=1)
    elapsed = _clamp_float(decision_elapsed_sec, default=0.0, minimum=0.0)

    reason = None
    if bool(timed_out):
        reason = "timeout"
        state["consecutive_slow_count"] = 0
    elif elapsed >= threshold:
        state["consecutive_slow_count"] = int(state.get("consecutive_slow_count", 0)) + 1
        if int(state["consecutive_slow_count"]) >= trigger_slow:
            reason = "slow_streak"
            state["consecutive_slow_count"] = 0
    else:
        state["consecutive_slow_count"] = 0

    if reason is not None:
        state["stage"] = _clamp_int(state.get("stage", 0), default=0, minimum=0) + 1
        state["penalty_events"] = _clamp_int(state.get("penalty_events", 0), default=0, minimum=0) + 1
        if reason == "timeout":
            state["timeout_triggers"] = _clamp_int(state.get("timeout_triggers", 0), default=0, minimum=0) + 1
        else:
            state["slow_triggers"] = _clamp_int(state.get("slow_triggers", 0), default=0, minimum=0) + 1

    state["effective_decision_timeout_sec"] = _effective_timeout_from_stage(state)
    return {
        "escalated": bool(reason is not None),
        "reason": reason,
        "stage": int(state["stage"]),
        "effective_decision_timeout_sec": float(state["effective_decision_timeout_sec"]),
        "consecutive_slow_count": int(state["consecutive_slow_count"]),
    }


def decision_timeout_penalty_snapshot(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {
            "enabled": False,
            "stage": 0,
            "penalty_events": 0,
            "timeout_triggers": 0,
            "slow_triggers": 0,
            "baseline_decision_timeout_sec": None,
            "effective_decision_timeout_sec": None,
            "min_timeout_sec": None,
            "halving_factor": None,
            "trigger_consecutive_slow": None,
            "slow_threshold_sec": None,
            # Compatibility aliases (deprecated)
            "baseline_native_timeout_sec": None,
            "effective_native_timeout_sec": None,
        }
    return {
        "enabled": bool(state.get("enabled", False)),
        "stage": int(state.get("stage", 0) or 0),
        "penalty_events": int(state.get("penalty_events", 0) or 0),
        "timeout_triggers": int(state.get("timeout_triggers", 0) or 0),
        "slow_triggers": int(state.get("slow_triggers", 0) or 0),
        "baseline_decision_timeout_sec": (
            float(state["baseline_decision_timeout_sec"])
            if state.get("baseline_decision_timeout_sec") is not None
            else None
        ),
        "effective_decision_timeout_sec": (
            float(state["effective_decision_timeout_sec"])
            if state.get("effective_decision_timeout_sec") is not None
            else None
        ),
        "min_timeout_sec": float(state["min_timeout_sec"]) if state.get("min_timeout_sec") is not None else None,
        "halving_factor": float(state["halving_factor"]) if state.get("halving_factor") is not None else None,
        "trigger_consecutive_slow": int(state["trigger_consecutive_slow"]) if state.get("trigger_consecutive_slow") is not None else None,
        "slow_threshold_sec": float(state["slow_threshold_sec"]) if state.get("slow_threshold_sec") is not None else None,
        # Compatibility aliases (deprecated)
        "baseline_native_timeout_sec": (
            float(state["baseline_decision_timeout_sec"])
            if state.get("baseline_decision_timeout_sec") is not None
            else None
        ),
        "effective_native_timeout_sec": (
            float(state["effective_decision_timeout_sec"])
            if state.get("effective_decision_timeout_sec") is not None
            else None
        ),
    }


# Compatibility wrappers. Deprecated names retained for one transition cycle.
def build_native_timeout_penalty_state(
    config: Dict[str, Any],
    provider: str,
    mode: str,
    baseline_native_timeout_sec: float,
) -> Dict[str, Any]:
    return build_decision_timeout_penalty_state(
        config=config,
        provider=provider,
        mode=mode,
        baseline_decision_timeout_sec=baseline_native_timeout_sec,
    )


def update_native_timeout_penalty_state(
    state: Optional[Dict[str, Any]],
    *,
    timed_out: bool,
    decision_elapsed_sec: float,
    slow_threshold_sec: Optional[float] = None,
) -> Dict[str, Any]:
    result = update_decision_timeout_penalty_state(
        state,
        timed_out=timed_out,
        decision_elapsed_sec=decision_elapsed_sec,
        slow_threshold_sec=slow_threshold_sec,
    )
    if "effective_decision_timeout_sec" in result:
        result["effective_native_timeout_sec"] = result["effective_decision_timeout_sec"]
    return result


def native_timeout_penalty_snapshot(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return decision_timeout_penalty_snapshot(state)
