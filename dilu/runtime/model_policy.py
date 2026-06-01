import fnmatch
import os
from typing import Any, Dict, List, Optional, Tuple


POLICY_TIMEOUT_FIELDS = ("decision_timeout_sec",)
DEPRECATED_POLICY_FIELDS = (
    "decision_max_output_tokens",
    "disable_streaming",
    "disable_checker_llm",
    "ollama_think_mode",
    "ollama_use_native_chat",
    "ollama_native_chat_timeout_sec",
)
EVAL_TIMEOUT_POLICY_MODE_DEFAULT = "laddered"
EVAL_TIMEOUT_LADDER_SEC_DEFAULT = [15.0, 20.0, 30.0]
EVAL_TIMEOUT_RECOVERY_SUCCESSES_DEFAULT = 3
EVAL_FIRST_RESPONSE_WARMUP_TIMEOUT_SEC_DEFAULT = 60.0
EVAL_FIRST_RESPONSE_WARMUP_MAX_FAILURES_DEFAULT = 2
REMOTE_WARMUP_PROVIDERS = {"openai", "azure", "gemini"}


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


def _normalize_eval_timeout_policy_mode(value: Any) -> str:
    text = str(value or EVAL_TIMEOUT_POLICY_MODE_DEFAULT).strip().lower()
    if text in {"off", "disabled", "none", "false", "0"}:
        return "disabled"
    return "legacy" if text in {"legacy", "adaptive", "shrink_only"} else "laddered"


def _normalize_timeout_seconds(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed <= 0.0:
        return 0.0
    return float(parsed)


def _normalize_timeout_ladder(value: Any) -> List[float]:
    raw = value if value is not None else EVAL_TIMEOUT_LADDER_SEC_DEFAULT
    if isinstance(raw, str):
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        values = [float(token) for token in tokens]
    else:
        values = [float(item) for item in list(raw)]
    if not values:
        values = list(EVAL_TIMEOUT_LADDER_SEC_DEFAULT)
    deduped: List[float] = []
    for item in values:
        item = max(1.0, float(item))
        if not deduped or float(item) != float(deduped[-1]):
            deduped.append(float(item))
    deduped = sorted(deduped)
    if len(deduped) < 2:
        deduped = list(EVAL_TIMEOUT_LADDER_SEC_DEFAULT)
    return deduped


def _normalize_provider(provider: Any) -> str:
    return str(provider or "").strip().lower()


def _provider_uses_eval_warmup(provider: str, mode: str, policy_mode: str) -> bool:
    return (
        str(mode or "").strip().lower() == "eval"
        and str(policy_mode or "").strip().lower() == "laddered"
        and str(provider or "").strip().lower() in REMOTE_WARMUP_PROVIDERS
    )


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
        decision_timeout_sec = _normalize_timeout_seconds(
            config.get("runtime_decision_timeout_sec", 60.0),
            default=60.0,
        )
    else:
        decision_timeout_sec = _normalize_timeout_seconds(
            config.get("eval_decision_timeout_sec", 60.0),
            default=60.0,
        )

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
            resolved["decision_timeout_sec"] = _normalize_timeout_seconds(
                shared_values["decision_timeout_sec"],
                default=float(resolved.get("decision_timeout_sec", 60.0) or 60.0),
            )
        deprecated_policy_fields_ignored.extend(
            _collect_deprecated_fields(shared_values, "model_policy_overrides")
        )

    legacy_key = None
    legacy_values: Dict[str, Any] = {}
    if norm_mode == "eval" and not shared_values:
        legacy_key, legacy_values = _match_override(model_name, legacy_eval_overrides)
        if legacy_values:
            if "decision_timeout_sec" in legacy_values and legacy_values["decision_timeout_sec"] is not None:
                resolved["decision_timeout_sec"] = _normalize_timeout_seconds(
                    legacy_values["decision_timeout_sec"],
                    default=float(resolved.get("decision_timeout_sec", 60.0) or 60.0),
                )
            deprecated_policy_fields_ignored.extend(
                _collect_deprecated_fields(legacy_values, "eval_model_overrides")
            )

    cli_values: Dict[str, Any] = {}
    if isinstance(cli_overrides, dict):
        for key in POLICY_TIMEOUT_FIELDS:
            if key in cli_overrides and cli_overrides[key] is not None:
                cli_values[key] = _normalize_timeout_seconds(
                    cli_overrides[key],
                    default=float(resolved.get(key, 60.0) or 60.0),
                )
        if cli_values:
            resolved.update(cli_values)
        deprecated_policy_fields_ignored.extend(_collect_deprecated_fields(cli_overrides, "cli"))

    resolved["decision_timeout_sec"] = _normalize_timeout_seconds(
        resolved.get("decision_timeout_sec", 60.0),
        default=60.0,
    )

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
    policy_mode = (
        _normalize_eval_timeout_policy_mode(config.get("eval_timeout_policy_mode"))
        if norm_mode == "eval"
        else "legacy"
    )
    baseline_timeout_sec = _normalize_timeout_seconds(
        baseline_decision_timeout_sec,
        default=60.0,
    )
    if policy_mode == "disabled" or baseline_timeout_sec <= 0.0:
        return {
            "enabled": False,
            "provider": norm_provider,
            "mode": norm_mode,
            "policy_mode": "disabled",
            "baseline_decision_timeout_sec": baseline_timeout_sec if baseline_timeout_sec > 0.0 else None,
            "effective_decision_timeout_sec": baseline_timeout_sec if baseline_timeout_sec > 0.0 else None,
            "min_timeout_sec": None,
            "halving_factor": None,
            "trigger_consecutive_slow": None,
            "slow_threshold_sec": None,
            "stage": 0,
            "timeout_ladder_sec": None,
            "current_level_index": None,
            "recovery_successes_required": None,
            "recovery_success_streak": None,
            "penalty_events": 0,
            "timeout_triggers": 0,
            "slow_triggers": 0,
            "warmup_active": False,
            "warmup_success_observed": True,
            "warmup_timeout_sec": None,
            "warmup_timeout_failures": 0,
            "warmup_max_failures": None,
            "timeout_phase": "disabled",
        }
    if norm_mode == "eval" and policy_mode == "laddered":
        ladder = _normalize_timeout_ladder(config.get("eval_timeout_ladder_sec"))
        recovery_successes = _clamp_int(
            config.get("eval_timeout_recovery_successes", EVAL_TIMEOUT_RECOVERY_SUCCESSES_DEFAULT),
            default=EVAL_TIMEOUT_RECOVERY_SUCCESSES_DEFAULT,
            minimum=1,
        )
        initial_timeout_sec = float(ladder[0])
        warmup_enabled = _provider_uses_eval_warmup(norm_provider, norm_mode, policy_mode)
        warmup_timeout_sec = _clamp_float(
            config.get(
                "eval_first_response_warmup_timeout_sec",
                EVAL_FIRST_RESPONSE_WARMUP_TIMEOUT_SEC_DEFAULT,
            ),
            default=EVAL_FIRST_RESPONSE_WARMUP_TIMEOUT_SEC_DEFAULT,
            minimum=1.0,
        )
        warmup_max_failures = _clamp_int(
            config.get(
                "eval_first_response_warmup_max_failures",
                EVAL_FIRST_RESPONSE_WARMUP_MAX_FAILURES_DEFAULT,
            ),
            default=EVAL_FIRST_RESPONSE_WARMUP_MAX_FAILURES_DEFAULT,
            minimum=1,
        )
        return {
            "enabled": True,
            "provider": norm_provider,
            "mode": norm_mode,
            "policy_mode": "laddered",
            "baseline_decision_timeout_sec": initial_timeout_sec,
            "effective_decision_timeout_sec": warmup_timeout_sec if warmup_enabled else initial_timeout_sec,
            "min_timeout_sec": initial_timeout_sec,
            "halving_factor": None,
            "trigger_consecutive_slow": None,
            "slow_threshold_sec": None,
            "stage": 0,
            "timeout_ladder_sec": list(ladder),
            "current_level_index": 0,
            "recovery_successes_required": int(recovery_successes),
            "recovery_success_streak": 0,
            "penalty_events": 0,
            "timeout_triggers": 0,
            "slow_triggers": 0,
            "warmup_active": bool(warmup_enabled),
            "warmup_success_observed": not bool(warmup_enabled),
            "warmup_timeout_sec": warmup_timeout_sec if warmup_enabled else None,
            "warmup_timeout_failures": 0,
            "warmup_max_failures": int(warmup_max_failures),
            "timeout_phase": "warmup" if warmup_enabled else "active",
        }

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
        baseline_timeout_sec,
        default=min_timeout_sec,
        minimum=min_timeout_sec,
    )
    return {
        "enabled": bool(enabled),
        "provider": norm_provider,
        "mode": norm_mode,
        "policy_mode": "legacy",
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
        "warmup_active": False,
        "warmup_success_observed": True,
        "warmup_timeout_sec": None,
        "warmup_timeout_failures": 0,
        "warmup_max_failures": None,
        "timeout_phase": "active",
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
            raw_effective = state.get("effective_decision_timeout_sec", state.get("baseline_decision_timeout_sec"))
            if raw_effective is not None:
                effective = _clamp_float(
                    raw_effective,
                    default=4.0,
                    minimum=1.0,
                )
        return {
            "escalated": False,
            "reason": None,
            "stage": 0 if not isinstance(state, dict) else int(state.get("stage", 0) or 0),
            "effective_decision_timeout_sec": effective,
            "consecutive_slow_count": 0 if not isinstance(state, dict) else int(state.get("consecutive_slow_count", 0) or 0),
            "policy_mode": None if not isinstance(state, dict) else state.get("policy_mode"),
            "recovered": False,
            "warmup_completed": False,
            "warmup_active": False if not isinstance(state, dict) else bool(state.get("warmup_active", False)),
            "warmup_timeout_failures": 0 if not isinstance(state, dict) else int(state.get("warmup_timeout_failures", 0) or 0),
            "timeout_phase": None if not isinstance(state, dict) else state.get("timeout_phase"),
        }

    if state.get("policy_mode") == "laddered":
        ladder = _normalize_timeout_ladder(state.get("timeout_ladder_sec"))
        current_idx = _clamp_int(state.get("current_level_index", 0), default=0, minimum=0)
        current_idx = min(current_idx, len(ladder) - 1)
        recovery_required = _clamp_int(
            state.get("recovery_successes_required", EVAL_TIMEOUT_RECOVERY_SUCCESSES_DEFAULT),
            default=EVAL_TIMEOUT_RECOVERY_SUCCESSES_DEFAULT,
            minimum=1,
        )
        escalated = False
        recovered = False
        reason = None
        warmup_completed = False

        if bool(state.get("warmup_active", False)):
            warmup_timeout_sec = _clamp_float(
                state.get("warmup_timeout_sec", EVAL_FIRST_RESPONSE_WARMUP_TIMEOUT_SEC_DEFAULT),
                default=EVAL_FIRST_RESPONSE_WARMUP_TIMEOUT_SEC_DEFAULT,
                minimum=1.0,
            )
            if bool(timed_out):
                state["warmup_timeout_failures"] = _clamp_int(
                    state.get("warmup_timeout_failures", 0),
                    default=0,
                    minimum=0,
                ) + 1
                state["effective_decision_timeout_sec"] = float(warmup_timeout_sec)
                state["timeout_phase"] = "warmup"
                return {
                    "escalated": False,
                    "recovered": False,
                    "reason": "warmup_timeout",
                    "stage": int(state.get("stage", 0) or 0),
                    "effective_decision_timeout_sec": float(state["effective_decision_timeout_sec"]),
                    "consecutive_slow_count": 0,
                    "policy_mode": "laddered",
                    "warmup_completed": False,
                    "warmup_active": True,
                    "warmup_timeout_failures": int(state.get("warmup_timeout_failures", 0) or 0),
                    "timeout_phase": "warmup",
                }

            state["warmup_active"] = False
            state["warmup_success_observed"] = True
            state["effective_decision_timeout_sec"] = float(ladder[current_idx])
            state["timeout_phase"] = "active"
            state["recovery_success_streak"] = 0
            warmup_completed = True

        if bool(timed_out):
            state["timeout_triggers"] = _clamp_int(state.get("timeout_triggers", 0), default=0, minimum=0) + 1
            state["recovery_success_streak"] = 0
            next_idx = min(current_idx + 1, len(ladder) - 1)
            if next_idx > current_idx:
                current_idx = next_idx
                escalated = True
                reason = "timeout"
                state["penalty_events"] = _clamp_int(state.get("penalty_events", 0), default=0, minimum=0) + 1
        else:
            state["recovery_success_streak"] = _clamp_int(
                state.get("recovery_success_streak", 0), default=0, minimum=0
            ) + 1
            if current_idx > 0 and int(state["recovery_success_streak"]) >= recovery_required:
                current_idx -= 1
                recovered = True
                reason = "recovery"
                state["recovery_success_streak"] = 0

        state["current_level_index"] = int(current_idx)
        state["stage"] = int(current_idx)
        state["effective_decision_timeout_sec"] = float(ladder[current_idx])
        state["baseline_decision_timeout_sec"] = float(ladder[0])
        state["min_timeout_sec"] = float(ladder[0])
        state["timeout_ladder_sec"] = list(ladder)
        state["timeout_phase"] = "active"
        return {
            "escalated": escalated,
            "recovered": recovered,
            "reason": reason,
            "stage": int(state["stage"]),
            "effective_decision_timeout_sec": float(state["effective_decision_timeout_sec"]),
            "consecutive_slow_count": 0,
            "policy_mode": "laddered",
            "warmup_completed": bool(warmup_completed),
            "warmup_active": bool(state.get("warmup_active", False)),
            "warmup_timeout_failures": int(state.get("warmup_timeout_failures", 0) or 0),
            "timeout_phase": str(state.get("timeout_phase") or "active"),
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
        "recovered": False,
        "reason": reason,
        "stage": int(state["stage"]),
        "effective_decision_timeout_sec": float(state["effective_decision_timeout_sec"]),
        "consecutive_slow_count": int(state["consecutive_slow_count"]),
        "policy_mode": "legacy",
        "warmup_completed": False,
        "warmup_active": False,
        "warmup_timeout_failures": int(state.get("warmup_timeout_failures", 0) or 0),
        "timeout_phase": str(state.get("timeout_phase") or "active"),
    }


def decision_timeout_penalty_snapshot(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {
            "enabled": False,
            "policy_mode": None,
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
            "timeout_ladder_sec": None,
            "current_level_index": None,
            "recovery_successes_required": None,
            "recovery_success_streak": None,
            "warmup_active": False,
            "warmup_success_observed": False,
            "warmup_timeout_sec": None,
            "warmup_timeout_failures": 0,
            "warmup_max_failures": None,
            "timeout_phase": None,
            # Compatibility aliases (deprecated)
            "baseline_native_timeout_sec": None,
            "effective_native_timeout_sec": None,
        }
    return {
        "enabled": bool(state.get("enabled", False)),
        "policy_mode": state.get("policy_mode"),
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
        "timeout_ladder_sec": (
            [float(value) for value in state.get("timeout_ladder_sec", [])]
            if state.get("timeout_ladder_sec") is not None
            else None
        ),
        "current_level_index": (
            int(state["current_level_index"])
            if state.get("current_level_index") is not None
            else None
        ),
        "recovery_successes_required": (
            int(state["recovery_successes_required"])
            if state.get("recovery_successes_required") is not None
            else None
        ),
        "recovery_success_streak": (
            int(state["recovery_success_streak"])
            if state.get("recovery_success_streak") is not None
            else None
        ),
        "warmup_active": bool(state.get("warmup_active", False)),
        "warmup_success_observed": bool(state.get("warmup_success_observed", False)),
        "warmup_timeout_sec": (
            float(state["warmup_timeout_sec"])
            if state.get("warmup_timeout_sec") is not None
            else None
        ),
        "warmup_timeout_failures": int(state.get("warmup_timeout_failures", 0) or 0),
        "warmup_max_failures": (
            int(state["warmup_max_failures"])
            if state.get("warmup_max_failures") is not None
            else None
        ),
        "timeout_phase": state.get("timeout_phase"),
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
