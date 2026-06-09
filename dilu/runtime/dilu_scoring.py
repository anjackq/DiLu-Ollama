import math
from typing import Any, Dict, Iterable, List, Optional


SPLIT_SCORING_POLICY_VERSION = "dilu_split_score_v1.2"
BALANCED_DRIVING_SCORE_POLICY_VERSION = "dilu_balanced_driving_score_v1"

SPLIT_SCORE_FIELDS = (
    "driving_score_balanced_v1",
    "driving_task_score_v2",
    "driving_behavior_task_gap_v1",
    "driving_score_behavior_v1",
    "driving_safety_score_v1",
    "driving_comfort_score_v1",
    "driving_efficiency_score_v1",
    "llm_driver_score_v1",
    "llm_output_contract_score_v1",
    "llm_runtime_reliability_score_v1",
    "llm_action_validity_score_v1",
    "llm_flow_recovery_independence_score_v1",
    "llm_safety_intervention_independence_score_v1",
    "llm_parser_independence_score_v1",
    "llm_intervention_independence_score_v1",
    "llm_latency_score_v1",
    "llm_resource_efficiency_score_v1",
    "dilu_joint_score_v1",
)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _int_or_zero(value: Any) -> int:
    number = _float_or_none(value)
    if number is None:
        return 0
    return int(number)


def _clamp01(value: Any) -> float:
    number = _float_or_none(value)
    if number is None:
        return 0.0
    return max(0.0, min(1.0, float(number)))


def _round_score(value: Any) -> float:
    return round(_clamp01(value), 4)


def _round_optional_score(value: Any) -> Optional[float]:
    number = _float_or_none(value)
    if number is None:
        return None
    return _round_score(number)


def _round_optional_float(value: Any) -> Optional[float]:
    number = _float_or_none(value)
    if number is None:
        return None
    return round(float(number), 4)


def _rate_score(value: Any, *, cap: float) -> float:
    cap = max(float(cap), 1e-9)
    return _clamp01(1.0 - float(_float_or_none(value) or 0.0) / cap)


def _less_is_better(value: Any, *, good: float, bad: float) -> Optional[float]:
    number = _float_or_none(value)
    if number is None:
        return None
    if number <= good:
        return 1.0
    if number >= bad:
        return 0.0
    return _clamp01(1.0 - (number - good) / max(1e-9, bad - good))


def _more_is_better(value: Any, *, poor: float, good: float) -> Optional[float]:
    number = _float_or_none(value)
    if number is None:
        return None
    if number >= good:
        return 1.0
    if number <= poor:
        return 0.0
    return _clamp01((number - poor) / max(1e-9, good - poor))


def _weighted_average(weighted_values: Iterable[tuple[float, Optional[float]]]) -> float:
    total_weight = 0.0
    total_value = 0.0
    for weight, value in weighted_values:
        if value is None:
            continue
        weight = max(0.0, float(weight))
        total_weight += weight
        total_value += weight * _clamp01(value)
    if total_weight <= 0.0:
        return 1.0
    return _clamp01(total_value / total_weight)


def _safe_rate(numerator: Any, denominator: Any) -> float:
    denom = max(1, _int_or_zero(denominator))
    return float(_int_or_zero(numerator)) / float(denom)


def _available_mean(values: List[Optional[float]]) -> Optional[float]:
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return None
    return _clamp01(sum(clean_values) / len(clean_values))


def _select_driving_task_score(episode: Dict[str, Any]) -> tuple[Optional[float], Optional[str]]:
    for key in ("driving_score_v2", "driving_score"):
        value = _float_or_none(episode.get(key))
        if value is not None:
            return _clamp01(value), key
    return None, None


def _episode_decisions(episode: Dict[str, Any]) -> int:
    return max(
        1,
        _int_or_zero(
            episode.get("decision_calls_total", episode.get("decisions_made", 0))
        ),
    )


def _driving_safety_score(episode: Dict[str, Any]) -> float:
    decisions = _episode_decisions(episode)
    unsafe_lane_rate = _safe_rate(
        episode.get(
            "unsafe_lane_change_attempt_count",
            episode.get("benchmark_unsafe_lane_change_attempts", 0),
        ),
        decisions,
    )
    unsafe_longitudinal_rate = _safe_rate(
        episode.get("unsafe_longitudinal_action_attempt_count", 0),
        decisions,
    )
    no_crash_score = 0.35 if bool(episode.get("crashed", False)) else 1.0
    return _weighted_average(
        (
            (0.30, no_crash_score),
            (0.20, _rate_score(episode.get("ttc_danger_rate"), cap=0.5)),
            (0.15, _rate_score(episode.get("headway_violation_rate"), cap=0.5)),
            (0.10, _rate_score(episode.get("rear_ttc_danger_rate"), cap=0.5)),
            (0.10, _rate_score(episode.get("rear_headway_violation_rate"), cap=0.5)),
            (0.075, _rate_score(unsafe_lane_rate, cap=0.2)),
            (0.075, _rate_score(unsafe_longitudinal_rate, cap=0.2)),
        )
    )


def _driving_efficiency_score(episode: Dict[str, Any]) -> float:
    task_score = 1.0 if bool(episode.get("task_completed", episode.get("success_no_collision", False))) else 0.0
    time_score = _float_or_none(episode.get("time_efficiency_score"))
    speed_score = _more_is_better(episode.get("avg_ego_speed_mps"), poor=8.0, good=25.0)
    progress_score = _more_is_better(episode.get("benchmark_max_progress_m"), poor=0.0, good=120.0)
    return _weighted_average(
        (
            (0.35, task_score),
            (0.25, time_score),
            (0.25, speed_score),
            (0.15, progress_score),
        )
    )


def _driving_comfort_score(episode: Dict[str, Any]) -> float:
    speed_variance_score = _float_or_none(episode.get("speed_variance_score"))
    if speed_variance_score is None and episode.get("benchmark_speed_std_mps") is not None:
        speed_variance_score = _rate_score(episode.get("benchmark_speed_std_mps"), cap=4.0)
    lane_change_rate = float(_float_or_none(episode.get("lane_change_rate")) or 0.0)
    excessive_lane_change_score = _rate_score(max(0.0, lane_change_rate - 0.15), cap=0.35)
    return _weighted_average(
        (
            (0.25, speed_variance_score),
            (0.20, _rate_score(episode.get("low_speed_blocking_rate"), cap=0.5)),
            (0.15, _rate_score(episode.get("stop_rate"), cap=0.3)),
            (0.10, _rate_score(episode.get("near_stop_rate"), cap=0.5)),
            (0.20, _rate_score(episode.get("flap_accel_decel_rate"), cap=0.3)),
            (0.10, excessive_lane_change_score),
        )
    )


def _llm_output_contract_score(episode: Dict[str, Any]) -> float:
    decisions = max(1, _int_or_zero(episode.get("decisions_made", episode.get("decision_calls_total", 0))))
    strict_rate = _safe_rate(episode.get("responses_strict_format", 0), decisions)
    delimiter_rate = _safe_rate(episode.get("responses_with_delimiter", 0), decisions)
    direct_rate = _safe_rate(episode.get("responses_direct_parseable", 0), decisions)
    format_success_rate = 1.0 - _safe_rate(episode.get("format_failure_count", 0), decisions)
    return _weighted_average(
        (
            (0.45, strict_rate),
            (0.20, delimiter_rate),
            (0.25, direct_rate),
            (0.10, format_success_rate),
        )
    )


def _llm_runtime_reliability_score(episode: Dict[str, Any]) -> float:
    decisions = _episode_decisions(episode)
    timeout_rate = _float_or_none(episode.get("decision_timeout_rate"))
    if timeout_rate is None:
        timeout_rate = _safe_rate(episode.get("decision_timeout_count", 0), decisions)
    fallback_rate = _float_or_none(episode.get("fallback_action_rate"))
    if fallback_rate is None:
        fallback_rate = _safe_rate(episode.get("fallback_action_count", 0), decisions)
    no_timeout_cap_score = 0.0 if episode.get("episode_stop_reason") == "episode_timeout_cap" else 1.0
    quarantine_score = 0.0 if bool(episode.get("model_quarantined_due_to_timeout_collapse", False)) else 1.0
    return _weighted_average(
        (
            (0.35, _rate_score(timeout_rate, cap=0.3)),
            (0.35, _rate_score(fallback_rate, cap=0.3)),
            (0.20, no_timeout_cap_score),
            (0.10, quarantine_score),
        )
    )


def _llm_action_validity_score(episode: Dict[str, Any]) -> float:
    diagnostics = _llm_intervention_independence_scores(episode)
    return diagnostics["llm_intervention_independence_score_v1"]


def _llm_intervention_independence_scores(episode: Dict[str, Any]) -> Dict[str, float]:
    decisions = _episode_decisions(episode)
    flow_recovery_rate = _safe_rate(episode.get("flow_recovery_shield_count", 0), decisions)
    lane_shield_rate = _safe_rate(episode.get("lane_change_shield_count", 0), decisions)
    longitudinal_shield_rate = _safe_rate(episode.get("longitudinal_safety_shield_count", 0), decisions)
    unsafe_lane_rate = _safe_rate(episode.get("unsafe_lane_change_attempt_count", 0), decisions)
    unsafe_longitudinal_rate = _safe_rate(episode.get("unsafe_longitudinal_action_attempt_count", 0), decisions)
    semantic_recovery_rate = _safe_rate(episode.get("semantic_recovery_count", 0), decisions)
    intent_resolver_rate = _safe_rate(episode.get("intent_resolver_used_count", 0), decisions)

    flow_recovery_independence = _rate_score(flow_recovery_rate, cap=0.20)
    safety_intervention_independence = _weighted_average(
        (
            (0.30, _rate_score(lane_shield_rate, cap=0.25)),
            (0.30, _rate_score(longitudinal_shield_rate, cap=0.25)),
            (0.20, _rate_score(unsafe_lane_rate, cap=0.25)),
            (0.20, _rate_score(unsafe_longitudinal_rate, cap=0.25)),
        )
    )
    parser_independence = _weighted_average(
        (
            (0.50, _rate_score(semantic_recovery_rate, cap=0.50)),
            (0.50, _rate_score(intent_resolver_rate, cap=0.50)),
        )
    )
    intervention_independence = min(
        flow_recovery_independence,
        safety_intervention_independence,
        parser_independence,
    )
    return {
        "llm_flow_recovery_independence_score_v1": _round_score(flow_recovery_independence),
        "llm_safety_intervention_independence_score_v1": _round_score(
            safety_intervention_independence
        ),
        "llm_parser_independence_score_v1": _round_score(parser_independence),
        "llm_intervention_independence_score_v1": _round_score(intervention_independence),
    }


def _llm_latency_score(episode: Dict[str, Any]) -> float:
    return _weighted_average(
        (
            (0.55, _less_is_better(episode.get("decision_latency_ms_avg"), good=1000.0, bad=8000.0)),
            (0.45, _less_is_better(episode.get("p95_decision_latency_sec"), good=1.5, bad=12.0)),
        )
    )


def _llm_resource_efficiency_score(episode: Dict[str, Any]) -> float:
    decisions = _episode_decisions(episode)
    total_tokens = _float_or_none(episode.get("total_tokens", episode.get("total_tokens_total")))
    completion_tokens = _float_or_none(
        episode.get("completion_tokens_total", episode.get("tokens_generated_total"))
    )
    token_per_decision_score = None
    if total_tokens is not None:
        token_per_decision_score = _less_is_better(total_tokens / decisions, good=500.0, bad=4000.0)
    completion_per_decision_score = None
    if completion_tokens is not None:
        completion_per_decision_score = _less_is_better(completion_tokens / decisions, good=64.0, bad=512.0)
    tokens_per_second_score = _more_is_better(episode.get("tokens_per_second"), poor=2.0, good=20.0)
    energy_per_decision_score = _less_is_better(
        episode.get("energy_per_decision_j"),
        good=0.1,
        bad=10.0,
    )
    energy_per_token_score = _less_is_better(
        episode.get("energy_per_token_j"),
        good=0.001,
        bad=0.1,
    )
    return _weighted_average(
        (
            (0.30, token_per_decision_score),
            (0.20, completion_per_decision_score),
            (0.25, tokens_per_second_score),
            (0.15, energy_per_decision_score),
            (0.10, energy_per_token_score),
        )
    )


def compute_split_scores_for_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    safety_score = _driving_safety_score(episode)
    efficiency_score = _driving_efficiency_score(episode)
    comfort_score = _driving_comfort_score(episode)
    driving_score = _weighted_average(
        (
            (0.45, safety_score),
            (0.30, efficiency_score),
            (0.25, comfort_score),
        )
    )

    output_contract_score = _llm_output_contract_score(episode)
    runtime_reliability_score = _llm_runtime_reliability_score(episode)
    action_validity_score = _llm_action_validity_score(episode)
    intervention_scores = _llm_intervention_independence_scores(episode)
    latency_score = _llm_latency_score(episode)
    resource_efficiency_score = _llm_resource_efficiency_score(episode)
    llm_score = _weighted_average(
        (
            (0.30, output_contract_score),
            (0.25, runtime_reliability_score),
            (0.20, action_validity_score),
            (0.15, latency_score),
            (0.10, resource_efficiency_score),
        )
    )
    joint_score = math.sqrt(_clamp01(driving_score) * _clamp01(llm_score))
    task_score, task_score_source = _select_driving_task_score(episode)
    balanced_driving_score = None
    behavior_task_gap = None
    if task_score is not None:
        balanced_driving_score = math.sqrt(_clamp01(driving_score) * _clamp01(task_score))
        behavior_task_gap = _clamp01(driving_score) - _clamp01(task_score)

    scored = dict(episode)
    scored.update(
        {
            "split_scoring_policy_version": SPLIT_SCORING_POLICY_VERSION,
            "balanced_driving_score_policy_version": BALANCED_DRIVING_SCORE_POLICY_VERSION,
            "driving_score_balanced_v1": _round_optional_score(balanced_driving_score),
            "driving_task_score_v2": _round_optional_score(task_score),
            "driving_task_score_source": task_score_source,
            "driving_behavior_task_gap_v1": _round_optional_float(behavior_task_gap),
            "driving_score_behavior_v1": _round_score(driving_score),
            "driving_safety_score_v1": _round_score(safety_score),
            "driving_comfort_score_v1": _round_score(comfort_score),
            "driving_efficiency_score_v1": _round_score(efficiency_score),
            "llm_driver_score_v1": _round_score(llm_score),
            "llm_output_contract_score_v1": _round_score(output_contract_score),
            "llm_runtime_reliability_score_v1": _round_score(runtime_reliability_score),
            "llm_action_validity_score_v1": _round_score(action_validity_score),
            **intervention_scores,
            "llm_latency_score_v1": _round_score(latency_score),
            "llm_resource_efficiency_score_v1": _round_score(resource_efficiency_score),
            "dilu_joint_score_v1": _round_score(joint_score),
        }
    )
    return scored
