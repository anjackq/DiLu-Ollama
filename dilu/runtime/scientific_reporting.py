import math
import random
import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .dilu_scoring import SPLIT_SCORING_POLICY_VERSION, SPLIT_SCORE_FIELDS
from .scientific_artifacts import write_scientific_analysis_artifacts


SCIENTIFIC_REPORTING_VERSION = "scientific_reporting_v1"
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260327

DEFAULT_PRIMARY_METRIC_THRESHOLDS = {
    "minimum_claim_episodes": 30,
    "max_crashes": 0,
    "min_no_collision_rate": 1.0,
    "max_ttc_danger_rate": 0.0,
    "max_headway_violation_rate": 0.0,
    "max_rear_ttc_danger_rate": 0.0,
    "max_rear_headway_violation_rate": 0.0,
    "max_decision_timeout_rate": 0.05,
    "max_fallback_action_rate": 0.05,
    "min_response_strict_format_rate": 0.80,
}

CONTINUOUS_EPISODE_METRICS: Tuple[Tuple[str, str], ...] = (
    ("ttc_danger_rate", "ttc_danger_rate"),
    ("headway_violation_rate", "headway_violation_rate"),
    ("rear_ttc_danger_rate", "rear_ttc_danger_rate"),
    ("rear_headway_violation_rate", "rear_headway_violation_rate"),
    ("avg_ego_speed_mps", "avg_ego_speed_mps"),
    ("avg_reward_per_step", "episode_reward_avg"),
    ("avg_steps", "steps"),
    ("task_completion_rate", "task_completed"),
    ("driving_score", "driving_score"),
    ("driving_score_v2", "driving_score_v2"),
    ("overall_score_v2", "overall_score_v2"),
    ("driving_score_balanced_v1", "driving_score_balanced_v1"),
    ("driving_task_score_v2", "driving_task_score_v2"),
    ("driving_behavior_task_gap_v1", "driving_behavior_task_gap_v1"),
    ("driving_score_behavior_v1", "driving_score_behavior_v1"),
    ("driving_safety_score_v1", "driving_safety_score_v1"),
    ("driving_comfort_score_v1", "driving_comfort_score_v1"),
    ("driving_efficiency_score_v1", "driving_efficiency_score_v1"),
    ("llm_driver_score_v1", "llm_driver_score_v1"),
    ("llm_output_contract_score_v1", "llm_output_contract_score_v1"),
    ("llm_runtime_reliability_score_v1", "llm_runtime_reliability_score_v1"),
    ("llm_action_validity_score_v1", "llm_action_validity_score_v1"),
    ("llm_flow_recovery_independence_score_v1", "llm_flow_recovery_independence_score_v1"),
    (
        "llm_safety_intervention_independence_score_v1",
        "llm_safety_intervention_independence_score_v1",
    ),
    ("llm_parser_independence_score_v1", "llm_parser_independence_score_v1"),
    ("llm_intervention_independence_score_v1", "llm_intervention_independence_score_v1"),
    ("llm_latency_score_v1", "llm_latency_score_v1"),
    ("llm_resource_efficiency_score_v1", "llm_resource_efficiency_score_v1"),
    ("dilu_joint_score_v1", "dilu_joint_score_v1"),
    ("decision_timeout_rate", "decision_timeout_rate"),
    ("fallback_action_rate", "fallback_action_rate"),
    ("response_strict_format_rate", "response_strict_format_rate"),
    ("decision_latency_ms_avg", "decision_latency_ms_avg"),
    ("low_speed_blocking_rate", "low_speed_blocking_rate"),
    ("flap_accel_decel_rate", "flap_accel_decel_rate"),
    ("lane_change_shield_rate", "lane_change_shield_rate"),
    ("longitudinal_safety_shield_rate", "longitudinal_safety_shield_rate"),
    ("flow_recovery_shield_rate", "flow_recovery_shield_rate"),
)

BINARY_EPISODE_METRICS: Tuple[str, ...] = (
    "crashed",
    "success_no_collision",
    "task_completed",
)


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


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


def bootstrap_ci95(
    values: Iterable[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Optional[List[float]]:
    clean_values = [float(value) for value in values if _float_or_none(value) is not None]
    if not clean_values:
        return None
    if len(clean_values) == 1:
        only = _round(clean_values[0])
        return [only, only]

    rng = random.Random(int(seed))
    sample_count = len(clean_values)
    boot_means = []
    for _ in range(max(1, int(iterations))):
        sample = [clean_values[rng.randrange(sample_count)] for _ in range(sample_count)]
        boot_means.append(sum(sample) / sample_count)
    boot_means.sort()
    lower_idx = max(0, min(len(boot_means) - 1, int(math.floor(0.025 * (len(boot_means) - 1)))))
    upper_idx = max(0, min(len(boot_means) - 1, int(math.ceil(0.975 * (len(boot_means) - 1)))))
    return [_round(boot_means[lower_idx]), _round(boot_means[upper_idx])]


def wilson_ci95(successes: int, total: int) -> Optional[List[float]]:
    total = int(total)
    successes = int(successes)
    if total <= 0:
        return None
    successes = max(0, min(successes, total))
    z = 1.96
    phat = successes / total
    denom = 1.0 + (z * z / total)
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return [_round(lower), _round(upper)]


def continuous_metric_summary(values: Iterable[Any]) -> Dict[str, Any]:
    clean_values = [float(value) for value in values if _float_or_none(value) is not None]
    warnings: List[str] = []
    if not clean_values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "ci95": None,
            "warnings": ["missing_sample_warning"],
        }
    if len(clean_values) == 1:
        warnings.append("low_n_warning")

    return {
        "n": len(clean_values),
        "mean": _round(statistics.mean(clean_values)),
        "std": _round(statistics.stdev(clean_values)) if len(clean_values) > 1 else 0.0,
        "median": _round(statistics.median(clean_values)),
        "min": _round(min(clean_values)),
        "max": _round(max(clean_values)),
        "ci95": bootstrap_ci95(clean_values),
        "warnings": warnings,
    }


def binary_metric_summary(values: Iterable[Any]) -> Dict[str, Any]:
    clean_values = [bool(value) for value in values if value is not None]
    total = len(clean_values)
    count = sum(1 for value in clean_values if value)
    return {
        "count": count,
        "total": total,
        "rate": _round(count / total) if total else None,
        "ci95": wilson_ci95(count, total),
        "warnings": [] if total else ["missing_sample_warning"],
    }


def build_primary_metric_spec(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = config or {}
    thresholds = dict(DEFAULT_PRIMARY_METRIC_THRESHOLDS)
    override_map = {
        "scientific_min_claim_episodes": "minimum_claim_episodes",
        "scientific_max_decision_timeout_rate": "max_decision_timeout_rate",
        "scientific_max_fallback_action_rate": "max_fallback_action_rate",
        "scientific_min_response_strict_format_rate": "min_response_strict_format_rate",
        "scientific_max_ttc_danger_rate": "max_ttc_danger_rate",
        "scientific_max_headway_violation_rate": "max_headway_violation_rate",
        "scientific_max_rear_ttc_danger_rate": "max_rear_ttc_danger_rate",
        "scientific_max_rear_headway_violation_rate": "max_rear_headway_violation_rate",
    }
    for config_key, threshold_key in override_map.items():
        if config.get(config_key) is not None:
            value = _float_or_none(config.get(config_key))
            if value is not None:
                thresholds[threshold_key] = int(value) if threshold_key == "minimum_claim_episodes" else value

    return {
        "version": SCIENTIFIC_REPORTING_VERSION,
        "policy": SPLIT_SCORING_POLICY_VERSION,
        "validity_policy": "safety_gated_claim_validity_v1",
        "minimum_claim_episodes": int(thresholds["minimum_claim_episodes"]),
        "safety_gates": {
            "max_crashes": int(thresholds["max_crashes"]),
            "min_no_collision_rate": float(thresholds["min_no_collision_rate"]),
            "max_ttc_danger_rate": float(thresholds["max_ttc_danger_rate"]),
            "max_headway_violation_rate": float(thresholds["max_headway_violation_rate"]),
            "max_rear_ttc_danger_rate": float(thresholds["max_rear_ttc_danger_rate"]),
            "max_rear_headway_violation_rate": float(thresholds["max_rear_headway_violation_rate"]),
        },
        "runtime_gates": {
            "max_decision_timeout_rate": float(thresholds["max_decision_timeout_rate"]),
            "max_fallback_action_rate": float(thresholds["max_fallback_action_rate"]),
            "min_response_strict_format_rate": float(thresholds["min_response_strict_format_rate"]),
        },
        "headline_metric_order": {
            "benchmark": [
                "driving_score_balanced_v1",
                "driving_score_behavior_v1",
                "driving_score_v2",
                "driving_score",
                "task_completion_rate",
            ],
            "seed_mode": ["avg_ego_speed_mps"],
        },
        "split_headline_metrics": {
            "primary_driving_metric": "driving_score_balanced_v1",
            "driving_behavior_component": "driving_score_behavior_v1",
            "driving_task_component": "driving_score_v2",
            "balanced_driving_formula": "sqrt(driving_score_behavior_v1 * driving_score_v2)",
            "primary_llm_metric": "llm_driver_score_v1",
            "secondary_joint_metric": "dilu_joint_score_v1",
            "score_fields": list(SPLIT_SCORE_FIELDS),
        },
        "legacy_compatibility_metrics": ["driving_score", "driving_score_v2"],
        "reward_primary_metric_allowed": False,
        "hypothesis_tests_enabled": False,
        "hypothesis_tests": [],
    }


def _episode_runtime_valid(episode: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    runtime_gates = spec["runtime_gates"]
    calls = max(1, _int_or_zero(episode.get("decision_calls_total", episode.get("decisions_made", 0))))
    decisions = max(1, _int_or_zero(episode.get("decisions_made", episode.get("decision_calls_total", 0))))

    timeout_rate = _float_or_none(episode.get("decision_timeout_rate"))
    if timeout_rate is None:
        timeout_rate = _int_or_zero(episode.get("decision_timeout_count")) / calls
    fallback_rate = _float_or_none(episode.get("fallback_action_rate"))
    if fallback_rate is None:
        fallback_rate = _int_or_zero(episode.get("fallback_action_count")) / calls
    strict_rate = _float_or_none(episode.get("response_strict_format_rate"))
    if strict_rate is None:
        strict_rate = _int_or_zero(episode.get("responses_strict_format")) / decisions

    return (
        timeout_rate <= float(runtime_gates["max_decision_timeout_rate"])
        and fallback_rate <= float(runtime_gates["max_fallback_action_rate"])
        and strict_rate >= float(runtime_gates["min_response_strict_format_rate"])
    )


def _episode_metric_values(episodes: List[Dict[str, Any]], metric_key: str) -> List[Any]:
    values = []
    for episode in episodes:
        if metric_key in episode:
            values.append(episode.get(metric_key))
    return values


def summarize_scientific_statistics(
    episodes: List[Dict[str, Any]],
    primary_metric_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spec = primary_metric_spec or build_primary_metric_spec()
    has_task_completion = any("task_completed" in episode for episode in episodes)
    return {
        "n_episodes": len(episodes),
        "n_completed": (
            sum(1 for episode in episodes if bool(episode.get("task_completed")))
            if has_task_completion
            else None
        ),
        "n_runtime_valid": sum(1 for episode in episodes if _episode_runtime_valid(episode, spec)),
        "continuous_metrics": {
            output_name: continuous_metric_summary(_episode_metric_values(episodes, episode_key))
            for output_name, episode_key in CONTINUOUS_EPISODE_METRICS
        },
        "binary_metrics": {
            metric: binary_metric_summary(_episode_metric_values(episodes, metric))
            for metric in BINARY_EPISODE_METRICS
            if metric != "task_completed" or has_task_completion
        },
        "hypothesis_tests_enabled": False,
        "hypothesis_tests": [],
    }


def _select_primary_metric(aggregate: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    benchmark_keys = spec["headline_metric_order"]["benchmark"]
    seed_keys = spec["headline_metric_order"]["seed_mode"]
    keys = benchmark_keys if any(aggregate.get(key) is not None for key in benchmark_keys) else seed_keys
    for key in keys:
        value = _float_or_none(aggregate.get(key))
        if value is not None:
            return key, _round(value)
    return None, None


def _select_named_metric(aggregate: Dict[str, Any], key: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    if not key:
        return None, None
    value = _float_or_none(aggregate.get(key))
    if value is None:
        return key, None
    return key, _round(value)


def _safety_invalid_reasons(aggregate: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    gates = spec["safety_gates"]
    reasons: List[str] = []
    crashes = _int_or_zero(aggregate.get("crashes"))
    if crashes > int(gates["max_crashes"]):
        reasons.append("crashes>0")
    no_collision_rate = _float_or_none(aggregate.get("no_collision_rate"))
    if no_collision_rate is not None and no_collision_rate < float(gates["min_no_collision_rate"]):
        reasons.append("no_collision_rate<1.0")

    metric_checks = (
        ("ttc_danger_rate_mean", "max_ttc_danger_rate"),
        ("headway_violation_rate_mean", "max_headway_violation_rate"),
        ("rear_ttc_danger_rate_mean", "max_rear_ttc_danger_rate"),
        ("rear_headway_violation_rate_mean", "max_rear_headway_violation_rate"),
    )
    for metric_name, gate_name in metric_checks:
        value = _float_or_none(aggregate.get(metric_name))
        threshold = float(gates[gate_name])
        if value is not None and value > threshold:
            reasons.append(f"{metric_name}>{threshold:g}")
    return reasons


def _runtime_invalid_reasons(aggregate: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    gates = spec["runtime_gates"]
    checks = (
        ("decision_timeout_rate_mean", "max_decision_timeout_rate", ">"),
        ("fallback_action_rate_mean", "max_fallback_action_rate", ">"),
        ("response_strict_format_rate", "min_response_strict_format_rate", "<"),
    )
    reasons: List[str] = []
    for metric_name, gate_name, op in checks:
        value = _float_or_none(aggregate.get(metric_name))
        threshold = float(gates[gate_name])
        if value is None:
            reasons.append(f"{metric_name}_missing")
        elif op == ">" and value > threshold:
            reasons.append(f"{metric_name}>{threshold:g}")
        elif op == "<" and value < threshold:
            reasons.append(f"{metric_name}<{threshold:g}")
    return reasons


def annotate_aggregate_with_scientific_reporting(
    aggregate: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    primary_metric_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spec = primary_metric_spec or build_primary_metric_spec()
    annotated = dict(aggregate)
    metric_name, metric_value = _select_primary_metric(annotated, spec)
    split_headlines = spec.get("split_headline_metrics") or {}
    llm_metric_name, llm_metric_value = _select_named_metric(
        annotated,
        split_headlines.get("primary_llm_metric"),
    )
    joint_metric_name, joint_metric_value = _select_named_metric(
        annotated,
        split_headlines.get("secondary_joint_metric"),
    )
    safety_reasons = _safety_invalid_reasons(annotated, spec)
    runtime_reasons = _runtime_invalid_reasons(annotated, spec)
    missing_reasons = [] if metric_name is not None else ["primary_metric_missing"]
    primary_invalid_reasons = safety_reasons + runtime_reasons + missing_reasons
    total_episodes = int(annotated.get("episodes") or len(episodes) or 0)
    sample_reasons = []
    if total_episodes <= 0:
        sample_reasons.append("no_executed_episodes")
    elif total_episodes < int(spec["minimum_claim_episodes"]):
        sample_reasons.append(f"n_episodes<{int(spec['minimum_claim_episodes'])}")

    if safety_reasons or total_episodes <= 0:
        validity_status = "failure_analysis_only"
    elif runtime_reasons or sample_reasons:
        validity_status = "exploratory_only"
    else:
        validity_status = "valid_for_claim"

    annotated.update(
        {
            "primary_metric_name": metric_name,
            "primary_metric_value": metric_value,
            "primary_driving_metric_name": metric_name,
            "primary_driving_metric_value": metric_value,
            "primary_llm_metric_name": llm_metric_name,
            "primary_llm_metric_value": llm_metric_value,
            "secondary_joint_metric_name": joint_metric_name,
            "secondary_joint_metric_value": joint_metric_value,
            "primary_metric_valid": bool(metric_name is not None and not safety_reasons and not runtime_reasons),
            "primary_metric_invalid_reasons": primary_invalid_reasons,
            "scientific_validity_status": validity_status,
            "scientific_validity_reasons": safety_reasons + runtime_reasons + sample_reasons,
            "scientific_stats": summarize_scientific_statistics(episodes, spec),
        }
    )
    return annotated
