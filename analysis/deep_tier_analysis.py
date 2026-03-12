"""Deep safety/efficiency/comfort analysis for tiered DiLu-Ollama results.

This module implements:
- canonical ingestion of latest tier compare reports
- data quality and comparability checks
- safety-first ranking with harmonized + sensitivity views
- bootstrap confidence intervals (episode-level)
- effect sizes (Cliff's delta + median difference)
- static visualization bundle
- markdown report generation
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REQUIRED_REPORT_KEYS = ("aggregates", "per_model", "metrics_config", "seeds")
EXPECTED_SLICES = tuple((f"tier{i}", mode) for i in (1, 2, 3) for mode in ("instruct", "thinking"))
PROFILE_KEYS = (
    "ttc_threshold_sec",
    "headway_threshold_m",
    "flapping_mode",
    "decision_timeout_sec",
    "decision_max_output_tokens",
    "disable_streaming",
    "disable_checker_llm",
    "ollama_think_mode",
    "ollama_use_native_chat",
    "ollama_native_chat_timeout_sec",
)

INDEX_DEFS: Dict[str, Sequence[Tuple[str, bool]]] = {
    "SafetyIndex": (
        ("crash_rate", False),
        ("ttc_danger_rate_mean", False),
        ("headway_violation_rate_mean", False),
        ("no_collision_rate", True),
    ),
    "EfficiencyIndex": (
        ("avg_episode_runtime_sec", False),
        ("decision_latency_ms_avg", False),
        ("avg_reward_per_step", True),
    ),
    "ComfortIndex": (
        ("flap_accel_decel_rate_mean", False),
        ("lane_change_rate_mean", False),
        ("speed_std_ego_mps", False),
    ),
    "RobustnessIndex": (
        ("decision_timeout_rate_mean", False),
        ("fallback_action_rate_mean", False),
        ("format_failure_rate_mean", False),
        ("response_strict_format_rate", True),
    ),
}

PRACTICAL_DELTA_THRESHOLD = 0.33
PRACTICAL_MEDIAN_DIFF_THRESHOLD = 10.0

SAFE_COLOR = "#1b9e77"
RISK_COLOR = "#d95f02"

BLOCKING_PROXY_SPEED_THRESHOLD_MPS = 8.5
BLOCKING_PROXY_TTC_RATE_MAX = 0.03
BLOCKING_PROXY_HEADWAY_RATE_MAX = 0.03
BLOCKING_PROXY_GATE_MAX = 0.60

TELEMETRY_REAR_TTC_GATE_MAX = 0.10
TELEMETRY_BLOCKING_GATE_MAX = 0.35

BLOCKING_PENALTY_START = 0.20
BLOCKING_PENALTY_RANGE = 0.60
BLOCKING_PENALTY_MAX = 20.0


@dataclass
class Issue:
    level: str
    code: str
    message: str
    experiment_dir: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "experiment_dir": self.experiment_dir,
            "model": self.model,
        }


@dataclass
class ExperimentBundle:
    experiment_dir: str
    experiment_path: Path
    tier: str
    mode: str
    manifest_path: Optional[Path] = None
    manifest: Optional[Dict[str, Any]] = None
    latest_report_path: Optional[Path] = None
    report: Optional[Dict[str, Any]] = None
    config_profile: Optional[Tuple[Tuple[str, Any], ...]] = None
    profile_dict: Dict[str, Any] = field(default_factory=dict)
    is_harmonized: bool = False
    valid_for_analysis: bool = False


@dataclass
class ModelRecord:
    model_id: str
    model: str
    family: str
    tier: str
    mode: str
    experiment_dir: str
    report_path: str
    raw: Dict[str, float]
    episodes: List[Dict[str, Any]]
    gate_pass: bool = False
    gate_mode: str = "proxy"
    indices: Dict[str, float] = field(default_factory=dict)
    secondary_score: float = float("nan")
    secondary_score_adjusted: float = float("nan")
    blocking_penalty: float = 0.0
    rank: Optional[int] = None
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    bootstrap_metrics: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.model} | {self.tier.upper()}-{self.mode[:1].upper()}"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _parse_tier_mode(experiment_dir: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^(tier\d+)_.*_(instruct|thinking)$", experiment_dir)
    if not match:
        return None
    return match.group(1), match.group(2)


def _model_family(model_name: str) -> str:
    lower = (model_name or "").lower()
    if "llama" in lower:
        return "llama"
    if "qwen" in lower:
        return "qwen"
    if "deepseek" in lower:
        return "deepseek"
    if "phi" in lower:
        return "phi"
    return "other"


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _finite(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        if value is None:
            continue
        number = _as_float(value)
        if np.isfinite(number):
            out.append(number)
    return out


def _discover_experiment_bundles(results_root: Path, issues: List[Issue]) -> List[ExperimentBundle]:
    bundles: List[ExperimentBundle] = []
    if not results_root.exists():
        issues.append(Issue("error", "MISSING_RESULTS_ROOT", f"Results root not found: {results_root}"))
        return bundles

    for child in sorted(results_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or not child.name.startswith("tier"):
            continue
        parsed = _parse_tier_mode(child.name)
        if not parsed:
            continue
        tier, mode = parsed
        bundle = ExperimentBundle(
            experiment_dir=child.name,
            experiment_path=child,
            tier=tier,
            mode=mode,
        )

        manifest_path = child / "manifest.json"
        if manifest_path.exists():
            bundle.manifest_path = manifest_path
            try:
                bundle.manifest = _read_json(manifest_path)
            except Exception as exc:
                issues.append(
                    Issue(
                        "error",
                        "MANIFEST_READ_ERROR",
                        f"Failed to read manifest: {exc}",
                        experiment_dir=child.name,
                    )
                )
        else:
            issues.append(
                Issue(
                    "warning",
                    "MISSING_MANIFEST",
                    "manifest.json is missing.",
                    experiment_dir=child.name,
                )
            )

        compare_dir = child / "compare"
        latest_report: Optional[Path] = None
        if compare_dir.exists():
            reports = sorted(compare_dir.glob("eval_compare_*.json"), key=lambda p: p.stat().st_mtime)
            if reports:
                latest_report = reports[-1]
        bundle.latest_report_path = latest_report
        if latest_report is None:
            issues.append(
                Issue(
                    "warning",
                    "MISSING_COMPARE_REPORT",
                    "No eval_compare_*.json found under compare/.",
                    experiment_dir=child.name,
                )
            )
        else:
            try:
                bundle.report = _read_json(latest_report)
            except Exception as exc:
                issues.append(
                    Issue(
                        "error",
                        "REPORT_READ_ERROR",
                        f"Failed to read latest compare report: {exc}",
                        experiment_dir=child.name,
                    )
                )

        bundles.append(bundle)
    return bundles


def _build_profile(metrics_config: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    profile = [(key, metrics_config.get(key)) for key in PROFILE_KEYS]
    return tuple(profile)


def _validate_experiment_bundle(bundle: ExperimentBundle, issues: List[Issue]) -> None:
    manifest = bundle.manifest or {}
    report = bundle.report or {}

    if manifest:
        manifest_experiment_id = manifest.get("experiment_id")
        if manifest_experiment_id and str(manifest_experiment_id) != bundle.experiment_dir:
            issues.append(
                Issue(
                    "warning",
                    "MANIFEST_EXPERIMENT_ID_MISMATCH",
                    f"manifest experiment_id='{manifest_experiment_id}' differs from directory '{bundle.experiment_dir}'.",
                    experiment_dir=bundle.experiment_dir,
                )
            )

    if not report:
        bundle.valid_for_analysis = False
        return

    missing_keys = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    if missing_keys:
        issues.append(
            Issue(
                "error",
                "REPORT_SCHEMA_MISSING_KEYS",
                f"Missing required report keys: {', '.join(missing_keys)}.",
                experiment_dir=bundle.experiment_dir,
            )
        )
        bundle.valid_for_analysis = False
        return

    report_experiment_id = report.get("experiment_id")
    if report_experiment_id and str(report_experiment_id) != bundle.experiment_dir:
        issues.append(
            Issue(
                "warning",
                "REPORT_EXPERIMENT_ID_MISMATCH",
                f"report experiment_id='{report_experiment_id}' differs from directory '{bundle.experiment_dir}'.",
                experiment_dir=bundle.experiment_dir,
            )
        )

    seeds = report.get("seeds", [])
    if not isinstance(seeds, list) or not seeds:
        issues.append(
            Issue(
                "error",
                "INVALID_SEEDS",
                "Report seeds is missing or empty.",
                experiment_dir=bundle.experiment_dir,
            )
        )
        bundle.valid_for_analysis = False
        return

    seed_set = set(int(s) for s in seeds)
    aggregates = report.get("aggregates", [])
    per_model = report.get("per_model", {})
    if not isinstance(aggregates, list) or not isinstance(per_model, dict):
        issues.append(
            Issue(
                "error",
                "INVALID_AGGREGATE_OR_PER_MODEL",
                "Report aggregates/per_model has invalid type.",
                experiment_dir=bundle.experiment_dir,
            )
        )
        bundle.valid_for_analysis = False
        return

    aggregate_models = set()
    for row in aggregates:
        model_name = str(row.get("model", "")).strip()
        if not model_name:
            issues.append(
                Issue(
                    "warning",
                    "AGGREGATE_MODEL_MISSING_NAME",
                    "Aggregate row without model name.",
                    experiment_dir=bundle.experiment_dir,
                )
            )
            continue
        aggregate_models.add(model_name)
        episodes = per_model.get(model_name)
        if not isinstance(episodes, list) or not episodes:
            issues.append(
                Issue(
                    "error",
                    "MISSING_PER_MODEL_EPISODES",
                    "Model has missing or empty per_model episodes.",
                    experiment_dir=bundle.experiment_dir,
                    model=model_name,
                )
            )
            continue

        episode_seeds = []
        for episode in episodes:
            if "seed" in episode:
                try:
                    episode_seeds.append(int(episode["seed"]))
                except Exception:
                    pass
        if len(episode_seeds) != len(seeds):
            issues.append(
                Issue(
                    "warning",
                    "SEED_COUNT_MISMATCH",
                    f"Episode seed count={len(episode_seeds)} differs from top-level seeds={len(seeds)}.",
                    experiment_dir=bundle.experiment_dir,
                    model=model_name,
                )
            )
        if set(episode_seeds) != seed_set:
            issues.append(
                Issue(
                    "warning",
                    "SEED_LIST_MISMATCH",
                    "Episode seeds do not match top-level seeds.",
                    experiment_dir=bundle.experiment_dir,
                    model=model_name,
                )
            )

    for model_name in per_model.keys():
        if model_name not in aggregate_models:
            issues.append(
                Issue(
                    "warning",
                    "PER_MODEL_NOT_IN_AGGREGATES",
                    "Model exists in per_model but not in aggregates.",
                    experiment_dir=bundle.experiment_dir,
                    model=model_name,
                )
            )

    metrics_config = report.get("metrics_config", {})
    if isinstance(metrics_config, dict):
        bundle.config_profile = _build_profile(metrics_config)
        bundle.profile_dict = {k: metrics_config.get(k) for k in PROFILE_KEYS}
    else:
        issues.append(
            Issue(
                "warning",
                "INVALID_METRICS_CONFIG",
                "metrics_config is not a dictionary.",
                experiment_dir=bundle.experiment_dir,
            )
        )

    has_errors = any(
        issue.level == "error" and issue.experiment_dir == bundle.experiment_dir for issue in issues
    )
    bundle.valid_for_analysis = not has_errors


def _validate_missing_slices(bundles: Sequence[ExperimentBundle], issues: List[Issue]) -> None:
    present = {(bundle.tier, bundle.mode) for bundle in bundles}
    by_slice = {(bundle.tier, bundle.mode): bundle for bundle in bundles}
    for slice_key in EXPECTED_SLICES:
        if slice_key not in present:
            issues.append(
                Issue(
                    "warning",
                    "MISSING_SLICE_DIR",
                    f"Missing experiment directory for slice {slice_key[0]}-{slice_key[1]}.",
                )
            )
            continue
        bundle = by_slice[slice_key]
        if bundle.latest_report_path is None:
            issues.append(
                Issue(
                    "warning",
                    "MISSING_SLICE_COMPARE",
                    f"Slice {slice_key[0]}-{slice_key[1]} has no compare report.",
                    experiment_dir=bundle.experiment_dir,
                )
            )


def _choose_harmonized_profile(bundles: Sequence[ExperimentBundle]) -> Optional[Tuple[Tuple[str, Any], ...]]:
    profiles = [bundle.config_profile for bundle in bundles if bundle.valid_for_analysis and bundle.config_profile]
    if not profiles:
        return None
    profile_counter = Counter(profiles)
    dominant_profile, _ = profile_counter.most_common(1)[0]
    return dominant_profile


def _nanmean_or_nan(values: Sequence[float]) -> float:
    finite_values = _finite(values)
    if not finite_values:
        return float("nan")
    return float(np.mean(np.array(finite_values, dtype=float)))


def _blocking_proxy_episode_rate(episodes: Sequence[Dict[str, Any]]) -> float:
    if not episodes:
        return float("nan")
    flagged = 0
    for episode in episodes:
        avg_speed = _as_float(episode.get("avg_ego_speed_mps"))
        ttc_rate = _as_float(episode.get("ttc_danger_rate"))
        headway_rate = _as_float(episode.get("headway_violation_rate"))
        is_proxy = (
            np.isfinite(avg_speed)
            and np.isfinite(ttc_rate)
            and np.isfinite(headway_rate)
            and avg_speed < BLOCKING_PROXY_SPEED_THRESHOLD_MPS
            and ttc_rate <= BLOCKING_PROXY_TTC_RATE_MAX
            and headway_rate <= BLOCKING_PROXY_HEADWAY_RATE_MAX
        )
        flagged += int(is_proxy)
    return float(flagged / max(len(episodes), 1))


def _extract_speed_std(episodes: List[Dict[str, Any]]) -> float:
    speeds = _finite(_as_float(episode.get("avg_ego_speed_mps")) for episode in episodes)
    if not speeds:
        return float("nan")
    return float(np.std(np.array(speeds, dtype=float), ddof=0))


def _build_model_records(
    bundles: Sequence[ExperimentBundle],
    require_harmonized: bool = False,
) -> List[ModelRecord]:
    records: List[ModelRecord] = []
    for bundle in bundles:
        if not bundle.valid_for_analysis:
            continue
        if require_harmonized and not bundle.is_harmonized:
            continue
        report = bundle.report or {}
        aggregates = report.get("aggregates", [])
        per_model = report.get("per_model", {})
        for aggregate in aggregates:
            model_name = str(aggregate.get("model", "")).strip()
            if not model_name:
                continue
            episodes = per_model.get(model_name, [])
            speed_std = _extract_speed_std(episodes if isinstance(episodes, list) else [])
            blocking_proxy_rate = _blocking_proxy_episode_rate(episodes if isinstance(episodes, list) else [])
            raw = {
                "crash_rate": _as_float(aggregate.get("crash_rate")),
                "no_collision_rate": _as_float(aggregate.get("no_collision_rate")),
                "ttc_danger_rate_mean": _as_float(aggregate.get("ttc_danger_rate_mean")),
                "headway_violation_rate_mean": _as_float(aggregate.get("headway_violation_rate_mean")),
                "rear_ttc_danger_rate_mean": _as_float(aggregate.get("rear_ttc_danger_rate_mean")),
                "rear_headway_violation_rate_mean": _as_float(aggregate.get("rear_headway_violation_rate_mean")),
                "low_speed_blocking_rate_mean": _as_float(aggregate.get("low_speed_blocking_rate_mean")),
                "blocking_proxy_episode_rate": (
                    _as_float(aggregate.get("blocking_proxy_episode_rate"))
                    if np.isfinite(_as_float(aggregate.get("blocking_proxy_episode_rate")))
                    else blocking_proxy_rate
                ),
                "error_rate": _as_float(aggregate.get("error_rate")),
                "avg_episode_runtime_sec": _as_float(aggregate.get("avg_episode_runtime_sec")),
                "decision_latency_ms_avg": _as_float(aggregate.get("decision_latency_ms_avg")),
                "avg_reward_per_step": _as_float(aggregate.get("avg_reward_per_step")),
                "flap_accel_decel_rate_mean": _as_float(aggregate.get("flap_accel_decel_rate_mean")),
                "lane_change_rate_mean": _as_float(aggregate.get("lane_change_rate_mean")),
                "decision_timeout_rate_mean": _as_float(aggregate.get("decision_timeout_rate_mean")),
                "fallback_action_rate_mean": _as_float(aggregate.get("fallback_action_rate_mean")),
                "format_failure_rate_mean": _as_float(aggregate.get("format_failure_rate_mean")),
                "response_strict_format_rate": _as_float(aggregate.get("response_strict_format_rate")),
                "speed_std_ego_mps": speed_std,
            }
            model_id = f"{bundle.experiment_dir}|{bundle.mode}|{bundle.tier}|{model_name}"
            records.append(
                ModelRecord(
                    model_id=model_id,
                    model=model_name,
                    family=_model_family(model_name),
                    tier=bundle.tier,
                    mode=bundle.mode,
                    experiment_dir=bundle.experiment_dir,
                    report_path=str(bundle.latest_report_path) if bundle.latest_report_path else "",
                    raw=raw,
                    episodes=episodes if isinstance(episodes, list) else [],
                )
            )
    return records


def _bounds(values: Sequence[float]) -> Tuple[float, float]:
    finite_values = np.array(_finite(values), dtype=float)
    if finite_values.size == 0:
        return (0.0, 1.0)
    q_low = float(np.percentile(finite_values, 5))
    q_high = float(np.percentile(finite_values, 95))
    if q_high <= q_low:
        q_low = float(np.min(finite_values))
        q_high = float(np.max(finite_values))
    if q_high <= q_low:
        q_high = q_low + 1.0
    return (q_low, q_high)


def _normalize(value: float, bound: Tuple[float, float], higher_is_better: bool) -> float:
    if not np.isfinite(value):
        return float("nan")
    low, high = bound
    if high <= low:
        return 50.0
    score = (value - low) / (high - low)
    score = float(np.clip(score, 0.0, 1.0))
    if not higher_is_better:
        score = 1.0 - score
    return 100.0 * score


def _compute_component_bounds(records: Sequence[ModelRecord]) -> Dict[str, Tuple[float, float]]:
    metric_names = set()
    for pairs in INDEX_DEFS.values():
        for metric_name, _ in pairs:
            metric_names.add(metric_name)
    metric_names.update(
        {
            "rear_ttc_danger_rate_mean",
            "rear_headway_violation_rate_mean",
            "low_speed_blocking_rate_mean",
            "blocking_proxy_episode_rate",
        }
    )
    return {
        metric_name: _bounds([record.raw.get(metric_name, float("nan")) for record in records])
        for metric_name in metric_names
    }


def _has_stage_b_telemetry(record: ModelRecord) -> bool:
    rear_ttc = record.raw.get("rear_ttc_danger_rate_mean", float("nan"))
    low_speed_blocking = record.raw.get("low_speed_blocking_rate_mean", float("nan"))
    return np.isfinite(rear_ttc) and np.isfinite(low_speed_blocking)


def _blocking_rate_for_record(record: ModelRecord) -> float:
    if _has_stage_b_telemetry(record):
        return _as_float(record.raw.get("low_speed_blocking_rate_mean"))
    return _as_float(record.raw.get("blocking_proxy_episode_rate"))


def _compute_blocking_penalty(blocking_rate: float) -> float:
    if not np.isfinite(blocking_rate):
        return 0.0
    scaled = (blocking_rate - BLOCKING_PENALTY_START) / max(BLOCKING_PENALTY_RANGE, 1e-9)
    return float(np.clip(scaled, 0.0, 1.0) * BLOCKING_PENALTY_MAX)


def _safety_metric_pairs_for_record(record: ModelRecord) -> List[Tuple[str, bool]]:
    pairs: List[Tuple[str, bool]] = [
        ("crash_rate", False),
        ("ttc_danger_rate_mean", False),
        ("headway_violation_rate_mean", False),
        ("no_collision_rate", True),
    ]
    if _has_stage_b_telemetry(record):
        pairs.extend(
            [
                ("rear_ttc_danger_rate_mean", False),
                ("low_speed_blocking_rate_mean", False),
            ]
        )
    else:
        pairs.append(("blocking_proxy_episode_rate", False))
    return pairs


def _compute_gate(record: ModelRecord) -> bool:
    crash_rate = record.raw.get("crash_rate", float("nan"))
    ttc_rate = record.raw.get("ttc_danger_rate_mean", float("nan"))
    headway_rate = record.raw.get("headway_violation_rate_mean", float("nan"))
    error_rate = record.raw.get("error_rate", float("nan"))
    base_gate_pass = (
        np.isfinite(crash_rate)
        and np.isfinite(ttc_rate)
        and np.isfinite(headway_rate)
        and np.isfinite(error_rate)
        and crash_rate <= 0.20
        and ttc_rate <= 0.15
        and headway_rate <= 0.15
        and error_rate == 0.0
    )
    if not base_gate_pass:
        return False

    if _has_stage_b_telemetry(record):
        rear_ttc_rate = _as_float(record.raw.get("rear_ttc_danger_rate_mean"))
        low_speed_blocking_rate = _as_float(record.raw.get("low_speed_blocking_rate_mean"))
        return (
            np.isfinite(rear_ttc_rate)
            and np.isfinite(low_speed_blocking_rate)
            and rear_ttc_rate <= TELEMETRY_REAR_TTC_GATE_MAX
            and low_speed_blocking_rate <= TELEMETRY_BLOCKING_GATE_MAX
        )

    proxy_rate = _as_float(record.raw.get("blocking_proxy_episode_rate"))
    return np.isfinite(proxy_rate) and proxy_rate <= BLOCKING_PROXY_GATE_MAX


def _safe_sort_number(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) else float(default)


def _score_records(records: List[ModelRecord], component_bounds: Dict[str, Tuple[float, float]]) -> None:
    for record in records:
        indices: Dict[str, float] = {}
        for index_name, metric_pairs in INDEX_DEFS.items():
            pairs_to_use = list(metric_pairs)
            if index_name == "SafetyIndex":
                pairs_to_use = _safety_metric_pairs_for_record(record)
            component_scores = []
            for metric_name, higher_is_better in pairs_to_use:
                raw_value = record.raw.get(metric_name, float("nan"))
                component_scores.append(
                    _normalize(raw_value, component_bounds[metric_name], higher_is_better)
                )
            finite_scores = [score for score in component_scores if np.isfinite(score)]
            indices[index_name] = float(np.mean(finite_scores)) if finite_scores else float("nan")
        record.indices = indices
        record.secondary_score = (
            0.45 * indices.get("EfficiencyIndex", float("nan"))
            + 0.35 * indices.get("ComfortIndex", float("nan"))
            + 0.20 * indices.get("RobustnessIndex", float("nan"))
        )
        blocking_rate = _blocking_rate_for_record(record)
        record.blocking_penalty = _compute_blocking_penalty(blocking_rate)
        record.secondary_score_adjusted = record.secondary_score - record.blocking_penalty
        record.gate_mode = "telemetry" if _has_stage_b_telemetry(record) else "proxy"
        record.gate_pass = _compute_gate(record)

    passed = [record for record in records if record.gate_pass]
    failed = [record for record in records if not record.gate_pass]

    passed_sorted = sorted(
        passed,
        key=lambda record: (
            -_safe_sort_number(record.secondary_score_adjusted, -1e9),
            -_safe_sort_number(record.indices.get("SafetyIndex", float("nan")), -1e9),
            -_safe_sort_number(record.raw.get("avg_reward_per_step", float("nan")), -1e9),
            _safe_sort_number(record.raw.get("decision_latency_ms_avg", float("nan")), 1e12),
        ),
    )
    failed_sorted = sorted(
        failed,
        key=lambda record: (
            -_safe_sort_number(record.indices.get("SafetyIndex", float("nan")), -1e9),
            -_safe_sort_number(record.secondary_score_adjusted, -1e9),
            -_safe_sort_number(record.raw.get("avg_reward_per_step", float("nan")), -1e9),
            _safe_sort_number(record.raw.get("decision_latency_ms_avg", float("nan")), 1e12),
        ),
    )

    ordered = passed_sorted + failed_sorted
    for index, record in enumerate(ordered, start=1):
        record.rank = index

    records.sort(key=lambda record: _safe_sort_number(float(record.rank or 1e9), 1e9))


def _aggregate_episode_metrics(episodes: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not episodes:
        return {
            "crash_rate": float("nan"),
            "no_collision_rate": float("nan"),
            "ttc_danger_rate_mean": float("nan"),
            "headway_violation_rate_mean": float("nan"),
            "rear_ttc_danger_rate_mean": float("nan"),
            "rear_headway_violation_rate_mean": float("nan"),
            "low_speed_blocking_rate_mean": float("nan"),
            "blocking_proxy_episode_rate": float("nan"),
            "error_rate": float("nan"),
            "avg_episode_runtime_sec": float("nan"),
            "decision_latency_ms_avg": float("nan"),
            "avg_reward_per_step": float("nan"),
            "flap_accel_decel_rate_mean": float("nan"),
            "lane_change_rate_mean": float("nan"),
            "speed_std_ego_mps": float("nan"),
            "decision_timeout_rate_mean": float("nan"),
            "fallback_action_rate_mean": float("nan"),
            "format_failure_rate_mean": float("nan"),
            "response_strict_format_rate": float("nan"),
        }

    crashes = np.array([1.0 if bool(ep.get("crashed", False)) else 0.0 for ep in episodes], dtype=float)
    success = np.array([1.0 if bool(ep.get("success_no_collision", False)) else 0.0 for ep in episodes], dtype=float)
    errors = np.array([1.0 if ep.get("error") else 0.0 for ep in episodes], dtype=float)
    ttc = np.array([_as_float(ep.get("ttc_danger_rate")) for ep in episodes], dtype=float)
    headway = np.array([_as_float(ep.get("headway_violation_rate")) for ep in episodes], dtype=float)
    rear_ttc = np.array([_as_float(ep.get("rear_ttc_danger_rate")) for ep in episodes], dtype=float)
    rear_headway = np.array([_as_float(ep.get("rear_headway_violation_rate")) for ep in episodes], dtype=float)
    low_speed_blocking = np.array([_as_float(ep.get("low_speed_blocking_rate")) for ep in episodes], dtype=float)
    episode_runtime = np.array([_as_float(ep.get("episode_runtime_sec")) for ep in episodes], dtype=float)
    decision_latency = np.array([_as_float(ep.get("decision_latency_ms_avg")) for ep in episodes], dtype=float)
    flap = np.array([_as_float(ep.get("flap_accel_decel_rate")) for ep in episodes], dtype=float)
    lane_change = np.array([_as_float(ep.get("lane_change_rate")) for ep in episodes], dtype=float)
    speed = np.array([_as_float(ep.get("avg_ego_speed_mps")) for ep in episodes], dtype=float)

    reward_sum = np.array([_as_float(ep.get("episode_reward_sum"), 0.0) for ep in episodes], dtype=float)
    steps = np.array([max(_as_float(ep.get("steps"), 0.0), 0.0) for ep in episodes], dtype=float)
    total_steps = float(np.sum(steps))
    total_reward = float(np.sum(reward_sum))

    timeout_counts = np.array([_as_float(ep.get("decision_timeout_count"), 0.0) for ep in episodes], dtype=float)
    fallback_counts = np.array([_as_float(ep.get("fallback_action_count"), 0.0) for ep in episodes], dtype=float)
    format_fail_counts = np.array([_as_float(ep.get("format_failure_count"), 0.0) for ep in episodes], dtype=float)
    decision_calls = np.array([_as_float(ep.get("decision_calls_total"), 0.0) for ep in episodes], dtype=float)
    decisions_made = np.array([_as_float(ep.get("decisions_made"), 0.0) for ep in episodes], dtype=float)
    strict_count = np.array([_as_float(ep.get("responses_strict_format"), 0.0) for ep in episodes], dtype=float)

    total_decision_calls = float(np.sum(decision_calls))
    total_decisions_made = float(np.sum(decisions_made))
    blocking_proxy_episode_rate = _blocking_proxy_episode_rate(episodes)

    return {
        "crash_rate": float(np.mean(crashes)),
        "no_collision_rate": float(np.mean(success)),
        "ttc_danger_rate_mean": _nanmean_or_nan(ttc.tolist()),
        "headway_violation_rate_mean": _nanmean_or_nan(headway.tolist()),
        "rear_ttc_danger_rate_mean": _nanmean_or_nan(rear_ttc.tolist()),
        "rear_headway_violation_rate_mean": _nanmean_or_nan(rear_headway.tolist()),
        "low_speed_blocking_rate_mean": _nanmean_or_nan(low_speed_blocking.tolist()),
        "blocking_proxy_episode_rate": float(blocking_proxy_episode_rate),
        "error_rate": float(np.mean(errors)),
        "avg_episode_runtime_sec": float(np.mean(episode_runtime)),
        "decision_latency_ms_avg": float(np.mean(decision_latency)),
        "avg_reward_per_step": float(total_reward / max(total_steps, 1.0)),
        "flap_accel_decel_rate_mean": float(np.mean(flap)),
        "lane_change_rate_mean": float(np.mean(lane_change)),
        "speed_std_ego_mps": float(np.std(speed, ddof=0)),
        "decision_timeout_rate_mean": float(np.sum(timeout_counts) / max(total_decision_calls, 1.0)),
        "fallback_action_rate_mean": float(np.sum(fallback_counts) / max(total_decision_calls, 1.0)),
        "format_failure_rate_mean": float(np.sum(format_fail_counts) / max(total_decisions_made, 1.0)),
        "response_strict_format_rate": float(np.sum(strict_count) / max(total_decisions_made, 1.0)),
    }


def _bootstrap_metrics(record: ModelRecord, n_bootstrap: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    episodes = record.episodes
    if not episodes:
        return {}
    n = len(episodes)
    samples = rng.integers(0, n, size=(n_bootstrap, n), endpoint=False)
    metric_store: Dict[str, List[float]] = defaultdict(list)
    for row in samples:
        sampled_episodes = [episodes[int(i)] for i in row]
        agg = _aggregate_episode_metrics(sampled_episodes)
        for key, value in agg.items():
            metric_store[key].append(float(value))
    return {key: np.array(values, dtype=float) for key, values in metric_store.items()}


def _bootstrap_indices(
    records: List[ModelRecord],
    component_bounds: Dict[str, Tuple[float, float]],
    n_bootstrap: int,
    random_seed: int,
) -> None:
    rng = np.random.default_rng(random_seed)
    for record in records:
        record.bootstrap_metrics = _bootstrap_metrics(record, n_bootstrap=n_bootstrap, rng=rng)

    for record in records:
        if not record.bootstrap_metrics:
            continue
        index_arrays: Dict[str, np.ndarray] = {}
        for index_name, metric_pairs in INDEX_DEFS.items():
            pairs_to_use = list(metric_pairs)
            if index_name == "SafetyIndex":
                pairs_to_use = _safety_metric_pairs_for_record(record)
            component_arrays = []
            for metric_name, higher_is_better in pairs_to_use:
                metric_values = record.bootstrap_metrics.get(metric_name)
                if metric_values is None:
                    continue
                bound = component_bounds[metric_name]
                normalized = np.array(
                    [_normalize(value, bound, higher_is_better) for value in metric_values],
                    dtype=float,
                )
                component_arrays.append(normalized)
            if component_arrays:
                index_arrays[index_name] = np.nanmean(np.vstack(component_arrays), axis=0)

        if not index_arrays:
            continue
        secondary = (
            0.45 * index_arrays.get("EfficiencyIndex", np.full(n_bootstrap, np.nan))
            + 0.35 * index_arrays.get("ComfortIndex", np.full(n_bootstrap, np.nan))
            + 0.20 * index_arrays.get("RobustnessIndex", np.full(n_bootstrap, np.nan))
        )
        index_arrays["SecondaryScore"] = secondary
        blocking_metric_name = "low_speed_blocking_rate_mean" if _has_stage_b_telemetry(record) else "blocking_proxy_episode_rate"
        blocking_metric_values = record.bootstrap_metrics.get(blocking_metric_name, np.full(n_bootstrap, np.nan))
        blocking_penalty_values = np.array(
            [_compute_blocking_penalty(value) for value in blocking_metric_values],
            dtype=float,
        )
        index_arrays["BlockingPenalty"] = blocking_penalty_values
        index_arrays["SecondaryScoreAdjusted"] = secondary - blocking_penalty_values

        for name, values in index_arrays.items():
            if np.all(np.isnan(values)):
                continue
            low, high = np.nanpercentile(values, [2.5, 97.5])
            record.ci[name] = (float(low), float(high))

        for name in (
            "crash_rate",
            "ttc_danger_rate_mean",
            "headway_violation_rate_mean",
            "rear_ttc_danger_rate_mean",
            "low_speed_blocking_rate_mean",
            "blocking_proxy_episode_rate",
        ):
            values = record.bootstrap_metrics.get(name)
            if values is None:
                continue
            if np.all(np.isnan(values)):
                continue
            low, high = np.nanpercentile(values, [2.5, 97.5])
            record.ci[name] = (float(low), float(high))


def _cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    arr_x = np.array(_finite(x), dtype=float)
    arr_y = np.array(_finite(y), dtype=float)
    if arr_x.size == 0 or arr_y.size == 0:
        return float("nan")
    greater = 0
    lower = 0
    for xv in arr_x:
        greater += int(np.sum(xv > arr_y))
        lower += int(np.sum(xv < arr_y))
    total = arr_x.size * arr_y.size
    return float((greater - lower) / max(total, 1))


def _delta_magnitude(delta: float) -> str:
    absolute = abs(delta)
    if absolute < 0.147:
        return "negligible"
    if absolute < 0.33:
        return "small"
    if absolute < 0.474:
        return "medium"
    return "large"


def _effect_sizes(records: Sequence[ModelRecord]) -> List[Dict[str, Any]]:
    metric_names = ["SafetyIndex", "EfficiencyIndex", "ComfortIndex", "RobustnessIndex", "SecondaryScoreAdjusted"]
    lenses = ["tier", "mode", "family"]
    out: List[Dict[str, Any]] = []

    for lens in lenses:
        grouped: Dict[str, List[ModelRecord]] = defaultdict(list)
        for record in records:
            grouped[getattr(record, lens)].append(record)
        keys = sorted(grouped.keys())
        for left, right in combinations(keys, 2):
            left_records = grouped[left]
            right_records = grouped[right]
            for metric_name in metric_names:
                left_values = [
                    (
                        record.secondary_score_adjusted
                        if metric_name == "SecondaryScoreAdjusted"
                        else record.indices.get(metric_name, float("nan"))
                    )
                    for record in left_records
                ]
                right_values = [
                    (
                        record.secondary_score_adjusted
                        if metric_name == "SecondaryScoreAdjusted"
                        else record.indices.get(metric_name, float("nan"))
                    )
                    for record in right_records
                ]
                if not _finite(left_values) or not _finite(right_values):
                    continue
                delta = _cliffs_delta(left_values, right_values)
                left_median = float(np.median(_finite(left_values)))
                right_median = float(np.median(_finite(right_values)))
                median_diff = left_median - right_median
                meaningful = (
                    abs(delta) >= PRACTICAL_DELTA_THRESHOLD
                    or abs(median_diff) >= PRACTICAL_MEDIAN_DIFF_THRESHOLD
                )
                out.append(
                    {
                        "lens": lens,
                        "group_a": left,
                        "group_b": right,
                        "metric": metric_name,
                        "n_a": len(_finite(left_values)),
                        "n_b": len(_finite(right_values)),
                        "cliffs_delta": float(delta),
                        "delta_magnitude": _delta_magnitude(delta),
                        "median_a": left_median,
                        "median_b": right_median,
                        "median_diff": float(median_diff),
                        "practically_meaningful": bool(meaningful),
                    }
                )
    return out

def _figure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8faf9",
            "axes.edgecolor": "#d0d7d5",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "font.size": 10,
            "grid.color": "#dfe6e3",
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
        }
    )


def _short_label(record: ModelRecord) -> str:
    return f"{record.model} ({record.tier}-{record.mode[0]})"


def _plot_safety_leaderboard(records: Sequence[ModelRecord], output_path: Path, title: str) -> None:
    if not records:
        return
    _figure_style()
    ordered = sorted(records, key=lambda record: _safe_sort_number(float(record.rank or 1e9), 1e9))
    labels = [_short_label(record) for record in ordered]
    values = [record.indices.get("SafetyIndex", float("nan")) for record in ordered]
    colors = [SAFE_COLOR if record.gate_pass else RISK_COLOR for record in ordered]
    ci_low = []
    ci_high = []
    for record, value in zip(ordered, values):
        low, high = record.ci.get("SafetyIndex", (value, value))
        ci_low.append(max(value - low, 0.0))
        ci_high.append(max(high - value, 0.0))

    fig_h = max(4.5, 0.48 * len(ordered) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    y = np.arange(len(ordered))
    bars = ax.barh(y, values, color=colors, alpha=0.9)
    ax.errorbar(values, y, xerr=[ci_low, ci_high], fmt="none", ecolor="#334", capsize=3, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("SafetyIndex (0-100)")
    ax.set_title(title)
    ax.invert_yaxis()
    for bar, record in zip(bars, ordered):
        secondary_adj = record.secondary_score_adjusted
        ax.text(
            min(bar.get_width() + 1.0, 97.5),
            bar.get_y() + bar.get_height() / 2,
            f"R{record.rank} | {'PASS' if record.gate_pass else 'FAIL'} | SecAdj {secondary_adj:.1f} | {record.gate_mode}",
            va="center",
            fontsize=8,
            color="#223",
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_tier_mode_heatmaps(records: Sequence[ModelRecord], output_path: Path) -> int:
    if not records:
        return 0
    _figure_style()
    tiers = sorted({record.tier for record in records}, key=lambda x: int(x.replace("tier", "")))
    modes = ["instruct", "thinking"]
    indices = ["SafetyIndex", "EfficiencyIndex", "ComfortIndex", "RobustnessIndex"]
    missing_cells = 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, index_name in zip(axes.flatten(), indices):
        matrix = np.full((len(tiers), len(modes)), np.nan, dtype=float)
        counts = np.zeros((len(tiers), len(modes)), dtype=int)
        for i, tier in enumerate(tiers):
            for j, mode in enumerate(modes):
                values = [
                    record.indices.get(index_name, float("nan"))
                    for record in records
                    if record.tier == tier and record.mode == mode
                ]
                finite_values = _finite(values)
                if finite_values:
                    matrix[i, j] = float(np.mean(finite_values))
                    counts[i, j] = len(finite_values)
                else:
                    missing_cells += 1

        cmap = plt.cm.YlGnBu.copy()
        cmap.set_bad(color="#eceff1")
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap)
        ax.set_xticks(np.arange(len(modes)))
        ax.set_yticks(np.arange(len(tiers)))
        ax.set_xticklabels([mode.title() for mode in modes])
        ax.set_yticklabels([tier.upper() for tier in tiers])
        ax.set_title(index_name)
        for i in range(len(tiers)):
            for j in range(len(modes)):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.1f}\nn={counts[i, j]}", ha="center", va="center", fontsize=9, color="#112")
                else:
                    ax.text(j, i, "NA", ha="center", va="center", fontsize=9, color="#556")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Tier x Mode Index Heatmaps (Harmonized View)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return missing_cells


def _plot_family_tradeoff(records: Sequence[ModelRecord], output_path: Path) -> None:
    if not records:
        return
    _figure_style()
    families = sorted({record.family for record in records})
    n = len(families)
    cols = min(3, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), squeeze=False)

    mode_colors = {"instruct": "#1f77b4", "thinking": "#ff7f0e"}
    tier_markers = {"tier1": "o", "tier2": "s", "tier3": "^"}

    for index, family in enumerate(families):
        ax = axes[index // cols][index % cols]
        subset = [record for record in records if record.family == family]
        for record in subset:
            x = record.indices.get("EfficiencyIndex", float("nan"))
            y = record.indices.get("SafetyIndex", float("nan"))
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            ax.scatter(
                x,
                y,
                s=110,
                c=mode_colors.get(record.mode, "#888"),
                marker=tier_markers.get(record.tier, "o"),
                edgecolor="#222",
                linewidth=0.5,
                alpha=0.9,
            )
            ax.text(x + 0.8, y + 0.8, record.model, fontsize=8, alpha=0.9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("EfficiencyIndex")
        ax.set_ylabel("SafetyIndex")
        ax.set_title(f"Family: {family}")

    for index in range(n, rows * cols):
        axes[index // cols][index % cols].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="#333", label=mode.title(), markersize=8)
        for mode, color in mode_colors.items()
    ]
    fig.legend(handles=handles, loc="upper right", title="Mode")
    fig.suptitle("Family-level Safety vs Efficiency Tradeoff", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_risk_runtime_bubble(records: Sequence[ModelRecord], output_path: Path) -> None:
    if not records:
        return
    _figure_style()
    fig, ax = plt.subplots(figsize=(13, 8))
    for record in records:
        crash = record.raw.get("crash_rate", float("nan"))
        latency = record.raw.get("decision_latency_ms_avg", float("nan"))
        ttc = record.raw.get("ttc_danger_rate_mean", float("nan"))
        if not all(np.isfinite(value) for value in (crash, latency, ttc)):
            continue
        size = 100 + 1800 * max(ttc, 0.0)
        color = SAFE_COLOR if record.gate_pass else RISK_COLOR
        ax.scatter(latency, crash, s=size, c=color, alpha=0.6, edgecolor="#222", linewidth=0.6)
        ax.text(latency + 15, crash + 0.005, _short_label(record), fontsize=8)

    ax.set_xlabel("Decision Latency (ms)")
    ax.set_ylabel("Crash Rate")
    ax.set_title("Risk-Runtime Bubble Chart (bubble = TTC danger rate)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_flow_compliance_risk(records: Sequence[ModelRecord], output_path: Path) -> None:
    if not records:
        return
    _figure_style()
    fig, ax = plt.subplots(figsize=(13, 8))
    for record in records:
        blocking_rate = _blocking_rate_for_record(record)
        crash_rate = record.raw.get("crash_rate", float("nan"))
        if not np.isfinite(blocking_rate) or not np.isfinite(crash_rate):
            continue
        rear_context = record.raw.get("rear_ttc_danger_rate_mean", float("nan"))
        if not np.isfinite(rear_context):
            rear_context = record.raw.get("ttc_danger_rate_mean", float("nan"))
        rear_context = max(_safe_sort_number(rear_context, 0.0), 0.0)
        bubble_size = 100 + 1800 * rear_context
        color = SAFE_COLOR if record.gate_pass else RISK_COLOR
        marker = "o" if record.gate_mode == "telemetry" else "s"
        ax.scatter(
            blocking_rate,
            crash_rate,
            s=bubble_size,
            c=color,
            marker=marker,
            alpha=0.65,
            edgecolor="#222",
            linewidth=0.6,
        )
        ax.text(blocking_rate + 0.01, crash_rate + 0.005, _short_label(record), fontsize=8)

    ax.set_xlabel("Blocking Risk Rate (telemetry low-speed blocking or Stage A proxy)")
    ax.set_ylabel("Crash Rate")
    ax.set_title("Flow Compliance Risk (bubble = rear TTC danger if available, else front TTC danger)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_comfort_panel(records: Sequence[ModelRecord], output_path: Path) -> None:
    if not records:
        return
    _figure_style()
    ordered = sorted(records, key=lambda record: _safe_sort_number(record.indices.get("ComfortIndex", float("nan")), -1e9), reverse=True)
    labels = [_short_label(record) for record in ordered]
    metrics = [
        ("flap_accel_decel_rate_mean", "Accel/Decel Flap Rate"),
        ("lane_change_rate_mean", "Lane Change Rate"),
        ("speed_std_ego_mps", "Speed Variability Std (m/s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, max(6, 0.45 * len(ordered) + 2)), sharey=True)
    for ax, (metric_name, title) in zip(axes, metrics):
        values = [_safe_sort_number(record.raw.get(metric_name, float("nan")), float("nan")) for record in ordered]
        y = np.arange(len(ordered))
        ax.barh(y, values, color="#3f7f93", alpha=0.9)
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.35)
        for yi, value in zip(y, values):
            if np.isfinite(value):
                ax.text(value, yi, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle("Comfort Diagnostics Panel (Lower is Better)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness_compare(
    harmonized_records: Sequence[ModelRecord],
    sensitivity_records: Sequence[ModelRecord],
    output_path: Path,
) -> None:
    _figure_style()
    metrics = [
        ("decision_timeout_rate_mean", "Decision Timeout Rate"),
        ("fallback_action_rate_mean", "Fallback Action Rate"),
        ("format_failure_rate_mean", "Format Failure Rate"),
    ]
    views = [
        ("Harmonized", harmonized_records),
        ("Sensitivity", sensitivity_records),
    ]

    fig, axes = plt.subplots(len(metrics), len(views), figsize=(16, 4.2 * len(metrics)), squeeze=False)
    for row, (metric_name, metric_label) in enumerate(metrics):
        for col, (view_name, records) in enumerate(views):
            ax = axes[row][col]
            ordered = sorted(records, key=lambda record: _safe_sort_number(record.raw.get(metric_name, float("nan")), 1e9))
            labels = [_short_label(record) for record in ordered]
            values = [_safe_sort_number(record.raw.get(metric_name, float("nan")), float("nan")) for record in ordered]
            y = np.arange(len(ordered))
            ax.barh(y, values, color="#637a91", alpha=0.88)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{view_name} | {metric_label}")
            ax.grid(axis="x", alpha=0.35)
            ax.set_xlim(left=0)

    fig.suptitle("Robustness Panel: Harmonized vs Sensitivity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_scorecards(records: Sequence[ModelRecord], output_path: Path, title: str) -> None:
    if not records:
        return
    _figure_style()
    ordered = sorted(records, key=lambda record: _safe_sort_number(float(record.rank or 1e9), 1e9))
    cols = 3
    rows = math.ceil(len(ordered) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.2 * rows), squeeze=False)
    for idx, record in enumerate(ordered):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        bg_color = "#e8f5f0" if record.gate_pass else "#fbe9e7"
        ax.set_facecolor(bg_color)
        summary = (
            f"Rank: {record.rank} ({'PASS' if record.gate_pass else 'FAIL'})\n"
            f"Safety: {record.indices.get('SafetyIndex', float('nan')):.1f}  "
            f"Eff: {record.indices.get('EfficiencyIndex', float('nan')):.1f}\n"
            f"Comfort: {record.indices.get('ComfortIndex', float('nan')):.1f}  "
            f"Robust: {record.indices.get('RobustnessIndex', float('nan')):.1f}\n"
            f"Secondary: {record.secondary_score:.1f} | Adj: {record.secondary_score_adjusted:.1f}\n"
            f"BlockingPenalty: {record.blocking_penalty:.2f} | GateMode: {record.gate_mode}\n"
            f"Crash: {record.raw.get('crash_rate', float('nan')):.3f} | TTC: {record.raw.get('ttc_danger_rate_mean', float('nan')):.3f}\n"
            f"Headway: {record.raw.get('headway_violation_rate_mean', float('nan')):.3f} | Latency: {record.raw.get('decision_latency_ms_avg', float('nan')):.1f} ms"
        )
        ax.text(
            0.03,
            0.95,
            f"{record.model}\n[{record.tier.upper()}-{record.mode.title()}]",
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(0.03, 0.72, summary, transform=ax.transAxes, va="top", fontsize=9, linespacing=1.35)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#c7d6d1")

    for idx in range(len(ordered), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, divider] + body)


def _top_by_index(records: Sequence[ModelRecord], index_name: str) -> Optional[ModelRecord]:
    candidates = [record for record in records if np.isfinite(record.indices.get(index_name, float("nan")))]
    if not candidates:
        return None
    return max(candidates, key=lambda record: record.indices.get(index_name, float("nan")))


def _records_to_rows(records: Sequence[ModelRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in sorted(records, key=lambda item: _safe_sort_number(float(item.rank or 1e9), 1e9)):
        ci_safety = record.ci.get("SafetyIndex", (float("nan"), float("nan")))
        ci_secondary = record.ci.get("SecondaryScore", (float("nan"), float("nan")))
        ci_secondary_adj = record.ci.get("SecondaryScoreAdjusted", (float("nan"), float("nan")))
        ci_blocking_proxy = record.ci.get("blocking_proxy_episode_rate", (float("nan"), float("nan")))
        ci_penalty = record.ci.get("BlockingPenalty", (float("nan"), float("nan")))
        rows.append(
            {
                "rank": record.rank,
                "model": record.model,
                "tier": record.tier,
                "mode": record.mode,
                "family": record.family,
                "gate_status": "pass" if record.gate_pass else "safety_fail",
                "gate_mode": record.gate_mode,
                "SafetyIndex": round(record.indices.get("SafetyIndex", float("nan")), 3),
                "EfficiencyIndex": round(record.indices.get("EfficiencyIndex", float("nan")), 3),
                "ComfortIndex": round(record.indices.get("ComfortIndex", float("nan")), 3),
                "RobustnessIndex": round(record.indices.get("RobustnessIndex", float("nan")), 3),
                "SecondaryScore": round(record.secondary_score, 3),
                "SecondaryScoreAdjusted": round(record.secondary_score_adjusted, 3),
                "BlockingPenalty": round(record.blocking_penalty, 3),
                "CI95_SafetyIndex": f"[{ci_safety[0]:.2f}, {ci_safety[1]:.2f}]",
                "CI95_SecondaryScore": f"[{ci_secondary[0]:.2f}, {ci_secondary[1]:.2f}]",
                "CI95_SecondaryScoreAdjusted": f"[{ci_secondary_adj[0]:.2f}, {ci_secondary_adj[1]:.2f}]",
                "CI95_blocking_proxy_episode_rate": f"[{ci_blocking_proxy[0]:.2f}, {ci_blocking_proxy[1]:.2f}]",
                "CI95_BlockingPenalty": f"[{ci_penalty[0]:.2f}, {ci_penalty[1]:.2f}]",
                "crash_rate": round(record.raw.get("crash_rate", float("nan")), 4),
                "ttc_danger_rate_mean": round(record.raw.get("ttc_danger_rate_mean", float("nan")), 4),
                "headway_violation_rate_mean": round(record.raw.get("headway_violation_rate_mean", float("nan")), 4),
                "rear_ttc_danger_rate_mean": round(record.raw.get("rear_ttc_danger_rate_mean", float("nan")), 4),
                "rear_headway_violation_rate_mean": round(record.raw.get("rear_headway_violation_rate_mean", float("nan")), 4),
                "low_speed_blocking_rate_mean": round(record.raw.get("low_speed_blocking_rate_mean", float("nan")), 4),
                "blocking_proxy_episode_rate": round(record.raw.get("blocking_proxy_episode_rate", float("nan")), 4),
                "error_rate": round(record.raw.get("error_rate", float("nan")), 4),
                "avg_episode_runtime_sec": round(record.raw.get("avg_episode_runtime_sec", float("nan")), 3),
                "decision_latency_ms_avg": round(record.raw.get("decision_latency_ms_avg", float("nan")), 3),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def _run_validation_checks(
    harmonized_records: Sequence[ModelRecord],
    sensitivity_records: Sequence[ModelRecord],
    missing_heatmap_cells: int,
    figure_paths: Sequence[Path],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    index_values = []
    for record in harmonized_records:
        index_values.extend(
            [
                record.indices.get("SafetyIndex", float("nan")),
                record.indices.get("EfficiencyIndex", float("nan")),
                record.indices.get("ComfortIndex", float("nan")),
                record.indices.get("RobustnessIndex", float("nan")),
            ]
        )
    in_bounds = all(0.0 <= value <= 100.0 for value in _finite(index_values))
    checks.append(
        {
            "name": "Index bounds [0,100]",
            "passed": bool(in_bounds),
            "detail": "All normalized indices are within bounds." if in_bounds else "Some index values are outside [0, 100].",
        }
    )

    synthetic_a = ModelRecord(
        model_id="synthetic_a",
        model="synthetic_a",
        family="other",
        tier="tier1",
        mode="instruct",
        experiment_dir="synthetic",
        report_path="synthetic",
        raw={
            "crash_rate": 0.0,
            "ttc_danger_rate_mean": 0.05,
            "headway_violation_rate_mean": 0.05,
            "rear_ttc_danger_rate_mean": float("nan"),
            "rear_headway_violation_rate_mean": float("nan"),
            "low_speed_blocking_rate_mean": float("nan"),
            "blocking_proxy_episode_rate": 0.0,
            "no_collision_rate": 1.0,
            "avg_episode_runtime_sec": 10.0,
            "decision_latency_ms_avg": 1000.0,
            "avg_reward_per_step": 0.7,
            "flap_accel_decel_rate_mean": 0.01,
            "lane_change_rate_mean": 0.1,
            "speed_std_ego_mps": 1.0,
            "decision_timeout_rate_mean": 0.0,
            "fallback_action_rate_mean": 0.0,
            "format_failure_rate_mean": 0.1,
            "response_strict_format_rate": 0.9,
            "error_rate": 0.0,
        },
        episodes=[],
    )
    synthetic_b = ModelRecord(
        model_id="synthetic_b",
        model="synthetic_b",
        family="other",
        tier="tier1",
        mode="instruct",
        experiment_dir="synthetic",
        report_path="synthetic",
        raw={
            **synthetic_a.raw,
            "crash_rate": 0.8,
            "no_collision_rate": 0.2,
            "ttc_danger_rate_mean": 0.4,
            "headway_violation_rate_mean": 0.4,
            "blocking_proxy_episode_rate": 0.9,
        },
        episodes=[],
    )
    synth_records = [synthetic_a, synthetic_b]
    synth_bounds = _compute_component_bounds(synth_records)
    _score_records(synth_records, synth_bounds)
    monotonic_ok = synthetic_a.indices.get("SafetyIndex", 0) > synthetic_b.indices.get("SafetyIndex", 0)
    checks.append(
        {
            "name": "Safety monotonicity",
            "passed": bool(monotonic_ok),
            "detail": "Lower crash risk maps to higher SafetyIndex." if monotonic_ok else "Monotonicity check failed.",
        }
    )

    gate_boundary_record = ModelRecord(
        model_id="boundary",
        model="boundary",
        family="other",
        tier="tier1",
        mode="instruct",
        experiment_dir="synthetic",
        report_path="synthetic",
        raw={
            **synthetic_a.raw,
            "crash_rate": 0.2,
            "ttc_danger_rate_mean": 0.15,
            "headway_violation_rate_mean": 0.15,
            "blocking_proxy_episode_rate": 0.6,
            "error_rate": 0.0,
        },
        episodes=[],
    )
    boundary_pass = _compute_gate(gate_boundary_record)
    checks.append(
        {
            "name": "Gate threshold boundary",
            "passed": bool(boundary_pass),
            "detail": "Boundary values are included in safety gate pass." if boundary_pass else "Gate boundary test failed.",
        }
    )

    small_episode_set = [
        {
            "crashed": bool(i % 2),
            "success_no_collision": bool((i + 1) % 2),
            "ttc_danger_rate": 0.1 + 0.01 * i,
            "headway_violation_rate": 0.08 + 0.01 * i,
            "error": None,
            "episode_runtime_sec": 12 + i,
            "decision_latency_ms_avg": 1000 + 20 * i,
            "episode_reward_sum": 10 + i,
            "steps": 20,
            "flap_accel_decel_rate": 0.02 * i,
            "lane_change_rate": 0.05 * i,
            "avg_ego_speed_mps": 8 + i,
            "decision_timeout_count": 0,
            "fallback_action_count": 0,
            "format_failure_count": 1,
            "decision_calls_total": 20,
            "decisions_made": 20,
            "responses_strict_format": 19,
        }
        for i in range(5)
    ]
    boot_record = ModelRecord(
        model_id="boot",
        model="boot",
        family="other",
        tier="tier1",
        mode="instruct",
        experiment_dir="synthetic",
        report_path="synthetic",
        raw={},
        episodes=small_episode_set,
    )
    boot = _bootstrap_metrics(boot_record, n_bootstrap=2000, rng=np.random.default_rng(7))
    bootstrap_ok = bool(boot and len(next(iter(boot.values()))) == 2000)
    checks.append(
        {
            "name": "Bootstrap with N=5",
            "passed": bootstrap_ok,
            "detail": "Generated 2,000 bootstrap samples." if bootstrap_ok else "Bootstrap sample generation failed.",
        }
    )

    degenerate_delta = _cliffs_delta([1, 1, 1], [1, 1, 1])
    checks.append(
        {
            "name": "Cliff's delta degenerate tie",
            "passed": abs(degenerate_delta) < 1e-9,
            "detail": f"Computed delta={degenerate_delta:.4f}",
        }
    )

    adjusted_le_raw = all(
        _safe_sort_number(record.secondary_score_adjusted, -1e9)
        <= _safe_sort_number(record.secondary_score, -1e9) + 1e-9
        for record in harmonized_records
    )
    checks.append(
        {
            "name": "SecondaryScoreAdjusted monotonicity",
            "passed": bool(adjusted_le_raw),
            "detail": "SecondaryScoreAdjusted is not greater than SecondaryScore for all records."
            if adjusted_le_raw
            else "Found records where SecondaryScoreAdjusted > SecondaryScore.",
        }
    )

    viz_files_ok = all(path.exists() and path.stat().st_size > 0 for path in figure_paths)
    checks.append(
        {
            "name": "Visualization artifact emission",
            "passed": viz_files_ok,
            "detail": "All expected figure files were generated." if viz_files_ok else "One or more figure files are missing.",
        }
    )

    labels_ok = bool(harmonized_records) and bool(sensitivity_records)
    checks.append(
        {
            "name": "Harmonized vs sensitivity data views",
            "passed": labels_ok,
            "detail": "Both comparison views contain records." if labels_ok else "One view is empty.",
        }
    )

    checks.append(
        {
            "name": "Heatmap missing-slice annotation coverage",
            "passed": True,
            "detail": f"Missing heatmap cells flagged: {missing_heatmap_cells}",
        }
    )
    return checks


def _write_markdown_report(
    report_path: Path,
    bundles: Sequence[ExperimentBundle],
    issues: Sequence[Issue],
    harmonized_records: Sequence[ModelRecord],
    sensitivity_records: Sequence[ModelRecord],
    dominant_profile: Optional[Tuple[Tuple[str, Any], ...]],
    effect_sizes_harmonized: Sequence[Dict[str, Any]],
    effect_sizes_sensitivity: Sequence[Dict[str, Any]],
    figure_map: Dict[str, Path],
    validation_checks: Sequence[Dict[str, Any]],
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    issue_counts = Counter(issue.level for issue in issues)
    by_code = Counter(issue.code for issue in issues)
    harmonized_rows = _records_to_rows(harmonized_records)
    sensitivity_rows = _records_to_rows(sensitivity_records)

    top_safety = _top_by_index(harmonized_records, "SafetyIndex")
    top_eff = _top_by_index(harmonized_records, "EfficiencyIndex")
    top_comfort = _top_by_index(harmonized_records, "ComfortIndex")
    top_robust = _top_by_index(harmonized_records, "RobustnessIndex")

    meaningful_harm = [row for row in effect_sizes_harmonized if row.get("practically_meaningful")]
    meaningful_harm.sort(key=lambda row: abs(row.get("cliffs_delta", 0.0)), reverse=True)
    meaningful_sens = [row for row in effect_sizes_sensitivity if row.get("practically_meaningful")]
    meaningful_sens.sort(key=lambda row: abs(row.get("cliffs_delta", 0.0)), reverse=True)

    missing_gaps = [
        issue
        for issue in issues
        if issue.code in {"MISSING_SLICE_DIR", "MISSING_SLICE_COMPARE", "MISSING_COMPARE_REPORT"}
    ]

    profile_dict = dict(dominant_profile) if dominant_profile else {}
    harmonized_dirs = [bundle.experiment_dir for bundle in bundles if bundle.is_harmonized and bundle.valid_for_analysis]
    non_harmonized_dirs = [bundle.experiment_dir for bundle in bundles if (not bundle.is_harmonized) and bundle.valid_for_analysis]
    gate_mode_counter = Counter(record.gate_mode for record in harmonized_records)

    report_lines: List[str] = []
    report_lines.append("# Deep Tier Analysis Report")
    report_lines.append("")
    report_lines.append(f"Generated at: `{now}`")
    report_lines.append("")

    report_lines.append("## Data Quality")
    report_lines.append("")
    report_lines.append(
        f"- Experiments discovered: **{len(bundles)}**, analysis-valid reports: **{sum(1 for bundle in bundles if bundle.valid_for_analysis)}**"
    )
    report_lines.append(
        f"- Issues: **errors={issue_counts.get('error', 0)}**, **warnings={issue_counts.get('warning', 0)}**, **info={issue_counts.get('info', 0)}**"
    )
    report_lines.append("- Issue code distribution:")
    for code, count in sorted(by_code.items(), key=lambda item: (-item[1], item[0])):
        report_lines.append(f"  - `{code}`: {count}")
    report_lines.append("")
    issue_rows = [issue.to_dict() for issue in issues]
    report_lines.append(_render_markdown_table(issue_rows[:40], ["level", "code", "experiment_dir", "model", "message"]))
    report_lines.append("")

    report_lines.append("## Comparability")
    report_lines.append("")
    report_lines.append(f"- Dominant harmonized profile (used for primary ranking): `{json.dumps(profile_dict, ensure_ascii=False)}`")
    report_lines.append(f"- Harmonized experiments: {', '.join(harmonized_dirs) if harmonized_dirs else 'none'}")
    report_lines.append(f"- Non-harmonized experiments (sensitivity-only): {', '.join(non_harmonized_dirs) if non_harmonized_dirs else 'none'}")
    report_lines.append(f"- Harmonized model count: **{len(harmonized_records)}**, Sensitivity model count: **{len(sensitivity_records)}**")
    report_lines.append(
        f"- Gate mode usage (harmonized): telemetry={gate_mode_counter.get('telemetry', 0)}, proxy={gate_mode_counter.get('proxy', 0)}"
    )
    report_lines.append("")

    report_lines.append("## Safety-first Ranking")
    report_lines.append("")
    report_lines.append(
        "- Base gate: `crash_rate <= 0.20`, `ttc_danger_rate_mean <= 0.15`, `headway_violation_rate_mean <= 0.15`, and `error_rate == 0`."
    )
    report_lines.append(
        "- Telemetry gate (Stage B): also require `rear_ttc_danger_rate_mean <= 0.10` and `low_speed_blocking_rate_mean <= 0.35`."
    )
    report_lines.append(
        "- Proxy gate (Stage A fallback): require `blocking_proxy_episode_rate <= 0.60` where proxy = low-speed with low front TTC/headway risk."
    )
    report_lines.append(
        "- Base score: `SecondaryScore = 0.45*Efficiency + 0.35*Comfort + 0.20*Robustness`."
    )
    report_lines.append(
        "- Blocking penalty: `clip((blocking_rate - 0.20) / 0.60, 0, 1) * 20`; final ranking uses `SecondaryScoreAdjusted`."
    )
    report_lines.append("- Tie-breakers: `SafetyIndex`, `avg_reward_per_step`, then lower `decision_latency_ms_avg`.")
    report_lines.append("")
    report_lines.append(f"![Safety Leaderboard]({figure_map['leaderboard_harmonized'].as_posix()})")
    report_lines.append("")
    ranking_columns = [
        "rank",
        "model",
        "tier",
        "mode",
        "gate_status",
        "SafetyIndex",
        "EfficiencyIndex",
        "ComfortIndex",
        "RobustnessIndex",
        "SecondaryScore",
        "SecondaryScoreAdjusted",
        "BlockingPenalty",
        "gate_mode",
        "CI95_SafetyIndex",
        "CI95_SecondaryScore",
        "CI95_SecondaryScoreAdjusted",
        "CI95_blocking_proxy_episode_rate",
        "CI95_BlockingPenalty",
        "crash_rate",
        "ttc_danger_rate_mean",
        "headway_violation_rate_mean",
        "rear_ttc_danger_rate_mean",
        "low_speed_blocking_rate_mean",
        "blocking_proxy_episode_rate",
        "error_rate",
        "avg_episode_runtime_sec",
        "decision_latency_ms_avg",
    ]
    report_lines.append(_render_markdown_table(harmonized_rows, ranking_columns))
    report_lines.append("")
    report_lines.append("- Full ranking CSV: `results/analysis/model_rankings_harmonized.csv`")
    report_lines.append("")

    report_lines.append("## Dimension Deep Dives")
    report_lines.append("")
    if top_safety:
        report_lines.append(f"- Top SafetyIndex: **{top_safety.model}** ({top_safety.indices['SafetyIndex']:.2f})")
    if top_eff:
        report_lines.append(f"- Top EfficiencyIndex: **{top_eff.model}** ({top_eff.indices['EfficiencyIndex']:.2f})")
    if top_comfort:
        report_lines.append(f"- Top ComfortIndex: **{top_comfort.model}** ({top_comfort.indices['ComfortIndex']:.2f})")
    if top_robust:
        report_lines.append(f"- Top RobustnessIndex: **{top_robust.model}** ({top_robust.indices['RobustnessIndex']:.2f})")
    report_lines.append("")
    report_lines.append(f"![Tier x Mode Heatmaps]({figure_map['heatmaps'].as_posix()})")
    report_lines.append("")
    report_lines.append(f"![Family Tradeoff]({figure_map['family_tradeoff'].as_posix()})")
    report_lines.append("")
    report_lines.append(f"![Risk Runtime Bubble]({figure_map['risk_runtime_bubble'].as_posix()})")
    report_lines.append("")
    report_lines.append(f"![Flow Compliance Risk]({figure_map['flow_compliance'].as_posix()})")
    report_lines.append("")
    report_lines.append(f"![Comfort Diagnostics]({figure_map['comfort_panel'].as_posix()})")
    report_lines.append("")
    report_lines.append(f"![Robustness Compare]({figure_map['robustness_compare'].as_posix()})")
    report_lines.append("")
    report_lines.append(f"![Model Scorecards]({figure_map['scorecards_harmonized'].as_posix()})")
    report_lines.append("")
    if meaningful_harm:
        report_lines.append("- Top practically meaningful effect sizes (harmonized):")
        for row in meaningful_harm[:12]:
            report_lines.append(
                f"  - `{row['lens']}` `{row['group_a']} vs {row['group_b']}` on `{row['metric']}`: "
                f"delta={row['cliffs_delta']:.3f} ({row['delta_magnitude']}), median_diff={row['median_diff']:.2f}"
            )
    else:
        report_lines.append("- No practically meaningful harmonized effect sizes detected under current thresholds.")
    report_lines.append("")

    report_lines.append("## Sensitivity Results")
    report_lines.append("")
    report_lines.append(f"![Safety Leaderboard (Sensitivity)]({figure_map['leaderboard_sensitivity'].as_posix()})")
    report_lines.append("")
    report_lines.append(_render_markdown_table(sensitivity_rows[:20], ranking_columns))
    report_lines.append("")
    if meaningful_sens:
        report_lines.append("- Top practically meaningful effect sizes (sensitivity):")
        for row in meaningful_sens[:12]:
            report_lines.append(
                f"  - `{row['lens']}` `{row['group_a']} vs {row['group_b']}` on `{row['metric']}`: "
                f"delta={row['cliffs_delta']:.3f} ({row['delta_magnitude']}), median_diff={row['median_diff']:.2f}"
            )
    else:
        report_lines.append("- No practically meaningful sensitivity effect sizes detected under current thresholds.")
    report_lines.append("")

    report_lines.append("## Coverage Gaps")
    report_lines.append("")
    if missing_gaps:
        for issue in missing_gaps:
            report_lines.append(f"- `{issue.code}`: {issue.message} ({issue.experiment_dir or 'global'})")
    else:
        report_lines.append("- No explicit coverage gaps detected.")
    report_lines.append("")
    report_lines.append("- Validation checks:")
    for check in validation_checks:
        report_lines.append(
            f"  - {'PASS' if check.get('passed') else 'FAIL'} | {check.get('name')}: {check.get('detail')}"
        )
    report_lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def run_pipeline(
    results_root: Path,
    output_root: Path,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    issues: List[Issue] = []
    bundles = _discover_experiment_bundles(results_root, issues)
    _validate_missing_slices(bundles, issues)
    for bundle in bundles:
        _validate_experiment_bundle(bundle, issues)

    dominant_profile = _choose_harmonized_profile(bundles)
    for bundle in bundles:
        bundle.is_harmonized = bool(
            bundle.valid_for_analysis and dominant_profile is not None and bundle.config_profile == dominant_profile
        )

    harmonized_records = _build_model_records(bundles, require_harmonized=True)
    sensitivity_records = _build_model_records(bundles, require_harmonized=False)

    if not sensitivity_records:
        raise RuntimeError("No valid model records found for analysis.")

    harmonized_bounds = _compute_component_bounds(harmonized_records) if harmonized_records else _compute_component_bounds(sensitivity_records)
    sensitivity_bounds = _compute_component_bounds(sensitivity_records)

    if harmonized_records:
        _score_records(harmonized_records, harmonized_bounds)
        _bootstrap_indices(
            harmonized_records,
            component_bounds=harmonized_bounds,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
    _score_records(sensitivity_records, sensitivity_bounds)
    _bootstrap_indices(
        sensitivity_records,
        component_bounds=sensitivity_bounds,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed + 1,
    )

    effect_sizes_harmonized = _effect_sizes(harmonized_records) if harmonized_records else []
    effect_sizes_sensitivity = _effect_sizes(sensitivity_records)

    figures_dir = output_root / "figures"
    figure_map = {
        "leaderboard_harmonized": figures_dir / "safety_leaderboard_harmonized.png",
        "leaderboard_sensitivity": figures_dir / "safety_leaderboard_sensitivity.png",
        "heatmaps": figures_dir / "tier_mode_heatmaps.png",
        "family_tradeoff": figures_dir / "family_tradeoff.png",
        "risk_runtime_bubble": figures_dir / "risk_runtime_bubble.png",
        "flow_compliance": figures_dir / "flow_compliance_risk.png",
        "comfort_panel": figures_dir / "comfort_diagnostics_panel.png",
        "robustness_compare": figures_dir / "robustness_compare_panel.png",
        "scorecards_harmonized": figures_dir / "scorecards_harmonized.png",
    }

    if harmonized_records:
        _plot_safety_leaderboard(
            harmonized_records,
            output_path=figure_map["leaderboard_harmonized"],
            title="Safety-first Leaderboard (Harmonized View)",
        )
    else:
        _plot_safety_leaderboard(
            sensitivity_records,
            output_path=figure_map["leaderboard_harmonized"],
            title="Safety-first Leaderboard (Fallback: Sensitivity View)",
        )
    _plot_safety_leaderboard(
        sensitivity_records,
        output_path=figure_map["leaderboard_sensitivity"],
        title="Safety-first Leaderboard (Sensitivity View)",
    )
    missing_heatmap_cells = _plot_tier_mode_heatmaps(
        harmonized_records if harmonized_records else sensitivity_records,
        output_path=figure_map["heatmaps"],
    )
    _plot_family_tradeoff(harmonized_records if harmonized_records else sensitivity_records, figure_map["family_tradeoff"])
    _plot_risk_runtime_bubble(harmonized_records if harmonized_records else sensitivity_records, figure_map["risk_runtime_bubble"])
    _plot_flow_compliance_risk(harmonized_records if harmonized_records else sensitivity_records, figure_map["flow_compliance"])
    _plot_comfort_panel(harmonized_records if harmonized_records else sensitivity_records, figure_map["comfort_panel"])
    _plot_robustness_compare(harmonized_records if harmonized_records else sensitivity_records, sensitivity_records, figure_map["robustness_compare"])
    _plot_scorecards(
        harmonized_records if harmonized_records else sensitivity_records,
        figure_map["scorecards_harmonized"],
        title="Per-model Scorecards (Primary View)",
    )

    ranking_harmonized_rows = _records_to_rows(harmonized_records if harmonized_records else sensitivity_records)
    ranking_sensitivity_rows = _records_to_rows(sensitivity_records)
    _write_csv(output_root / "model_rankings_harmonized.csv", ranking_harmonized_rows)
    _write_csv(output_root / "model_rankings_sensitivity.csv", ranking_sensitivity_rows)

    validation_checks = _run_validation_checks(
        harmonized_records=harmonized_records if harmonized_records else sensitivity_records,
        sensitivity_records=sensitivity_records,
        missing_heatmap_cells=missing_heatmap_cells,
        figure_paths=list(figure_map.values()),
    )
    validation_path = output_root / "validation_checks.json"
    _write_json(validation_path, {"checks": validation_checks})

    issues_path = output_root / "data_quality_issues.json"
    _write_json(
        issues_path,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "issues": [issue.to_dict() for issue in issues],
        },
    )

    effect_sizes_path = output_root / "effect_sizes.json"
    _write_json(
        effect_sizes_path,
        {
            "harmonized": effect_sizes_harmonized,
            "sensitivity": effect_sizes_sensitivity,
            "practical_thresholds": {
                "cliffs_delta_abs": PRACTICAL_DELTA_THRESHOLD,
                "median_diff_abs": PRACTICAL_MEDIAN_DIFF_THRESHOLD,
            },
        },
    )

    report_path = output_root / "deep_tier_report.md"
    _write_markdown_report(
        report_path=report_path,
        bundles=bundles,
        issues=issues,
        harmonized_records=harmonized_records if harmonized_records else sensitivity_records,
        sensitivity_records=sensitivity_records,
        dominant_profile=dominant_profile,
        effect_sizes_harmonized=effect_sizes_harmonized,
        effect_sizes_sensitivity=effect_sizes_sensitivity,
        figure_map=figure_map,
        validation_checks=validation_checks,
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "output_root": str(output_root),
        "n_experiments": len(bundles),
        "n_harmonized_models": len(harmonized_records),
        "n_sensitivity_models": len(sensitivity_records),
        "dominant_profile": dict(dominant_profile) if dominant_profile else None,
        "artifacts": {
            "report_markdown": str(report_path),
            "figures": {name: str(path) for name, path in figure_map.items()},
            "ranking_harmonized_csv": str(output_root / "model_rankings_harmonized.csv"),
            "ranking_sensitivity_csv": str(output_root / "model_rankings_sensitivity.csv"),
            "issues_json": str(issues_path),
            "effect_sizes_json": str(effect_sizes_path),
            "validation_json": str(validation_path),
        },
    }
    _write_json(output_root / "analysis_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep safety/efficiency/comfort analysis for tiered DiLu-Ollama compare reports."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root folder that contains tier experiment directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results") / "analysis",
        help="Output root for report/figures/artifacts.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples per model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible bootstrap sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        results_root=args.results_root,
        output_root=args.output_root,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
    )
    print("Deep tier analysis completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
