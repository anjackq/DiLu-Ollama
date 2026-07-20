import importlib

from .llm_env import (
    configure_runtime_env,
    openai_compatible_default_headers_from_env,
    openai_compatible_default_headers_from_config,
)
from .config_loader import load_runtime_config
from .ollama_transport import (
    OllamaNativeChatResolution,
    normalize_ollama_native_chat_mode,
    normalize_ollama_think_mode,
    ollama_model_maybe_supports_thinking,
    resolve_ollama_native_chat_mode,
)
from .token_usage import (
    aggregate_episode_token_usage,
    combine_token_usage_records,
    build_token_usage_record_from_langchain_message,
    build_token_usage_record_from_ollama_native_payload,
    build_whitespace_estimate_token_usage,
)
from .highway_env_config import (
    build_highway_env_config,
    resolve_simulation_env_mode,
    build_native_highway_env_config,
    resolve_simulation_env_bundle,
)
from .highway_scenario_spec import (
    apply_highway_scenario_spec,
    normalize_scenario_spec,
    scenario_spec_summary,
)
from .constants import DEFAULT_DILU_SEEDS
from .task_benchmark import (
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
)
from .model_policy import (
    resolve_model_policy,
    apply_model_policy_to_env,
    build_decision_timeout_penalty_state,
    update_decision_timeout_penalty_state,
    decision_timeout_penalty_snapshot,
    build_native_timeout_penalty_state,
    update_native_timeout_penalty_state,
    native_timeout_penalty_snapshot,
)
from .path_utils import (
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
from .energy_monitor import (
    TOKEN_COUNT_METHOD,
    estimate_generated_tokens,
    load_idle_calibration,
    save_idle_calibration,
    enrich_episode_energy_metrics,
    summarize_energy_latency_episodes,
    create_energy_monitor,
    system_hardware_snapshot,
    build_energy_tradeoff_summary,
)
from .safety_shields import (
    SafetyShieldResult,
    apply_lane_change_safety_shield,
    apply_low_speed_recovery_shield,
    apply_longitudinal_safety_shield,
)
from .scientific_reporting import (
    annotate_aggregate_with_scientific_reporting,
    build_primary_metric_spec,
    write_scientific_analysis_artifacts,
)
from .dilu_scoring import (
    BALANCED_DRIVING_SCORE_POLICY_VERSION,
    SPLIT_SCORING_POLICY_VERSION,
    SPLIT_SCORE_FIELDS,
    compute_split_scores_for_episode,
)
from .non_llm_baselines import (
    BaselineDecision,
    BaselinePolicy,
    BaselineSpec,
    DEFAULT_BASELINE_NAMES,
    EXPERT_BASELINE_NAME,
    baseline_names_for_levels,
    configure_true_idm_ego,
    get_baseline_spec,
    iter_baseline_specs,
    parse_baseline_levels,
    resolve_baseline_names,
)
from .harness_config import (
    ConditionSpec,
    ExecutionMode,
    FallbackPolicy,
    HarnessConfig,
    OutputEnforcement,
    PolicyContent,
    ThinkMode,
    TransportProfile,
    resolve_main_conditions,
)

_LAZY_SCIENTIFIC_EXPORTS = {
    "AttemptStatus": (".campaign_attempts", "AttemptStatus"),
    "ScientificAttemptLedger": (".campaign_attempts", "ScientificAttemptLedger"),
    "ScientificAttemptRecord": (".campaign_attempts", "ScientificAttemptRecord"),
    "ScientificAttemptWriteError": (
        ".campaign_attempts",
        "ScientificAttemptWriteError",
    ),
    "RuntimeLockBinding": (".scientific_runtime", "RuntimeLockBinding"),
    "ScientificEpisodeIdentity": (
        ".scientific_runtime",
        "ScientificEpisodeIdentity",
    ),
    "ScientificEpisodeRuntime": (
        ".scientific_runtime",
        "ScientificEpisodeRuntime",
    ),
    "VerifiedRuntimeLockBinding": (
        ".scientific_runtime",
        "VerifiedRuntimeLockBinding",
    ),
    "load_verified_runtime_lock_binding": (
        "._scientific_runtime_binding",
        "load_verified_runtime_lock_binding",
    ),
    "build_scientific_episode_runtime": (
        ".scientific_runtime",
        "build_scientific_episode_runtime",
    ),
    "ScientificTraceWriter": (".scientific_trace", "ScientificTraceWriter"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_SCIENTIFIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "configure_runtime_env",
    "openai_compatible_default_headers_from_env",
    "openai_compatible_default_headers_from_config",
    "load_runtime_config",
    "OllamaNativeChatResolution",
    "normalize_ollama_native_chat_mode",
    "normalize_ollama_think_mode",
    "ollama_model_maybe_supports_thinking",
    "resolve_ollama_native_chat_mode",
    "aggregate_episode_token_usage",
    "combine_token_usage_records",
    "build_token_usage_record_from_langchain_message",
    "build_token_usage_record_from_ollama_native_payload",
    "build_whitespace_estimate_token_usage",
    "build_highway_env_config",
    "resolve_simulation_env_mode",
    "build_native_highway_env_config",
    "resolve_simulation_env_bundle",
    "apply_highway_scenario_spec",
    "normalize_scenario_spec",
    "scenario_spec_summary",
    "DEFAULT_DILU_SEEDS",
    "DEFAULT_BENCHMARK_CASE_SET",
    "load_benchmark_case_set",
    "build_benchmark_case_set_fingerprint",
    "build_case_env_config",
    "benchmark_max_steps",
    "build_benchmark_instruction",
    "benchmark_metric_config",
    "validate_benchmark_case_set",
    "summarize_benchmark_episodes",
    "benchmark_result_validity",
    "BenchmarkEpisodeEvaluator",
    "augment_behavior_aware_benchmark_episode",
    "resolve_model_policy",
    "apply_model_policy_to_env",
    "build_decision_timeout_penalty_state",
    "update_decision_timeout_penalty_state",
    "decision_timeout_penalty_snapshot",
    "build_native_timeout_penalty_state",
    "update_native_timeout_penalty_state",
    "native_timeout_penalty_snapshot",
    "ensure_dir",
    "ensure_parent_dir",
    "timestamped_results_path",
    "current_timestamp",
    "slugify_model_name",
    "build_experiment_root",
    "build_model_root",
    "build_model_run_dir",
    "ensure_experiment_layout",
    "write_json_atomic",
    "read_json",
    "TOKEN_COUNT_METHOD",
    "estimate_generated_tokens",
    "load_idle_calibration",
    "save_idle_calibration",
    "enrich_episode_energy_metrics",
    "summarize_energy_latency_episodes",
    "create_energy_monitor",
    "system_hardware_snapshot",
    "build_energy_tradeoff_summary",
    "SafetyShieldResult",
    "apply_lane_change_safety_shield",
    "apply_low_speed_recovery_shield",
    "apply_longitudinal_safety_shield",
    "annotate_aggregate_with_scientific_reporting",
    "build_primary_metric_spec",
    "write_scientific_analysis_artifacts",
    "BALANCED_DRIVING_SCORE_POLICY_VERSION",
    "SPLIT_SCORING_POLICY_VERSION",
    "SPLIT_SCORE_FIELDS",
    "compute_split_scores_for_episode",
    "BaselineDecision",
    "BaselinePolicy",
    "BaselineSpec",
    "DEFAULT_BASELINE_NAMES",
    "EXPERT_BASELINE_NAME",
    "baseline_names_for_levels",
    "configure_true_idm_ego",
    "get_baseline_spec",
    "iter_baseline_specs",
    "parse_baseline_levels",
    "resolve_baseline_names",
    "AttemptStatus",
    "ConditionSpec",
    "ExecutionMode",
    "FallbackPolicy",
    "HarnessConfig",
    "OutputEnforcement",
    "PolicyContent",
    "RuntimeLockBinding",
    "ScientificAttemptLedger",
    "ScientificAttemptRecord",
    "ScientificAttemptWriteError",
    "ScientificEpisodeIdentity",
    "ScientificEpisodeRuntime",
    "ScientificTraceWriter",
    "VerifiedRuntimeLockBinding",
    "build_scientific_episode_runtime",
    "load_verified_runtime_lock_binding",
    "resolve_main_conditions",
    "ThinkMode",
    "TransportProfile",
]
