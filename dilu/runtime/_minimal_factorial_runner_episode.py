"""One-episode binding for the minimal-factorial campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping


def run_scheduled_episode(
    prepared: Any,
    scheduled: Any,
    *,
    ledger: Any,
    trace_writer: Any,
    client: Any,
    episode_temp_dir: Path,
    lock_loader: Callable[..., Any],
    runtime_builder: Callable[..., Any],
    env_builder: Callable[..., tuple[Mapping[str, Any], Mapping[str, Any]]],
    instruction_builder: Callable[[Mapping[str, Any]], str],
    max_steps_builder: Callable[..., int],
    evaluator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    lock_root = prepared.lock_root / scheduled.model_slot / scheduled.condition_id
    runtime_lock = lock_loader(
        runtime_lock_path=lock_root / "RUNTIME_PROTOCOL_LOCK.json",
        authorization_path=lock_root / "PROTOCOL_FROZEN.json",
    )
    scientific_runtime = runtime_builder(
        harness_config=scheduled.condition,
        identity=scheduled.identity(),
        runtime_lock=runtime_lock,
        transport_client=client,
        trace_writer=trace_writer,
        attempt_ledger=ledger,
    )
    benchmark_case = prepared.case_by_id[scheduled.case_id]
    case_env_config, case_env_snapshot = env_builder(
        prepared.environment_config,
        prepared.target_env_id,
        benchmark_case,
    )
    thresholds = prepared.thresholds
    result = evaluator(
        config=prepared.runtime_config,
        env_config=case_env_config,
        env_type=prepared.target_env_id,
        agent_memory=None,
        seed=scheduled.simulator_seed,
        few_shot_num=0,
        temp_dir=str(episode_temp_dir),
        ttc_threshold_sec=thresholds.ttc_threshold_sec,
        headway_threshold_m=thresholds.headway_threshold_m,
        rear_ttc_threshold_sec=thresholds.rear_ttc_threshold_sec,
        rear_headway_threshold_m=thresholds.rear_headway_threshold_m,
        low_speed_blocking_threshold_mps=(thresholds.low_speed_blocking_threshold_mps),
        blocking_front_gap_safe_m=thresholds.blocking_front_gap_safe_m,
        blocking_front_ttc_safe_sec=thresholds.blocking_front_ttc_safe_sec,
        stop_threshold_mps=thresholds.stop_threshold_mps,
        near_stop_threshold_mps=thresholds.near_stop_threshold_mps,
        alignment_sample_rate=0.0,
        alignment_max_samples=0,
        slow_decision_threshold_sec=thresholds.slow_decision_threshold_sec,
        timeout_penalty_state=None,
        save_artifacts=False,
        record_video=False,
        run_dir=None,
        run_id=None,
        model_name=scheduled.model_tag,
        quiet_mode=True,
        enable_db_logging=False,
        on_step=None,
        on_decision=None,
        benchmark_case=benchmark_case,
        driving_instruction=instruction_builder(benchmark_case),
        max_steps_override=max_steps_builder(
            benchmark_case,
            case_env_snapshot,
            prepared.default_max_steps,
        ),
        timeout_early_stop_policy=None,
        execution_mode=scheduled.condition.condition.execution_mode,
        shield_config=scheduled.condition.shield,
        scientific_trace_writer=None,
        scientific_trace_record_factory=None,
        scientific_runtime=scientific_runtime,
    )
    return {
        **result,
        "runtime_lock_source_artifact_sha256": (runtime_lock.source_artifact_sha256),
        "runtime_lock_authorization_artifact_sha256": (
            runtime_lock.authorization_artifact_sha256
        ),
        "runtime_lock_binding_sha256": runtime_lock.binding_sha256,
        "prompt_sha256": runtime_lock.prompt_sha256,
        "capability_artifact_sha256": runtime_lock.capability_artifact_sha256,
        "capability_snapshot_sha256": runtime_lock.capability_snapshot_sha256,
        "trace_schema_sha256": runtime_lock.trace_schema_sha256,
    }


__all__ = ["run_scheduled_episode"]
