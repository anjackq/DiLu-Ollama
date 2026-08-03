"""Thin orchestration for the locked ICLR 2027 minimal factorial campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from evaluate_models_ollama import run_episode

from ._minimal_factorial_runner_campaign import (
    FrozenThresholds,
    load_checked_case_set as _load_checked_case_set_impl,
    load_frozen_s1 as _load_frozen_s1_impl,
    open_frozen_campaign as _open_frozen_campaign_impl,
    validate_live_snapshot as _validate_live_snapshot_impl,
    verify_frozen_campaign as _verify_frozen_campaign_impl,
)
from ._minimal_factorial_runner_execution import (
    RunSummary,
    build_model_clients,
    execute_campaign as _execute_campaign_impl,
)
from ._minimal_factorial_runner_episode import (
    run_scheduled_episode as _run_scheduled_episode_impl,
)
from ._minimal_factorial_runner_status import (
    campaign_status as _campaign_status_impl,
    summarize_status as _summarize_status_impl,
)
from ._minimal_factorial_runner_summaries import append_summary_record
from ._scientific_runtime_binding import load_verified_runtime_lock_binding
from .campaign_attempts import (
    AttemptStatus,
    ScientificAttemptLedger,
)
from .minimal_factorial_schedule import (
    ExperimentManifest,
    RuntimeSnapshot,
    build_smoke_schedule,
    build_runtime_snapshot,
    build_union_schedule,
    load_experiment_manifest,
)
from .runtime_lock_authoring import author_verified_runtime_locks
from .scientific_runtime import build_scientific_episode_runtime
from .scientific_trace import ScientificTraceWriter
from .task_benchmark import (
    benchmark_max_steps,
    build_benchmark_instruction,
    build_case_env_config,
)


def run_probe_lock(manifest_path: Path) -> Path:
    result = author_verified_runtime_locks(_repo_root(manifest_path))
    return result.preflight_path


def run_smoke(manifest_path: Path, *, resume: bool) -> RunSummary:
    prepared = _prepare_campaign(manifest_path, campaign="smoke")
    return _execute_campaign(
        prepared,
        scheduled_rows=prepared.schedule,
        denominator_rows=prepared.schedule,
        resume=resume,
        stage="smoke",
    )


def run_claim_stage(
    manifest_path: Path,
    *,
    stage: Literal["stage1", "stage2"],
    resume: bool,
) -> RunSummary:
    if stage not in {"stage1", "stage2"}:
        raise ValueError("stage must be 'stage1' or 'stage2'.")
    prepared = _prepare_campaign(manifest_path, campaign="claim")
    stage_name = "stage1" if stage == "stage1" else "stage2_additional"
    scheduled = tuple(row for row in prepared.schedule if row.stage == stage_name)
    denominator = scheduled if stage == "stage1" else tuple(prepared.schedule)
    return _execute_campaign(
        prepared,
        scheduled_rows=scheduled,
        denominator_rows=denominator,
        resume=resume,
        stage=stage,
    )


def campaign_status(manifest_path: Path) -> dict[str, Any]:
    validated = _validate_live_snapshot(manifest_path)
    base = validated.repo_root / validated.manifest.outputs.root
    campaigns = []
    for campaign, directory in (
        ("smoke", validated.manifest.outputs.smoke),
        ("claim", validated.manifest.outputs.llm_campaign),
    ):
        if (base / directory / "campaign_manifest.json").is_file():
            campaigns.append(_open_frozen_campaign(validated, campaign))
    return _campaign_status_impl(tuple(campaigns))


def _summarize_status(
    rows: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, AttemptStatus],
) -> dict[str, Any]:
    return _summarize_status_impl(rows, statuses)


def _select_pending_rows(
    rows: Sequence[Any],
    statuses: Mapping[str, AttemptStatus],
    *,
    resume: bool,
) -> tuple[Any, ...]:
    terminal = {
        AttemptStatus.COMPLETED,
        AttemptStatus.BLOCKED,
        AttemptStatus.FAILED,
        AttemptStatus.WRITE_AMBIGUOUS,
    }
    if not resume and statuses:
        raise ValueError("Existing attempts require resume=True.")
    return tuple(
        row for row in rows if statuses.get(row.episode_attempt_id) not in terminal
    )


def _prepare_campaign(manifest_path: Path, *, campaign: str) -> Any:
    validated = _validate_live_snapshot(manifest_path)
    return _open_frozen_campaign(validated, campaign)


def _execute_campaign(
    prepared: Any,
    *,
    scheduled_rows: Sequence[Any],
    denominator_rows: Sequence[Any],
    resume: bool,
    stage: str,
) -> RunSummary:
    _require_unique_scheduled_ids(scheduled_rows)
    _require_unique_scheduled_ids(denominator_rows)
    return _execute_campaign_impl(
        prepared,
        scheduled_rows=scheduled_rows,
        denominator_rows=denominator_rows,
        resume=resume,
        stage=stage,
        ledger_type=ScientificAttemptLedger,
        trace_type=ScientificTraceWriter,
        client_builder=build_model_clients,
        episode_runner=_run_scheduled_episode,
        pending_selector=_select_pending_rows,
        summary_appender=_append_episode_summary,
        failure_recorder=_record_infrastructure_failure,
        completion_checker=_completion_errors,
    )


def _run_scheduled_episode(
    prepared: Any,
    scheduled: Any,
    *,
    ledger: Any,
    trace_writer: Any,
    client: Any,
    episode_temp_dir: Path,
) -> dict[str, Any]:
    return _run_scheduled_episode_impl(
        prepared,
        scheduled,
        ledger=ledger,
        trace_writer=trace_writer,
        client=client,
        episode_temp_dir=episode_temp_dir,
        lock_loader=load_verified_runtime_lock_binding,
        runtime_builder=build_scientific_episode_runtime,
        env_builder=build_case_env_config,
        instruction_builder=build_benchmark_instruction,
        max_steps_builder=benchmark_max_steps,
        evaluator=run_episode,
    )


def _record_infrastructure_failure(
    ledger: Any,
    scheduled: Any,
    error: Exception,
) -> None:
    attempt_id = scheduled.episode_attempt_id
    status = ledger.attempt_status(attempt_id)
    if status is None:
        ledger.append_started(attempt_id)
    elif status is AttemptStatus.COMPLETED:
        ledger.append_summary_failure(
            attempt_id,
            failure_class="summary_durability_failure",
            failure_message=f"{type(error).__name__}: {error}",
        )
        return
    elif status is not AttemptStatus.STARTED:
        return
    ledger.append_terminal(
        attempt_id,
        status=AttemptStatus.FAILED,
        decision_count=0,
        failure_class="infrastructure_exception",
        failure_message=f"{type(error).__name__}: {error}",
        trace_absence_reason="aborted_before_first_decision",
    )


def _completion_errors(
    rows: Sequence[Any],
    summaries: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, AttemptStatus],
) -> tuple[str, ...]:
    scheduled_ids = [row.episode_attempt_id for row in rows]
    expected = set(scheduled_ids)
    observed = [item.get("episode_attempt_id") for item in summaries]
    errors: list[str] = []
    if len(scheduled_ids) != len(expected):
        errors.append("duplicate scheduled episode IDs")
    if len(observed) != len(set(observed)):
        errors.append("duplicate episode summaries")
    if set(observed) != expected or len(observed) != len(expected):
        errors.append("episode summary denominator mismatch")
    if any(
        statuses.get(attempt_id) is not AttemptStatus.COMPLETED
        for attempt_id in expected
    ):
        errors.append("non-completed attempt in denominator")
    return tuple(errors)


def _require_unique_scheduled_ids(rows: Sequence[Any]) -> None:
    identities = [row.episode_attempt_id for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Campaign contains duplicate scheduled episode IDs.")


def _append_episode_summary(
    path: Path,
    summary: Mapping[str, Any],
    ledger: Any,
) -> None:
    attempt_id = summary.get("episode_attempt_id")
    if (
        not isinstance(attempt_id, str)
        or ledger.attempt_status(attempt_id) not in _TERMINAL_STATUSES
    ):
        raise RuntimeError("Episode summary requires terminal attempt evidence.")
    append_summary_record(path, summary)


def _validate_live_snapshot(manifest_path: Path) -> Any:
    return _validate_live_snapshot_impl(
        manifest_path,
        repo_root=_repo_root,
        load_manifest=load_experiment_manifest,
        load_cases=_load_checked_case_set,
        build_snapshot=build_runtime_snapshot,
    )


def _open_frozen_campaign(validated: Any, campaign: str) -> Any:
    return _open_frozen_campaign_impl(
        validated,
        campaign,
        load_s1=_load_frozen_s1,
        build_smoke=build_smoke_schedule,
        build_union=build_union_schedule,
        verify=_verify_frozen_campaign,
    )


def _load_frozen_s1(
    root: Path,
    manifest: ExperimentManifest,
    snapshot: RuntimeSnapshot,
) -> Any:
    return _load_frozen_s1_impl(root, manifest, snapshot)


def _verify_frozen_campaign(
    path: Path,
    manifest: ExperimentManifest,
    snapshot: RuntimeSnapshot,
    schedule: Sequence[Any],
    case_set: Mapping[str, Any],
    *,
    union_path: Path | None = None,
) -> None:
    _verify_frozen_campaign_impl(
        path,
        manifest,
        snapshot,
        schedule,
        case_set,
        union_path=union_path,
    )


def _load_checked_case_set(
    repo_root: Path,
    manifest: ExperimentManifest,
) -> dict[str, Any]:
    return _load_checked_case_set_impl(repo_root, manifest)


def _repo_root(manifest_path: Path) -> Path:
    if not isinstance(manifest_path, Path):
        raise TypeError("manifest_path must be a pathlib.Path.")
    resolved = manifest_path.resolve(strict=True)
    expected_suffix = Path("configs") / "iclr2027" / "minimal_factorial.yaml"
    if Path(*resolved.parts[-3:]) != expected_suffix:
        raise ValueError("manifest_path must identify the registered manifest.")
    return resolved.parents[2]


_TERMINAL_STATUSES = frozenset(
    {
        AttemptStatus.COMPLETED,
        AttemptStatus.BLOCKED,
        AttemptStatus.FAILED,
        AttemptStatus.WRITE_AMBIGUOUS,
    }
)


__all__ = [
    "FrozenThresholds",
    "RunSummary",
    "campaign_status",
    "run_claim_stage",
    "run_probe_lock",
    "run_smoke",
]
