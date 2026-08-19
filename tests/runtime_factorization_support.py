from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest import mock

import evaluate_models_ollama as evaluator

from dilu.runtime.campaign_attempts import ScientificAttemptLedger
from dilu.runtime.harness_config import HarnessConfig, OutputEnforcement
from dilu.runtime.ollama_scientific_client import OllamaScientificClient
from dilu.runtime.scientific_runtime import (
    RuntimeLockBinding,
    ScientificEpisodeIdentity,
    ScientificEpisodeRuntime,
    VerifiedRuntimeLockBinding,
    build_scientific_episode_runtime,
)
from dilu.runtime._scientific_runtime_binding import (
    load_verified_runtime_lock_binding,
)
from dilu.runtime.scientific_trace import (
    ScientificTraceCommitAmbiguousError,
    ScientificTraceWriter,
    TraceReference,
)
from tests.scientific_transport_support import (
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_retry_policy,
    success_payload,
)
from tests.test_scientific_driver_action_resolution import _scientific_config


def identity(attempt: int = 0) -> ScientificEpisodeIdentity:
    return ScientificEpisodeIdentity(
        campaign_id="campaign-runtime-001",
        episode_attempt_id=f"episode-attempt-{attempt:03d}",
        case_id="case-001",
        pair_id="pair-001",
        template_id="template-001",
        replicate_id=0,
        simulator_seed=7,
        primary_snapshot_id="snapshot-001",
        benchmark_fingerprint="sha256:" + "b" * 64,
        code_revision="git:" + "d" * 40,
    )


def client(*, schema: bool = False) -> OllamaScientificClient:
    content = '"Response to user:#### 3"' if schema else "Response to user:#### 3"
    return OllamaScientificClient(
        capabilities=make_capabilities(),
        retry_policy=make_retry_policy(),
        identity_inspector=identity_inspector_for(),
        post=lambda *args, **kwargs: FakeResponse(success_payload(content)),
        sleep=lambda _: None,
    )


def runtime(
    root: Path,
    *,
    attempt: int = 0,
    episode_identity: ScientificEpisodeIdentity | None = None,
    config: HarnessConfig | None = None,
    ledger: ScientificAttemptLedger | None = None,
    transport_client: OllamaScientificClient | None = None,
    writer: ScientificTraceWriter | None = None,
) -> ScientificEpisodeRuntime:
    config = config or _scientific_config()
    resolved_identity = episode_identity or identity(attempt)
    capabilities = make_capabilities()
    resolved_client = transport_client or client(
        schema=config.condition.output_enforcement is OutputEnforcement.BACKEND_SCHEMA
    )
    captured_lock = RuntimeLockBinding.from_runtime(
        harness_config=config,
        identity=resolved_identity,
        capabilities=capabilities,
    )
    external_lock = verified_runtime_lock(root, captured_lock.to_dict())
    return build_scientific_episode_runtime(
        harness_config=config,
        identity=resolved_identity,
        runtime_lock=external_lock,
        transport_client=resolved_client,
        trace_writer=writer
        or ScientificTraceWriter(
            root / "decision_traces.jsonl", artifact_root=root, resume=attempt > 0
        ),
        attempt_ledger=ledger
        or ScientificAttemptLedger(
            root / "campaign_attempts.jsonl",
            campaign_id=resolved_identity.campaign_id,
            resume=attempt > 0,
        ),
    )


def verified_runtime_lock(
    root: Path,
    mapping: dict[str, str],
    *,
    expected_source_artifact_sha256: str | None = None,
) -> VerifiedRuntimeLockBinding:
    root.mkdir(parents=True, exist_ok=True)
    runtime_bytes = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    source_artifact_sha256 = "sha256:" + hashlib.sha256(runtime_bytes).hexdigest()
    authorization = {
        "artifact_type": "runtime_lock_authorization_v1",
        "runtime_lock_sha256": (
            expected_source_artifact_sha256 or source_artifact_sha256
        ),
    }
    runtime_path = root / "RUNTIME_PROTOCOL_LOCK.json"
    authorization_path = root / "PROTOCOL_FROZEN.json"
    runtime_path.write_bytes(runtime_bytes)
    authorization_path.write_text(
        json.dumps(authorization, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return load_verified_runtime_lock_binding(
        runtime_lock_path=runtime_path,
        authorization_path=authorization_path,
    )


class FakeEnvironment:
    def __init__(
        self,
        *,
        fail_step: bool = False,
        terminate_after: int = 1,
    ) -> None:
        self.unwrapped = self
        self.fail_step = fail_step
        self.terminate_after = terminate_after
        self.step_count = 0
        self.closed = False

    def configure(self, config: dict[str, Any]) -> None:
        del config

    def reset(self, *, seed: int) -> tuple[list[float], dict[str, Any]]:
        del seed
        return [0.0], {}

    def step(
        self,
        action_id: int,
    ) -> tuple[list[float], float, bool, bool, dict[str, bool]]:
        if self.fail_step:
            raise RuntimeError("simulator failed")
        self.step_count += 1
        terminated = self.step_count >= self.terminate_after
        return [float(action_id)], 1.0, terminated, False, {"crashed": False}

    def close(self) -> None:
        self.closed = True


class AmbiguousTraceWriter(ScientificTraceWriter):
    def append(self, record: object) -> None:
        del record
        raise ScientificTraceCommitAmbiguousError("trace fsync outcome is ambiguous")


class SecondAppendAmbiguousTraceWriter(ScientificTraceWriter):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._append_attempts = 0

    def append(self, record: object) -> TraceReference:
        self._append_attempts += 1
        if self._append_attempts == 2:
            self._poisoned = True
            self._poison_path.write_text(
                "trace_commit_ambiguous:second-append\n",
                encoding="utf-8",
            )
            raise ScientificTraceCommitAmbiguousError(
                "second trace fsync outcome is ambiguous"
            )
        return super().append(record)


class FakeScenario:
    def available_action_ids(self) -> list[int]:
        return [0, 1, 2, 3, 4]

    def action_id_for_token(self, token: str) -> int:
        if token != "IDLE":
            raise KeyError(token)
        return 1

    def describe(self, frame_id: int) -> str:
        return f"deterministic scenario {frame_id}"

    def availableActionsDescription(self) -> str:
        return "0 left; 1 idle; 2 right; 3 faster; 4 slower"


def run_episode(
    root: Path,
    scientific_runtime: ScientificEpisodeRuntime,
    *,
    timeout_state: dict | None = None,
    environment: FakeEnvironment | None = None,
    max_steps_override: int | None = 1,
    score_side_effect: Any = None,
) -> dict[str, Any]:
    environment = environment or FakeEnvironment()
    resolved_score_side_effect = score_side_effect or (lambda result: result)
    with (
        mock.patch.object(evaluator.gym, "make", return_value=environment),
        mock.patch.object(evaluator, "EnvScenario", return_value=FakeScenario()),
        mock.patch.object(
            evaluator,
            "extract_step_traffic_metrics",
            side_effect=lambda *args, **kwargs: traffic_metrics(),
        ),
        mock.patch.object(
            evaluator,
            "compute_split_scores_for_episode",
            side_effect=resolved_score_side_effect,
        ),
    ):
        return evaluator.run_episode(
            config={},
            env_config={"fake-env": {}},
            env_type="fake-env",
            agent_memory=None,
            seed=7,
            few_shot_num=0,
            temp_dir=str(root),
            ttc_threshold_sec=2.0,
            headway_threshold_m=8.0,
            rear_ttc_threshold_sec=2.0,
            rear_headway_threshold_m=8.0,
            low_speed_blocking_threshold_mps=5.0,
            blocking_front_gap_safe_m=20.0,
            blocking_front_ttc_safe_sec=4.0,
            stop_threshold_mps=0.5,
            near_stop_threshold_mps=2.0,
            alignment_sample_rate=0.0,
            alignment_max_samples=0,
            slow_decision_threshold_sec=1.0,
            save_artifacts=False,
            record_video=False,
            quiet_mode=True,
            enable_db_logging=False,
            on_decision=None,
            max_steps_override=max_steps_override,
            timeout_penalty_state=timeout_state,
            scientific_runtime=scientific_runtime,
        )


def traffic_metrics() -> dict[str, object]:
    return {
        "ego_speed_mps": 25.0,
        "front_gap_m": None,
        "relative_speed_mps": None,
        "ttc_sec": None,
        "ttc_danger": False,
        "headway_violation": False,
        "rear_gap_m": None,
        "rear_closing_speed_mps": None,
        "rear_ttc_sec": None,
        "rear_ttc_danger": False,
        "rear_headway_violation": False,
        "low_speed_blocking": False,
        "stopped": False,
        "near_stop": False,
    }


def terminal_attempt_payload(root: Path) -> dict[str, object]:
    path = root / "campaign_attempts.jsonl"
    payloads = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    attempts = [
        payload["attempt"]
        for payload in payloads
        if payload["event_type"] == "attempt_lifecycle"
    ]
    return attempts[-1]


__all__ = [
    "AmbiguousTraceWriter",
    "FakeEnvironment",
    "SecondAppendAmbiguousTraceWriter",
    "client",
    "identity",
    "run_episode",
    "runtime",
    "terminal_attempt_payload",
]
