from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from dilu.driver_agent.prompt_modules import PromptArtifact

from ._scientific_runtime_binding import (
    RuntimeLockBinding,
    ScientificEpisodeIdentity,
    VerifiedRuntimeLockBinding,
)
from .action_resolution import ActionResolutionResult
from .campaign_attempts import (
    TRACE_COMMIT_AMBIGUOUS,
    AttemptStatus,
    ScientificAttemptLedger,
)
from .generation_seed import (
    post_divergence_generation_seed,
    primary_snapshot_generation_seed,
)
from .harness_config import HarnessConfig
from .ollama_scientific_client import (
    OllamaScientificClient,
    ScientificGenerationAbort,
)
from .runtime_failures import (
    ProtocolInvariantCode,
    ProtocolInvariantViolation,
    RuntimeProtocolError,
)
from .scientific_trace import (
    DecisionTraceContext,
    DecisionTraceKey,
    DecisionTraceRecord,
    GenerationSeedScope,
    ScientificTraceWriter,
    ScientificSimulatorAbort,
    ScientificTraceCommitAmbiguousError,
    ScientificTraceWriteError,
    TraceDisposition,
    TraceReference,
)
from .scientific_transport_records import GenerationResult
from .scientific_transport_types import ScientificGenerationContext
from .shield_stack import ShieldStackResult


@dataclass(frozen=True)
class ScientificEpisodeRuntime:
    harness_config: HarnessConfig
    identity: ScientificEpisodeIdentity
    runtime_lock: VerifiedRuntimeLockBinding
    transport_client: OllamaScientificClient
    trace_writer: ScientificTraceWriter
    attempt_ledger: ScientificAttemptLedger
    completion_publisher: (
        Callable[[Mapping[str, Any], tuple[TraceReference, ...]], None] | None
    ) = None

    def __post_init__(self) -> None:
        expected_types = (
            (self.harness_config, HarnessConfig, "harness_config"),
            (self.identity, ScientificEpisodeIdentity, "identity"),
            (self.runtime_lock, VerifiedRuntimeLockBinding, "runtime_lock"),
            (self.transport_client, OllamaScientificClient, "transport_client"),
            (self.trace_writer, ScientificTraceWriter, "trace_writer"),
            (self.attempt_ledger, ScientificAttemptLedger, "attempt_ledger"),
        )
        for value, expected, name in expected_types:
            if not isinstance(value, expected):
                raise ValueError(f"{name} must be {expected.__name__}.")
        if self.attempt_ledger.campaign_id != self.identity.campaign_id:
            raise ValueError("Attempt ledger belongs to another campaign.")

    @property
    def model_tag(self) -> str:
        return self.runtime_lock.model_tag

    @property
    def model_digest(self) -> str:
        return self.runtime_lock.model_digest

    def validate_binding(self) -> None:
        try:
            self.harness_config.validate_scientific()
            if self.transport_client.retry_policy != self.harness_config.retry_policy:
                raise ValueError(
                    "Transport retry policy drifted from the harness config."
                )
            expected = RuntimeLockBinding.from_runtime(
                harness_config=self.harness_config,
                identity=self.identity,
                capabilities=self.transport_client.capabilities,
            )
            if self.runtime_lock.to_dict() != expected.to_dict():
                raise ValueError("Runtime lock fields do not match the bound runtime.")
        except Exception as exc:
            violation = ProtocolInvariantViolation.from_mapping(
                ProtocolInvariantCode.RUNTIME_LOCK_MISMATCH,
                "Scientific runtime lock validation failed.",
                {"error_type": type(exc).__name__},
            )
            raise RuntimeProtocolError(violation) from exc

    def begin_attempt(self) -> None:
        attempt_id = self.identity.episode_attempt_id
        if self.attempt_ledger.attempt_status(attempt_id) is AttemptStatus.STARTED:
            if not self.attempt_ledger.can_resume(attempt_id):
                raise RuntimeError("Started attempt is not ledger-approved to resume.")
            return
        self.attempt_ledger.append_started(attempt_id)

    def generation_context(self, decision_index: int) -> ScientificGenerationContext:
        _require_nonnegative_int("decision_index", decision_index)
        request_id = self._request_id(decision_index)
        self.attempt_ledger.register_request_id(
            request_id,
            self.identity.episode_attempt_id,
        )
        return ScientificGenerationContext(
            request_id=request_id,
            model_digest=self.model_digest,
            generation_seed=self._generation_seed(decision_index),
        )

    def build_trace_record(
        self,
        *,
        decision_index: int,
        env_step_index: int,
        available_action_ids: tuple[int, ...],
        prompt_artifact: PromptArtifact,
        generation: GenerationResult,
        resolution: ActionResolutionResult | None,
        shield_stack: ShieldStackResult | None,
        disposition: TraceDisposition,
        decision_latency_ms: float,
        benchmark_event_meta: Mapping[str, Any],
    ) -> DecisionTraceRecord:
        self._validate_generation(decision_index, generation)
        identity = self.identity
        key = DecisionTraceKey(
            campaign_id=identity.campaign_id,
            episode_attempt_id=identity.episode_attempt_id,
            condition_id=self.harness_config.condition_id(),
            case_id=identity.case_id,
            pair_id=identity.pair_id,
            template_id=identity.template_id,
            replicate_id=identity.replicate_id,
            decision_index=decision_index,
            env_step_index=env_step_index,
        )
        scope = (
            GenerationSeedScope.PRIMARY_SNAPSHOT
            if decision_index == 0
            else GenerationSeedScope.POST_DIVERGENCE
        )
        event_ids = tuple(
            sorted(
                str(item)
                for item in benchmark_event_meta.get("benchmark_event_ids", ())
            )
        )
        context = DecisionTraceContext(
            key=key,
            benchmark_fingerprint=identity.benchmark_fingerprint,
            code_revision=identity.code_revision,
            simulator_seed=identity.simulator_seed,
            generation_seed_master=self.harness_config.transport.generation_seed_master,
            generation_seed_scope=scope,
            decision_snapshot_id=(
                identity.primary_snapshot_id if decision_index == 0 else None
            ),
            available_action_ids=available_action_ids,
            event_phase="pre_step",
            applied_event_ids=event_ids,
        )
        return DecisionTraceRecord(
            context=context,
            harness_config=self.harness_config,
            prompt_artifact=prompt_artifact,
            generation=generation,
            resolution=resolution,
            shield_stack=shield_stack,
            disposition=disposition,
            decision_latency_ms=decision_latency_ms,
        )

    def complete_attempt(
        self,
        references: tuple[TraceReference, ...],
        *,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        if self.completion_publisher is not None:
            if result is None:
                raise ValueError("Completion publisher requires the episode result.")
            expected_references = [reference.to_dict() for reference in references]
            if result.get("scientific_trace_references") != expected_references:
                raise ValueError(
                    "Episode result trace references do not match ordered evidence."
                )
            self.completion_publisher(result, references)
            return
        self.attempt_ledger.append_terminal(
            self.identity.episode_attempt_id,
            status=AttemptStatus.COMPLETED,
            decision_count=len(references),
            trace_references=references,
        )

    def current_trace_references(self) -> tuple[TraceReference, ...]:
        return self.trace_writer.references_for_attempt(
            self.identity.campaign_id,
            self.identity.episode_attempt_id,
        )

    def cached_trace_references(self) -> tuple[TraceReference, ...]:
        return self.trace_writer.cached_references_for_attempt(
            self.identity.campaign_id,
            self.identity.episode_attempt_id,
        )

    def abort_attempt(
        self,
        error: Exception,
        references: tuple[TraceReference, ...],
    ) -> None:
        if not isinstance(error, Exception):
            raise ValueError("error must be an Exception.")
        evidence = list(references)
        status = AttemptStatus.FAILED
        failure_class = type(error).__name__
        absence_reason = None
        if isinstance(error, ScientificTraceCommitAmbiguousError):
            status = AttemptStatus.WRITE_AMBIGUOUS
            failure_class = TRACE_COMMIT_AMBIGUOUS
            absence_reason = TRACE_COMMIT_AMBIGUOUS
        elif isinstance(error, ScientificTraceWriteError):
            failure_class = error.failure_class.value
        elif isinstance(error, RuntimeProtocolError):
            status = AttemptStatus.BLOCKED
            failure_class = error.invariant_code.value
        elif isinstance(error, ScientificGenerationAbort):
            status = AttemptStatus.BLOCKED
            failure_class = error.result.error_class.value
            _append_reference_once(evidence, error.trace_reference)
        elif isinstance(error, ScientificSimulatorAbort):
            failure_class = error.failure_class.value
            _append_reference_once(evidence, error.trace_reference)
        if not evidence and absence_reason is None:
            absence_reason = "aborted_before_first_decision"
        self.attempt_ledger.append_terminal(
            self.identity.episode_attempt_id,
            status=status,
            decision_count=len(evidence),
            trace_references=tuple(evidence),
            failure_class=failure_class,
            failure_message=str(error) or type(error).__name__,
            trace_absence_reason=absence_reason,
        )

    def _request_id(self, decision_index: int) -> str:
        identity = self.identity
        payload = "|".join(
            (
                identity.campaign_id,
                identity.episode_attempt_id,
                identity.case_id,
                str(decision_index),
            )
        ).encode("utf-8")
        return "req-" + hashlib.sha256(payload).hexdigest()

    def _generation_seed(self, decision_index: int) -> int:
        identity = self.identity
        master = self.harness_config.transport.generation_seed_master
        if decision_index == 0:
            return primary_snapshot_generation_seed(
                master,
                self.model_digest,
                identity.pair_id,
                identity.primary_snapshot_id,
                identity.replicate_id,
            )
        return post_divergence_generation_seed(
            master,
            self.model_digest,
            identity.case_id,
            decision_index,
            identity.replicate_id,
        )

    def _validate_generation(
        self,
        decision_index: int,
        generation: GenerationResult,
    ) -> None:
        if not isinstance(generation, GenerationResult):
            raise ValueError("generation must be GenerationResult.")
        request = generation.request
        if (
            request.request_id != self._request_id(decision_index)
            or request.model_digest != self.model_digest
            or request.options.seed != self._generation_seed(decision_index)
        ):
            raise ValueError("Generation evidence drifted from episode ownership.")


def build_scientific_episode_runtime(
    *,
    harness_config: HarnessConfig,
    identity: ScientificEpisodeIdentity,
    runtime_lock: VerifiedRuntimeLockBinding,
    transport_client: OllamaScientificClient,
    trace_writer: ScientificTraceWriter,
    attempt_ledger: ScientificAttemptLedger,
    completion_publisher: (
        Callable[[Mapping[str, Any], tuple[TraceReference, ...]], None] | None
    ) = None,
) -> ScientificEpisodeRuntime:
    """Build a claim-bearing episode only from externally bound resources."""
    runtime = ScientificEpisodeRuntime(
        harness_config=harness_config,
        identity=identity,
        runtime_lock=runtime_lock,
        transport_client=transport_client,
        trace_writer=trace_writer,
        attempt_ledger=attempt_ledger,
        completion_publisher=completion_publisher,
    )
    attempt_ledger.validate_trace_evidence(trace_writer)
    runtime.validate_binding()
    return runtime


def _append_reference_once(
    references: list[TraceReference],
    candidate: TraceReference | None,
) -> None:
    if candidate is not None and candidate not in references:
        references.append(candidate)


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


__all__ = [
    "RuntimeLockBinding",
    "ScientificEpisodeIdentity",
    "ScientificEpisodeRuntime",
    "VerifiedRuntimeLockBinding",
    "build_scientific_episode_runtime",
]
