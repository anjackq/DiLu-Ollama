from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from dilu.driver_agent.prompt_modules import PromptArtifact, build_prompt_artifact

from .action_resolution import (
    ActionAvailability,
    ActionResolutionResult,
    ActionSyntaxStatus,
    resolve_action,
)
from .generation_seed import (
    post_divergence_generation_seed,
    primary_snapshot_generation_seed,
)
from .harness_config import HarnessConfig
from .runtime_failures import RuntimeFailureClass
from .scientific_transport_records import GenerationResult
from .shield_stack import ShieldStackResult


_SHA256_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z")


class GenerationSeedScope(str, Enum):
    PRIMARY_SNAPSHOT = "primary_snapshot"
    POST_DIVERGENCE = "post_divergence"


class TraceDisposition(str, Enum):
    READY_FOR_ENV_STEP = "ready_for_env_step"
    BLOCKED_BEFORE_EXECUTION = "blocked_before_execution"


@dataclass(frozen=True)
class DecisionTraceKey:
    campaign_id: str
    episode_attempt_id: str
    condition_id: str
    case_id: str
    pair_id: str
    template_id: str
    replicate_id: int
    decision_index: int
    env_step_index: int

    def __post_init__(self) -> None:
        for field_name in (
            "campaign_id",
            "episode_attempt_id",
            "condition_id",
            "case_id",
            "pair_id",
            "template_id",
        ):
            _require_canonical_text(field_name, getattr(self, field_name))
        for field_name in ("replicate_id", "decision_index", "env_step_index"):
            _require_nonnegative_int(field_name, getattr(self, field_name))

    def identity(self) -> tuple[object, ...]:
        return (
            self.campaign_id,
            self.episode_attempt_id,
            self.condition_id,
            self.case_id,
            self.pair_id,
            self.template_id,
            self.replicate_id,
            self.decision_index,
            self.env_step_index,
        )

    def episode_identity(self) -> tuple[str, str]:
        return self.campaign_id, self.episode_attempt_id


@dataclass(frozen=True)
class DecisionTraceContext:
    key: DecisionTraceKey
    benchmark_fingerprint: str
    code_revision: str
    simulator_seed: int
    generation_seed_master: int
    generation_seed_scope: GenerationSeedScope
    decision_snapshot_id: str | None
    available_action_ids: tuple[int, ...]
    event_phase: str
    applied_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, DecisionTraceKey):
            raise ValueError("key must be a DecisionTraceKey.")
        if not _SHA256_PATTERN.fullmatch(self.benchmark_fingerprint):
            raise ValueError("benchmark_fingerprint must be a full sha256 digest.")
        _require_canonical_text("code_revision", self.code_revision)
        _require_uint32("simulator_seed", self.simulator_seed)
        _require_uint32("generation_seed_master", self.generation_seed_master)
        if not isinstance(self.generation_seed_scope, GenerationSeedScope):
            raise ValueError("generation_seed_scope must be resolved.")
        if self.generation_seed_scope is GenerationSeedScope.PRIMARY_SNAPSHOT:
            _require_canonical_text("decision_snapshot_id", self.decision_snapshot_id)
        elif self.decision_snapshot_id is not None:
            raise ValueError("Post-divergence traces cannot carry a snapshot ID.")
        if (self.key.decision_index == 0) != (
            self.generation_seed_scope is GenerationSeedScope.PRIMARY_SNAPSHOT
        ):
            raise ValueError(
                "Decision zero must use the primary snapshot seed scope; later "
                "decisions must use post-divergence scope."
            )
        if (
            not isinstance(self.available_action_ids, tuple)
            or not self.available_action_ids
            or self.available_action_ids
            != tuple(sorted(set(self.available_action_ids)))
            or any(
                isinstance(action, bool)
                or not isinstance(action, int)
                or action not in range(5)
                for action in self.available_action_ids
            )
        ):
            raise ValueError("available_action_ids must be sorted unique IDs in 0..4.")
        if 1 not in self.available_action_ids:
            raise ValueError(
                "Scientific action availability must include fixed IDLE=1."
            )
        _require_canonical_text("event_phase", self.event_phase)
        if not isinstance(self.applied_event_ids, tuple):
            raise ValueError("applied_event_ids must be an immutable tuple.")
        for event_id in self.applied_event_ids:
            _require_canonical_text("applied_event_id", event_id)


@dataclass(frozen=True)
class DecisionTraceRecord:
    context: DecisionTraceContext
    harness_config: HarnessConfig
    prompt_artifact: PromptArtifact
    generation: GenerationResult
    resolution: ActionResolutionResult | None
    shield_stack: ShieldStackResult | None
    disposition: TraceDisposition
    decision_latency_ms: float

    def __post_init__(self) -> None:
        self._validate_types()
        self.harness_config.validate_scientific()
        self._validate_context_and_prompt()
        self._validate_generation()
        self._validate_disposition()
        if self.resolution is not None:
            self._validate_resolution()
        if self.shield_stack is not None:
            self._validate_shield_stack()

    def to_dict(self) -> dict[str, object]:
        from ._scientific_trace_serialization import serialize_trace_record

        return serialize_trace_record(self)

    def _validate_types(self) -> None:
        expected = (
            (self.context, DecisionTraceContext, "context"),
            (self.harness_config, HarnessConfig, "harness_config"),
            (self.prompt_artifact, PromptArtifact, "prompt_artifact"),
            (self.generation, GenerationResult, "generation"),
        )
        for value, expected_type, name in expected:
            if not isinstance(value, expected_type):
                raise ValueError(f"{name} must be {expected_type.__name__}.")
        if not isinstance(self.disposition, TraceDisposition):
            raise ValueError("disposition must be resolved.")
        if (
            isinstance(self.decision_latency_ms, bool)
            or not isinstance(self.decision_latency_ms, (int, float))
            or not math.isfinite(float(self.decision_latency_ms))
            or float(self.decision_latency_ms) < self.generation.latency_ms
        ):
            raise ValueError("decision_latency_ms must include generation latency.")

    def _validate_context_and_prompt(self) -> None:
        config = self.harness_config
        context = self.context
        if context.key.condition_id != config.condition_id():
            raise ValueError("Trace condition ID does not match the harness config.")
        if context.generation_seed_master != config.transport.generation_seed_master:
            raise ValueError("Trace master seed does not match the harness config.")
        prompt = self.prompt_artifact
        expected_prompt = build_prompt_artifact(
            config.condition.policy_content,
            output_enforcement=config.condition.output_enforcement,
        )
        if (
            prompt.policy_content is not config.condition.policy_content
            or prompt.output_enforcement is not config.condition.output_enforcement
            or prompt.few_shot_num != 0
            or prompt != expected_prompt
        ):
            raise ValueError("Prompt artifact does not match the harness condition.")

    def _validate_generation(self) -> None:
        request = self.generation.request
        config = self.harness_config
        transport = config.transport
        if (
            request.output_enforcement is not config.condition.output_enforcement
            or request.think_mode is not transport.think_mode
            or float(request.options.temperature) != transport.temperature
            or request.options.num_ctx != transport.context_tokens
            or request.options.num_predict != transport.max_output_tokens
            or float(request.timeout_sec) != transport.timeout_sec
        ):
            raise ValueError("Generation request drifted from the harness config.")
        if (
            len(request.messages) != 2
            or request.messages[0] != ("system", self.prompt_artifact.system_prompt())
            or request.messages[1][0] != "user"
        ):
            raise ValueError("Generation request does not bind the prompt artifact.")
        expected_seed = self._expected_generation_seed()
        if request.options.seed != expected_seed:
            raise ValueError("Generation seed does not match the trace context.")

    def _expected_generation_seed(self) -> int:
        key = self.context.key
        master = self.context.generation_seed_master
        digest = self.generation.model_digest
        if self.context.generation_seed_scope is GenerationSeedScope.PRIMARY_SNAPSHOT:
            assert self.context.decision_snapshot_id is not None
            return primary_snapshot_generation_seed(
                master,
                digest,
                key.pair_id,
                self.context.decision_snapshot_id,
                key.replicate_id,
            )
        return post_divergence_generation_seed(
            master,
            digest,
            key.case_id,
            key.decision_index,
            key.replicate_id,
        )

    def _validate_disposition(self) -> None:
        if self.disposition is TraceDisposition.BLOCKED_BEFORE_EXECUTION:
            if not self.generation.requires_cell_abort:
                raise ValueError("Blocked traces require a cell-aborting generation.")
            if self.resolution is not None or self.shield_stack is not None:
                raise ValueError("Blocked traces cannot contain execution evidence.")
            return
        if self.generation.requires_cell_abort:
            raise ValueError("Cell-aborting generations cannot be ready for env.step.")
        if not isinstance(self.resolution, ActionResolutionResult) or not isinstance(
            self.shield_stack, ShieldStackResult
        ):
            raise ValueError("Ready traces require resolution and shield evidence.")

    def _validate_resolution(self) -> None:
        assert self.resolution is not None
        resolution = self.resolution
        parser_input = (
            self.generation.contract_text or self.generation.raw_response or ""
        )
        if resolution.raw_response != parser_input:
            raise ValueError("Action parser input does not match generation evidence.")
        timed_out = (
            self.generation.error_class is RuntimeFailureClass.GENERATION_TIMEOUT
        )
        expected_resolution = resolve_action(
            parser_input,
            available_action_ids=self.context.available_action_ids,
            timed_out=timed_out,
            parser_mode=self.harness_config.parser_mode,
            resolver_mode=self.harness_config.resolver_mode,
            fallback_policy=self.harness_config.fallback_policy,
        )
        if resolution != expected_resolution:
            raise ValueError("Action resolution was not deterministically reproduced.")
        if timed_out != (resolution.syntax_status is ActionSyntaxStatus.TIMEOUT):
            raise ValueError("Timeout generation and action syntax status disagree.")
        available = set(self.context.available_action_ids)
        if resolution.strict_action is None:
            expected_availability = ActionAvailability.NOT_APPLICABLE
        elif resolution.strict_action in available:
            expected_availability = ActionAvailability.AVAILABLE
        else:
            expected_availability = ActionAvailability.UNAVAILABLE
        if resolution.action_available is not expected_availability:
            raise ValueError("Action availability is inconsistent with the context.")
        if (
            resolution.fallback_action is not None
            and resolution.fallback_action not in available
        ):
            raise ValueError("Fallback action is unavailable in the traced state.")

    def _validate_shield_stack(self) -> None:
        assert self.resolution is not None and self.shield_stack is not None
        resolution = self.resolution
        stack = self.shield_stack
        proposal = (
            resolution.recovered_action
            if resolution.recovered_action is not None
            else resolution.strict_action
        )
        if stack.proposed_action_id != proposal:
            raise ValueError("Shield proposal does not preserve model action evidence.")
        if stack.fallback_modified_action_id != resolution.final_resolved_action:
            raise ValueError("Shield input does not preserve action resolution.")
        if stack.execution_mode is not self.harness_config.condition.execution_mode:
            raise ValueError("Shield execution mode does not match the condition.")


def _require_canonical_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text.")


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _require_uint32(name: str, value: object) -> None:
    _require_nonnegative_int(name, value)
    if int(value) > (1 << 32) - 1:
        raise ValueError(f"{name} must fit in uint32.")


__all__ = [
    "DecisionTraceContext",
    "DecisionTraceKey",
    "DecisionTraceRecord",
    "GenerationSeedScope",
    "TraceDisposition",
]
