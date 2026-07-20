from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import product
from typing import Any, Mapping

from ._harness_config_support import (
    ConditionSpec,
    ExecutionMode,
    FallbackPolicy,
    OutputEnforcement,
    ParserMode,
    PolicyContent,
    ResolverMode,
    ThinkMode,
    TraceLevel,
    TransportProfile,
    canonical_sha256,
    canonicalize_dataclass,
    flatten_mapping,
    parse_bool,
    parse_enum,
    parse_float,
    parse_int,
    parse_str_tuple,
    require_enum_instance,
    strict_mapping,
    validate_bool,
    validate_finite_float,
    validate_nonnegative_int,
    validate_positive_int,
    validate_uint32,
)


SHIELD_STAGE_ORDER = ("lane_change", "longitudinal_safety", "low_speed_recovery")
SHIELD_NUMERIC_FIELDS = (
    "target_front_gap_required_m",
    "target_rear_gap_required_m",
    "target_rear_ttc_required_sec",
    "front_critical_gap_m",
    "front_caution_gap_m",
    "acceleration_caution_gap_m",
    "front_critical_ttc_sec",
    "front_caution_ttc_sec",
    "acceleration_caution_ttc_sec",
    "acceleration_projected_speed_gain_mps",
    "projected_time_horizon_sec",
    "low_speed_recovery_floor_mps",
    "low_speed_recovery_front_gap_m",
    "low_speed_recovery_ttc_sec",
)


@dataclass(frozen=True)
class ShieldConfig:
    stage_order: tuple[str, ...]
    target_front_gap_required_m: float
    target_rear_gap_required_m: float
    target_rear_ttc_required_sec: float
    front_critical_gap_m: float
    front_caution_gap_m: float
    acceleration_caution_gap_m: float
    front_critical_ttc_sec: float
    front_caution_ttc_sec: float
    acceleration_caution_ttc_sec: float
    acceleration_projected_speed_gain_mps: float
    projected_time_horizon_sec: float
    low_speed_recovery_floor_mps: float
    low_speed_recovery_front_gap_m: float
    low_speed_recovery_ttc_sec: float

    @classmethod
    def implementation_defaults(cls) -> "ShieldConfig":
        return cls(
            stage_order=SHIELD_STAGE_ORDER,
            target_front_gap_required_m=14.0,
            target_rear_gap_required_m=10.0,
            target_rear_ttc_required_sec=2.5,
            front_critical_gap_m=8.0,
            front_caution_gap_m=12.0,
            acceleration_caution_gap_m=18.0,
            front_critical_ttc_sec=2.0,
            front_caution_ttc_sec=3.0,
            acceleration_caution_ttc_sec=5.0,
            acceleration_projected_speed_gain_mps=2.0,
            projected_time_horizon_sec=1.0,
            low_speed_recovery_floor_mps=16.7,
            low_speed_recovery_front_gap_m=25.0,
            low_speed_recovery_ttc_sec=4.0,
        )

    def validate_scientific(self) -> None:
        if self.stage_order != SHIELD_STAGE_ORDER:
            raise ValueError(
                "Scientific shield stage order does not match the protocol."
            )
        for field_name in SHIELD_NUMERIC_FIELDS:
            validate_finite_float(
                getattr(self, field_name), f"shield.{field_name}", positive=True
            )
        gap_thresholds = (
            self.front_critical_gap_m,
            self.front_caution_gap_m,
            self.acceleration_caution_gap_m,
        )
        ttc_thresholds = (
            self.front_critical_ttc_sec,
            self.front_caution_ttc_sec,
            self.acceleration_caution_ttc_sec,
        )
        if not _strictly_increasing(gap_thresholds):
            raise ValueError(
                "Scientific front-gap thresholds must be strictly increasing."
            )
        if not _strictly_increasing(ttc_thresholds):
            raise ValueError(
                "Scientific front-TTC thresholds must be strictly increasing."
            )
        if self != self.implementation_defaults():
            raise ValueError(
                "Scientific shield thresholds must match the executed primitives."
            )


@dataclass(frozen=True)
class TransportConfig:
    profile: TransportProfile
    think_mode: ThinkMode
    temperature: float
    context_tokens: int
    max_output_tokens: int
    timeout_sec: float
    generation_seed_master: int
    allow_transport_fallback: bool
    adaptive_timeout: bool

    def validate_scientific(self) -> None:
        require_enum_instance(self.profile, TransportProfile, "transport.profile")
        require_enum_instance(self.think_mode, ThinkMode, "transport.think_mode")
        if self.profile is not TransportProfile.OLLAMA_NATIVE_CHAT:
            raise ValueError("Scientific transport must use Ollama native chat.")
        if self.think_mode is ThinkMode.AUTO:
            raise ValueError("Scientific think mode must be explicit, not auto.")
        validate_finite_float(self.temperature, "transport.temperature", minimum=0.0)
        validate_positive_int(self.context_tokens, "transport.context_tokens")
        validate_positive_int(self.max_output_tokens, "transport.max_output_tokens")
        validate_finite_float(self.timeout_sec, "transport.timeout_sec", positive=True)
        validate_uint32(self.generation_seed_master, "transport.generation_seed_master")
        validate_bool(
            self.allow_transport_fallback, "transport.allow_transport_fallback"
        )
        validate_bool(self.adaptive_timeout, "transport.adaptive_timeout")
        if self.allow_transport_fallback or self.adaptive_timeout:
            raise ValueError(
                "Scientific transport fallback and adaptation are disabled."
            )


@dataclass(frozen=True)
class RetryPolicy:
    max_transport_unavailable_retries: int
    retry_cooldown_sec: float
    retry_on_timeout: bool
    retry_on_empty_output: bool
    retry_on_schema_rejection: bool

    def validate_scientific(self) -> None:
        validate_nonnegative_int(
            self.max_transport_unavailable_retries,
            "retry_policy.max_transport_unavailable_retries",
        )
        validate_finite_float(
            self.retry_cooldown_sec, "retry_policy.retry_cooldown_sec", positive=True
        )
        retry_flags = (
            self.retry_on_timeout,
            self.retry_on_empty_output,
            self.retry_on_schema_rejection,
        )
        for field_name, value in zip(
            ("retry_on_timeout", "retry_on_empty_output", "retry_on_schema_rejection"),
            retry_flags,
        ):
            validate_bool(value, f"retry_policy.{field_name}")
        if self.max_transport_unavailable_retries != 1:
            raise ValueError("Scientific transport policy permits exactly one retry.")
        if self.retry_cooldown_sec != 10.0:
            raise ValueError("Scientific retry cooldown must be exactly 10 seconds.")
        if any(retry_flags):
            raise ValueError("Only pre-accept transport unavailability is retryable.")


@dataclass(frozen=True)
class HarnessConfig:
    condition: ConditionSpec
    parser_mode: ParserMode
    resolver_mode: ResolverMode
    fallback_policy: FallbackPolicy
    shield: ShieldConfig
    transport: TransportConfig
    retry_policy: RetryPolicy
    trace_level: TraceLevel

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HarnessConfig":
        mapping = strict_mapping(value, _field_names(cls), "harness")
        condition = strict_mapping(
            mapping["condition"], _field_names(ConditionSpec), "condition"
        )
        shield = strict_mapping(mapping["shield"], _field_names(ShieldConfig), "shield")
        transport = strict_mapping(
            mapping["transport"], _field_names(TransportConfig), "transport"
        )
        retry = strict_mapping(
            mapping["retry_policy"], _field_names(RetryPolicy), "retry_policy"
        )
        return cls(
            condition=_parse_condition(condition),
            parser_mode=parse_enum(ParserMode, mapping["parser_mode"], "parser_mode"),
            resolver_mode=parse_enum(
                ResolverMode, mapping["resolver_mode"], "resolver_mode"
            ),
            fallback_policy=parse_enum(
                FallbackPolicy, mapping["fallback_policy"], "fallback_policy"
            ),
            shield=_parse_shield(shield),
            transport=_parse_transport(transport),
            retry_policy=_parse_retry_policy(retry),
            trace_level=parse_enum(TraceLevel, mapping["trace_level"], "trace_level"),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return canonicalize_dataclass(self)

    def config_hash(self) -> str:
        self.validate_scientific()
        return canonical_sha256(self)

    def condition_id(self) -> str:
        return self.condition.condition_id()

    def validate_scientific(self) -> None:
        if not isinstance(self.condition, ConditionSpec):
            raise ValueError("Scientific condition is unresolved.")
        self.condition.validate()
        require_enum_instance(self.parser_mode, ParserMode, "parser_mode")
        require_enum_instance(self.resolver_mode, ResolverMode, "resolver_mode")
        require_enum_instance(self.fallback_policy, FallbackPolicy, "fallback_policy")
        require_enum_instance(self.trace_level, TraceLevel, "trace_level")
        if self.parser_mode is not ParserMode.STRICT_ONLY:
            raise ValueError("Scientific parser mode must be strict_only.")
        if self.resolver_mode is not ResolverMode.DISABLED:
            raise ValueError("Scientific resolver assistance must be disabled.")
        if self.fallback_policy is not FallbackPolicy.FIXED_IDLE:
            raise ValueError("Scientific fallback policy must be fixed_idle.")
        if self.trace_level is not TraceLevel.MANDATORY_SCIENTIFIC:
            raise ValueError("Scientific trace level must be mandatory_scientific.")
        for value, expected_type, field_name in (
            (self.shield, ShieldConfig, "shield"),
            (self.transport, TransportConfig, "transport"),
            (self.retry_policy, RetryPolicy, "retry_policy"),
        ):
            if not isinstance(value, expected_type):
                raise ValueError(f"Scientific {field_name} config is unresolved.")
        self.shield.validate_scientific()
        self.transport.validate_scientific()
        self.retry_policy.validate_scientific()


def resolve_main_conditions(base: HarnessConfig) -> tuple[HarnessConfig, ...]:
    conditions = []
    for policy, output, execution in product(
        (PolicyContent.HISTORICAL_DILU_2024, PolicyContent.MODULAR_HARNESS),
        (OutputEnforcement.PROMPT_ONLY, OutputEnforcement.BACKEND_SCHEMA),
        (ExecutionMode.UNSHIELDED_OPERATIONAL, ExecutionMode.SHIELDED),
    ):
        condition = replace(base, condition=ConditionSpec(policy, output, execution))
        condition.validate_scientific()
        conditions.append(condition)
    return tuple(conditions)


def diff_conditions(
    left: HarnessConfig, right: HarnessConfig
) -> dict[str, tuple[Any, Any]]:
    left_flat = flatten_mapping(left.to_canonical_dict())
    right_flat = flatten_mapping(right.to_canonical_dict())
    keys = sorted(left_flat.keys() | right_flat.keys())
    return {
        key: (left_flat.get(key), right_flat.get(key))
        for key in keys
        if left_flat.get(key) != right_flat.get(key)
    }


def _field_names(dataclass_type: type[Any]) -> set[str]:
    return {field.name for field in fields(dataclass_type)}


def _strictly_increasing(values: tuple[float, ...]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def _parse_condition(value: Mapping[str, Any]) -> ConditionSpec:
    return ConditionSpec(
        policy_content=parse_enum(
            PolicyContent, value["policy_content"], "condition.policy_content"
        ),
        output_enforcement=parse_enum(
            OutputEnforcement,
            value["output_enforcement"],
            "condition.output_enforcement",
        ),
        execution_mode=parse_enum(
            ExecutionMode, value["execution_mode"], "condition.execution_mode"
        ),
    )


def _parse_shield(value: Mapping[str, Any]) -> ShieldConfig:
    numbers = {
        name: parse_float(value[name], f"shield.{name}")
        for name in SHIELD_NUMERIC_FIELDS
    }
    return ShieldConfig(
        stage_order=parse_str_tuple(value["stage_order"], "shield.stage_order"),
        **numbers,
    )


def _parse_transport(value: Mapping[str, Any]) -> TransportConfig:
    return TransportConfig(
        profile=parse_enum(TransportProfile, value["profile"], "transport.profile"),
        think_mode=parse_enum(ThinkMode, value["think_mode"], "transport.think_mode"),
        temperature=parse_float(value["temperature"], "transport.temperature"),
        context_tokens=parse_int(value["context_tokens"], "transport.context_tokens"),
        max_output_tokens=parse_int(
            value["max_output_tokens"], "transport.max_output_tokens"
        ),
        timeout_sec=parse_float(value["timeout_sec"], "transport.timeout_sec"),
        generation_seed_master=parse_int(
            value["generation_seed_master"], "transport.generation_seed_master"
        ),
        allow_transport_fallback=parse_bool(
            value["allow_transport_fallback"], "transport.allow_transport_fallback"
        ),
        adaptive_timeout=parse_bool(
            value["adaptive_timeout"], "transport.adaptive_timeout"
        ),
    )


def _parse_retry_policy(value: Mapping[str, Any]) -> RetryPolicy:
    return RetryPolicy(
        max_transport_unavailable_retries=parse_int(
            value["max_transport_unavailable_retries"],
            "retry_policy.max_transport_unavailable_retries",
        ),
        retry_cooldown_sec=parse_float(
            value["retry_cooldown_sec"], "retry_policy.retry_cooldown_sec"
        ),
        retry_on_timeout=parse_bool(
            value["retry_on_timeout"], "retry_policy.retry_on_timeout"
        ),
        retry_on_empty_output=parse_bool(
            value["retry_on_empty_output"], "retry_policy.retry_on_empty_output"
        ),
        retry_on_schema_rejection=parse_bool(
            value["retry_on_schema_rejection"],
            "retry_policy.retry_on_schema_rejection",
        ),
    )


__all__ = [
    "ConditionSpec",
    "ExecutionMode",
    "FallbackPolicy",
    "HarnessConfig",
    "OutputEnforcement",
    "ParserMode",
    "PolicyContent",
    "ResolverMode",
    "RetryPolicy",
    "ShieldConfig",
    "ThinkMode",
    "TraceLevel",
    "TransportConfig",
    "TransportProfile",
    "diff_conditions",
    "resolve_main_conditions",
]
