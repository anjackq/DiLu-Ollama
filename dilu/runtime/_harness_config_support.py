from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, TypeVar


class PolicyContent(str, Enum):
    HISTORICAL_DILU_2024 = "historical_dilu_2024"
    MODULAR_HARNESS = "modular_harness"


class OutputEnforcement(str, Enum):
    PROMPT_ONLY = "prompt_only"
    BACKEND_SCHEMA = "backend_schema"
    BACKEND_SCHEMA_GROUNDED = "backend_schema_grounded"


class ExecutionMode(str, Enum):
    UNSHIELDED_OPERATIONAL = "unshielded_operational"
    SHIELDED = "shielded"


class ParserMode(str, Enum):
    STRICT_ONLY = "strict_only"
    DETERMINISTIC_RECOVERY = "deterministic_recovery"
    RESOLVER_ASSISTED = "resolver_assisted"


class ResolverMode(str, Enum):
    DISABLED = "disabled"
    ASSISTED = "assisted"


class FallbackPolicy(str, Enum):
    INVALID_TERMINATE = "invalid_terminate"
    FIXED_IDLE = "fixed_idle"
    FIXED_SLOWER = "fixed_slower"
    STATE_AWARE = "state_aware"


class TransportProfile(str, Enum):
    OLLAMA_NATIVE_CHAT = "ollama_native_chat"
    OLLAMA_OPENAI_COMPATIBLE = "ollama_openai_compatible"


class ThinkMode(str, Enum):
    THINK = "think"
    NO_THINK = "no_think"
    AUTO = "auto"


class TraceLevel(str, Enum):
    MANDATORY_SCIENTIFIC = "mandatory_scientific"
    STANDARD = "standard"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ConditionSpec:
    policy_content: PolicyContent
    output_enforcement: OutputEnforcement
    execution_mode: ExecutionMode

    def validate(self) -> None:
        require_enum_instance(self.policy_content, PolicyContent, "policy_content")
        require_enum_instance(
            self.output_enforcement, OutputEnforcement, "output_enforcement"
        )
        require_enum_instance(self.execution_mode, ExecutionMode, "execution_mode")

    def to_canonical_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "policy_content": self.policy_content.value,
            "output_enforcement": self.output_enforcement.value,
            "execution_mode": self.execution_mode.value,
        }

    def condition_id(self) -> str:
        self.validate()
        bits = (
            _POLICY_DIGITS[self.policy_content],
            _OUTPUT_DIGITS[self.output_enforcement],
            _EXECUTION_DIGITS[self.execution_mode],
        )
        return f"c{''.join(bits)}"

    @classmethod
    def from_condition_id(cls, condition_id: str) -> "ConditionSpec":
        """Invert :meth:`condition_id`: parse ``"c" + 3 digits`` back to a spec.

        This is a general-purpose inverse, not special-cased per campaign:
        any digit combination registered in the digit maps round-trips,
        which is what lets V8's ``c120``/``c121`` (grounded, digit ``"2"``)
        resolve exactly like the V5/V7 binary grid (digits ``"0"``/``"1"``)
        without separate parsing logic.
        """
        if (
            not isinstance(condition_id, str)
            or len(condition_id) != 4
            or condition_id[0] != "c"
        ):
            raise ValueError(f"Invalid condition_id: {condition_id!r}.")
        policy_digit, output_digit, execution_digit = condition_id[1], condition_id[2], condition_id[3]
        if (
            policy_digit not in _POLICY_BY_DIGIT
            or output_digit not in _OUTPUT_BY_DIGIT
            or execution_digit not in _EXECUTION_BY_DIGIT
        ):
            raise ValueError(f"Invalid condition_id: {condition_id!r}.")
        spec = cls(
            _POLICY_BY_DIGIT[policy_digit],
            _OUTPUT_BY_DIGIT[output_digit],
            _EXECUTION_BY_DIGIT[execution_digit],
        )
        if spec.condition_id() != condition_id:
            raise ValueError(f"condition_id round-trip drifted for {condition_id!r}.")
        return spec


_POLICY_DIGITS = {
    PolicyContent.HISTORICAL_DILU_2024: "0",
    PolicyContent.MODULAR_HARNESS: "1",
}
_OUTPUT_DIGITS = {
    OutputEnforcement.PROMPT_ONLY: "0",
    OutputEnforcement.BACKEND_SCHEMA: "1",
    OutputEnforcement.BACKEND_SCHEMA_GROUNDED: "2",
}
_EXECUTION_DIGITS = {
    ExecutionMode.UNSHIELDED_OPERATIONAL: "0",
    ExecutionMode.SHIELDED: "1",
}
_POLICY_BY_DIGIT = {digit: value for value, digit in _POLICY_DIGITS.items()}
_OUTPUT_BY_DIGIT = {digit: value for value, digit in _OUTPUT_DIGITS.items()}
_EXECUTION_BY_DIGIT = {digit: value for value, digit in _EXECUTION_DIGITS.items()}


EnumType = TypeVar("EnumType", bound=Enum)


def parse_enum(enum_type: type[EnumType], value: Any, field_name: str) -> EnumType:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc


def strict_mapping(
    value: Any, expected_keys: set[str], field_name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    actual_keys = {str(key) for key in value}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Invalid {field_name} keys; missing={missing}, unknown={unknown}."
        )
    return value


def parse_bool(value: Any, field_name: str) -> bool:
    validate_bool(value, field_name)
    return value


def parse_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def parse_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def parse_str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{field_name} must be a string sequence.")
    return tuple(value)


def require_enum_instance(value: Any, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be a resolved {enum_type.__name__}.")


def validate_bool(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")


def validate_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def validate_nonnegative_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def validate_uint32(value: Any, field_name: str) -> None:
    validate_nonnegative_int(value, field_name)
    if value > (1 << 32) - 1:
        raise ValueError(f"{field_name} must fit in uint32.")


def validate_finite_float(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite float.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive.")


def canonicalize_dataclass(value: Any) -> dict[str, Any]:
    return {
        field.name: _canonicalize(getattr(value, field.name)) for field in fields(value)
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(flatten_mapping(item, path))
        else:
            flattened[path] = item
    return flattened


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize_dataclass(value)
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    return value
