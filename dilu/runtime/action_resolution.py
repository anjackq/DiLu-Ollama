from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from .harness_config import FallbackPolicy, ParserMode, ResolverMode
from .runtime_failures import (
    ProtocolInvariantCode,
    ProtocolInvariantViolation,
    RuntimeFailureClass,
    RuntimeProtocolError,
)


CANONICAL_ACTION_IDS = (0, 1, 2, 3, 4)
FIXED_IDLE_ACTION_ID = 1
CANONICAL_ACTION_PATTERN = re.compile(r"\AResponse to user:\#{4} ([0-4])\Z")


class ActionSyntaxStatus(str, Enum):
    STRICT_VALID = "strict_valid"
    INVALID = "invalid"
    EMPTY = "empty"
    TIMEOUT = "timeout"


class RecoveryStage(str, Enum):
    NONE = "none"
    DETERMINISTIC = "deterministic"
    RESOLVER = "resolver"


class ActionAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class ActionResolutionResult:
    raw_response: str
    syntax_status: ActionSyntaxStatus
    strict_action: int | None
    recovered_action: int | None
    recovery_stage: RecoveryStage
    violation: RuntimeFailureClass | None
    action_available: ActionAvailability
    fallback_action: int | None
    final_resolved_action: int

    def __post_init__(self) -> None:
        if self.final_resolved_action not in CANONICAL_ACTION_IDS:
            raise ValueError("Final resolved action must be canonical.")
        if self.syntax_status is ActionSyntaxStatus.STRICT_VALID:
            if self.strict_action not in CANONICAL_ACTION_IDS:
                raise ValueError(
                    "Strict-valid results require a canonical strict action."
                )
            if self.action_available is ActionAvailability.NOT_APPLICABLE:
                raise ValueError(
                    "Strict-valid results require an availability outcome."
                )
        elif self.strict_action is not None:
            raise ValueError("Non-strict results cannot carry a strict action.")
        elif self.action_available is not ActionAvailability.NOT_APPLICABLE:
            raise ValueError("Non-strict results require not_applicable availability.")
        if (self.recovered_action is None) != (
            self.recovery_stage is RecoveryStage.NONE
        ):
            raise ValueError(
                "Recovered action and recovery stage must be set together."
            )
        if (
            self.recovered_action is not None
            and self.recovered_action not in CANONICAL_ACTION_IDS
        ):
            raise ValueError("Recovered action must be canonical.")
        if self.fallback_action is not None:
            if (
                self.final_resolved_action != self.fallback_action
                or self.violation is None
            ):
                raise ValueError(
                    "Fallback results must preserve the violation and final action."
                )
        elif self.final_resolved_action not in {
            self.strict_action,
            self.recovered_action,
        }:
            raise ValueError(
                "Non-fallback final action must come from parsing or recovery."
            )

    @property
    def used_fallback(self) -> bool:
        return self.fallback_action is not None


def parse_canonical_action(raw_response: str) -> int:
    if not isinstance(raw_response, str):
        raise ValueError("Canonical action response must be a string.")
    match = CANONICAL_ACTION_PATTERN.fullmatch(raw_response)
    if match is None:
        raise ValueError("Response does not match the canonical action grammar.")
    return int(match.group(1))


def backend_action_domain() -> tuple[int, ...]:
    return CANONICAL_ACTION_IDS


def require_fixed_idle_available(available_action_ids: Iterable[int]) -> int:
    available = _normalize_available_actions(available_action_ids)
    if FIXED_IDLE_ACTION_ID not in available:
        violation = ProtocolInvariantViolation.from_mapping(
            ProtocolInvariantCode.FIXED_FALLBACK_UNAVAILABLE,
            "Scientific fixed IDLE fallback is not available in the current state.",
            {"available_action_ids": tuple(sorted(available))},
        )
        raise RuntimeProtocolError(violation)
    return FIXED_IDLE_ACTION_ID


def resolve_action(
    raw_response: str,
    *,
    available_action_ids: Iterable[int],
    timed_out: bool = False,
    parser_mode: ParserMode = ParserMode.STRICT_ONLY,
    resolver_mode: ResolverMode = ResolverMode.DISABLED,
    fallback_policy: FallbackPolicy = FallbackPolicy.FIXED_IDLE,
    deterministic_recovery: Callable[[str], int] | None = None,
    resolver: Callable[[str], int] | None = None,
) -> ActionResolutionResult:
    _validate_confirmatory_modes(parser_mode, resolver_mode, fallback_policy)
    if not isinstance(raw_response, str):
        raise ValueError("raw_response must be a string.")
    if not isinstance(timed_out, bool):
        raise ValueError("timed_out must be a boolean.")
    available = _normalize_available_actions(available_action_ids)

    if timed_out:
        return _fallback_result(
            raw_response,
            ActionSyntaxStatus.TIMEOUT,
            RuntimeFailureClass.GENERATION_TIMEOUT,
            available,
        )
    if not raw_response.strip():
        return _fallback_result(
            raw_response,
            ActionSyntaxStatus.EMPTY,
            RuntimeFailureClass.MODEL_EMPTY_OUTPUT,
            available,
        )
    try:
        strict_action = parse_canonical_action(raw_response)
    except ValueError:
        return _fallback_result(
            raw_response,
            ActionSyntaxStatus.INVALID,
            RuntimeFailureClass.SYNTAX_INVALID,
            available,
        )

    if strict_action not in available:
        return _fallback_result(
            raw_response,
            ActionSyntaxStatus.STRICT_VALID,
            RuntimeFailureClass.ACTION_UNAVAILABLE,
            available,
            strict_action=strict_action,
            action_available=ActionAvailability.UNAVAILABLE,
        )
    return ActionResolutionResult(
        raw_response=raw_response,
        syntax_status=ActionSyntaxStatus.STRICT_VALID,
        strict_action=strict_action,
        recovered_action=None,
        recovery_stage=RecoveryStage.NONE,
        violation=None,
        action_available=ActionAvailability.AVAILABLE,
        fallback_action=None,
        final_resolved_action=strict_action,
    )


def _fallback_result(
    raw_response: str,
    syntax_status: ActionSyntaxStatus,
    violation: RuntimeFailureClass,
    available_actions: frozenset[int],
    *,
    strict_action: int | None = None,
    action_available: ActionAvailability = ActionAvailability.NOT_APPLICABLE,
) -> ActionResolutionResult:
    fallback_action = require_fixed_idle_available(available_actions)
    return ActionResolutionResult(
        raw_response=raw_response,
        syntax_status=syntax_status,
        strict_action=strict_action,
        recovered_action=None,
        recovery_stage=RecoveryStage.NONE,
        violation=violation,
        action_available=action_available,
        fallback_action=fallback_action,
        final_resolved_action=fallback_action,
    )


def _validate_confirmatory_modes(
    parser_mode: ParserMode,
    resolver_mode: ResolverMode,
    fallback_policy: FallbackPolicy,
) -> None:
    if parser_mode is not ParserMode.STRICT_ONLY:
        raise ValueError("Confirmatory action resolution requires strict_only parsing.")
    if resolver_mode is not ResolverMode.DISABLED:
        raise ValueError("Confirmatory action resolution disables resolver assistance.")
    if fallback_policy is not FallbackPolicy.FIXED_IDLE:
        raise ValueError("Confirmatory action resolution requires fixed_idle fallback.")


def _normalize_available_actions(values: Iterable[int]) -> frozenset[int]:
    normalized = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Available action IDs must be integers.")
        if value not in CANONICAL_ACTION_IDS:
            raise ValueError(
                f"Available action ID is outside the canonical domain: {value}."
            )
        normalized.add(value)
    if not normalized:
        raise ValueError("Available action IDs must be non-empty.")
    return frozenset(normalized)


__all__ = [
    "ActionAvailability",
    "ActionResolutionResult",
    "ActionSyntaxStatus",
    "CANONICAL_ACTION_IDS",
    "FIXED_IDLE_ACTION_ID",
    "RecoveryStage",
    "backend_action_domain",
    "parse_canonical_action",
    "require_fixed_idle_available",
    "resolve_action",
]
