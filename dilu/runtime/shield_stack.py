from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import safety_shields as _safety_shields
from .harness_config import ExecutionMode, SHIELD_STAGE_ORDER, ShieldConfig
from .safety_shields import (
    SafetyShieldResult,
    apply_lane_change_safety_shield,
    apply_longitudinal_safety_shield,
    apply_low_speed_recovery_shield,
)


_METADATA_PREFIX = {
    "lane_change": "lane_change",
    "longitudinal_safety": "longitudinal_safety",
    "low_speed_recovery": "flow_recovery",
}
_PRIMITIVE_SHIELD_TYPE = {
    "lane_change": "lane_change",
    "longitudinal_safety": "longitudinal",
    "low_speed_recovery": "flow_recovery",
}
_LIVE_CONSTANTS = {
    "target_front_gap_required_m": "TARGET_FRONT_GAP_REQUIRED_M",
    "target_rear_gap_required_m": "TARGET_REAR_GAP_REQUIRED_M",
    "target_rear_ttc_required_sec": "TARGET_REAR_TTC_REQUIRED_SEC",
    "front_critical_gap_m": "FRONT_CRITICAL_GAP_M",
    "front_caution_gap_m": "FRONT_CAUTION_GAP_M",
    "acceleration_caution_gap_m": "ACCELERATION_CAUTION_GAP_M",
    "front_critical_ttc_sec": "FRONT_CRITICAL_TTC_SEC",
    "front_caution_ttc_sec": "FRONT_CAUTION_TTC_SEC",
    "acceleration_caution_ttc_sec": "ACCELERATION_CAUTION_TTC_SEC",
    "acceleration_projected_speed_gain_mps": ("ACCELERATION_PROJECTED_SPEED_GAIN_MPS"),
    "projected_time_horizon_sec": "PROJECTED_TIME_HORIZON_SEC",
    "low_speed_recovery_floor_mps": "LOW_SPEED_RECOVERY_FLOOR_MPS",
    "low_speed_recovery_front_gap_m": "LOW_SPEED_RECOVERY_FRONT_GAP_M",
    "low_speed_recovery_ttc_sec": "LOW_SPEED_RECOVERY_TTC_SEC",
}


@dataclass(frozen=True)
class ShieldStageResult:
    stage_name: str
    input_action_id: int
    output_action_id: int
    applied: bool
    bypassed: bool
    reason: str
    primitive_result: SafetyShieldResult | None

    def __post_init__(self) -> None:
        if self.stage_name not in SHIELD_STAGE_ORDER:
            raise ValueError("Unknown shield stage.")
        _validate_action_id(self.input_action_id, "input_action_id")
        _validate_action_id(self.output_action_id, "output_action_id")
        if not isinstance(self.applied, bool) or not isinstance(self.bypassed, bool):
            raise ValueError("Shield stage flags must be booleans.")
        if (
            not isinstance(self.reason, str)
            or not self.reason
            or self.reason != self.reason.strip()
        ):
            raise ValueError("Shield stage reason must be canonical text.")
        if self.bypassed:
            if self.applied or self.primitive_result is not None:
                raise ValueError("Bypassed stages cannot apply a primitive.")
            if self.input_action_id != self.output_action_id:
                raise ValueError("Bypassed stages cannot change the action.")
            return
        if not isinstance(self.primitive_result, SafetyShieldResult):
            raise ValueError("Executed stages require typed primitive evidence.")
        if (
            self.primitive_result.original_action_id != self.input_action_id
            or self.primitive_result.action_id != self.output_action_id
            or self.primitive_result.applied is not self.applied
            or self.primitive_result.reason != self.reason
            or self.primitive_result.shield_type
            != _PRIMITIVE_SHIELD_TYPE[self.stage_name]
        ):
            raise ValueError("Shield stage summary must preserve primitive evidence.")

    def to_metadata(self) -> dict[str, Any]:
        prefix = _METADATA_PREFIX[self.stage_name]
        primitive = self.primitive_result or SafetyShieldResult(
            original_action_id=self.input_action_id,
            action_id=self.output_action_id,
            applied=False,
            reason=self.reason,
            shield_type=prefix,
        )
        metadata = primitive.to_metadata(prefix)
        metadata.update(
            {
                f"{self.stage_name}_stage_input_action_id": self.input_action_id,
                f"{self.stage_name}_stage_output_action_id": self.output_action_id,
                f"{self.stage_name}_stage_applied": self.applied,
                f"{self.stage_name}_stage_bypassed": self.bypassed,
                f"{self.stage_name}_stage_reason": self.reason,
            }
        )
        return metadata


@dataclass(frozen=True)
class ShieldStackResult:
    proposed_action_id: int | None
    fallback_modified_action_id: int
    unshielded_action_id: int
    shielded_action_id: int | None
    executed_action_id: int
    execution_mode: ExecutionMode
    stages: tuple[ShieldStageResult, ...]

    def __post_init__(self) -> None:
        _validate_optional_action_id(self.proposed_action_id, "proposed_action_id")
        _validate_action_id(
            self.fallback_modified_action_id,
            "fallback_modified_action_id",
        )
        _validate_action_id(self.unshielded_action_id, "unshielded_action_id")
        _validate_optional_action_id(self.shielded_action_id, "shielded_action_id")
        _validate_action_id(self.executed_action_id, "executed_action_id")
        if self.unshielded_action_id != self.fallback_modified_action_id:
            raise ValueError(
                "Unshielded action must preserve the fallback-modified action."
            )
        if not isinstance(self.execution_mode, ExecutionMode):
            raise ValueError("execution_mode must be an ExecutionMode.")
        if not isinstance(self.stages, tuple):
            raise ValueError("stages must be an immutable tuple.")
        if not all(isinstance(stage, ShieldStageResult) for stage in self.stages):
            raise ValueError("stages must contain ShieldStageResult values.")
        if tuple(stage.stage_name for stage in self.stages) != SHIELD_STAGE_ORDER:
            raise ValueError("Shield stages must preserve the frozen order.")
        expected_input = self.unshielded_action_id
        for stage in self.stages:
            if stage.input_action_id != expected_input:
                raise ValueError("Shield stages must form one action chain.")
            expected_input = stage.output_action_id
        if self.execution_mode is ExecutionMode.UNSHIELDED_OPERATIONAL:
            if not all(stage.bypassed for stage in self.stages):
                raise ValueError("Unshielded mode requires bypassed stages.")
            if self.shielded_action_id is not None:
                raise ValueError("Unshielded mode cannot report a shielded action.")
            if self.executed_action_id != self.unshielded_action_id:
                raise ValueError("Unshielded mode must execute the unshielded action.")
        else:
            if any(stage.bypassed for stage in self.stages):
                raise ValueError("Shielded mode requires executed stages.")
            if self.shielded_action_id != expected_input:
                raise ValueError("Shielded action must match the stage chain.")
            if self.executed_action_id != self.shielded_action_id:
                raise ValueError("Shielded mode must execute the shielded action.")

    @property
    def final_action_id(self) -> int:
        """Legacy alias for the action sent to the environment."""
        return self.executed_action_id

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "shield_execution_mode": self.execution_mode.value,
            "shield_proposed_action_id": self.proposed_action_id,
            "shield_fallback_modified_action_id": self.fallback_modified_action_id,
            "shield_unshielded_action_id": self.unshielded_action_id,
            "shield_shielded_action_id": self.shielded_action_id,
            "shield_executed_action_id": self.executed_action_id,
            "shield_final_action_id": self.executed_action_id,
            "shield_stage_order": SHIELD_STAGE_ORDER,
        }
        for stage in self.stages:
            metadata.update(stage.to_metadata())
        lane_applied = self.stages[0].applied
        longitudinal_applied = self.stages[1].applied
        metadata.update(
            {
                "reactive_safety_shield_applied": lane_applied or longitudinal_applied,
                "reactive_safety_original_action_id": self.fallback_modified_action_id,
                "reactive_safety_final_action_id": self.executed_action_id,
                "flow_recovery_reason": self.stages[2].reason,
            }
        )
        return metadata


def execute_shield_stack(
    *,
    scenario: Any,
    proposed_action_id: int | None,
    fallback_modified_action_id: int,
    execution_mode: ExecutionMode,
    shield_config: ShieldConfig,
) -> ShieldStackResult:
    if not isinstance(shield_config, ShieldConfig):
        raise ValueError("shield_config must be a ShieldConfig.")
    shield_config.validate_scientific()
    _validate_live_primitive_binding(shield_config)
    if not isinstance(execution_mode, ExecutionMode):
        raise ValueError("execution_mode must be an ExecutionMode.")
    _validate_optional_action_id(proposed_action_id, "proposed_action_id")
    _validate_action_id(fallback_modified_action_id, "fallback_modified_action_id")

    if execution_mode is ExecutionMode.UNSHIELDED_OPERATIONAL:
        stages = tuple(
            ShieldStageResult(
                stage_name=stage_name,
                input_action_id=fallback_modified_action_id,
                output_action_id=fallback_modified_action_id,
                applied=False,
                bypassed=True,
                reason="execution_mode_unshielded",
                primitive_result=None,
            )
            for stage_name in SHIELD_STAGE_ORDER
        )
        return ShieldStackResult(
            proposed_action_id=proposed_action_id,
            fallback_modified_action_id=fallback_modified_action_id,
            unshielded_action_id=fallback_modified_action_id,
            shielded_action_id=None,
            executed_action_id=fallback_modified_action_id,
            execution_mode=execution_mode,
            stages=stages,
        )

    lane = apply_lane_change_safety_shield(scenario, fallback_modified_action_id)
    lane_stage = _executed_stage("lane_change", lane)
    longitudinal = apply_longitudinal_safety_shield(scenario, lane.action_id)
    longitudinal_stage = _executed_stage("longitudinal_safety", longitudinal)
    flow = apply_low_speed_recovery_shield(
        scenario,
        longitudinal.action_id,
        safety_shield_applied=bool(lane.applied or longitudinal.applied),
    )
    flow_stage = _executed_stage("low_speed_recovery", flow)
    stages = (lane_stage, longitudinal_stage, flow_stage)
    return ShieldStackResult(
        proposed_action_id=proposed_action_id,
        fallback_modified_action_id=fallback_modified_action_id,
        unshielded_action_id=fallback_modified_action_id,
        shielded_action_id=flow.action_id,
        executed_action_id=flow.action_id,
        execution_mode=execution_mode,
        stages=stages,
    )


def _executed_stage(
    stage_name: str,
    result: SafetyShieldResult,
) -> ShieldStageResult:
    if not isinstance(result, SafetyShieldResult):
        raise ValueError("Shield primitives must return SafetyShieldResult.")
    return ShieldStageResult(
        stage_name=stage_name,
        input_action_id=result.original_action_id,
        output_action_id=result.action_id,
        applied=result.applied,
        bypassed=False,
        reason=result.reason,
        primitive_result=result,
    )


def _validate_action_id(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
        raise ValueError(f"{name} must be an action ID in 0..4.")


def _validate_optional_action_id(value: int | None, name: str) -> None:
    if value is not None:
        _validate_action_id(value, name)


def _validate_live_primitive_binding(config: ShieldConfig) -> None:
    drift = tuple(
        field_name
        for field_name, constant_name in _LIVE_CONSTANTS.items()
        if getattr(config, field_name) != getattr(_safety_shields, constant_name)
    )
    if drift:
        raise ValueError(
            "Scientific shield config does not match live primitives: "
            + ",".join(drift)
        )


__all__ = [
    "ShieldStackResult",
    "ShieldStageResult",
    "execute_shield_stack",
]
