import dataclasses
import unittest
from unittest import mock

from dilu.runtime.harness_config import ExecutionMode, ShieldConfig
from dilu.runtime.safety_shields import SafetyShieldResult
from dilu.runtime.shield_stack import execute_shield_stack


def _shield_result(
    input_action: int,
    output_action: int,
    *,
    applied: bool,
    reason: str,
    shield_type: str,
) -> SafetyShieldResult:
    return SafetyShieldResult(
        original_action_id=input_action,
        action_id=output_action,
        applied=applied,
        reason=reason,
        shield_type=shield_type,
    )


class ShieldStackTests(unittest.TestCase):
    def test_unshielded_mode_bypasses_every_stage_and_preserves_action_layers(
        self,
    ) -> None:
        with mock.patch(
            "dilu.runtime.shield_stack.apply_lane_change_safety_shield",
            side_effect=AssertionError("lane stage must be bypassed"),
        ), mock.patch(
            "dilu.runtime.shield_stack.apply_longitudinal_safety_shield",
            side_effect=AssertionError("longitudinal stage must be bypassed"),
        ), mock.patch(
            "dilu.runtime.shield_stack.apply_low_speed_recovery_shield",
            side_effect=AssertionError("flow stage must be bypassed"),
        ):
            result = execute_shield_stack(
                scenario=object(),
                proposed_action_id=3,
                fallback_modified_action_id=1,
                execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
                shield_config=ShieldConfig.implementation_defaults(),
            )

        self.assertEqual(result.proposed_action_id, 3)
        self.assertEqual(result.fallback_modified_action_id, 1)
        self.assertEqual(result.unshielded_action_id, 1)
        self.assertIsNone(result.shielded_action_id)
        self.assertEqual(result.executed_action_id, 1)
        self.assertEqual(result.final_action_id, 1)
        self.assertEqual(
            tuple(stage.stage_name for stage in result.stages),
            ("lane_change", "longitudinal_safety", "low_speed_recovery"),
        )
        self.assertTrue(all(stage.bypassed for stage in result.stages))
        self.assertTrue(all(not stage.applied for stage in result.stages))
        self.assertTrue(
            all(
                stage.input_action_id == stage.output_action_id == 1
                for stage in result.stages
            )
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.final_action_id = 4

    def test_shielded_mode_executes_frozen_stage_order(self) -> None:
        calls: list[tuple[str, int, bool | None]] = []

        def lane(_scenario, action):
            calls.append(("lane_change", action, None))
            return _shield_result(
                action, 1, applied=True, reason="unsafe_lane", shield_type="lane_change"
            )

        def longitudinal(_scenario, action):
            calls.append(("longitudinal_safety", action, None))
            return _shield_result(
                action, 4, applied=True, reason="front_risk", shield_type="longitudinal"
            )

        def flow(_scenario, action, *, safety_shield_applied):
            calls.append(("low_speed_recovery", action, safety_shield_applied))
            return _shield_result(
                action,
                action,
                applied=False,
                reason="safety_shield_already_applied",
                shield_type="flow_recovery",
            )

        with mock.patch(
            "dilu.runtime.shield_stack.apply_lane_change_safety_shield", lane
        ), mock.patch(
            "dilu.runtime.shield_stack.apply_longitudinal_safety_shield",
            longitudinal,
        ), mock.patch(
            "dilu.runtime.shield_stack.apply_low_speed_recovery_shield", flow
        ):
            result = execute_shield_stack(
                scenario=object(),
                proposed_action_id=0,
                fallback_modified_action_id=0,
                execution_mode=ExecutionMode.SHIELDED,
                shield_config=ShieldConfig.implementation_defaults(),
            )

        self.assertEqual(
            calls,
            [
                ("lane_change", 0, None),
                ("longitudinal_safety", 1, None),
                ("low_speed_recovery", 4, True),
            ],
        )
        self.assertEqual(result.final_action_id, 4)
        self.assertEqual(result.unshielded_action_id, 0)
        self.assertEqual(result.shielded_action_id, 4)
        self.assertEqual(result.executed_action_id, 4)
        self.assertEqual(
            tuple(
                (stage.input_action_id, stage.output_action_id)
                for stage in result.stages
            ),
            ((0, 1), (1, 4), (4, 4)),
        )
        self.assertTrue(all(not stage.bypassed for stage in result.stages))

    def test_stack_rejects_ignored_threshold_drift(self) -> None:
        config = dataclasses.replace(
            ShieldConfig.implementation_defaults(),
            target_front_gap_required_m=15.0,
        )

        with self.assertRaises(ValueError):
            execute_shield_stack(
                scenario=object(),
                proposed_action_id=1,
                fallback_modified_action_id=1,
                execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
                shield_config=config,
            )

    def test_stack_rejects_live_primitive_constant_drift(self) -> None:
        with mock.patch(
            "dilu.runtime.safety_shields.TARGET_FRONT_GAP_REQUIRED_M",
            15.0,
        ), self.assertRaises(ValueError):
            execute_shield_stack(
                scenario=object(),
                proposed_action_id=1,
                fallback_modified_action_id=1,
                execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
                shield_config=ShieldConfig.implementation_defaults(),
            )

    def test_missing_proposal_and_action_layers_remain_explicit(self) -> None:
        result = execute_shield_stack(
            scenario=object(),
            proposed_action_id=None,
            fallback_modified_action_id=1,
            execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
            shield_config=ShieldConfig.implementation_defaults(),
        )

        self.assertIsNone(result.proposed_action_id)
        self.assertEqual(result.unshielded_action_id, 1)
        self.assertIsNone(result.shielded_action_id)
        self.assertEqual(result.executed_action_id, 1)
        metadata = result.to_metadata()
        self.assertIsNone(metadata["shield_proposed_action_id"])
        self.assertEqual(metadata["shield_unshielded_action_id"], 1)
        self.assertIsNone(metadata["shield_shielded_action_id"])
        self.assertEqual(metadata["shield_executed_action_id"], 1)

    def test_mode_and_stage_evidence_cannot_be_relabelled(self) -> None:
        result = execute_shield_stack(
            scenario=object(),
            proposed_action_id=None,
            fallback_modified_action_id=1,
            execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
            shield_config=ShieldConfig.implementation_defaults(),
        )

        with self.assertRaises(ValueError):
            dataclasses.replace(result, execution_mode=ExecutionMode.SHIELDED)

    def test_stage_rejects_wrong_primitive_type_label(self) -> None:
        wrong_lane = _shield_result(
            1,
            1,
            applied=False,
            reason="safe",
            shield_type="wrong_stage",
        )
        with mock.patch(
            "dilu.runtime.shield_stack.apply_lane_change_safety_shield",
            return_value=wrong_lane,
        ), self.assertRaises(ValueError):
            execute_shield_stack(
                scenario=object(),
                proposed_action_id=1,
                fallback_modified_action_id=1,
                execution_mode=ExecutionMode.SHIELDED,
                shield_config=ShieldConfig.implementation_defaults(),
            )

    def test_evaluator_compatibility_helper_can_select_unshielded_mode(self) -> None:
        from evaluate_models_ollama import (
            _apply_reactive_safety_shields,
            _decision_trace_item,
        )

        decision_meta: dict[str, object] = {}
        action, metadata = _apply_reactive_safety_shields(
            1,
            object(),
            decision_meta,
            execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
            proposed_action_id=3,
        )

        self.assertEqual(action, 1)
        self.assertEqual(metadata["shield_execution_mode"], "unshielded_operational")
        self.assertEqual(metadata["shield_proposed_action_id"], 3)
        self.assertEqual(metadata["shield_fallback_modified_action_id"], 1)
        self.assertEqual(metadata["shield_unshielded_action_id"], 1)
        self.assertIsNone(metadata["shield_shielded_action_id"])
        self.assertEqual(metadata["shield_executed_action_id"], 1)
        self.assertTrue(metadata["lane_change_stage_bypassed"])
        self.assertTrue(metadata["longitudinal_safety_stage_bypassed"])
        self.assertTrue(metadata["low_speed_recovery_stage_bypassed"])
        self.assertEqual(decision_meta["selected_action"], 1)
        trace = _decision_trace_item(
            step_idx=0,
            action_id=action,
            response_text="Response to user:#### 3",
            decision_meta=decision_meta,
        )
        self.assertEqual(trace["model_action_id"], 3)
        self.assertEqual(trace["shield_fallback_modified_action_id"], 1)
        self.assertEqual(trace["final_action_id"], 1)
        self.assertTrue(trace["lane_change_stage_bypassed"])


if __name__ == "__main__":
    unittest.main()
