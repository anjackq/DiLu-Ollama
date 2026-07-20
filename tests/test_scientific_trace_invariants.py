from __future__ import annotations

import dataclasses
import json
import unittest
from unittest import mock

import jsonschema

from dilu.driver_agent.prompt_modules import build_prompt_artifact
from dilu.runtime.action_resolution import (
    ActionAvailability,
    ActionResolutionResult,
    ActionSyntaxStatus,
    RecoveryStage,
    resolve_action,
)
from dilu.runtime.harness_config import (
    ConditionSpec,
    ExecutionMode,
    OutputEnforcement,
    PolicyContent,
)
from dilu.runtime.ollama_scientific_client import (
    OllamaScientificClient,
)
from dilu.runtime.runtime_failures import RuntimeFailureClass
from dilu.runtime.scientific_trace import (
    DecisionTraceRecord,
    GenerationSeedScope,
    ScientificTraceWriter,
    TraceDisposition,
    append_trace_before_step,
    trace_schema_path,
)
from dilu.runtime.safety_shields import SafetyShieldResult
from dilu.runtime.shield_stack import execute_shield_stack
from tests.scientific_transport_support import (
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_request,
    make_retry_policy,
    success_payload,
)
from tests.scientific_trace_support import blocked_record
from tests.test_scientific_driver_action_resolution import _scientific_config
from tests.test_scientific_trace import _generation, _record


class DecisionTraceInvariantTests(unittest.TestCase):
    def test_canonical_output_cannot_be_relabeled_as_invalid_fallback(self) -> None:
        record = _record()
        bad_resolution = ActionResolutionResult(
            raw_response="Response to user:#### 3",
            syntax_status=ActionSyntaxStatus.INVALID,
            strict_action=None,
            recovered_action=None,
            recovery_stage=RecoveryStage.NONE,
            violation=RuntimeFailureClass.SYNTAX_INVALID,
            action_available=ActionAvailability.NOT_APPLICABLE,
            fallback_action=1,
            final_resolved_action=1,
        )
        bad_stack = execute_shield_stack(
            scenario=object(),
            proposed_action_id=None,
            fallback_modified_action_id=1,
            execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
            shield_config=record.harness_config.shield,
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(
                record,
                resolution=bad_resolution,
                shield_stack=bad_stack,
            )

    def test_hidden_prompt_messages_and_missing_idle_are_rejected(self) -> None:
        record = _record()
        hidden_request = dataclasses.replace(
            record.generation.request,
            messages=record.generation.request.messages
            + (("assistant", "hidden steering"),),
        )
        hidden_generation = dataclasses.replace(
            record.generation,
            request=hidden_request,
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(record, generation=hidden_generation)
        with self.assertRaises(ValueError):
            dataclasses.replace(
                record.context,
                available_action_ids=(0, 2, 3, 4),
            )

    def test_cross_stage_tampering_is_rejected(self) -> None:
        record = _record()
        bad_generation = dataclasses.replace(
            record.generation,
            request=dataclasses.replace(
                record.generation.request,
                messages=(("system", "tampered"), ("user", "scenario")),
            ),
        )
        bad_resolution = dataclasses.replace(
            record.resolution,
            raw_response="Response to user:#### 2",
        )
        bad_availability = dataclasses.replace(
            record.context,
            available_action_ids=(0, 1, 2, 4),
        )
        bad_stack = execute_shield_stack(
            scenario=object(),
            proposed_action_id=3,
            fallback_modified_action_id=2,
            execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
            shield_config=record.harness_config.shield,
        )
        replacements = {
            "condition_id": {
                "context": dataclasses.replace(
                    record.context,
                    key=dataclasses.replace(record.context.key, condition_id="c111"),
                )
            },
            "prompt_request": {"generation": bad_generation},
            "parser_input": {"resolution": bad_resolution},
            "availability": {"context": bad_availability},
            "shield_input": {"shield_stack": bad_stack},
        }
        for name, replacement in replacements.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    dataclasses.replace(record, **replacement)
        with self.assertRaises(ValueError):
            dataclasses.replace(
                record.context,
                generation_seed_scope=GenerationSeedScope.POST_DIVERGENCE,
                decision_snapshot_id=None,
            )

    def test_blocked_transport_failure_is_persisted_without_execution(self) -> None:
        record = blocked_record()
        payload = record.to_dict()
        schema = json.loads(trace_schema_path().read_text(encoding="utf-8"))
        jsonschema.validate(payload, schema)

        self.assertEqual(payload["disposition"], "blocked_before_execution")
        self.assertEqual(payload["failure"]["failure_class"], "transport_drift")
        self.assertIsNone(payload["action_resolution"])
        self.assertIsNone(payload["shield_stack"])
        with self.assertRaises(ValueError):
            append_trace_before_step(
                mock.Mock(spec=ScientificTraceWriter), record, mock.Mock()
            )

    def test_backend_schema_and_shielded_condition_serializes_all_stages(self) -> None:
        config = dataclasses.replace(
            _scientific_config(),
            condition=ConditionSpec(
                PolicyContent.HISTORICAL_DILU_2024,
                OutputEnforcement.BACKEND_SCHEMA,
                ExecutionMode.SHIELDED,
            ),
        )
        prompt = build_prompt_artifact(
            config.condition.policy_content,
            output_enforcement=config.condition.output_enforcement,
        )
        primary = _generation(decision_index=0)
        request = make_request(OutputEnforcement.BACKEND_SCHEMA)
        request = dataclasses.replace(
            request,
            messages=(("system", prompt.system_prompt()), ("user", "scenario")),
            options=dataclasses.replace(
                request.options,
                seed=primary.request.options.seed,
            ),
        )
        generation = OllamaScientificClient(
            capabilities=make_capabilities(),
            retry_policy=make_retry_policy(),
            identity_inspector=identity_inspector_for(),
            post=lambda *args, **kwargs: FakeResponse(
                success_payload('"Response to user:#### 3"')
            ),
            sleep=lambda _: None,
        ).generate(request)
        resolution = resolve_action(
            generation.contract_text or "",
            available_action_ids=(0, 1, 2, 3, 4),
        )

        def no_change(action_id: int) -> SafetyShieldResult:
            return SafetyShieldResult(
                original_action_id=action_id,
                action_id=action_id,
                applied=False,
                reason="test_no_change",
                shield_type="unused",
                front_ttc_sec=float("inf"),
            )

        with (
            mock.patch(
                "dilu.runtime.shield_stack.apply_lane_change_safety_shield",
                side_effect=lambda scenario, action: dataclasses.replace(
                    no_change(action), shield_type="lane_change"
                ),
            ),
            mock.patch(
                "dilu.runtime.shield_stack.apply_longitudinal_safety_shield",
                side_effect=lambda scenario, action: dataclasses.replace(
                    no_change(action), shield_type="longitudinal"
                ),
            ),
            mock.patch(
                "dilu.runtime.shield_stack.apply_low_speed_recovery_shield",
                side_effect=lambda scenario, action, **kwargs: dataclasses.replace(
                    no_change(action), shield_type="flow_recovery"
                ),
            ),
        ):
            stack = execute_shield_stack(
                scenario=object(),
                proposed_action_id=resolution.strict_action,
                fallback_modified_action_id=resolution.final_resolved_action,
                execution_mode=ExecutionMode.SHIELDED,
                shield_config=config.shield,
            )
        base = _record()
        context = dataclasses.replace(
            base.context,
            key=dataclasses.replace(
                base.context.key, condition_id=config.condition_id()
            ),
        )
        record = DecisionTraceRecord(
            context=context,
            harness_config=config,
            prompt_artifact=prompt,
            generation=generation,
            resolution=resolution,
            shield_stack=stack,
            disposition=TraceDisposition.READY_FOR_ENV_STEP,
            decision_latency_ms=generation.latency_ms,
        )
        payload = record.to_dict()

        self.assertEqual(payload["factors"]["output_enforcement"], "backend_schema")
        self.assertEqual(payload["shield_stack"]["execution_mode"], "shielded")
        self.assertEqual(
            [stage["stage_name"] for stage in payload["shield_stack"]["stages"]],
            ["lane_change", "longitudinal_safety", "low_speed_recovery"],
        )
        first_primitive = payload["shield_stack"]["stages"][0]["primitive"]
        self.assertIsNone(first_primitive["front_ttc_sec"])
        self.assertEqual(
            first_primitive["nonfinite_values"],
            {"front_ttc_sec": "positive_infinity"},
        )


if __name__ == "__main__":
    unittest.main()
