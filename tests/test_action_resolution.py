import dataclasses
import unittest

from dilu.runtime.action_resolution import (
    ActionAvailability,
    ActionResolutionResult,
    ActionSyntaxStatus,
    RecoveryStage,
    backend_action_domain,
    parse_canonical_action,
    resolve_action,
)
from dilu.runtime.harness_config import ParserMode, ResolverMode
from dilu.runtime.runtime_failures import (
    ProtocolInvariantCode,
    RuntimeFailureClass,
    RuntimeProtocolError,
)


class CanonicalGrammarTests(unittest.TestCase):
    def test_exact_canonical_grammar_accepts_only_one_action_id(self) -> None:
        for action_id in range(5):
            self.assertEqual(
                parse_canonical_action(f"Response to user:#### {action_id}"),
                action_id,
            )

    def test_exact_canonical_grammar_rejects_noncanonical_output(self) -> None:
        invalid_responses = (
            "",
            "Response to user:#### 5",
            "Response to user:#### -1",
            "Response to user: 3",
            "Response to user:#### 3\n",
            " Response to user:#### 3",
            "Response to user:####  3",
            "response to user:#### 3",
            "Response to user:####\t3",
            "Response to user:#### ３",
            "Response to user:#### 3\nReason: clear",
            "Reasoning\nResponse to user:#### 3",
            "Response to user:#### 3 or 4",
            "`Response to user:#### 3`",
            '{"action_id": 3}',
            "Response to user:#### <3>",
        )

        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    parse_canonical_action(response)

    def test_schema_mode_does_not_mask_unavailable_actions(self) -> None:
        self.assertEqual(backend_action_domain(), (0, 1, 2, 3, 4))
        result = resolve_action(
            "Response to user:#### 3",
            available_action_ids=(1,),
        )
        self.assertEqual(result.strict_action, 3)
        self.assertEqual(result.action_available, ActionAvailability.UNAVAILABLE)


class ActionResolutionTests(unittest.TestCase):
    def test_strict_action_and_state_availability_remain_separate(self) -> None:
        result = resolve_action(
            "Response to user:#### 3",
            available_action_ids=(1, 4),
        )

        self.assertEqual(result.syntax_status, ActionSyntaxStatus.STRICT_VALID)
        self.assertEqual(result.strict_action, 3)
        self.assertEqual(result.action_available, ActionAvailability.UNAVAILABLE)
        self.assertEqual(result.violation, RuntimeFailureClass.ACTION_UNAVAILABLE)
        self.assertEqual(result.fallback_action, 1)
        self.assertEqual(result.final_resolved_action, 1)

    def test_valid_available_action_does_not_apply_fallback(self) -> None:
        result = resolve_action(
            "Response to user:#### 3",
            available_action_ids=(1, 3, 4),
        )

        self.assertIsInstance(result, ActionResolutionResult)
        self.assertEqual(result.strict_action, 3)
        self.assertEqual(result.action_available, ActionAvailability.AVAILABLE)
        self.assertIsNone(result.violation)
        self.assertIsNone(result.fallback_action)
        self.assertEqual(result.final_resolved_action, 3)

    def test_invalid_empty_and_timeout_responses_stay_distinct(self) -> None:
        cases = (
            (
                "not an action",
                False,
                ActionSyntaxStatus.INVALID,
                RuntimeFailureClass.SYNTAX_INVALID,
            ),
            (
                "",
                False,
                ActionSyntaxStatus.EMPTY,
                RuntimeFailureClass.MODEL_EMPTY_OUTPUT,
            ),
            (
                " \t",
                False,
                ActionSyntaxStatus.EMPTY,
                RuntimeFailureClass.MODEL_EMPTY_OUTPUT,
            ),
            (
                "partial output",
                True,
                ActionSyntaxStatus.TIMEOUT,
                RuntimeFailureClass.GENERATION_TIMEOUT,
            ),
        )

        for raw_response, timed_out, status, failure in cases:
            with self.subTest(status=status):
                result = resolve_action(
                    raw_response,
                    available_action_ids=(0, 1, 2, 3, 4),
                    timed_out=timed_out,
                )
                self.assertEqual(result.raw_response, raw_response)
                self.assertEqual(result.syntax_status, status)
                self.assertEqual(result.violation, failure)
                self.assertEqual(result.recovery_stage, RecoveryStage.NONE)
                self.assertEqual(
                    result.action_available,
                    ActionAvailability.NOT_APPLICABLE,
                )
                self.assertEqual(result.fallback_action, 1)
                self.assertEqual(result.final_resolved_action, 1)

    def test_strict_only_never_invokes_recovery_or_resolver(self) -> None:
        def forbidden(_: str) -> int:
            raise AssertionError("strict_only invoked a recovery callback")

        result = resolve_action(
            "Action id: 2",
            available_action_ids=(1, 2),
            parser_mode=ParserMode.STRICT_ONLY,
            resolver_mode=ResolverMode.DISABLED,
            deterministic_recovery=forbidden,
            resolver=forbidden,
        )

        self.assertEqual(result.final_resolved_action, 1)
        self.assertIsNone(result.recovered_action)

    def test_fixed_idle_unavailable_raises_typed_protocol_failure(self) -> None:
        with self.assertRaises(RuntimeProtocolError) as context:
            resolve_action("invalid", available_action_ids=(0, 2, 3, 4))

        self.assertEqual(
            context.exception.invariant_code,
            ProtocolInvariantCode.FIXED_FALLBACK_UNAVAILABLE,
        )

    def test_action_resolution_result_is_frozen(self) -> None:
        result = resolve_action(
            "Response to user:#### 1",
            available_action_ids=(1,),
        )

        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.final_resolved_action = 4

    def test_result_rejects_internally_inconsistent_stage_fields(self) -> None:
        with self.assertRaises(ValueError):
            ActionResolutionResult(
                raw_response="invalid",
                syntax_status=ActionSyntaxStatus.STRICT_VALID,
                strict_action=None,
                recovered_action=None,
                recovery_stage=RecoveryStage.NONE,
                violation=None,
                action_available=ActionAvailability.NOT_APPLICABLE,
                fallback_action=None,
                final_resolved_action=1,
            )

    def test_available_actions_must_be_nonempty_canonical_domain(self) -> None:
        for available_actions in ((), (1, 9), (True, 1)):
            with self.subTest(actions=available_actions):
                with self.assertRaises(ValueError):
                    resolve_action(
                        "Response to user:#### 1",
                        available_action_ids=available_actions,
                    )


if __name__ == "__main__":
    unittest.main()
