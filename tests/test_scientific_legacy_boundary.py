from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evaluate_models_ollama as evaluator

from dilu.runtime.harness_config import resolve_main_conditions
from dilu.runtime.llm_env import configure_runtime_env
from dilu.runtime.model_policy import build_decision_timeout_penalty_state
from tests.runtime_factorization_support import run_episode, runtime
from tests.test_scientific_driver_action_resolution import _scientific_config


class ScientificLegacyBoundaryTests(unittest.TestCase):
    def test_legacy_environment_bridge_rejects_scientific_mode(self) -> None:
        with self.assertRaises(ValueError):
            configure_runtime_env({}, mode="scientific")

    def test_legacy_timeout_builder_rejects_scientific_mode(self) -> None:
        with self.assertRaises(ValueError):
            build_decision_timeout_penalty_state(
                {},
                provider="ollama",
                mode="scientific",
                baseline_decision_timeout_sec=60.0,
            )

    def test_eight_cells_never_call_legacy_factor_helpers(self) -> None:
        forbidden = (
            "configure_runtime_env",
            "resolve_model_policy",
            "apply_model_policy_to_env",
            "build_decision_timeout_penalty_state",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches = [
                mock.patch.object(
                    evaluator,
                    name,
                    side_effect=AssertionError(f"legacy helper called: {name}"),
                )
                for name in forbidden
            ]
            for patcher in patches:
                patcher.start()
            self.addCleanup(lambda: [patcher.stop() for patcher in patches])

            for index, config in enumerate(
                resolve_main_conditions(_scientific_config())
            ):
                bound_runtime = runtime(
                    root / config.condition_id(),
                    attempt=index,
                    config=config,
                )
                run_episode(
                    root / config.condition_id(),
                    bound_runtime,
                )


if __name__ == "__main__":
    unittest.main()
