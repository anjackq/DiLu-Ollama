from __future__ import annotations

import json
import tempfile
import unittest
from itertools import product
from pathlib import Path

from dilu.runtime.harness_config import (
    ExecutionMode,
    OutputEnforcement,
    PolicyContent,
    resolve_main_conditions,
)
from tests.runtime_factorization_support import run_episode, runtime
from tests.test_scientific_driver_action_resolution import _scientific_config


class ScientificFactorMatrixTests(unittest.TestCase):
    def test_eight_cell_results_and_traces_bind_declared_factors(self) -> None:
        observed: set[tuple[str, str, str]] = set()
        config_hashes: set[str] = set()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, config in enumerate(
                resolve_main_conditions(_scientific_config())
            ):
                cell_root = root / config.condition_id()
                result = run_episode(
                    cell_root,
                    runtime(cell_root, attempt=index, config=config),
                )
                trace = json.loads(
                    (cell_root / "decision_traces.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()[0]
                )
                factors = (
                    result["policy_content"],
                    result["output_enforcement"],
                    result["execution_mode"],
                )
                observed.add(factors)
                config_hashes.add(result["config_sha256"])
                self.assertEqual(
                    trace["factors"],
                    {
                        "policy_content": factors[0],
                        "output_enforcement": factors[1],
                        "execution_mode": factors[2],
                    },
                )
                self.assertEqual(trace["prompt"]["policy_content"], factors[0])
                self.assertEqual(
                    trace["generation"]["request"]["output_enforcement"],
                    factors[1],
                )
                self.assertEqual(trace["shield_stack"]["execution_mode"], factors[2])
                self.assertEqual(result["condition_id"], config.condition_id())

        expected = set(
            product(
                (item.value for item in PolicyContent),
                (
                    OutputEnforcement.PROMPT_ONLY.value,
                    OutputEnforcement.BACKEND_SCHEMA.value,
                ),
                (item.value for item in ExecutionMode),
            )
        )
        self.assertEqual(observed, expected)
        self.assertEqual(len(config_hashes), 8)


if __name__ == "__main__":
    unittest.main()
