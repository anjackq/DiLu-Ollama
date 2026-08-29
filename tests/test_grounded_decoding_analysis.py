"""Registered V8 grounded-decoding analysis: validation gate and contrasts.

Covers task-6-brief.md steps 1-2: the blocked-output contract (missing
rows, duplicate ids, digest mismatch, fingerprint drift, and the Family M
manipulation-check gate), the Family A/B/C/D paired contrasts against
constructed constant offsets, Holm correction within a family, bootstrap
determinism, and a direct byte-for-byte comparison of the sign-flip/Holm
implementation against the registered V7 reference script.

Fix-verification tests from the Task 6 review round (real-episode-schema
guard, fail-closed trace join, action-distribution shift, Family M's
condition-id filter) live in the sibling file
``test_grounded_decoding_analysis_fixes.py`` to keep both files under the
400-line-per-file limit.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import unittest
from pathlib import Path

import numpy as np

from dilu.runtime._grounded_decoding_analysis_stats import (
    SIGN_FLIP_DRAWS,
    holm,
    sign_flip_p,
)
from dilu.runtime._grounded_decoding_analysis_validation import validate_v8_rows
from dilu.runtime._minimal_factorial_analysis_bootstrap import (
    BOOTSTRAP_DRAWS,
    derive_bootstrap_seed,
)
from dilu.runtime.grounded_decoding_analysis import (
    blocked_payload,
    run_registered_v8_analysis,
)
from tests.grounded_decoding_analysis_support import (
    MODEL_SLOTS,
    SMALL_MODEL_SLOTS,
    build_v8_fixture,
    drift_a_fingerprint,
    drop_a_row,
    duplicate_a_row,
    mismatch_a_digest,
    trigger_family_m,
)

ROOT = Path(__file__).resolve().parents[1]
V7_REFERENCE_SCRIPT = (
    ROOT
    / "results"
    / "iclr2027_model_breadth_factorial_v7"
    / "analysis-prototype"
    / "analyze_v7_full_factorial.py"
)
MANIFEST_SHA256 = "sha256:" + "1" * 64


class GroundedDecodingValidationGateTests(unittest.TestCase):
    """Step 1: the fail-closed blocked-output contract with synthetic fixtures."""

    def setUp(self) -> None:
        self.v8_rows, self.frozen_rows = build_v8_fixture()

    def test_happy_path_480_rows_validate_complete(self) -> None:
        self.assertEqual(len(self.v8_rows), 480)
        result = validate_v8_rows(self.v8_rows, self.frozen_rows)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.errors, ())
        self.assertTrue(result.contrast_artifacts_written)

    def test_missing_row_blocks(self) -> None:
        result = validate_v8_rows(drop_a_row(self.v8_rows), self.frozen_rows)
        self._assert_blocked(result)
        self.assertTrue(any("480" in error for error in result.errors))

    def test_duplicate_id_blocks(self) -> None:
        result = validate_v8_rows(duplicate_a_row(self.v8_rows), self.frozen_rows)
        self._assert_blocked(result)
        self.assertTrue(any("duplicate" in error.lower() for error in result.errors))

    def test_digest_mismatch_blocks(self) -> None:
        result = validate_v8_rows(mismatch_a_digest(self.v8_rows), self.frozen_rows)
        self._assert_blocked(result)
        self.assertTrue(any("digest" in error.lower() for error in result.errors))

    def test_fingerprint_drift_blocks(self) -> None:
        result = validate_v8_rows(self.v8_rows, drift_a_fingerprint(self.frozen_rows))
        self._assert_blocked(result)
        self.assertTrue(any("fingerprint" in error.lower() for error in result.errors))

    def test_family_m_gate_blocks_on_any_action_unavailable_violation(self) -> None:
        result = validate_v8_rows(
            trigger_family_m(self.v8_rows, "qwen_8b"), self.frozen_rows
        )
        self._assert_blocked(result)
        self.assertTrue(any("Family M" in error for error in result.errors))
        self.assertTrue(any("qwen_8b" in error for error in result.errors))

    def test_family_m_gate_passes_when_all_counts_are_zero(self) -> None:
        # The happy-path fixture already carries all-zero counts; this pins
        # that the gate does not fire on a clean run (no false positive).
        result = validate_v8_rows(self.v8_rows, self.frozen_rows)
        self.assertFalse(any("Family M" in error for error in result.errors))

    def test_blocked_payload_shape_is_exactly_the_registered_three_keys(self) -> None:
        result = run_registered_v8_analysis(
            drop_a_row(self.v8_rows), self.frozen_rows, manifest_sha256=MANIFEST_SHA256
        )
        payload = blocked_payload(result.validation)
        self.assertEqual(set(payload), {"status", "errors", "contrast_artifacts_written"})
        self.assertEqual(payload["status"], "blocked")
        self.assertFalse(payload["contrast_artifacts_written"])
        self.assertIsInstance(payload["errors"], list)
        self.assertTrue(payload["errors"])
        self.assertEqual(result.family_a, ())
        self.assertEqual(result.family_b, ())
        self.assertEqual(result.family_c, ())
        self.assertEqual(result.family_d, ())
        self.assertEqual(result.descriptive, ())

    def _assert_blocked(self, result) -> None:
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.contrast_artifacts_written)
        self.assertTrue(result.errors)


class GroundedDecodingContrastTests(unittest.TestCase):
    """Step 2: Families A-D against constructed constant offsets."""

    def test_family_a_and_b_paired_means_equal_constructed_constants(self) -> None:
        offsets_a = {slot: 0.10 + 0.01 * index for index, slot in enumerate(MODEL_SLOTS)}
        offsets_b = {slot: -0.05 - 0.01 * index for index, slot in enumerate(MODEL_SLOTS)}
        v8_rows, frozen_rows = build_v8_fixture(
            family_a_offsets=offsets_a, family_b_offsets=offsets_b
        )
        result = run_registered_v8_analysis(
            v8_rows, frozen_rows, manifest_sha256=MANIFEST_SHA256
        )
        self.assertEqual(result.validation.status, "complete")
        by_model_a = {row["model_slot"]: row for row in result.family_a}
        by_model_b = {row["model_slot"]: row for row in result.family_b}
        self.assertEqual(set(by_model_a), set(MODEL_SLOTS))
        for slot in MODEL_SLOTS:
            self.assertAlmostEqual(by_model_a[slot]["estimate"], offsets_a[slot], places=9)
            self.assertAlmostEqual(by_model_b[slot]["estimate"], offsets_b[slot], places=9)
            self.assertEqual(by_model_a[slot]["n_paired_cases"], 30)
            self.assertEqual(by_model_b[slot]["n_paired_cases"], 30)

    def test_family_c_equals_constructed_difference_in_differences(self) -> None:
        offsets_a = {slot: 0.20 for slot in MODEL_SLOTS}
        offsets_b = {slot: 0.05 for slot in MODEL_SLOTS}
        v8_rows, frozen_rows = build_v8_fixture(
            family_a_offsets=offsets_a, family_b_offsets=offsets_b
        )
        result = run_registered_v8_analysis(
            v8_rows, frozen_rows, manifest_sha256=MANIFEST_SHA256
        )
        self.assertEqual(len(result.family_c), len(MODEL_SLOTS))
        for row in result.family_c:
            self.assertAlmostEqual(row["estimate"], 0.15, places=9)
            self.assertEqual(row["n_paired_cases"], 30)

    def test_family_d_endpoint_equals_constant_over_all_120_cases(self) -> None:
        offsets_a = {slot: 0.30 for slot in SMALL_MODEL_SLOTS}
        v8_rows, frozen_rows = build_v8_fixture(family_a_offsets=offsets_a)
        result = run_registered_v8_analysis(
            v8_rows, frozen_rows, manifest_sha256=MANIFEST_SHA256
        )
        self.assertEqual({row["model_slot"] for row in result.family_d}, set(SMALL_MODEL_SLOTS))
        for row in result.family_d:
            self.assertAlmostEqual(row["estimate"], 0.30, places=9)
            self.assertEqual(row["n_paired_cases"], 120)

    def test_interval_metadata_says_fixed_suite_sensitivity_never_confidence(self) -> None:
        v8_rows, frozen_rows = build_v8_fixture()
        result = run_registered_v8_analysis(
            v8_rows, frozen_rows, manifest_sha256=MANIFEST_SHA256
        )
        for row in (*result.family_a, *result.family_b, *result.family_c, *result.family_d):
            self.assertIn("fixed-suite sensitivity", row["evidence_scope"])
            self.assertNotIn("confidence", row["evidence_scope"])

    def test_holm_applied_within_family_one_strong_four_null(self) -> None:
        models = list(MODEL_SLOTS)
        strong = models[0]
        offsets = {strong: 0.5, **{slot: 0.0 for slot in models[1:]}}
        v8_rows, frozen_rows = build_v8_fixture(family_a_offsets=offsets)
        result = run_registered_v8_analysis(
            v8_rows, frozen_rows, manifest_sha256=MANIFEST_SHA256
        )
        by_model = {row["model_slot"]: row for row in result.family_a}
        strong_row = by_model[strong]
        self.assertLess(strong_row["p_value"], 0.01)
        self.assertAlmostEqual(
            strong_row["p_holm"], min(1.0, len(models) * strong_row["p_value"]), places=12
        )
        for slot in models[1:]:
            null_row = by_model[slot]
            self.assertEqual(null_row["p_value"], 1.0)
            self.assertEqual(null_row["p_holm"], 1.0)

    def test_descriptive_secondary_outcomes_carry_no_inferential_fields(self) -> None:
        v8_rows, frozen_rows = build_v8_fixture()
        result = run_registered_v8_analysis(
            v8_rows, frozen_rows, manifest_sha256=MANIFEST_SHA256
        )
        self.assertTrue(result.descriptive)
        for row in result.descriptive:
            self.assertTrue(math.isnan(row["p_value"]))
            self.assertTrue(math.isnan(row["p_holm"]))
            self.assertEqual(row["sign_flip_draws"], 0)
        outcomes = {row["outcome"] for row in result.descriptive}
        self.assertEqual(
            outcomes,
            {
                "task_completion",
                "crash",
                "fallback_rate",
                "shield_intervention_rate",
                "decision_latency_ms_avg",
            },
        )


class GroundedDecodingBootstrapDeterminismTests(unittest.TestCase):
    """Bootstrap seed recipe and rerun determinism."""

    def test_bootstrap_seed_matches_registered_recipe(self) -> None:
        seed = derive_bootstrap_seed(
            MANIFEST_SHA256,
            "qwen_06b",
            "FAMILY_A_O2E1_MINUS_O1E1",
            "driving_score_balanced_v1",
        )
        parts = (
            MANIFEST_SHA256,
            "qwen_06b",
            "FAMILY_A_O2E1_MINUS_O1E1",
            "driving_score_balanced_v1",
            "bootstrap-v1",
        )
        expected = int.from_bytes(
            hashlib.sha256("|".join(parts).encode()).digest()[:8], "big"
        )
        self.assertEqual(seed, expected)

    def test_20000_draws_and_rerun_is_byte_identical(self) -> None:
        v8_rows, frozen_rows = build_v8_fixture(family_a_offsets={"qwen_06b": 0.2})
        result_1 = run_registered_v8_analysis(
            copy.deepcopy(v8_rows), copy.deepcopy(frozen_rows), manifest_sha256=MANIFEST_SHA256
        )
        result_2 = run_registered_v8_analysis(
            copy.deepcopy(v8_rows), copy.deepcopy(frozen_rows), manifest_sha256=MANIFEST_SHA256
        )
        for row in result_1.family_a:
            self.assertEqual(row["bootstrap_draws"], BOOTSTRAP_DRAWS)
            self.assertEqual(row["bootstrap_draws"], 20_000)
            self.assertEqual(row["sign_flip_draws"], SIGN_FLIP_DRAWS)
            self.assertEqual(row["sign_flip_draws"], 20_000)
        self.assertEqual(
            json.dumps(result_1.family_a, sort_keys=True),
            json.dumps(result_2.family_a, sort_keys=True),
        )
        self.assertEqual(
            json.dumps(result_1.family_d, sort_keys=True),
            json.dumps(result_2.family_d, sort_keys=True),
        )
        self.assertEqual(
            json.dumps(result_1.descriptive, sort_keys=True),
            json.dumps(result_2.descriptive, sort_keys=True),
        )


class SignFlipHolmMatchesV7ReferenceTests(unittest.TestCase):
    """Direct comparison against the registered V7 prototype implementation."""

    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "v7_reference_analysis_prototype", V7_REFERENCE_SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.reference = module

    def test_sign_flip_p_matches_reference_exactly(self) -> None:
        values = np.array([0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.05, 0.3, -0.02, 0.12])
        for seed in (123456789, 0, 42, 2**31 - 1):
            with self.subTest(seed=seed):
                mine = sign_flip_p(values, seed)
                theirs = self.reference.sign_flip_p(values, seed)
                self.assertEqual(mine, theirs)

    def test_holm_matches_reference_exactly(self) -> None:
        base_rows = [
            {"p_value": 0.001},
            {"p_value": 0.5},
            {"p_value": 0.02},
            {"p_value": 0.9},
            {"p_value": 0.5},
        ]
        rows_mine = copy.deepcopy(base_rows)
        rows_theirs = copy.deepcopy(base_rows)
        holm(rows_mine)
        self.reference.holm(rows_theirs)
        self.assertEqual(
            [row["p_holm"] for row in rows_mine],
            [row["p_holm"] for row in rows_theirs],
        )


if __name__ == "__main__":
    unittest.main()
