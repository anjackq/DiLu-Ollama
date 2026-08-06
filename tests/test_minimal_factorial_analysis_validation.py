from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

from dilu.runtime._minimal_factorial_analysis_locks import (
    validate_authorized_runtime_locks,
)
from dilu.runtime._minimal_factorial_analysis_metrics import validate_analysis_metrics
from dilu.runtime._minimal_factorial_schedule_support import canonical_sha256
from dilu.runtime._minimal_factorial_analysis_validation import (
    validate_joined_rows,
)
from tests.minimal_factorial_analysis_support import synthetic_analysis_bundle


class MinimalFactorialAnalysisValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.claim, self.episodes, self.baselines = synthetic_analysis_bundle()

    def test_exact_registered_denominators_validate(self) -> None:
        result = validate_joined_rows(self.claim, self.episodes, self.baselines)

        self.assertEqual(
            self.claim["manifest"]["campaign_id"],
            "iclr2027-minimal-factorial-v3",
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.errors, ())
        self.assertTrue(result.contrast_artifacts_written)
        self.assertEqual(result.expected_episode_rows, 840)
        self.assertEqual(result.observed_episode_rows, 840)
        self.assertEqual(result.expected_baseline_rows, 360)
        self.assertEqual(result.observed_baseline_rows, 360)

    def test_episode_mutations_are_blocked(self) -> None:
        variants: dict[str, list[dict[str, object]]] = {}
        variants["missing"] = copy.deepcopy(self.episodes[:-1])
        variants["duplicate"] = copy.deepcopy(self.episodes)
        variants["duplicate"].append(copy.deepcopy(variants["duplicate"][0]))
        variants["blocked"] = copy.deepcopy(self.episodes)
        variants["blocked"][0]["status"] = "blocked"
        variants["error"] = copy.deepcopy(self.episodes)
        variants["error"][0]["error"] = "synthetic failure"
        variants["trace_invalid"] = copy.deepcopy(self.episodes)
        variants["trace_invalid"][0]["scientific_trace_references"][0][
            "record_sha256"
        ] = "invalid"
        variants["mixed_digest"] = copy.deepcopy(self.episodes)
        variants["mixed_digest"][0]["model_digest"] = "sha256:" + "9" * 64
        variants["seed_drift"] = copy.deepcopy(self.episodes)
        variants["seed_drift"][0]["simulator_seed"] = 99_999
        variants["config_drift"] = copy.deepcopy(self.episodes)
        variants["config_drift"][0]["config_sha256"] = "sha256:" + "8" * 64
        variants["old_extra"] = copy.deepcopy(self.episodes)
        old = copy.deepcopy(variants["old_extra"][0])
        old["episode_attempt_id"] = "episode-" + "7" * 64
        old["campaign_id"] = "development-one-shot"
        variants["old_extra"].append(old)
        variants["development_stage"] = copy.deepcopy(self.episodes)
        variants["development_stage"][30]["stage"] = "development-one-shot"

        for name, rows in variants.items():
            with self.subTest(name=name):
                result = validate_joined_rows(self.claim, rows, self.baselines)
                self.assertEqual(result.status, "blocked")
                self.assertFalse(result.contrast_artifacts_written)
                self.assertTrue(result.errors)

    def test_category_and_runtime_lock_drift_are_blocked(self) -> None:
        category_rows = copy.deepcopy(self.episodes)
        category_rows[0]["category"] = "category-wrong"
        lock_rows = copy.deepcopy(self.episodes)
        lock_rows[0]["runtime_lock_binding_sha256"] = "sha256:" + "1" * 64

        for rows in (category_rows, lock_rows):
            with self.subTest(field=next(iter(rows[0]))):
                self.assertEqual(
                    validate_joined_rows(self.claim, rows, self.baselines).status,
                    "blocked",
                )

    def test_unregistered_campaign_id_is_blocked(self) -> None:
        claim = copy.deepcopy(self.claim)
        claim["manifest"]["campaign_id"] = "development-one-shot"

        result = validate_joined_rows(claim, self.episodes, self.baselines)

        self.assertEqual(result.status, "blocked")
        self.assertTrue(any("campaign ID" in error for error in result.errors))

    def test_v2_campaign_is_not_registered_for_analysis(self) -> None:
        claim = copy.deepcopy(self.claim)
        claim["manifest"]["campaign_id"] = "iclr2027-minimal-factorial-v2"

        result = validate_joined_rows(claim, self.episodes, self.baselines)

        self.assertEqual(result.status, "blocked")
        self.assertTrue(any("campaign ID" in error for error in result.errors))

    def test_joined_unregistered_stage_label_is_blocked(self) -> None:
        claim = copy.deepcopy(self.claim)
        episodes = copy.deepcopy(self.episodes)
        claim["schedule"][30]["stage"] = "development-one-shot"
        episodes[30]["stage"] = "development-one-shot"

        result = validate_joined_rows(claim, episodes, self.baselines)

        self.assertEqual(result.status, "blocked")
        self.assertTrue(any("stage label" in error for error in result.errors))

    def test_baseline_mutations_are_blocked(self) -> None:
        variants = {
            "missing": copy.deepcopy(self.baselines[:-1]),
            "duplicate": copy.deepcopy(self.baselines),
            "seed": copy.deepcopy(self.baselines),
            "shield": copy.deepcopy(self.baselines),
            "provenance": copy.deepcopy(self.baselines),
            "policy": copy.deepcopy(self.baselines),
        }
        variants["duplicate"].append(copy.deepcopy(variants["duplicate"][0]))
        variants["seed"][0]["simulator_seed"] = 99_999
        variants["shield"][0]["safety_shields_enabled"] = False
        variants["provenance"][0]["scoring_sha256"] = "sha256:" + "6" * 64
        variants["policy"][0]["baseline_policy"] = "old-development-policy"

        for name, rows in variants.items():
            with self.subTest(name=name):
                result = validate_joined_rows(self.claim, self.episodes, rows)
                self.assertEqual(result.status, "blocked")
                self.assertFalse(result.contrast_artifacts_written)
                self.assertTrue(result.errors)

    def test_metric_schema_is_fail_closed(self) -> None:
        missing = copy.deepcopy(self.episodes)
        del missing[0]["decision_calls_total"]
        invalid = copy.deepcopy(self.baselines)
        invalid[0]["task_completed"] = "yes"
        invalid[1]["driving_score_balanced_v1"] = 2.0

        errors = validate_analysis_metrics(missing, invalid)

        self.assertIn(
            "episode decision_calls_total must be a nonnegative integer",
            errors,
        )
        self.assertIn("baseline task_completed must be a boolean", errors)
        self.assertIn(
            "baseline driving_score_balanced_v1 must be within [0, 1]",
            errors,
        )

    def test_authorized_runtime_locks_are_reloaded_and_matched(self) -> None:
        claim = copy.deepcopy(self.claim)
        claim["manifest"]["transport"] = {
            "native_endpoint": "http://localhost:11434/api/chat"
        }
        by_cell = {}
        for row in self.episodes:
            by_cell.setdefault((row["model_slot"], row["condition_id"]), row)
        bindings = {cell: _binding_for(row, claim) for cell, row in by_cell.items()}

        def loader(*, runtime_lock_path: Path, authorization_path: Path) -> object:
            self.assertEqual(runtime_lock_path.name, "RUNTIME_PROTOCOL_LOCK.json")
            self.assertEqual(authorization_path.name, "PROTOCOL_FROZEN.json")
            return bindings[
                (runtime_lock_path.parent.parent.name, runtime_lock_path.parent.name)
            ]

        self.assertEqual(
            validate_authorized_runtime_locks(
                Path("locks"), claim, self.episodes, lock_loader=loader
            ),
            (),
        )
        drifted = copy.deepcopy(self.episodes)
        drifted[0]["runtime_lock_authorization_artifact_sha256"] = "sha256:" + "9" * 64
        self.assertTrue(
            validate_authorized_runtime_locks(
                Path("locks"), claim, drifted, lock_loader=loader
            )
        )


def _binding_for(row: dict[str, object], claim: dict[str, object]) -> object:
    condition = row["condition"]
    snapshot = claim["runtime_snapshot"]
    return SimpleNamespace(
        condition_id=row["condition_id"],
        config_sha256="sha256:" + canonical_sha256(condition),
        prompt_sha256=row["prompt_sha256"],
        model_tag=row["model_tag"],
        model_digest=row["model_digest"],
        native_endpoint="http://localhost:11434/api/chat",
        think_mode=SimpleNamespace(value=condition["transport"]["think_mode"]),
        capability_artifact_sha256=row["capability_artifact_sha256"],
        capability_snapshot_sha256=row["capability_snapshot_sha256"],
        trace_schema_sha256="sha256:" + snapshot["trace_schema_sha256"],
        benchmark_fingerprint=row["benchmark_fingerprint"],
        code_revision=row["code_revision"],
        source_artifact_sha256=row["runtime_lock_source_artifact_sha256"],
        authorization_artifact_sha256=(
            row["runtime_lock_authorization_artifact_sha256"]
        ),
        binding_sha256=row["runtime_lock_binding_sha256"],
    )


if __name__ == "__main__":
    unittest.main()
