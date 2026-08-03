from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from statistics import fmean
from unittest import mock

from dilu.runtime.minimal_factorial_analysis import (
    BootstrapInterval,
    _draw_stratified,
    derive_bootstrap_seed,
    endpoint_contrast,
    factorial_contrasts,
    stratified_bootstrap,
)
from dilu.runtime._minimal_factorial_analysis_tables import compute_registered_tables
from dilu.runtime._minimal_factorial_analysis_io import (
    _enrich_trace_metrics,
    _trace_metrics,
    _validate_trace_join,
)
from tests.minimal_factorial_analysis_support import synthetic_analysis_bundle


class _RecordingRandom:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, ...], int]] = []

    def choices(self, population: object, *, k: int) -> list[float]:
        values = tuple(float(value) for value in population)
        self.calls.append((values, k))
        return [values[0]] * k


def _factorial_values() -> dict[str, float]:
    values = {}
    for policy in (0, 1):
        for output in (0, 1):
            for execution in (0, 1):
                values[f"c{policy}{output}{execution}"] = (
                    1.0
                    + 2.0 * policy
                    + 3.0 * output
                    + 5.0 * execution
                    + 7.0 * policy * output
                    + 11.0 * policy * execution
                    + 13.0 * output * execution
                    + 17.0 * policy * output * execution
                )
    return values


class MinimalFactorialContrastTests(unittest.TestCase):
    def test_registered_averaged_factorial_formulas(self) -> None:
        contrasts = factorial_contrasts(_factorial_values())

        self.assertEqual(
            contrasts,
            {
                "P_MAIN": 15.25,
                "O_MAIN": 17.25,
                "E_MAIN": 21.25,
                "PO_INTERACTION": 15.5,
                "PE_INTERACTION": 19.5,
                "OE_INTERACTION": 21.5,
                "POE_INTERACTION": 17.0,
            },
        )
        self.assertEqual(endpoint_contrast(_factorial_values()), 58.0)


class MinimalFactorialBootstrapTests(unittest.TestCase):
    def test_seed_uses_registered_sha256_material(self) -> None:
        material = "manifest|qwen|P_MAIN|completion|bootstrap-v1"
        expected = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")

        observed = derive_bootstrap_seed(
            "manifest",
            "qwen",
            "P_MAIN",
            "completion",
        )

        self.assertEqual(observed, expected)

    def test_stratified_draw_uses_registered_category_sample_sizes(self) -> None:
        categories = {f"category-{index}": (float(index),) * 12 for index in range(10)}
        stage1_rng = _RecordingRandom()
        endpoint_rng = _RecordingRandom()

        _draw_stratified(categories, stage1_rng, samples_per_category=3)
        _draw_stratified(categories, endpoint_rng, samples_per_category=12)

        self.assertEqual(len(stage1_rng.calls), 10)
        self.assertEqual({call[1] for call in stage1_rng.calls}, {3})
        self.assertEqual(len(endpoint_rng.calls), 10)
        self.assertEqual({call[1] for call in endpoint_rng.calls}, {12})

    def test_bootstrap_is_20k_draw_byte_stable_fixed_suite_sensitivity(self) -> None:
        categories = {
            f"category-{index}": tuple(float(index + offset) for offset in range(3))
            for index in range(10)
        }
        seed = derive_bootstrap_seed(
            "manifest",
            "qwen",
            "P_MAIN",
            "driving_score_balanced_v1",
        )

        first = stratified_bootstrap(
            categories,
            samples_per_category=3,
            draws=20_000,
            seed=seed,
        )
        second = stratified_bootstrap(
            categories,
            samples_per_category=3,
            draws=20_000,
            seed=seed,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.draws, 20_000)
        self.assertEqual(first.evidence_scope, "fixed-suite sensitivity interval")
        self.assertNotIn("population", first.evidence_scope)


class MinimalFactorialTableTests(unittest.TestCase):
    def test_primary_snapshot_trace_join_is_fail_closed(self) -> None:
        episode = {
            "campaign_id": "campaign",
            "condition_id": "c000",
            "case_id": "case-001",
            "pair_id": "pair-001",
            "template_id": "template-001",
            "replicate_id": 0,
            "simulator_seed": 17,
            "primary_snapshot_id": "snapshot-registered",
            "benchmark_fingerprint": "sha256:" + "a" * 64,
            "code_revision": "b" * 40,
            "config_sha256": "sha256:" + "c" * 64,
            "prompt_sha256": "sha256:" + "d" * 64,
            "model_tag": "qwen3:0.6b",
            "model_digest": "sha256:" + "e" * 64,
            "capability_artifact_sha256": "sha256:" + "f" * 64,
            "capability_snapshot_sha256": "sha256:" + "1" * 64,
            "condition": {"transport": {"think_mode": "no_think"}},
        }
        record = {
            "trace_key": {
                key: episode[key]
                for key in (
                    "campaign_id",
                    "condition_id",
                    "case_id",
                    "pair_id",
                    "template_id",
                    "replicate_id",
                )
            }
            | {"decision_index": 0},
            "context": {
                "simulator_seed": episode["simulator_seed"],
                "decision_snapshot_id": "snapshot-wrong",
                "benchmark_fingerprint": episode["benchmark_fingerprint"],
                "code_revision": episode["code_revision"],
            },
            "config_sha256": episode["config_sha256"],
            "prompt": {"prompt_sha256": episode["prompt_sha256"]},
            "generation": {
                "request": {
                    "model_tag": episode["model_tag"],
                    "model_digest": episode["model_digest"],
                    "native_endpoint": "http://localhost:11434/api/chat",
                    "think_mode": "no_think",
                },
                "transport_evidence": {
                    "capability_artifact_sha256": episode["capability_artifact_sha256"],
                    "capability_snapshot_sha256": episode["capability_snapshot_sha256"],
                },
            },
        }

        with self.assertRaisesRegex(ValueError, "decision_snapshot_id"):
            _validate_trace_join(
                record,
                episode,
                "http://localhost:11434/api/chat",
            )

    def test_trace_reference_semantic_hash_mismatch_is_blocked(self) -> None:
        reference = {
            "relative_path": "traces/decision_traces.jsonl",
            "line_number": 1,
            "record_sha256": "sha256:" + "a" * 64,
            "schema_version": "iclr2027.scientific_trace.v1",
            "schema_sha256": "sha256:" + "b" * 64,
        }
        actual_reference = mock.Mock(line_number=1)
        actual_reference.to_dict.return_value = {
            **reference,
            "record_sha256": "sha256:" + "c" * 64,
        }
        record = {
            "trace_key": {"episode_attempt_id": "episode-001"},
            "shield_stack": {
                "proposed_action_id": 0,
                "executed_action_id": 0,
                "stages": [
                    {"stage_name": name, "applied": False}
                    for name in (
                        "lane_change",
                        "longitudinal_safety",
                        "low_speed_recovery",
                    )
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces" / "decision_traces.jsonl"
            trace_path.parent.mkdir()
            trace_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            writer = mock.Mock()
            writer.reference_snapshot.return_value = (actual_reference,)
            with (
                mock.patch(
                    "dilu.runtime._minimal_factorial_analysis_io.ScientificTraceWriter",
                    return_value=writer,
                ),
                self.assertRaisesRegex(ValueError, "validated record"),
            ):
                _enrich_trace_metrics(
                    (
                        {
                            "episode_attempt_id": "episode-001",
                            "scientific_trace_references": [reference],
                        },
                    ),
                    trace_path,
                    native_endpoint="http://localhost:11434/api/chat",
                )

    def test_trace_metrics_count_any_stage_once_and_proposal_changes(self) -> None:
        def record(
            applied: tuple[bool, bool, bool],
            proposed: int,
            executed: int,
        ) -> dict[str, object]:
            return {
                "shield_stack": {
                    "proposed_action_id": proposed,
                    "executed_action_id": executed,
                    "stages": [
                        {"stage_name": name, "applied": flag}
                        for name, flag in zip(
                            (
                                "lane_change",
                                "longitudinal_safety",
                                "low_speed_recovery",
                            ),
                            applied,
                        )
                    ],
                }
            }

        metrics = _trace_metrics(
            [
                record((True, True, False), 0, 1),
                record((False, False, True), 1, 1),
            ]
        )

        self.assertEqual(metrics["analysis_any_shield_intervention_count"], 2)
        self.assertEqual(metrics["analysis_lane_change_shield_count"], 1)
        self.assertEqual(metrics["analysis_longitudinal_safety_shield_count"], 1)
        self.assertEqual(metrics["analysis_low_speed_recovery_shield_count"], 1)
        self.assertEqual(metrics["analysis_proposal_action_change_count"], 1)

    def test_registered_tables_are_model_separated_complete_and_provenanced(
        self,
    ) -> None:
        claim, episodes, baselines = synthetic_analysis_bundle()

        def interval(
            values: object, *, draws: int, seed: int, **_kwargs: object
        ) -> BootstrapInterval:
            flattened = [value for category in values.values() for value in category]
            effect = fmean(flattened)
            return BootstrapInterval(
                effect,
                effect,
                effect,
                draws,
                seed,
                "fixed-suite sensitivity interval",
            )

        with mock.patch(
            "dilu.runtime._minimal_factorial_analysis_tables.stratified_bootstrap",
            side_effect=interval,
        ):
            tables = compute_registered_tables(
                claim,
                episodes,
                baselines,
                manifest_sha256="sha256:" + "a" * 64,
            )

        self.assertEqual(len(tables.condition_summary), 192)
        self.assertEqual(len(tables.factor_contrasts), 168)
        self.assertEqual(len(tables.endpoint_contrasts), 24)
        self.assertEqual(len(tables.calibration_contrasts), 18)
        self.assertEqual(len(tables.category_summary), 1920)
        provenance = {
            "campaign_id",
            "manifest_sha256",
            "case_set_sha256",
            "selected_30_sha256",
            "config_sha256",
            "runtime_lock_sha256",
            "source_revision",
            "trace_schema_sha256",
            "scoring_sha256",
            "environment_sha256",
            "model_digest",
        }
        for row in (
            tables.factor_contrasts
            + tables.endpoint_contrasts
            + tables.calibration_contrasts
        ):
            self.assertTrue(provenance.issubset(row))
            self.assertIn("numerator", row)
            self.assertIn("denominator", row)
            self.assertNotIn("p_value", row)
        self.assertEqual(
            {row["denominator"] for row in tables.factor_contrasts},
            {30},
        )
        self.assertEqual(
            {row["denominator"] for row in tables.endpoint_contrasts},
            {120},
        )
        manifest_sha256 = "sha256:" + "a" * 64
        for row in tables.calibration_contrasts:
            subject = row["bootstrap_seed_subject"]
            self.assertEqual(
                row["bootstrap_seed"],
                derive_bootstrap_seed(
                    manifest_sha256,
                    subject,
                    row["contrast_id"],
                    row["outcome"],
                ),
            )


if __name__ == "__main__":
    unittest.main()
