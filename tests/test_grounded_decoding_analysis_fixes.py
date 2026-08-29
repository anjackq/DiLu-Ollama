"""Fix-verification tests from the Task 6 review round.

Split out of ``test_grounded_decoding_analysis.py`` to keep both files
under the 400-line-per-file limit. Covers, one class per finding:

- CRITICAL 1: ``SECONDARY_OUTCOMES`` must only read fields that exist on a
  real episode row (the old "shield_intervention_rate" did not), and the
  computed replacement must reach *both* sides of the paired V8-minus-O1
  difference, not just the V8 side.
- CRITICAL 2: the V8 episode<->trace join must fail closed (zero-match
  episode, wrong trace file / identity drift, malformed trace_key all
  raise -- none may silently default to zero).
- IMPORTANT 3: the descriptive O2-vs-O1 executed-action distribution
  shift, previously unimplemented.
- MINOR 4: Family M must only sum over O2 (c120/c121) condition ids, so a
  registered digest-drift-contingency O1 rerun row appended to the V8
  schedule can never trip it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dilu.runtime._grounded_decoding_analysis_action_shift import (
    compute_action_distribution_shift,
)
from dilu.runtime._grounded_decoding_analysis_families import SECONDARY_OUTCOMES
from dilu.runtime._grounded_decoding_analysis_validation import _family_m_errors
from dilu.runtime.grounded_decoding_analysis import run_registered_v8_analysis
from scripts.analyze_iclr2027_grounded_decoding import (
    _enrich_frozen_o1_rows,
    _enrich_v8_rows,
    _frozen_o1_rows,
)
from tests.grounded_decoding_analysis_support import MODEL_SLOTS, build_v8_fixture

ROOT = Path(__file__).resolve().parents[1]
V7_REAL_EPISODES = (
    ROOT / "results" / "iclr2027_model_breadth_factorial_v7" / "llm_campaign" / "episodes.jsonl"
)
# The smoke campaign is a small, real (non-synthetic) frozen O1 fixture --
# fast enough to join in a unit test, unlike the multi-hundred-MB llm_campaign
# trace file.
V5_SMOKE_EPISODES = (
    ROOT / "results" / "iclr2027_minimal_factorial_v5" / "smoke" / "episodes.jsonl"
)
V5_SMOKE_TRACES = (
    ROOT
    / "results"
    / "iclr2027_minimal_factorial_v5"
    / "smoke"
    / "traces"
    / "decision_traces.jsonl"
)
MANIFEST_SHA256 = "sha256:" + "1" * 64


class RealEpisodeSchemaGuardTests(unittest.TestCase):
    """Fix for CRITICAL 1: guards SECONDARY_OUTCOMES against invented fields.

    Reads one real row straight from the frozen V7 episodes.jsonl and
    asserts every non-CLI-enrichment field SECONDARY_OUTCOMES reads
    actually exists there, so a field like the old "shield_intervention_rate"
    (never a real episode field -- the real ones are
    lane_change_shield_rate/longitudinal_safety_shield_rate/
    flow_recovery_shield_rate) cannot silently reappear.
    """

    def test_secondary_outcome_fields_exist_on_a_real_frozen_episode_row(self) -> None:
        with V7_REAL_EPISODES.open("r", encoding="utf-8") as handle:
            real_row = json.loads(handle.readline())
        for outcome, (field, _direction) in SECONDARY_OUTCOMES.items():
            with self.subTest(outcome=outcome, field=field):
                if field.startswith("analysis_"):
                    # CLI-enrichment field (computed from decision_traces.jsonl,
                    # see scripts/analyze_iclr2027_grounded_decoding.py); not
                    # expected to exist on a raw, un-enriched episode row.
                    continue
                self.assertIn(field, real_row, f"{field!r} is not a real episode field")


class FrozenComparatorEnrichmentTests(unittest.TestCase):
    """Fix for CRITICAL 1 (join completeness): the O1 side needs the field too.

    ``_grounded_decoding_analysis_families._RowIndex._o1_value`` reads
    ``analysis_shield_intervention_rate`` off the *frozen* comparator row,
    not just the V8 row, when computing the paired shield-intervention-rate
    difference. A raw frozen row (straight from ``episodes.jsonl``) never
    carries that field -- only ``_enrich_frozen_o1_rows`` (the CLI's O1-side
    counterpart to ``_enrich_v8_rows``) adds it. Skipping that enrichment
    would raise ``KeyError`` the first time the shield-intervention-rate
    descriptive contrast actually ran against real data.
    """

    def test_real_frozen_o1_rows_lack_the_field_before_enrichment(self) -> None:
        raw_rows = _frozen_o1_rows(V7_REAL_EPISODES)
        self.assertTrue(raw_rows)
        self.assertNotIn("analysis_shield_intervention_rate", raw_rows[0])

    def test_enrich_frozen_o1_rows_against_real_v5_smoke_data(self) -> None:
        raw_rows = _frozen_o1_rows(V5_SMOKE_EPISODES)
        self.assertTrue(raw_rows, "expected at least one real O1 row in the smoke campaign")
        enriched = _enrich_frozen_o1_rows(raw_rows, V5_SMOKE_TRACES)
        self.assertEqual(len(enriched), len(raw_rows))
        for row in enriched:
            rate = row["analysis_shield_intervention_rate"]
            self.assertIsInstance(rate, float)
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)


class V8TraceJoinFailClosedTests(unittest.TestCase):
    """Fix for CRITICAL 2: the V8 episode<->trace join must fail closed."""

    def _episode(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "episode_attempt_id": "episode-abc",
            "model_tag": "qwen3:0.6b",
            "model_digest": "sha256:" + "0" * 64,
            "case_id": "case1",
            "simulator_seed": 1234,
            "decision_calls_total": 5,
        }
        row.update(overrides)
        return row

    def _trace_record(self, **overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "trace_key": {
                "episode_attempt_id": "episode-abc",
                "case_id": "case1",
                "condition_id": "c121",
            },
            "context": {"simulator_seed": 1234},
            "generation": {
                "request": {"model_tag": "qwen3:0.6b", "model_digest": "sha256:" + "0" * 64}
            },
            "action_resolution": {"violation": None, "final_resolved_action": 1},
            "shield_stack": {
                "stages": [
                    {"stage_name": name, "applied": False}
                    for name in ("lane_change", "longitudinal_safety", "low_speed_recovery")
                ]
            },
        }
        record.update(overrides)
        return record

    def _write_traces(self, directory: Path, records: list[dict[str, object]]) -> Path:
        path = directory / "decision_traces.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return path

    def test_zero_matching_trace_records_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = self._write_traces(Path(directory), [])
            with self.assertRaises(ValueError) as ctx:
                _enrich_v8_rows([self._episode()], trace_path)
            self.assertIn("no decision trace records found", str(ctx.exception))

    def test_wrong_trace_file_identity_drift_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = self._trace_record(
                generation={
                    "request": {
                        "model_tag": "llama3.2:1b",
                        "model_digest": "sha256:" + "9" * 64,
                    }
                }
            )
            trace_path = self._write_traces(Path(directory), [record])
            with self.assertRaises(ValueError) as ctx:
                _enrich_v8_rows([self._episode()], trace_path)
            self.assertIn("identity drifted", str(ctx.exception))

    def test_malformed_trace_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = self._write_traces(
                Path(directory), [{"trace_key": {"case_id": "case1"}}]
            )
            with self.assertRaises(ValueError) as ctx:
                _enrich_v8_rows([self._episode()], trace_path)
            self.assertIn("trace_key", str(ctx.exception))

    def test_valid_join_computes_counts_shield_rate_and_action_histogram(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unavailable_record = self._trace_record(
                action_resolution={"violation": "action_unavailable", "final_resolved_action": 2},
                shield_stack={
                    "stages": [
                        {"stage_name": "lane_change", "applied": True},
                        {"stage_name": "longitudinal_safety", "applied": False},
                        {"stage_name": "low_speed_recovery", "applied": False},
                    ]
                },
            )
            clean_record = self._trace_record(
                action_resolution={"violation": None, "final_resolved_action": 1}
            )
            trace_path = self._write_traces(Path(directory), [unavailable_record, clean_record])
            enriched, histograms = _enrich_v8_rows(
                [self._episode(decision_calls_total=2)], trace_path
            )
            self.assertEqual(enriched[0]["analysis_action_unavailable_count"], 1)
            self.assertAlmostEqual(enriched[0]["analysis_shield_intervention_rate"], 0.5)
            self.assertEqual(dict(histograms["qwen3:0.6b"]), {2: 1, 1: 1})


class ActionDistributionShiftTests(unittest.TestCase):
    """Fix for IMPORTANT 3: descriptive O2-vs-O1 executed-action distribution shift."""

    def test_shares_and_shift_computed_per_model_and_action(self) -> None:
        o2_counts = {"qwen_06b": {1: 9, 2: 1}}
        o1_counts = {"qwen_06b": {1: 5, 2: 5}}
        rows = compute_action_distribution_shift(o2_counts, o1_counts)
        by_action = {row["action_id"]: row for row in rows}
        self.assertAlmostEqual(by_action[1]["o2_share"], 0.9)
        self.assertAlmostEqual(by_action[1]["o1_share"], 0.5)
        self.assertAlmostEqual(by_action[1]["share_shift"], 0.4)
        self.assertAlmostEqual(by_action[2]["o2_share"], 0.1)
        self.assertAlmostEqual(by_action[2]["o1_share"], 0.5)
        self.assertAlmostEqual(by_action[2]["share_shift"], -0.4)
        for row in rows:
            self.assertEqual(
                row["evidence_scope"],
                "descriptive fixed-suite action-distribution summary, no test",
            )
            self.assertNotIn("p_value", row)
            self.assertNotIn("p_holm", row)

    def test_wired_into_run_registered_v8_analysis_result(self) -> None:
        v8_rows, frozen_rows = build_v8_fixture()
        o2_counts = {slot: {1: 10} for slot in MODEL_SLOTS}
        o1_counts = {slot: {1: 8, 2: 2} for slot in MODEL_SLOTS}
        result = run_registered_v8_analysis(
            v8_rows,
            frozen_rows,
            manifest_sha256=MANIFEST_SHA256,
            o2_action_counts=o2_counts,
            o1_action_counts=o1_counts,
        )
        self.assertTrue(result.action_distribution)
        self.assertEqual(
            {row["model_slot"] for row in result.action_distribution}, set(MODEL_SLOTS)
        )

    def test_omitted_histograms_default_to_empty_action_distribution(self) -> None:
        v8_rows, frozen_rows = build_v8_fixture()
        result = run_registered_v8_analysis(
            v8_rows, frozen_rows, manifest_sha256=MANIFEST_SHA256
        )
        self.assertEqual(result.action_distribution, ())


class FamilyMConditionFilterTests(unittest.TestCase):
    """Fix for MINOR 4: Family M must only sum over O2 (c120/c121) rows."""

    def test_family_m_ignores_appended_o1_rerun_rows(self) -> None:
        rows = [
            {
                "model_slot": "qwen_06b",
                "condition_id": "c121",
                "analysis_action_unavailable_count": 0,
            },
            # A registered digest-drift-contingency rerun row: O1, appended
            # to the V8 schedule. A nonzero count here must NOT trip the
            # "under O2" gate.
            {
                "model_slot": "qwen_06b",
                "condition_id": "c111",
                "analysis_action_unavailable_count": 5,
            },
        ]
        self.assertEqual(_family_m_errors(rows), set())

    def test_family_m_still_fires_for_genuine_o2_violations(self) -> None:
        rows = [
            {
                "model_slot": "qwen_06b",
                "condition_id": "c121",
                "analysis_action_unavailable_count": 3,
            },
        ]
        errors = _family_m_errors(rows)
        self.assertTrue(any("Family M" in error and "qwen_06b" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
