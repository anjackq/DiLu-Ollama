from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dilu.runtime._qwen_stage1_240_analysis import (
    CLAIM_ELIGIBLE,
    EXPECTED_ROWS,
    build_qwen_stage1_tables,
    publish_qwen_stage1_bundle,
)


def _episodes() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for condition_index in range(8):
        for case_index in range(30):
            rows.append(
                {
                    "campaign_id": "iclr2027-minimal-factorial-v4",
                    "model_slot": "qwen_06b",
                    "model_digest": "sha256:" + "a" * 64,
                    "stage": "stage1",
                    "condition_id": f"c{condition_index:03b}",
                    "case_id": f"case-{case_index:02d}",
                    "category": f"category-{case_index // 3}",
                    "driving_score_balanced_v1": case_index / 30,
                    "task_completed": case_index % 2 == 0,
                    "crashed": case_index % 3 == 0,
                    "responses_strict_format": 2,
                    "decisions_made": 2,
                    "fallback_action_count": 0,
                    "decision_calls_total": 2,
                    "decision_timeout_count": 0,
                    "decision_latency_ms_avg": 10.0 + condition_index,
                }
            )
    return tuple(rows)


class QwenStage1240AnalysisTests(unittest.TestCase):
    def test_fixed_tables_have_only_registered_fast_outcomes(self) -> None:
        episodes = _episodes()
        tables = build_qwen_stage1_tables(
            episodes,
            provenance={
                "campaign_id": "iclr2027-minimal-factorial-v4",
                "analysis_scope": "qwen_stage1_240",
                "claim_eligible": "false",
            },
            manifest_sha256="sha256:" + "b" * 64,
        )

        self.assertEqual(EXPECTED_ROWS, 240)
        self.assertFalse(CLAIM_ELIGIBLE)
        self.assertEqual(len(tables.condition_summary), 56)
        self.assertEqual(len(tables.factor_contrasts), 49)
        self.assertEqual(
            {row["outcome"] for row in tables.condition_summary},
            {
                "driving_score_balanced_v1",
                "task_completion",
                "crash",
                "strict_format_rate",
                "fallback_rate",
                "timeout_rate",
                "decision_latency_ms_avg",
            },
        )
        reversed_tables = build_qwen_stage1_tables(
            tuple(reversed(episodes)),
            provenance={
                "campaign_id": "iclr2027-minimal-factorial-v4",
                "analysis_scope": "qwen_stage1_240",
                "claim_eligible": "false",
            },
            manifest_sha256="sha256:" + "b" * 64,
        )
        self.assertEqual(tables, reversed_tables)

    def test_publisher_has_exact_three_file_layout(self) -> None:
        tables = build_qwen_stage1_tables(
            _episodes(),
            provenance={"analysis_scope": "qwen_stage1_240"},
            manifest_sha256="sha256:" + "b" * 64,
        )
        validation = {
            "status": "complete",
            "errors": [],
            "scope": "qwen_stage1_240",
            "claim_eligible": False,
            "expected_rows": 240,
            "observed_rows": 240,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "analysis"
            path = publish_qwen_stage1_bundle(root, validation, tables)

            self.assertEqual(
                {item.name for item in root.iterdir()},
                {"validation.json", "condition_summary.csv", "factor_contrasts.csv"},
            )
            self.assertEqual(json.loads(path.read_text())["status"], "complete")
            with (root / "factor_contrasts.csv").open(newline="") as handle:
                self.assertEqual(len(tuple(csv.DictReader(handle))), 49)

    def test_blocked_publisher_writes_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "analysis"
            publish_qwen_stage1_bundle(
                root,
                {
                    "status": "blocked",
                    "errors": ["missing row"],
                    "scope": "qwen_stage1_240",
                    "claim_eligible": False,
                    "expected_rows": 240,
                    "observed_rows": 239,
                },
            )
            self.assertEqual({item.name for item in root.iterdir()}, {"validation.json"})

    def test_cli_exposes_no_denominator_or_scope_switches(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/analyze_iclr2027_qwen_stage1_240.py", "--help"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-root", result.stdout)
        for forbidden in ("--model", "--stage", "--expected-rows", "--scope"):
            self.assertNotIn(forbidden, result.stdout)


if __name__ == "__main__":
    unittest.main()
