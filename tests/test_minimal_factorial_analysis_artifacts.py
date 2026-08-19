from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime._minimal_factorial_analysis_artifacts import (
    EXACT_SUCCESS_FILES,
    AnalysisTables,
    publish_analysis_bundle,
)
from dilu.runtime._minimal_factorial_analysis_validation import AnalysisValidation
from dilu.runtime.minimal_factorial_analysis import (
    AnalysisInputPaths,
    run_registered_analysis,
)


def _tables() -> AnalysisTables:
    provenance = {"campaign_id": "iclr2027-minimal-factorial-v4"}
    contrast = {
        **provenance,
        "contrast_id": "P_MAIN",
        "outcome": "completion",
        "effect": 0.1,
        "lower_2_5": 0.0,
        "upper_97_5": 0.2,
    }
    return AnalysisTables(
        condition_summary=({**provenance, "condition_id": "c000", "mean": 0.5},),
        factor_contrasts=(contrast,),
        endpoint_contrasts=({**contrast, "contrast_id": "ENDPOINT"},),
        calibration_contrasts=({**contrast, "contrast_id": "CAL_QWEN"},),
        category_summary=({**provenance, "category": "merge", "mean": 0.5},),
        analysis_report="# Analysis report\n\nFixed-suite registered analysis.\n",
        stats_appendix="# Statistical appendix\n\nNo p-values were calculated.\n",
    )


def _files(root: Path) -> frozenset[str]:
    return frozenset(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


class MinimalFactorialAnalysisArtifactTests(unittest.TestCase):
    def test_public_runner_publishes_only_blocked_validation(self) -> None:
        validation = AnalysisValidation(
            "blocked", ("missing row",), False, 840, 839, 360, 360
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "analysis"
            with mock.patch(
                "dilu.runtime.minimal_factorial_analysis.load_analysis_inputs",
                return_value=(validation, None),
            ):
                path = run_registered_analysis(
                    AnalysisInputPaths(*(Path(name) for name in ("m", "e", "r", "b"))),
                    output_root=output,
                )

            self.assertEqual(_files(output), {"analysis_validation.json"})
            self.assertEqual(json.loads(path.read_text())["status"], "blocked")

    def test_cli_help_runs_from_repository_checkout(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/analyze_iclr2027_minimal_factorial.py", "--help"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--baseline-episodes", result.stdout)

    def test_blocked_publication_writes_only_canonical_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "analysis"

            validation_path = publish_analysis_bundle(
                output,
                {
                    "status": "blocked",
                    "errors": ["seed drift", "missing row", "seed drift"],
                    "contrast_artifacts_written": False,
                },
            )

            self.assertEqual(_files(output), {"analysis_validation.json"})
            self.assertEqual(
                json.loads(validation_path.read_text(encoding="utf-8")),
                {
                    "status": "blocked",
                    "errors": ["missing row", "seed drift"],
                    "contrast_artifacts_written": False,
                },
            )

    def test_complete_publication_has_exact_byte_stable_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "analysis-a"
            second = root / "analysis-b"
            validation = AnalysisValidation("complete", (), True, 840, 840, 360, 360)

            publish_analysis_bundle(first, validation, _tables())
            publish_analysis_bundle(second, validation, _tables())

            self.assertEqual(_files(first), EXACT_SUCCESS_FILES)
            with (first / "condition_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                condition_rows = tuple(csv.DictReader(handle))
            self.assertEqual(
                {row["campaign_id"] for row in condition_rows},
                {"iclr2027-minimal-factorial-v4"},
            )
            self.assertEqual(
                {
                    path.relative_to(first).as_posix(): path.read_bytes()
                    for path in first.rglob("*")
                    if path.is_file()
                },
                {
                    path.relative_to(second).as_posix(): path.read_bytes()
                    for path in second.rglob("*")
                    if path.is_file()
                },
            )


if __name__ == "__main__":
    unittest.main()
