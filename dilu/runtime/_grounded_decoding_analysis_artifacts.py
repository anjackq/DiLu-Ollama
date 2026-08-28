"""Deterministic atomic publication for the V8 grounded-decoding analysis bundle.

Reuses the generic CSV/JSON row writers already proven by the V5/V7
registered analysis (``_write_csv``/``_write_json`` from
``_minimal_factorial_analysis_artifacts``) instead of reimplementing atomic
file writing; only the bundle *shape* is new here, because V8's table set
(Families A-D plus one descriptive table) does not match the V5/V7
``condition_summary``/``factor_contrasts``/... layout that module's
``EXACT_SUCCESS_FILES`` is pinned to.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._grounded_decoding_analysis_validation import GroundedAnalysisValidation
from ._minimal_factorial_analysis_artifacts import _write_csv, _write_json

TABLE_FILES = (
    "family_a_contrasts.csv",
    "family_b_contrasts.csv",
    "family_c_contrasts.csv",
    "family_d_contrasts.csv",
    "descriptive_secondary_outcomes.csv",
)
EXACT_SUCCESS_FILES = frozenset(
    {"analysis_validation.json", "analysis-report.md", "stats-appendix.md", *TABLE_FILES}
)


@dataclass(frozen=True)
class V8AnalysisTables:
    family_a: tuple[Mapping[str, Any], ...]
    family_b: tuple[Mapping[str, Any], ...]
    family_c: tuple[Mapping[str, Any], ...]
    family_d: tuple[Mapping[str, Any], ...]
    descriptive: tuple[Mapping[str, Any], ...]
    analysis_report: str
    stats_appendix: str


def publish_v8_analysis_bundle(
    output_root: Path,
    validation: GroundedAnalysisValidation,
    tables: V8AnalysisTables | None = None,
) -> Path:
    target = Path(output_root).resolve()
    if target.exists():
        raise FileExistsError(f"Analysis output already exists: {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-stage-", dir=target.parent))
    try:
        payload = {
            "status": validation.status,
            "errors": sorted(set(validation.errors)),
            "contrast_artifacts_written": validation.contrast_artifacts_written,
        }
        _write_json(stage / "analysis_validation.json", payload)
        if payload["status"] == "complete":
            if tables is None or payload["contrast_artifacts_written"] is not True:
                raise ValueError(
                    "Complete V8 analysis publication requires complete tables."
                )
            _write_tables(stage, tables)
            observed = _relative_files(stage)
            if observed != EXACT_SUCCESS_FILES:
                raise RuntimeError(f"V8 analysis artifact layout drifted: {sorted(observed)}.")
        elif tables is not None or payload["contrast_artifacts_written"] is not False:
            raise ValueError("Blocked V8 analysis must not publish contrast artifacts.")
        elif _relative_files(stage) != {"analysis_validation.json"}:
            raise RuntimeError("Blocked V8 analysis staging contains claim artifacts.")
        os.rename(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target / "analysis_validation.json"


def _write_tables(root: Path, tables: V8AnalysisTables) -> None:
    rows_by_name = {
        "family_a_contrasts.csv": tables.family_a,
        "family_b_contrasts.csv": tables.family_b,
        "family_c_contrasts.csv": tables.family_c,
        "family_d_contrasts.csv": tables.family_d,
        "descriptive_secondary_outcomes.csv": tables.descriptive,
    }
    for name, rows in rows_by_name.items():
        _write_csv(root / name, rows)
    (root / "analysis-report.md").write_text(
        tables.analysis_report, encoding="utf-8", newline="\n"
    )
    (root / "stats-appendix.md").write_text(
        tables.stats_appendix, encoding="utf-8", newline="\n"
    )


def _relative_files(root: Path) -> frozenset[str]:
    return frozenset(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


__all__ = ["EXACT_SUCCESS_FILES", "TABLE_FILES", "V8AnalysisTables", "publish_v8_analysis_bundle"]
