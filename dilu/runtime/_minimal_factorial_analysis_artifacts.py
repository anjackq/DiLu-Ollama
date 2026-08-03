"""Deterministic atomic publication for registered analysis artifacts."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

TABLE_FILES = (
    "condition_summary.csv",
    "factor_contrasts.csv",
    "endpoint_contrasts.csv",
    "calibration_contrasts.csv",
    "category_summary.csv",
)
EXACT_SUCCESS_FILES = frozenset(
    {
        "analysis_validation.json",
        *TABLE_FILES,
        "analysis-report.md",
        "stats-appendix.md",
        "figure-data/registered_contrasts.csv",
    }
)


@dataclass(frozen=True)
class AnalysisTables:
    condition_summary: tuple[Mapping[str, Any], ...]
    factor_contrasts: tuple[Mapping[str, Any], ...]
    endpoint_contrasts: tuple[Mapping[str, Any], ...]
    calibration_contrasts: tuple[Mapping[str, Any], ...]
    category_summary: tuple[Mapping[str, Any], ...]
    analysis_report: str
    stats_appendix: str


def publish_analysis_bundle(
    output_root: Path,
    validation: Any,
    tables: AnalysisTables | None = None,
) -> Path:
    target = Path(output_root).resolve()
    if target.exists():
        raise FileExistsError(f"Analysis output already exists: {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-stage-", dir=target.parent))
    try:
        payload = _validation_payload(validation)
        _write_json(stage / "analysis_validation.json", payload)
        if payload["status"] == "complete":
            if tables is None or payload["contrast_artifacts_written"] is not True:
                raise ValueError(
                    "Complete analysis publication requires complete tables."
                )
            _write_tables(stage, tables)
            observed = _relative_files(stage)
            if observed != EXACT_SUCCESS_FILES:
                raise RuntimeError(
                    f"Analysis artifact layout drifted: {sorted(observed)}."
                )
        elif tables is not None or payload["contrast_artifacts_written"] is not False:
            raise ValueError("Blocked analysis must not publish contrast artifacts.")
        elif _relative_files(stage) != {"analysis_validation.json"}:
            raise RuntimeError("Blocked analysis staging contains claim artifacts.")
        os.rename(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target / "analysis_validation.json"


def _write_tables(root: Path, tables: AnalysisTables) -> None:
    rows_by_name = {
        "condition_summary.csv": tables.condition_summary,
        "factor_contrasts.csv": tables.factor_contrasts,
        "endpoint_contrasts.csv": tables.endpoint_contrasts,
        "calibration_contrasts.csv": tables.calibration_contrasts,
        "category_summary.csv": tables.category_summary,
    }
    for name, rows in rows_by_name.items():
        _write_csv(root / name, rows)
    (root / "analysis-report.md").write_text(
        tables.analysis_report,
        encoding="utf-8",
        newline="\n",
    )
    (root / "stats-appendix.md").write_text(
        tables.stats_appendix,
        encoding="utf-8",
        newline="\n",
    )
    figure_rows = tuple(
        tables.factor_contrasts
        + tables.endpoint_contrasts
        + tables.calibration_contrasts
    )
    _write_csv(root / "figure-data" / "registered_contrasts.csv", figure_rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Registered table cannot be empty: {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validation_payload(validation: Any) -> dict[str, Any]:
    value = asdict(validation) if is_dataclass(validation) else dict(validation)
    errors = sorted({str(error) for error in value.get("errors", ())})
    payload = {
        "status": value.get("status"),
        "errors": errors,
        "contrast_artifacts_written": value.get("contrast_artifacts_written"),
    }
    if payload["status"] == "complete":
        counts = {
            "expected_episode_rows": value.get("expected_episode_rows"),
            "observed_episode_rows": value.get("observed_episode_rows"),
            "expected_baseline_rows": value.get("expected_baseline_rows"),
            "observed_baseline_rows": value.get("observed_baseline_rows"),
        }
        if counts != {
            "expected_episode_rows": 840,
            "observed_episode_rows": 840,
            "expected_baseline_rows": 360,
            "observed_baseline_rows": 360,
        }:
            raise ValueError(
                "Complete analysis requires exact registered denominators."
            )
        payload.update(
            {
                **counts,
                "hash_gates": [
                    "campaign",
                    "registered_manifest",
                    "case_set",
                    "selected_30",
                    "runtime_snapshot",
                    "model_digest",
                    "config",
                    "authorized_runtime_lock",
                    "source_revision",
                    "trace_schema",
                    "scoring",
                    "environment",
                    "baseline",
                ],
            }
        )
    return payload


def _relative_files(root: Path) -> frozenset[str]:
    return frozenset(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


__all__ = [
    "AnalysisTables",
    "EXACT_SUCCESS_FILES",
    "publish_analysis_bundle",
]
