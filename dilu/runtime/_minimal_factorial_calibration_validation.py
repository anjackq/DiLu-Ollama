"""Bundle validation for matched non-LLM calibration evidence."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CalibrationValidation:
    valid: bool
    expected_rows: int
    observed_rows: int
    errors: tuple[str, ...]


def validate_baseline_bundle(
    report_path: Path,
    episodes_path: Path,
    contract: Any,
) -> CalibrationValidation:
    errors: list[str] = []
    try:
        report = _load_object(Path(report_path))
        report_rows = _rows(report.get("episodes"), "report episodes")
        if report.get("artifact_type") != "matched_non_llm_calibration_report_v1":
            errors.append("report artifact type drifted")
        if tuple(report.get("baselines") or ()) != contract.policies:
            errors.append("report baseline policy set drifted")
        if report.get("episode_count") != contract.expected_rows:
            errors.append("report episode count drifted")
        if report.get("safety_shields_enabled") is not True:
            errors.append("report shield declaration drifted")
        if report.get("provenance") != dict(contract.provenance):
            errors.append("report top-level provenance drifted")
    except Exception as exc:
        report_rows = ()
        errors.append(f"report invalid: {exc}")
    try:
        with Path(episodes_path).open("r", newline="", encoding="utf-8") as handle:
            csv_rows = tuple(dict(row) for row in csv.DictReader(handle))
    except Exception as exc:
        csv_rows = ()
        errors.append(f"episode CSV invalid: {exc}")
    errors.extend(_row_errors(report_rows, contract, "report"))
    errors.extend(_row_errors(csv_rows, contract, "CSV"))
    if _identities(report_rows) != _identities(csv_rows):
        errors.append("report and CSV episode identities differ")
    elif _csv_projection(report_rows) != _csv_projection(csv_rows):
        errors.append("report and CSV episode contents differ")
    return CalibrationValidation(
        not errors,
        contract.expected_rows,
        len(report_rows),
        tuple(errors),
    )


def _row_errors(
    rows: Sequence[Mapping[str, Any]],
    contract: Any,
    source: str,
) -> tuple[str, ...]:
    expected = {
        (policy, case_id, seed)
        for policy in contract.policies
        for case_id, seed in contract.case_seeds
    }
    observed = _identities(rows)
    errors: list[str] = []
    if len(rows) != contract.expected_rows or observed != expected:
        errors.append(f"{source} calibration denominator mismatch")
    if len(observed) != len(rows):
        errors.append(f"{source} contains duplicate baseline rows")
    for row in rows:
        if (
            row.get("error") not in (None, "")
            or row.get("episode_stop_reason") == "error"
        ):
            errors.append(f"{source} contains an evaluator error row")
            break
        if not _boolean(row.get("safety_shields_enabled")):
            errors.append(f"{source} contains an unshielded row")
            break
        if row.get("balanced_driving_score_policy_version") != contract.scoring_version:
            errors.append(f"{source} scoring policy version drifted")
            break
        if any(
            str(row.get(key)) != value for key, value in contract.provenance.items()
        ):
            errors.append(f"{source} provenance drifted")
            break
    return tuple(errors)


def _identities(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, int]]:
    identities: set[tuple[str, str, int]] = set()
    for row in rows:
        try:
            identities.add(
                (
                    str(row.get("baseline_policy")),
                    str(row.get("case_id")),
                    int(row.get("simulator_seed")),
                )
            )
        except (TypeError, ValueError):
            continue
    return identities


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object.")
    return dict(value)


def _csv_projection(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    keys = sorted({key for row in rows for key in row})
    projected = []
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                encoded = json.dumps(value, sort_keys=True)
            elif value is None:
                encoded = ""
            else:
                encoded = str(value)
            values.append((key, encoded))
        projected.append(tuple(values))
    return tuple(sorted(projected))


def _rows(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError(f"{name} must be a list of objects.")
    return tuple(value)


def _boolean(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


__all__ = ["CalibrationValidation", "validate_baseline_bundle"]
