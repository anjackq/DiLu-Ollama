from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from ._minimal_factorial_schedule_support import canonical_sha256
from ._minimal_factorial_analysis_design import ENDPOINTS, STAGES, category_errors
from .minimal_factorial_calibration import build_calibration_contract

EXPECTED_EPISODE_ROWS = 840
EXPECTED_BASELINE_ROWS = 360
_SHA256_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_RUNTIME_HASH_FIELDS = (
    "runtime_lock_source_artifact_sha256",
    "runtime_lock_authorization_artifact_sha256",
    "runtime_lock_binding_sha256",
    "prompt_sha256",
    "capability_artifact_sha256",
    "capability_snapshot_sha256",
    "trace_schema_sha256",
)


@dataclass(frozen=True)
class AnalysisValidation:
    status: str
    errors: tuple[str, ...]
    contrast_artifacts_written: bool
    expected_episode_rows: int
    observed_episode_rows: int
    expected_baseline_rows: int
    observed_baseline_rows: int


@dataclass(frozen=True)
class _ClaimContract:
    campaign_id: str
    snapshot: Mapping[str, Any]
    snapshot_sha256: str
    schedule: tuple[Mapping[str, Any], ...]
    by_attempt: Mapping[str, Mapping[str, Any]]


def validate_joined_rows(
    claim_manifest: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> AnalysisValidation:
    """Validate in-memory claim, episode, and calibration rows without I/O."""
    errors: list[str] = []
    episode_values = _safe_rows(episodes, "episode", errors)
    baseline_values = _safe_rows(baseline_rows, "baseline", errors)
    try:
        contract = _claim_contract(claim_manifest)
    except (TypeError, ValueError) as exc:
        errors.append(f"claim manifest invalid: {exc}")
        contract = None
    if contract is not None:
        errors.extend(_schedule_errors(contract))
        errors.extend(_episode_errors(contract, episode_values))
        errors.extend(_baseline_errors(contract, episode_values, baseline_values))
    unique_errors = tuple(sorted(set(errors)))
    valid = not unique_errors
    return AnalysisValidation(
        "complete" if valid else "blocked",
        unique_errors,
        valid,
        EXPECTED_EPISODE_ROWS,
        len(episode_values),
        EXPECTED_BASELINE_ROWS,
        len(baseline_values),
    )


def _claim_contract(claim: Mapping[str, Any]) -> _ClaimContract:
    if not isinstance(claim, Mapping):
        raise TypeError("claim manifest must be an object")
    manifest = _mapping(claim.get("manifest"), "registered manifest")
    campaign_id = _text(manifest, "campaign_id")
    if campaign_id != "iclr2027-minimal-factorial-v5":
        raise ValueError("claim campaign ID is not registered")
    models = _rows(manifest.get("models"), "registered models")
    if len(models) != 2 or len({_text(row, "slot") for row in models}) != 2:
        raise ValueError("registered analysis requires exactly two models")
    snapshot = _mapping(claim.get("runtime_snapshot"), "runtime snapshot")
    snapshot_sha256 = _text(claim, "runtime_snapshot_sha256")
    if snapshot_sha256 != canonical_sha256(snapshot):
        raise ValueError("runtime snapshot hash drifted")
    schedule = _rows(claim.get("schedule"), "claim schedule")
    if len(schedule) != EXPECTED_EPISODE_ROWS:
        raise ValueError("claim schedule must contain exactly 840 rows")
    by_attempt = _unique_index(schedule, "episode_attempt_id", "claim schedule")
    if any(row.get("campaign_id") != campaign_id for row in schedule):
        raise ValueError("claim schedule contains an unapproved campaign")
    registered = {_text(row, "slot"): _text(row, "tag") for row in models}
    observed = {_text(row, "model_slot"): _text(row, "model_tag") for row in schedule}
    if observed != registered:
        raise ValueError("scheduled model slots or tags drifted")
    return _ClaimContract(
        campaign_id,
        snapshot,
        snapshot_sha256,
        schedule,
        by_attempt,
    )


def _schedule_errors(contract: _ClaimContract) -> tuple[str, ...]:
    errors: list[str] = []
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in contract.schedule:
        cells[(str(row.get("model_slot")), str(row.get("condition_id")))].append(row)
        if not _is_sha256(row.get("model_digest")):
            errors.append("scheduled model digest is invalid")
    if len(cells) != 16:
        errors.append("schedule is not the registered 2 x 8 design")
    stage1_sets: list[set[tuple[str, int]]] = []
    endpoint_sets: list[set[tuple[str, int]]] = []
    if any(row.get("stage") not in STAGES for row in contract.schedule):
        errors.append("schedule contains an unregistered stage label")
    for (_model, condition), rows in cells.items():
        stage1 = [row for row in rows if row.get("stage") == "stage1"]
        if len(stage1) != 30:
            errors.append("Stage 1 cell does not contain exactly 30 rows")
        stage1_sets.append(_case_seed_set(stage1))
        if condition in ENDPOINTS:
            if len(rows) != 120:
                errors.append("endpoint cell does not contain exactly 120 rows")
            endpoint_sets.append(_case_seed_set(rows))
        elif len(rows) != 30:
            errors.append("non-endpoint condition contains extra Stage 2 rows")
    if stage1_sets and any(values != stage1_sets[0] for values in stage1_sets[1:]):
        errors.append("Stage 1 identities are not reused across factorial cells")
    if endpoint_sets and any(
        values != endpoint_sets[0] for values in endpoint_sets[1:]
    ):
        errors.append("endpoint identities are not reused across models and conditions")
    if stage1_sets and endpoint_sets and not stage1_sets[0].issubset(endpoint_sets[0]):
        errors.append("Stage 1 endpoint identities are not reused in Stage 2")
    return tuple(errors)


def _episode_errors(
    contract: _ClaimContract,
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        by_attempt = _unique_index(episodes, "episode_attempt_id", "episodes")
    except ValueError as exc:
        by_attempt = {}
        errors.append(str(exc))
    if len(episodes) != EXPECTED_EPISODE_ROWS or set(by_attempt) != set(
        contract.by_attempt
    ):
        errors.append("episode rows do not exactly join the 840-row claim schedule")
    expected_snapshot = "sha256:" + contract.snapshot_sha256
    expected_trace = _normalized_sha(contract.snapshot.get("trace_schema_sha256"))
    grouped_hashes: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    trace_locations: set[tuple[str, int]] = set()
    trace_records: set[str] = set()
    categories_by_case: dict[str, set[str]] = defaultdict(set)
    for attempt_id, row in by_attempt.items():
        scheduled = contract.by_attempt.get(attempt_id)
        if scheduled is None:
            continue
        if any(row.get(key) != value for key, value in scheduled.items()):
            errors.append("episode identity or frozen condition drifted from schedule")
        if row.get("status") not in (None, "completed"):
            errors.append("episode rows contain a blocked or non-completed attempt")
        if (
            row.get("error") not in (None, "")
            or row.get("episode_stop_reason") == "error"
        ):
            errors.append("episode rows contain an evaluator error")
        if row.get("runtime_snapshot_sha256") != expected_snapshot:
            errors.append("episode runtime snapshot hash drifted")
        expected_config = "sha256:" + canonical_sha256(scheduled.get("condition"))
        if row.get("config_sha256") != expected_config:
            errors.append("episode config hash drifted")
        if row.get("trace_schema_sha256") != expected_trace:
            errors.append("episode trace schema hash drifted")
        cell = (str(row.get("model_slot")), str(row.get("condition_id")))
        for field in _RUNTIME_HASH_FIELDS:
            value = row.get(field)
            if not _is_sha256(value):
                errors.append(f"episode {field} is invalid")
            else:
                grouped_hashes[cell][field].add(str(value))
        _trace_errors(row, expected_trace, trace_locations, trace_records, errors)
        categories_by_case[str(row.get("case_id"))].add(str(row.get("category") or ""))
    if any(
        len(values) != 1
        for fields in grouped_hashes.values()
        for values in fields.values()
    ):
        errors.append(
            "runtime-lock or trace hashes are mixed within a model-condition cell"
        )
    if any(len(values) != 1 or "" in values for values in categories_by_case.values()):
        errors.append("case category labels drifted across joined rows")
    errors.extend(category_errors(episodes))
    return tuple(errors)


def _trace_errors(
    row: Mapping[str, Any],
    expected_schema: str,
    locations: set[tuple[str, int]],
    records: set[str],
    errors: list[str],
) -> None:
    references = row.get("scientific_trace_references")
    decisions = row.get("decisions_made")
    if (
        not isinstance(references, list)
        or not references
        or decisions != len(references)
    ):
        errors.append("episode trace references are missing or incomplete")
        return
    for reference in references:
        if not isinstance(reference, Mapping):
            errors.append("episode trace reference is not an object")
            continue
        path = str(reference.get("relative_path") or "")
        line = reference.get("line_number")
        record = reference.get("record_sha256")
        relative = PurePosixPath(path)
        valid_path = (
            bool(path) and not relative.is_absolute() and ".." not in relative.parts
        )
        if (
            not valid_path
            or isinstance(line, bool)
            or not isinstance(line, int)
            or line < 1
            or not _is_sha256(record)
            or reference.get("schema_sha256") != expected_schema
            or not isinstance(reference.get("schema_version"), str)
            or not reference.get("schema_version")
        ):
            errors.append("episode trace reference is invalid")
            continue
        location = (path, line)
        if location in locations or str(record) in records:
            errors.append("episode trace reference is duplicated")
        locations.add(location)
        records.add(str(record))


def _baseline_errors(
    contract: _ClaimContract,
    episodes: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        calibration = build_calibration_contract(
            {
                "manifest": {},
                "runtime_snapshot": contract.snapshot,
                "runtime_snapshot_sha256": contract.snapshot_sha256,
                "schedule": list(contract.schedule),
            },
            [row for row in contract.schedule if row.get("condition_id") == "c111"],
        )
    except (TypeError, ValueError) as exc:
        return (f"calibration contract invalid: {exc}",)
    expected = {
        (policy, case_id, seed)
        for policy in calibration.policies
        for case_id, seed in calibration.case_seeds
    }
    observed = [_baseline_identity(row) for row in rows]
    if len(rows) != EXPECTED_BASELINE_ROWS or set(observed) != expected:
        errors.append("baseline rows do not match the exact 360-row denominator")
    if len(set(observed)) != len(observed):
        errors.append("baseline rows contain duplicate identities")
    category_by_case = {
        str(row.get("case_id")): str(row.get("category")) for row in episodes
    }
    for row in rows:
        if row.get("error") not in (None, ""):
            errors.append("baseline rows contain an evaluator error")
        if row.get("safety_shields_enabled") is not True:
            errors.append("baseline rows contain an unshielded episode")
        if (
            row.get("balanced_driving_score_policy_version")
            != calibration.scoring_version
        ):
            errors.append("baseline scoring policy version drifted")
        if any(
            str(row.get(key)) != value for key, value in calibration.provenance.items()
        ):
            errors.append("baseline provenance drifted")
        if str(row.get("category")) != category_by_case.get(str(row.get("case_id"))):
            errors.append("baseline category drifted from the matched endpoint case")
    return tuple(errors)


def _safe_rows(
    value: Any, name: str, errors: list[str]
) -> tuple[Mapping[str, Any], ...]:
    try:
        return _rows(value, f"{name} rows")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return ()


def _rows(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError(f"{name} must be a sequence of objects")
    return tuple(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item or item != item.strip():
        raise ValueError(f"{name} must be canonical text")
    return item


def _unique_index(
    rows: Sequence[Mapping[str, Any]], key: str, name: str
) -> dict[str, Mapping[str, Any]]:
    values = [_text(row, key) for row in rows]
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate {key} values")
    return dict(zip(values, rows))


def _case_seed_set(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, int]]:
    values: set[tuple[str, int]] = set()
    for row in rows:
        try:
            values.add((str(row.get("case_id")), int(row.get("simulator_seed"))))
        except (TypeError, ValueError):
            continue
    return values


def _baseline_identity(row: Mapping[str, Any]) -> tuple[str, str, int]:
    try:
        seed = int(row.get("simulator_seed"))
    except (TypeError, ValueError):
        seed = -1
    return str(row.get("baseline_policy")), str(row.get("case_id")), seed


def _normalized_sha(value: Any) -> str:
    text = str(value or "")
    return text if text.startswith("sha256:") else "sha256:" + text


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


__all__ = ["AnalysisValidation", "validate_joined_rows"]
