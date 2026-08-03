"""Registered summaries and contrasts over validated episode rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from ._minimal_factorial_analysis_artifacts import AnalysisTables
from ._minimal_factorial_analysis_bootstrap import (
    BOOTSTRAP_DRAWS,
    derive_bootstrap_seed,
    stratified_bootstrap,
)
from ._minimal_factorial_analysis_contrasts import (
    endpoint_contrast,
    factorial_contrasts,
)
from ._minimal_factorial_analysis_summaries import (
    OutcomeReader,
    category_summaries,
    condition_summaries,
    outcomes,
)
from ._minimal_factorial_schedule_support import canonical_sha256


def compute_registered_tables(
    claim: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
) -> AnalysisTables:
    models = sorted({_text(row, "model_slot") for row in episodes})
    model_digests = {
        model: _one(
            [row for row in episodes if row.get("model_slot") == model],
            "model_digest",
        )
        for model in models
    }
    outcome_readers = outcomes()
    provenance = _provenance(claim, episodes, manifest_sha256)
    condition_rows = _with_model_digests(
        condition_summaries(episodes, models, outcome_readers, provenance),
        model_digests,
    )
    factor_rows = _with_model_digests(
        _factor_rows(episodes, models, outcome_readers, provenance, manifest_sha256),
        model_digests,
    )
    endpoint_rows = _with_model_digests(
        _endpoint_rows(episodes, models, outcome_readers, provenance, manifest_sha256),
        model_digests,
    )
    calibration_rows = _with_model_digests(
        _calibration_rows(
            episodes,
            baseline_rows,
            models,
            outcome_readers,
            provenance,
            manifest_sha256,
        ),
        model_digests,
    )
    category_rows = _with_model_digests(
        category_summaries(episodes, models, outcome_readers, provenance),
        model_digests,
    )
    report = (
        "# Registered minimal-factorial analysis\n\n"
        "This bundle reports model-separated fixed-suite effects. "
        "Non-LLM comparisons are behavior calibration only.\n"
    )
    appendix = (
        "# Statistical appendix\n\n"
        "Intervals use 20,000 category-stratified bootstrap draws and describe "
        "fixed-suite sensitivity, not population confidence. No p-values were "
        "calculated and models were not pooled.\n"
    )
    return AnalysisTables(
        tuple(condition_rows),
        tuple(factor_rows),
        tuple(endpoint_rows),
        tuple(calibration_rows),
        tuple(category_rows),
        report,
        appendix,
    )


def _factor_rows(
    episodes: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    outcomes: Mapping[str, tuple[OutcomeReader, str]],
    provenance: Mapping[str, str],
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in models:
        rows = [
            row
            for row in episodes
            if row.get("model_slot") == model and row.get("stage") == "stage1"
        ]
        by_case = _case_cells(rows)
        for outcome, (reader, direction) in outcomes.items():
            effects: dict[str, dict[str, float]] = defaultdict(dict)
            for case_id, cells in by_case.items():
                contrasts = factorial_contrasts(
                    {condition: reader(row) for condition, row in cells.items()}
                )
                for contrast_id, value in contrasts.items():
                    effects[contrast_id][case_id] = value
            categories = _case_categories(by_case)
            for contrast_id in sorted(effects):
                interval = _interval(
                    effects[contrast_id],
                    categories,
                    3,
                    manifest_sha256,
                    model,
                    contrast_id,
                    outcome,
                )
                output.append(
                    _contrast_row(
                        provenance,
                        model,
                        contrast_id,
                        outcome,
                        direction,
                        interval,
                        30,
                        model,
                    )
                )
    return output


def _endpoint_rows(
    episodes: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    outcomes: Mapping[str, tuple[OutcomeReader, str]],
    provenance: Mapping[str, str],
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in models:
        rows = [row for row in episodes if row.get("model_slot") == model]
        by_case = _case_cells(
            row for row in rows if row.get("condition_id") in {"c000", "c111"}
        )
        categories = _case_categories(by_case)
        for outcome, (reader, direction) in outcomes.items():
            effects = {
                case_id: endpoint_contrast(
                    {
                        condition: reader(cells[condition])
                        for condition in ("c000", "c111")
                    }
                    | {condition: 0.0 for condition in _non_endpoints()}
                )
                for case_id, cells in by_case.items()
            }
            contrast_id = "ENDPOINT_C111_MINUS_C000"
            interval = _interval(
                effects,
                categories,
                12,
                manifest_sha256,
                model,
                contrast_id,
                outcome,
            )
            output.append(
                _contrast_row(
                    provenance,
                    model,
                    contrast_id,
                    outcome,
                    direction,
                    interval,
                    120,
                    model,
                )
            )
    return output


def _calibration_rows(
    episodes: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    outcomes: Mapping[str, tuple[OutcomeReader, str]],
    provenance: Mapping[str, str],
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    baseline_by_policy = _group(baseline_rows, "baseline_policy")
    for model in models:
        model_rows = {
            _text(row, "case_id"): row
            for row in episodes
            if row.get("model_slot") == model and row.get("condition_id") == "c111"
        }
        categories = {
            case_id: _text(row, "category") for case_id, row in model_rows.items()
        }
        for policy, reference_rows in sorted(baseline_by_policy.items()):
            references = {_text(row, "case_id"): row for row in reference_rows}
            for outcome in (
                "driving_score_balanced_v1",
                "task_completion",
                "crash",
            ):
                reader, direction = outcomes[outcome]
                effects = {
                    case_id: reader(model_rows[case_id]) - reader(references[case_id])
                    for case_id in model_rows
                }
                contrast_id = f"CAL_{model.upper()}_C111_MINUS_{policy.upper()}"
                interval = _interval(
                    effects,
                    categories,
                    12,
                    manifest_sha256,
                    f"{model}:{policy}",
                    contrast_id,
                    outcome,
                )
                output.append(
                    _contrast_row(
                        provenance,
                        model,
                        contrast_id,
                        outcome,
                        direction,
                        interval,
                        120,
                        f"{model}:{policy}",
                    )
                )
    return output


def _interval(
    effects: Mapping[str, float],
    categories: Mapping[str, str],
    sample_size: int,
    manifest_sha256: str,
    model: str,
    contrast_id: str,
    outcome: str,
) -> Any:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, value in effects.items():
        grouped[categories[case_id]].append(value)
    seed = derive_bootstrap_seed(
        manifest_sha256,
        model,
        contrast_id,
        outcome,
    )
    return stratified_bootstrap(
        grouped,
        samples_per_category=sample_size,
        draws=BOOTSTRAP_DRAWS,
        seed=seed,
    )


def _contrast_row(
    provenance: Mapping[str, str],
    model: str,
    contrast_id: str,
    outcome: str,
    direction: str,
    interval: Any,
    case_count: int,
    bootstrap_seed_subject: str,
) -> dict[str, Any]:
    return {
        **provenance,
        "model_or_reference": model,
        "contrast_id": contrast_id,
        "outcome": outcome,
        "outcome_direction": direction,
        "numerator": interval.effect * case_count,
        "denominator": case_count,
        "effect": interval.effect,
        "lower_2_5": interval.lower_2_5,
        "upper_97_5": interval.upper_97_5,
        "bootstrap_draws": interval.draws,
        "bootstrap_seed": interval.seed,
        "bootstrap_seed_subject": bootstrap_seed_subject,
        "evidence_scope": interval.evidence_scope,
    }


def _case_cells(rows: Sequence[Mapping[str, Any]] | Any) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        cells[_text(row, "case_id")][_text(row, "condition_id")] = row
    return dict(cells)


def _case_categories(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {
        case_id: _text(next(iter(rows.values())), "category")
        for case_id, rows in cells.items()
    }


def _non_endpoints() -> tuple[str, ...]:
    return ("c001", "c010", "c011", "c100", "c101", "c110")


def _group(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_text(row, key)].append(row)
    return dict(grouped)


def _with_model_digests(
    rows: Sequence[Mapping[str, Any]],
    model_digests: Mapping[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "model_digest": model_digests[_text(row, "model_or_reference")],
        }
        for row in rows
    ]


def _provenance(
    claim: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
) -> dict[str, str]:
    snapshot = claim["runtime_snapshot"]
    stage1_cases = sorted(
        {_text(row, "case_id") for row in episodes if row.get("stage") == "stage1"}
    )
    return {
        "campaign_id": _text(episodes[0], "campaign_id"),
        "manifest_sha256": manifest_sha256,
        "case_set_sha256": str(snapshot["case_set_fingerprint"]),
        "selected_30_sha256": "sha256:" + canonical_sha256(stage1_cases),
        "config_sha256": "sha256:"
        + canonical_sha256(sorted({_text(row, "config_sha256") for row in episodes})),
        "runtime_lock_sha256": "sha256:"
        + canonical_sha256(
            sorted({_text(row, "runtime_lock_binding_sha256") for row in episodes})
        ),
        "source_revision": str(snapshot["code_revision"]),
        "trace_schema_sha256": _one(episodes, "trace_schema_sha256"),
        "scoring_sha256": "sha256:" + str(snapshot["scoring_fingerprint"]),
        "environment_sha256": "sha256:"
        + canonical_sha256(snapshot["environment_config"]),
    }


def _one(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    values = {_text(row, key) for row in rows}
    if len(values) != 1:
        raise ValueError(f"Validated analysis rows disagree on {key}.")
    return values.pop()


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text.")
    return value


__all__ = ["compute_registered_tables"]
