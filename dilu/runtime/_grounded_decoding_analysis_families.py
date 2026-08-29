"""Registered V8 contrast families A-D: paired O2-minus-O1 differences.

Unlike the V5/V7 factorial's 7-way ``factorial_contrasts`` algebra, every
V8 family here is a simple paired two-condition difference (grounded O2
minus its frozen O1 counterpart, same case and simulator seed), so this
module composes -- rather than reuses -- ``_minimal_factorial_analysis_
contrasts.factorial_contrasts`` (which does not apply to V8's design). What
*is* reused unmodified: :func:`stratified_bootstrap`/
:func:`derive_bootstrap_seed` for interval estimation, and
:func:`sign_flip_p`/:func:`holm` from
:mod:`dilu.runtime._grounded_decoding_analysis_stats` for the registered
inferential layer.

Family A/B/C are three-cases-per-category paired contrasts over the 30
Stage-1 cases; Family D is a twelve-cases-per-category paired contrast over
all 120 cases for the two small models. Holm is applied *within* each
family separately (never across families, never pooling models).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from ._grounded_decoding_analysis_pairing import paired_o1_condition_id
from ._grounded_decoding_analysis_stats import (
    SIGN_FLIP_DRAWS,
    derive_sign_flip_seed,
    holm,
    sign_flip_p,
)
from ._minimal_factorial_analysis_bootstrap import (
    BOOTSTRAP_DRAWS,
    derive_bootstrap_seed,
    stratified_bootstrap,
)

PRIMARY_OUTCOME = "driving_score_balanced_v1"
PRIMARY_DIRECTION = "higher"
# Field names on the right must exist on the real episode row (see
# tests/test_grounded_decoding_analysis_fixes.py::RealEpisodeSchemaGuardTests,
# which guards this table against another invented field). "shield_intervention_rate"
# is not itself a real episode field -- it is an aggregate the CLI's
# ``_enrich_v8_rows`` computes from real per-shield-stage trace records
# (mirroring the ``analysis_any_shield_intervention_count`` convention in
# ``_minimal_factorial_analysis_io.py``) and writes onto the row under the
# ``analysis_`` prefix used for every other enrichment field.
SECONDARY_OUTCOMES: dict[str, tuple[str, str]] = {
    "task_completion": ("task_completed", "higher"),
    "crash": ("crashed", "lower"),
    "fallback_rate": ("fallback_action_rate", "lower"),
    "shield_intervention_rate": ("analysis_shield_intervention_rate", "diagnostic"),
    "decision_latency_ms_avg": ("decision_latency_ms_avg", "lower"),
}

FAMILY_A = "FAMILY_A_O2E1_MINUS_O1E1"
FAMILY_B = "FAMILY_B_O2E0_MINUS_O1E0"
FAMILY_C = "FAMILY_C_O2XE_DID"
FAMILY_D = "FAMILY_D_ENDPOINT_O2E1_MINUS_O1E1"
STAGE1_SAMPLES_PER_CATEGORY = 3
ENDPOINT_SAMPLES_PER_CATEGORY = 12


def compute_family_tables(
    v8_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    """Compute registered Families A-D for the primary outcome plus descriptives.

    Returns a mapping ``{"FAMILY_A": [...], ..., "descriptive": [...]}``:
    each of the four family keys holds the Holm-corrected primary-outcome
    test rows for that family (one row per model); ``"descriptive"`` holds
    every family x secondary-outcome combination (estimate and bootstrap
    interval only, no test, no Holm).
    """
    index = _RowIndex(v8_rows, frozen_rows)
    tables: dict[str, list[dict[str, Any]]] = {}
    for family_id, contrast_id, models, sample_size in (
        ("FAMILY_A", FAMILY_A, index.stage1_models, STAGE1_SAMPLES_PER_CATEGORY),
        ("FAMILY_B", FAMILY_B, index.stage1_models, STAGE1_SAMPLES_PER_CATEGORY),
        ("FAMILY_C", FAMILY_C, index.stage1_models, STAGE1_SAMPLES_PER_CATEGORY),
        ("FAMILY_D", FAMILY_D, index.endpoint_models, ENDPOINT_SAMPLES_PER_CATEGORY),
    ):
        rows = [
            _contrast_row(
                index,
                model,
                family_id,
                contrast_id,
                PRIMARY_OUTCOME,
                PRIMARY_DIRECTION,
                sample_size,
                manifest_sha256,
                inferential=True,
            )
            for model in models
        ]
        holm(rows)
        tables[family_id] = rows
    tables["descriptive"] = _descriptive_rows(index, manifest_sha256)
    return tables


def _descriptive_rows(index: "_RowIndex", manifest_sha256: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_id, contrast_id, models, sample_size in (
        ("FAMILY_A", FAMILY_A, index.stage1_models, STAGE1_SAMPLES_PER_CATEGORY),
        ("FAMILY_B", FAMILY_B, index.stage1_models, STAGE1_SAMPLES_PER_CATEGORY),
        ("FAMILY_C", FAMILY_C, index.stage1_models, STAGE1_SAMPLES_PER_CATEGORY),
        ("FAMILY_D", FAMILY_D, index.endpoint_models, ENDPOINT_SAMPLES_PER_CATEGORY),
    ):
        for model in models:
            for outcome, (field, direction) in SECONDARY_OUTCOMES.items():
                rows.append(
                    _contrast_row(
                        index,
                        model,
                        family_id,
                        contrast_id,
                        outcome,
                        direction,
                        sample_size,
                        manifest_sha256,
                        inferential=False,
                        field=field,
                    )
                )
    return rows


def _contrast_row(
    index: "_RowIndex",
    model: str,
    family_id: str,
    contrast_id: str,
    outcome: str,
    direction: str,
    sample_size: int,
    manifest_sha256: str,
    *,
    inferential: bool,
    field: str | None = None,
) -> dict[str, Any]:
    values = index.paired_values(family_id, model, field or PRIMARY_OUTCOME)
    categories = index.categories(model)
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, value in values.items():
        grouped[categories[case_id]].append(value)
    bootstrap_seed = derive_bootstrap_seed(manifest_sha256, model, contrast_id, outcome)
    interval = stratified_bootstrap(
        grouped,
        samples_per_category=sample_size,
        draws=BOOTSTRAP_DRAWS,
        seed=bootstrap_seed,
    )
    ordered_values = [values[case_id] for case_id in sorted(values)]
    p_value = float("nan")
    sign_flip_draws = 0
    if inferential:
        sign_flip_seed = derive_sign_flip_seed(manifest_sha256, model, contrast_id, outcome)
        p_value = sign_flip_p(ordered_values, sign_flip_seed)
        sign_flip_draws = SIGN_FLIP_DRAWS
    return {
        "family": family_id,
        "model_slot": model,
        "model_tag": index.model_tag(model),
        "contrast_id": contrast_id,
        "outcome": outcome,
        "outcome_direction": direction,
        "n_paired_cases": len(values),
        "estimate": interval.effect,
        "lower_2_5": interval.lower_2_5,
        "upper_97_5": interval.upper_97_5,
        "bootstrap_draws": interval.draws,
        "bootstrap_seed": interval.seed,
        "evidence_scope": interval.evidence_scope,
        "p_value": p_value,
        "p_holm": float("nan"),
        "sign_flip_draws": sign_flip_draws,
    }


class _RowIndex:
    """Lookups shared by every family: paired case values, categories, model tags."""

    def __init__(
        self,
        v8_rows: Sequence[Mapping[str, Any]],
        frozen_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self._v8_by_model_condition: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = (
            defaultdict(dict)
        )
        self._categories: dict[str, dict[str, str]] = defaultdict(dict)
        self._model_tags: dict[str, str] = {}
        self._v8_stage1_by_model_condition: dict[
            tuple[str, str], dict[str, Mapping[str, Any]]
        ] = defaultdict(dict)
        for row in v8_rows:
            model = str(row["model_slot"])
            condition = str(row["condition_id"])
            case_id = str(row["case_id"])
            self._v8_by_model_condition[(model, condition)][case_id] = row
            if str(row["stage"]) == "stage1":
                self._v8_stage1_by_model_condition[(model, condition)][case_id] = row
            self._categories[model][case_id] = str(row["category"])
            self._model_tags[model] = str(row["model_tag"])
        self._frozen_by_key: dict[tuple[str, str, str, int], Mapping[str, Any]] = {
            (
                str(row["model_tag"]),
                str(row["condition_id"]),
                str(row["case_id"]),
                int(row["simulator_seed"]),
            ): row
            for row in frozen_rows
        }
        self.stage1_models = tuple(
            sorted(
                {
                    model
                    for (model, condition) in self._v8_by_model_condition
                    if condition in ("c120", "c121")
                }
            )
        )
        self.endpoint_models = tuple(
            sorted(model for model in self.stage1_models if self._has_endpoint_rows(model))
        )

    def _has_endpoint_rows(self, model: str) -> bool:
        # Stage-2-additional rows share condition_id "c121" with Stage 1, so a
        # model that ran Stage 2 has more than the 30 Stage-1 c121 rows here.
        return len(self._v8_by_model_condition.get((model, "c121"), {})) > 30

    def model_tag(self, model: str) -> str:
        return self._model_tags[model]

    def categories(self, model: str) -> Mapping[str, str]:
        return self._categories[model]

    def _o1_value(self, v8_row: Mapping[str, Any], condition_id: str, field: str) -> float:
        key = (
            v8_row["model_tag"],
            paired_o1_condition_id(condition_id),
            v8_row["case_id"],
            int(v8_row["simulator_seed"]),
        )
        frozen_row = self._frozen_by_key[key]
        return _read(frozen_row, field)

    def _paired(
        self,
        rows_by_model_condition: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
        model: str,
        condition_id: str,
        field: str,
    ) -> dict[str, float]:
        rows = rows_by_model_condition.get((model, condition_id), {})
        return {
            case_id: _read(row, field) - self._o1_value(row, condition_id, field)
            for case_id, row in rows.items()
        }

    def paired_values(self, family_id: str, model: str, field: str) -> dict[str, float]:
        if family_id == "FAMILY_A":
            return self._paired(self._v8_stage1_by_model_condition, model, "c121", field)
        if family_id == "FAMILY_B":
            return self._paired(self._v8_stage1_by_model_condition, model, "c120", field)
        if family_id == "FAMILY_C":
            a = self._paired(self._v8_stage1_by_model_condition, model, "c121", field)
            b = self._paired(self._v8_stage1_by_model_condition, model, "c120", field)
            return {case_id: a[case_id] - b[case_id] for case_id in a if case_id in b}
        if family_id == "FAMILY_D":
            return self._paired(self._v8_by_model_condition, model, "c121", field)
        raise ValueError(f"Unknown family_id: {family_id!r}.")


def _read(row: Mapping[str, Any], field: str) -> float:
    return float(row[field])


__all__ = [
    "ENDPOINT_SAMPLES_PER_CATEGORY",
    "FAMILY_A",
    "FAMILY_B",
    "FAMILY_C",
    "FAMILY_D",
    "PRIMARY_DIRECTION",
    "PRIMARY_OUTCOME",
    "SECONDARY_OUTCOMES",
    "STAGE1_SAMPLES_PER_CATEGORY",
    "compute_family_tables",
]
