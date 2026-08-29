"""Public registered-analysis API for the ICLR 2027 grounded-decoding V8 campaign.

Composes the V8-specific validation gate
(:mod:`dilu.runtime._grounded_decoding_analysis_validation`) with the
registered contrast families
(:mod:`dilu.runtime._grounded_decoding_analysis_families`), mirroring the
shape of :func:`dilu.runtime.minimal_factorial_analysis.run_registered_
analysis`: a blocked run publishes only the validation payload, a complete
run publishes validation plus every table. Kept as a pure, in-memory
function -- :func:`run_registered_v8_analysis` never touches the
filesystem -- so it is safe to call from tests; disk I/O and publication
are the caller's responsibility (see ``scripts/analyze_iclr2027_grounded_
decoding.py`` and :func:`dilu.runtime._grounded_decoding_analysis_
artifacts.publish_v8_analysis_bundle`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._grounded_decoding_analysis_action_shift import compute_action_distribution_shift
from ._grounded_decoding_analysis_families import compute_family_tables
from ._grounded_decoding_analysis_stats import (
    SIGN_FLIP_DRAWS,
    derive_sign_flip_seed,
    holm,
    sign_flip_p,
)
from ._grounded_decoding_analysis_validation import (
    GroundedAnalysisValidation,
    validate_v8_rows,
)
from ._minimal_factorial_analysis_bootstrap import (
    BootstrapInterval,
    derive_bootstrap_seed,
    stratified_bootstrap,
)


@dataclass(frozen=True)
class V8AnalysisResult:
    validation: GroundedAnalysisValidation
    family_a: tuple[Mapping[str, Any], ...]
    family_b: tuple[Mapping[str, Any], ...]
    family_c: tuple[Mapping[str, Any], ...]
    family_d: tuple[Mapping[str, Any], ...]
    descriptive: tuple[Mapping[str, Any], ...]
    action_distribution: tuple[Mapping[str, Any], ...] = ()


def run_registered_v8_analysis(
    v8_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
    o2_action_counts: Mapping[str, Mapping[int, int]] | None = None,
    o1_action_counts: Mapping[str, Mapping[int, int]] | None = None,
) -> V8AnalysisResult:
    """Validate the 480 V8 rows, then compute Families A-D if the gate passes.

    Returns a blocked ``V8AnalysisResult`` (empty tables) when any gate --
    row shape, duplicate id, comparator pairing (missing/digest/fingerprint/
    scoring), or the Family M manipulation check -- fails. A blocked result
    is the *only* thing that should ever reach disk in that case (see
    :func:`blocked_payload`).

    ``o2_action_counts``/``o1_action_counts`` are optional per-model executed-
    action histograms (see :func:`compute_action_distribution_shift`); when
    omitted the descriptive action-distribution table is simply empty, since
    that table is purely descriptive and never gates completion.
    """
    validation = validate_v8_rows(v8_rows, frozen_rows)
    if validation.status != "complete":
        return V8AnalysisResult(validation, (), (), (), (), (), ())
    tables = compute_family_tables(v8_rows, frozen_rows, manifest_sha256=manifest_sha256)
    action_distribution = compute_action_distribution_shift(
        o2_action_counts or {}, o1_action_counts or {}
    )
    return V8AnalysisResult(
        validation,
        tuple(tables["FAMILY_A"]),
        tuple(tables["FAMILY_B"]),
        tuple(tables["FAMILY_C"]),
        tuple(tables["FAMILY_D"]),
        tuple(tables["descriptive"]),
        tuple(action_distribution),
    )


def blocked_payload(validation: GroundedAnalysisValidation) -> dict[str, Any]:
    """The exact registered blocked-output shape (three keys, nothing else)."""
    return {
        "status": validation.status,
        "errors": sorted(set(validation.errors)),
        "contrast_artifacts_written": validation.contrast_artifacts_written,
    }


__all__ = [
    "BootstrapInterval",
    "GroundedAnalysisValidation",
    "SIGN_FLIP_DRAWS",
    "V8AnalysisResult",
    "blocked_payload",
    "compute_action_distribution_shift",
    "derive_bootstrap_seed",
    "derive_sign_flip_seed",
    "holm",
    "run_registered_v8_analysis",
    "sign_flip_p",
    "stratified_bootstrap",
    "validate_v8_rows",
]
