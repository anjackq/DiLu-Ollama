"""Public registered-analysis API for the ICLR 2027 minimal factorial."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ._minimal_factorial_analysis_bootstrap import (
    BootstrapInterval,
    _draw_stratified,
    derive_bootstrap_seed,
    stratified_bootstrap,
)
from ._minimal_factorial_analysis_contrasts import (
    endpoint_contrast,
    factorial_contrasts,
)
from ._minimal_factorial_analysis_artifacts import (
    AnalysisTables,
    publish_analysis_bundle,
)
from ._minimal_factorial_analysis_io import (
    AnalysisInputPaths,
    load_analysis_inputs,
)
from ._minimal_factorial_analysis_tables import compute_registered_tables
from ._minimal_factorial_analysis_validation import AnalysisValidation


def run_registered_analysis(
    paths: AnalysisInputPaths,
    *,
    output_root: Path,
) -> Path:
    validation, inputs = load_analysis_inputs(paths)
    if inputs is None:
        return publish_analysis_bundle(output_root, validation)
    try:
        tables = compute_registered_tables(
            inputs.claim,
            inputs.episodes,
            inputs.baseline_rows,
            manifest_sha256=inputs.manifest_sha256,
        )
    except Exception as exc:
        blocked = dataclasses.replace(
            validation,
            status="blocked",
            errors=(f"registered analysis failed: {exc}",),
            contrast_artifacts_written=False,
        )
        return publish_analysis_bundle(output_root, blocked)
    completed = dataclasses.replace(
        validation,
        status="complete",
        errors=(),
        contrast_artifacts_written=True,
    )
    return publish_analysis_bundle(output_root, completed, tables)


__all__ = [
    "BootstrapInterval",
    "AnalysisInputPaths",
    "AnalysisTables",
    "AnalysisValidation",
    "_draw_stratified",
    "derive_bootstrap_seed",
    "endpoint_contrast",
    "factorial_contrasts",
    "run_registered_analysis",
    "stratified_bootstrap",
]
