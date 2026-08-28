"""Registered sign-flip test and Holm correction for the V8 grounded-decoding analysis.

Neither procedure lives in the shared ``_minimal_factorial_analysis_*``
helpers (the V5/V7 registered analysis reports bootstrap intervals only, no
p-values). The reference implementation is the standalone prototype script
``results/iclr2027_model_breadth_factorial_v7/analysis-prototype/
analyze_v7_full_factorial.py`` (``sign_flip_p`` at lines 99-106, ``holm`` at
lines 120-125). Both functions here are byte-for-byte ports of that
reference -- same draw count, same ``+1/+1`` continuity correction, same
``1e-15`` tolerance, same step-down running-max Holm recipe -- so V8's
inferential layer matches the already-reviewed V7 prototype exactly rather
than a reimplementation that could silently drift from it.

The registered bootstrap *interval* seed recipe
(``<manifest_sha256>|<model>|<contrast_id>|<outcome>|bootstrap-v1``) lives in
:mod:`dilu.runtime._minimal_factorial_analysis_bootstrap` and is imported
and reused unmodified. That recipe has no sign-flip-specific counterpart, so
:func:`derive_sign_flip_seed` reuses the exact same ``derive_bootstrap_seed``
hash construction with a distinct ``version`` tag (``"signflip-v1"``) so the
permutation-test seed can never collide with a bootstrap-interval seed for
the same (model, contrast, outcome) triple.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._minimal_factorial_analysis_bootstrap import derive_bootstrap_seed

SIGN_FLIP_DRAWS = 20_000
SIGN_FLIP_VERSION = "signflip-v1"


def sign_flip_p(values: Sequence[float] | np.ndarray, seed: int) -> float:
    """Two-sided paired sign-flip test p-value. Exact port of the V7 reference."""
    array = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(array)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(SIGN_FLIP_DRAWS):
        signs = rng.choice((-1.0, 1.0), size=len(array))
        exceed += abs(float(np.mean(array * signs))) >= observed - 1e-15
    return (exceed + 1) / (SIGN_FLIP_DRAWS + 1)


def holm(rows: list[dict[str, Any]]) -> None:
    """Step-down Holm correction, mutating ``p_holm`` in place. Exact port of V7."""
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_value"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p_value"]))
        rows[index]["p_holm"] = running


def derive_sign_flip_seed(
    manifest_sha256: str,
    model_or_reference: str,
    contrast_id: str,
    outcome: str,
) -> int:
    """Deterministic sign-flip permutation seed, keyed like the bootstrap seed."""
    return derive_bootstrap_seed(
        manifest_sha256,
        model_or_reference,
        contrast_id,
        outcome,
        SIGN_FLIP_VERSION,
    )


__all__ = [
    "SIGN_FLIP_DRAWS",
    "SIGN_FLIP_VERSION",
    "derive_sign_flip_seed",
    "holm",
    "sign_flip_p",
]
