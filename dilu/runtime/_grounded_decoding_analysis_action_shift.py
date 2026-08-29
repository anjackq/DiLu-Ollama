"""Descriptive action-distribution shift under grounded decoding (O2 vs O1).

Registered as purely descriptive (no significance test, no Holm, no gate):
the per-model share of executed actions under V8's grounded decoding (O2,
``c120``+``c121`` combined) against the same model's frozen O1 comparator
(``c110``+``c111``). This lives here rather than the paper repo's Task 1
frozen-trace tool because the O2 side of the comparison only exists in V8
data, which only this module's caller (the CLI script) reads.

Pure function over precomputed per-model action histograms so it stays
testable without any trace I/O: the CLI script builds those histograms
from the *same* ``decision_traces.jsonl`` scan already performed for the
Family M gate and the shield-intervention-rate enrichment (see
``scripts/analyze_iclr2027_grounded_decoding.py``), rather than a second
pass over the trace file.
"""

from __future__ import annotations

from typing import Any, Mapping

EVIDENCE_SCOPE = "descriptive fixed-suite action-distribution summary, no test"


def compute_action_distribution_shift(
    o2_action_counts: Mapping[str, Mapping[int, int]],
    o1_action_counts: Mapping[str, Mapping[int, int]],
) -> list[dict[str, Any]]:
    """Per (model, action_id) executed-action share under O2 vs. its O1 comparator."""
    rows: list[dict[str, Any]] = []
    for model in sorted(set(o2_action_counts) | set(o1_action_counts)):
        o2_counts = o2_action_counts.get(model, {})
        o1_counts = o1_action_counts.get(model, {})
        o2_total = sum(o2_counts.values())
        o1_total = sum(o1_counts.values())
        for action_id in sorted(set(o2_counts) | set(o1_counts)):
            o2_count = int(o2_counts.get(action_id, 0))
            o1_count = int(o1_counts.get(action_id, 0))
            o2_share = o2_count / o2_total if o2_total else 0.0
            o1_share = o1_count / o1_total if o1_total else 0.0
            rows.append(
                {
                    "model_slot": model,
                    "action_id": action_id,
                    "o2_count": o2_count,
                    "o2_total": o2_total,
                    "o2_share": o2_share,
                    "o1_count": o1_count,
                    "o1_total": o1_total,
                    "o1_share": o1_share,
                    "share_shift": o2_share - o1_share,
                    "evidence_scope": EVIDENCE_SCOPE,
                }
            )
    return rows


__all__ = ["EVIDENCE_SCOPE", "compute_action_distribution_shift"]
