"""CLI for the registered ICLR 2027 grounded-decoding (V8) analysis.

Wires real filesystem I/O around the pure library in
``dilu.runtime.grounded_decoding_analysis``: reads the 480 V8 episode rows,
enriches them with the per-episode ``action_unavailable`` violation count
the Family M gate needs and a real ``shield_intervention_rate`` (both
derived from ``decision_traces.jsonl`` -- the trace schema itself is
untouched), reads the frozen V5/V7 comparator rows (only the O1 cells V8
pairs against: c110/c111), and builds the O2-vs-O1 executed-action
histograms the descriptive action-distribution-shift table needs.

The V8 episode-to-trace join is fail-closed: an episode with zero matching
trace records, or a matched record whose identity (model_tag,
model_digest, case_id, simulator_seed) drifts from the episode row, raises
rather than silently defaulting counts to zero -- Family M is the one gate
standing between a broken O2 transport and a published false result, so a
broken join must never look like "genuinely zero violations."

The frozen O1 comparator rows go through the *same* fail-closed enrichment
(against their own campaign's ``--v5-traces``/``--v7-traces`` file) so they
carry a real ``analysis_shield_intervention_rate`` too: the descriptive
shield-intervention-rate contrast reads this field on both the V8 and the
frozen side (see ``_grounded_decoding_analysis_families.SECONDARY_OUTCOMES``),
and a raw, un-enriched frozen row does not carry it -- only the three
disjoint per-shield-stage rate/count fields
(``lane_change_shield_rate``/``longitudinal_safety_shield_rate``/
``flow_recovery_shield_rate``), which are not interchangeable with "any
shield intervened" (they are not mutually exclusive, so summing them
overcounts). Skipping this enrichment on the frozen side would raise a
``KeyError`` the first time a real V8 campaign's analysis actually ran.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.runtime._grounded_decoding_analysis_artifacts import (  # noqa: E402
    V8AnalysisTables,
    publish_v8_analysis_bundle,
)
from dilu.runtime.grounded_decoding_analysis import (  # noqa: E402
    blocked_payload,
    run_registered_v8_analysis,
)

_FROZEN_O1_CONDITIONS = frozenset({"c110", "c111"})
_ACTION_UNAVAILABLE_FIELD = "analysis_action_unavailable_count"
_SHIELD_INTERVENTION_RATE_FIELD = "analysis_shield_intervention_rate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-manifest", type=Path, required=True)
    parser.add_argument("--v8-episodes", type=Path, required=True)
    parser.add_argument("--v8-traces", type=Path, required=True)
    parser.add_argument("--v5-episodes", type=Path, required=True)
    parser.add_argument("--v5-traces", type=Path, required=True)
    parser.add_argument("--v7-episodes", type=Path, required=True)
    parser.add_argument("--v7-traces", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _trace_key(record: Mapping[str, Any], trace_path: Path) -> Mapping[str, Any]:
    key = record.get("trace_key")
    if not isinstance(key, Mapping):
        raise ValueError(f"{trace_path}: decision trace record has an invalid trace_key")
    attempt_id = key.get("episode_attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError(
            f"{trace_path}: decision trace record trace_key.episode_attempt_id is invalid"
        )
    return key


def _violation(record: Mapping[str, Any], trace_path: Path) -> Any:
    resolution = record.get("action_resolution")
    if not isinstance(resolution, Mapping):
        raise ValueError(f"{trace_path}: decision trace record is missing action_resolution")
    return resolution.get("violation")


def _final_action(record: Mapping[str, Any], trace_path: Path) -> Any:
    resolution = record.get("action_resolution")
    if not isinstance(resolution, Mapping):
        raise ValueError(f"{trace_path}: decision trace record is missing action_resolution")
    return resolution.get("final_resolved_action")


def _any_shield_applied(record: Mapping[str, Any], trace_path: Path) -> bool:
    stack = record.get("shield_stack")
    if not isinstance(stack, Mapping):
        raise ValueError(f"{trace_path}: decision trace record is missing shield_stack")
    stages = stack.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"{trace_path}: decision trace record shield_stack.stages is invalid")
    return any(isinstance(stage, Mapping) and stage.get("applied") is True for stage in stages)


def _group_by_episode(trace_path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _read_jsonl(trace_path):
        key = _trace_key(record, trace_path)
        grouped[str(key["episode_attempt_id"])].append(record)
    return dict(grouped)


def _validate_trace_identity(
    record: Mapping[str, Any], row: Mapping[str, Any], trace_path: Path
) -> None:
    """Cross-check a subset of identity fields (model_tag/digest/case/seed).

    Mirrors ``_validate_trace_join`` in ``_minimal_factorial_analysis_io.py``
    at a smaller scope, so pointing ``--v8-traces`` at the wrong file (or a
    truncated/regenerated one whose records no longer line up with the
    episodes) raises instead of silently joining mismatched records.
    """
    key = record.get("trace_key", {})
    generation = record.get("generation")
    request = generation.get("request") if isinstance(generation, Mapping) else None
    context = record.get("context")
    if not isinstance(request, Mapping) or not isinstance(context, Mapping):
        raise ValueError(f"{trace_path}: decision trace record is missing generation/context")
    pairs = {
        "model_tag": (request.get("model_tag"), row.get("model_tag")),
        "model_digest": (request.get("model_digest"), row.get("model_digest")),
        "case_id": (key.get("case_id"), row.get("case_id")),
        "simulator_seed": (context.get("simulator_seed"), row.get("simulator_seed")),
    }
    drifted = sorted(
        name for name, (trace_value, row_value) in pairs.items() if trace_value != row_value
    )
    if drifted:
        raise ValueError(
            f"{trace_path}: decision trace identity drifted from V8 episode "
            f"{row.get('episode_attempt_id')!r}: {', '.join(drifted)}"
        )


def _enrich_v8_rows(
    episodes: list[dict[str, Any]], trace_path: Path
) -> tuple[list[dict[str, Any]], dict[str, "Counter[Any]"]]:
    """Fail-closed join of V8 episodes to their decision traces.

    Also builds the per-model-tag executed-action histogram in the same
    pass (one scan of ``trace_path`` serves the Family M count, the
    shield-intervention rate, and the O2 side of the action-distribution
    table -- no second pass over this file).
    """
    grouped = _group_by_episode(trace_path)
    action_histograms: dict[str, Counter[Any]] = defaultdict(Counter)
    enriched: list[dict[str, Any]] = []
    for row in episodes:
        attempt_id = str(row.get("episode_attempt_id"))
        records = grouped.get(attempt_id)
        if not records:
            raise ValueError(
                f"{trace_path}: no decision trace records found for V8 episode "
                f"{attempt_id!r}; check --v8-traces points at the matching, complete "
                "trace file for this campaign"
            )
        for record in records:
            _validate_trace_identity(record, row, trace_path)
        unavailable = sum(
            1 for record in records if _violation(record, trace_path) == "action_unavailable"
        )
        shield_hits = sum(1 for record in records if _any_shield_applied(record, trace_path))
        calls = row.get("decision_calls_total")
        shield_rate = (
            shield_hits / calls
            if isinstance(calls, int) and not isinstance(calls, bool) and calls > 0
            else 0.0
        )
        model_tag = str(row.get("model_tag"))
        for record in records:
            action = _final_action(record, trace_path)
            if action is not None:
                action_histograms[model_tag][action] += 1
        enriched.append(
            {
                **row,
                _ACTION_UNAVAILABLE_FIELD: unavailable,
                _SHIELD_INTERVENTION_RATE_FIELD: shield_rate,
            }
        )
    return enriched, dict(action_histograms)


def _o1_action_histograms(trace_paths: Sequence[Path]) -> dict[str, Counter[Any]]:
    """Executed-action histogram per model_tag over the frozen O1 (c110/c111) traces."""
    histograms: dict[str, Counter[Any]] = defaultdict(Counter)
    for path in trace_paths:
        for record in _read_jsonl(path):
            key = record.get("trace_key")
            if not isinstance(key, Mapping) or key.get("condition_id") not in _FROZEN_O1_CONDITIONS:
                continue
            generation = record.get("generation")
            request = generation.get("request") if isinstance(generation, Mapping) else None
            resolution = record.get("action_resolution")
            model_tag = request.get("model_tag") if isinstance(request, Mapping) else None
            action = (
                resolution.get("final_resolved_action")
                if isinstance(resolution, Mapping)
                else None
            )
            if isinstance(model_tag, str) and action is not None:
                histograms[model_tag][action] += 1
    return dict(histograms)


def _frozen_o1_rows(path: Path) -> list[dict[str, Any]]:
    return [
        row for row in _read_jsonl(path) if row.get("condition_id") in _FROZEN_O1_CONDITIONS
    ]


def _enrich_frozen_o1_rows(
    episodes: list[dict[str, Any]], trace_path: Path
) -> list[dict[str, Any]]:
    """Fail-closed join of frozen O1 comparator rows to their own decision traces.

    Reuses :func:`_enrich_v8_rows` unmodified -- it is agnostic to which
    campaign the rows came from, it only needs the identity fields every
    episode row (V8 or frozen) already carries. Without this, the frozen
    rows would keep only the raw ``lane_change_shield_rate``/
    ``longitudinal_safety_shield_rate``/``flow_recovery_shield_rate``
    fields and lack ``analysis_shield_intervention_rate`` entirely, which
    the descriptive shield-intervention-rate contrast reads on *both*
    sides of the paired V8-minus-O1 difference (see
    ``_grounded_decoding_analysis_families._RowIndex._o1_value``). The
    per-episode ``analysis_action_unavailable_count`` this also computes
    is unused on the frozen side (Family M only ever walks V8 rows) but
    harmless to carry.
    """
    enriched, _histograms = _enrich_v8_rows(episodes, trace_path)
    return enriched


def _by_model_slot(
    histograms_by_tag: Mapping[str, Mapping[Any, int]], tag_to_slot: Mapping[str, str]
) -> dict[str, dict[Any, int]]:
    return {
        tag_to_slot[tag]: dict(counts)
        for tag, counts in histograms_by_tag.items()
        if tag in tag_to_slot
    }


def _manifest_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _report(result: Any) -> str:
    families = {
        "Family A (primary, c121-c111)": result.family_a,
        "Family B (secondary, c120-c110)": result.family_b,
        "Family C (secondary, O2xE DiD)": result.family_c,
        "Family D (endpoint, 120 cases)": result.family_d,
    }
    lines = ["# Registered V8 grounded-decoding analysis", ""]
    for name, rows in families.items():
        lines.append(f"## {name}")
        for row in rows:
            lines.append(
                f"- {row['model_slot']}: estimate={row['estimate']:.4f} "
                f"[{row['lower_2_5']:.4f}, {row['upper_97_5']:.4f}] "
                f"p={row['p_value']:.4f} p_holm={row['p_holm']:.4f}"
            )
        lines.append("")
    lines.append("## Descriptive: action-distribution shift under O2 (no test)")
    for row in result.action_distribution:
        lines.append(
            f"- {row['model_slot']} action={row['action_id']}: "
            f"O2 share={row['o2_share']:.4f} O1 share={row['o1_share']:.4f} "
            f"shift={row['share_shift']:+.4f}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _stats_appendix(result: Any) -> str:
    return (
        "# Statistical appendix\n\n"
        "Intervals use 20,000 category-stratified bootstrap draws and describe "
        "fixed-suite sensitivity, not population confidence. Tests are two-sided "
        "paired sign-flip with 20,000 Monte Carlo draws; Holm correction is "
        "applied within each family separately. Models are never pooled. The "
        "action-distribution-shift table is purely descriptive: no test, no "
        "Holm, no gate.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    v8_rows, o2_histograms_by_tag = _enrich_v8_rows(
        _read_jsonl(args.v8_episodes), args.v8_traces
    )
    frozen_rows = _enrich_frozen_o1_rows(
        _frozen_o1_rows(args.v5_episodes), args.v5_traces
    ) + _enrich_frozen_o1_rows(_frozen_o1_rows(args.v7_episodes), args.v7_traces)
    o1_histograms_by_tag = _o1_action_histograms([args.v5_traces, args.v7_traces])
    tag_to_slot = {str(row["model_tag"]): str(row["model_slot"]) for row in v8_rows}
    manifest_sha256 = _manifest_sha256(args.v8_manifest)

    result = run_registered_v8_analysis(
        v8_rows,
        frozen_rows,
        manifest_sha256=manifest_sha256,
        o2_action_counts=_by_model_slot(o2_histograms_by_tag, tag_to_slot),
        o1_action_counts=_by_model_slot(o1_histograms_by_tag, tag_to_slot),
    )
    if result.validation.status != "complete":
        payload = blocked_payload(result.validation)
        publish_v8_analysis_bundle(args.output_root, result.validation)
        print(json.dumps(payload, sort_keys=True))
        return 2

    tables = V8AnalysisTables(
        result.family_a,
        result.family_b,
        result.family_c,
        result.family_d,
        result.descriptive,
        result.action_distribution,
        _report(result),
        _stats_appendix(result),
    )
    validation_path = publish_v8_analysis_bundle(
        args.output_root, result.validation, tables
    )
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
