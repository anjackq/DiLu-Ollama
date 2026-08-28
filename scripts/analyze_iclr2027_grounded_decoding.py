"""CLI for the registered ICLR 2027 grounded-decoding (V8) analysis.

Wires real filesystem I/O around the pure library in
``dilu.runtime.grounded_decoding_analysis``: reads the 480 V8 episode rows,
enriches them with the per-episode ``action_unavailable`` violation count
the Family M gate needs (read-only over ``decision_traces.jsonl`` -- the
trace schema itself is untouched), reads the frozen V5/V7 comparator rows
(only the O1 cells V8 pairs against: c110/c111), then runs and publishes
the registered analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-manifest", type=Path, required=True)
    parser.add_argument("--v8-episodes", type=Path, required=True)
    parser.add_argument("--v8-traces", type=Path, required=True)
    parser.add_argument("--v5-episodes", type=Path, required=True)
    parser.add_argument("--v7-episodes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _action_unavailable_counts(trace_path: Path) -> dict[str, int]:
    """Per-episode count of decisions whose violation was ``action_unavailable``."""
    counts: dict[str, int] = defaultdict(int)
    for record in _read_jsonl(trace_path):
        key = record.get("trace_key", {})
        episode_attempt_id = key.get("episode_attempt_id")
        if not isinstance(episode_attempt_id, str):
            continue
        if record.get("action_resolution", {}).get("violation") == "action_unavailable":
            counts[episode_attempt_id] += 1
        else:
            counts.setdefault(episode_attempt_id, 0)
    return dict(counts)


def _enrich_v8_rows(episodes: list[dict[str, Any]], trace_path: Path) -> list[dict[str, Any]]:
    counts = _action_unavailable_counts(trace_path)
    enriched = []
    for row in episodes:
        attempt_id = str(row.get("episode_attempt_id"))
        enriched.append({**row, _ACTION_UNAVAILABLE_FIELD: counts.get(attempt_id, 0)})
    return enriched


def _frozen_o1_rows(path: Path) -> list[dict[str, Any]]:
    return [
        row for row in _read_jsonl(path) if row.get("condition_id") in _FROZEN_O1_CONDITIONS
    ]


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
    return "\n".join(lines) + "\n"


def _stats_appendix(result: Any) -> str:
    return (
        "# Statistical appendix\n\n"
        "Intervals use 20,000 category-stratified bootstrap draws and describe "
        "fixed-suite sensitivity, not population confidence. Tests are two-sided "
        "paired sign-flip with 20,000 Monte Carlo draws; Holm correction is "
        "applied within each family separately. Models are never pooled.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    v8_rows = _enrich_v8_rows(_read_jsonl(args.v8_episodes), args.v8_traces)
    frozen_rows = _frozen_o1_rows(args.v5_episodes) + _frozen_o1_rows(args.v7_episodes)
    manifest_sha256 = _manifest_sha256(args.v8_manifest)

    result = run_registered_v8_analysis(
        v8_rows, frozen_rows, manifest_sha256=manifest_sha256
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
