"""CLI for the registered ICLR 2027 minimal-factorial analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.runtime.minimal_factorial_analysis import (  # noqa: E402
    AnalysisInputPaths,
    run_registered_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-episodes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_path = run_registered_analysis(
        AnalysisInputPaths(
            args.manifest,
            args.episodes,
            args.baseline_report,
            args.baseline_episodes,
        ),
        output_root=args.output_root,
    )
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
