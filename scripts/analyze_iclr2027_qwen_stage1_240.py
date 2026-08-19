"""CLI for the fixed, non-promotable Qwen Stage-1 diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.runtime._qwen_stage1_240_analysis import (
    run_qwen_stage1_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "iclr2027_minimal_factorial_v5"
            / "diagnostics"
            / "qwen_stage1_240"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_path = run_qwen_stage1_analysis(
        REPO_ROOT,
        output_root=args.output_root,
    )
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
