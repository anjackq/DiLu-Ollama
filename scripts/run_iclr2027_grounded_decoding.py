"""Command-line entry point for the ICLR 2027 grounded-decoding V8 campaign."""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.runtime.grounded_decoding_runner import (
    campaign_status,
    run_claim_stage,
    run_probe_lock,
    run_smoke,
)

DEFAULT_MANIFEST = Path("configs/iclr2027/grounded_decoding_v8.yaml")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe-lock")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--resume", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-episodes", type=_positive_int)
    commands.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if not faulthandler.is_enabled():
        faulthandler.enable()
    args = build_parser().parse_args(argv)
    if args.command == "probe-lock":
        result: object = run_probe_lock(args.manifest)
    elif args.command == "smoke":
        result = run_smoke(args.manifest, resume=args.resume)
    elif args.command == "run":
        result = run_claim_stage(
            args.manifest,
            stage=args.stage,
            resume=args.resume,
            max_episodes=args.max_episodes,
        )
    else:
        result = campaign_status(args.manifest)
    print(json.dumps(_json_value(result), sort_keys=True, default=str))
    return 0


def _json_value(value: object) -> object:
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


if __name__ == "__main__":
    raise SystemExit(main())
