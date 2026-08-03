"""Command-line entry point for the ICLR 2027 minimal factorial."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Sequence

from dilu.runtime.minimal_factorial_runner import (
    campaign_status,
    run_claim_stage,
    run_probe_lock,
    run_smoke,
)

DEFAULT_MANIFEST = Path("configs/iclr2027/minimal_factorial.yaml")


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
    commands.add_parser("baselines")
    commands.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest
    if args.command == "probe-lock":
        result: object = run_probe_lock(manifest)
    elif args.command == "smoke":
        result = run_smoke(manifest, resume=args.resume)
    elif args.command == "run":
        result = run_claim_stage(
            manifest,
            stage=args.stage,
            resume=args.resume,
        )
    elif args.command == "baselines":
        result = _run_baselines(manifest)
    else:
        result = campaign_status(manifest)
    print(json.dumps(_json_value(result), sort_keys=True, default=str))
    return 0


def _run_baselines(manifest_path: Path) -> Path:
    try:
        from dilu.runtime.minimal_factorial_calibration import (
            run_baseline_campaign,
        )
    except ImportError as exc:
        raise RuntimeError("Baseline calibration support is not installed.") from exc
    from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

    manifest = load_experiment_manifest(manifest_path)
    repo_root = manifest_path.resolve().parents[2]
    base = repo_root / manifest.outputs.root
    return run_baseline_campaign(
        base / manifest.outputs.llm_campaign / "campaign_manifest.json",
        output_root=base / manifest.outputs.baselines,
    )


def _json_value(value: object) -> object:
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


if __name__ == "__main__":
    raise SystemExit(main())
