"""Atomic three-file publisher for the fixed Qwen Stage-1 diagnostic."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_ROWS = 240
EXACT_FILES = frozenset(
    {"validation.json", "condition_summary.csv", "factor_contrasts.csv"}
)


@dataclass(frozen=True)
class QwenStage1Tables:
    condition_summary: tuple[Mapping[str, Any], ...]
    factor_contrasts: tuple[Mapping[str, Any], ...]


def publish_qwen_stage1_bundle(
    output_root: Path,
    validation: Mapping[str, Any],
    tables: QwenStage1Tables | None = None,
) -> Path:
    target = Path(output_root).resolve()
    if target.exists():
        raise FileExistsError(f"Diagnostic output already exists: {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-stage-", dir=target.parent))
    try:
        payload = dict(validation)
        payload["errors"] = sorted({str(error) for error in payload.get("errors", ())})
        _write_json(stage / "validation.json", payload)
        if payload.get("status") == "complete":
            if (
                tables is None
                or payload.get("claim_eligible") is not False
                or payload.get("expected_rows") != EXPECTED_ROWS
                or payload.get("observed_rows") != EXPECTED_ROWS
            ):
                raise ValueError("Complete diagnostic requires the exact fixed scope.")
            _write_csv(stage / "condition_summary.csv", tables.condition_summary)
            _write_csv(stage / "factor_contrasts.csv", tables.factor_contrasts)
            if _relative_files(stage) != EXACT_FILES:
                raise RuntimeError("Diagnostic artifact layout drifted.")
        elif tables is not None or _relative_files(stage) != {"validation.json"}:
            raise ValueError("Blocked diagnostic must publish validation only.")
        os.rename(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target / "validation.json"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Diagnostic table cannot be empty: {path.name}.")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _relative_files(root: Path) -> frozenset[str]:
    return frozenset(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


__all__ = ["EXACT_FILES", "QwenStage1Tables", "publish_qwen_stage1_bundle"]
