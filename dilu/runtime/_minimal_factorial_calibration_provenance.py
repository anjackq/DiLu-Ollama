"""Live-checkout provenance gates for matched calibration."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._minimal_factorial_schedule_support import canonical_sha256

_REVISION_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def require_frozen_checkout(
    repo_root: Path,
    snapshot: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> None:
    revision = str(snapshot.get("code_revision") or "")
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("Frozen source revision is invalid.")
    observed_revision = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    if observed_revision != revision:
        raise ValueError("Current checkout differs from the frozen source revision.")
    status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=no",
    )
    if status.stdout:
        raise ValueError("Baseline calibration requires a clean frozen checkout.")

    scoring_path = repo_root / "dilu" / "runtime" / "dilu_scoring.py"
    scoring_sha256 = hashlib.sha256(scoring_path.read_bytes()).hexdigest()
    if scoring_sha256 != snapshot.get("scoring_fingerprint"):
        raise ValueError("Current scoring implementation drifted from the claim.")
    success_criteria = [case.get("success_criteria") for case in cases]
    if canonical_sha256(success_criteria) != snapshot.get("predicate_fingerprint"):
        raise ValueError("Current predicate inputs drifted from the claim.")


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Git provenance check failed: {' '.join(args)}.")
    return result


__all__ = ["require_frozen_checkout"]
