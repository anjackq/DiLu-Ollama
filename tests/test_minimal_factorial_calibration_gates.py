from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime._campaign_attempt_state import AttemptLedgerSnapshot
from dilu.runtime._minimal_factorial_calibration_completion import (
    require_complete_claim_campaign,
)
from dilu.runtime._minimal_factorial_calibration_provenance import (
    require_frozen_checkout,
)
from dilu.runtime._minimal_factorial_schedule_support import canonical_sha256
from dilu.runtime.campaign_attempts import AttemptStatus, ScientificAttemptWriteError
from dilu.runtime.minimal_factorial_calibration import run_baseline_campaign
from tests.test_minimal_factorial_calibration import _claim_bundle


class MinimalFactorialCalibrationGateTests(unittest.TestCase):
    def test_runner_rejects_evaluator_error_before_publishing(self) -> None:
        claim, cases = _claim_bundle()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            claim_path = (
                repo
                / "results"
                / "iclr2027"
                / "llm_campaign"
                / "campaign_manifest.json"
            )
            claim_path.parent.mkdir(parents=True)
            claim_path.write_text(json.dumps(claim, sort_keys=True), encoding="utf-8")
            (repo / "cases.json").write_text(
                json.dumps(cases, sort_keys=True), encoding="utf-8"
            )
            output = repo / "results" / "iclr2027" / "baselines"
            environment = claim["runtime_snapshot"]["environment_config"]
            failed_episode = {
                "baseline_safety_shields_enabled": True,
                "episode_stop_reason": "error",
                "error": "simulator unavailable",
            }
            with (
                mock.patch(
                    "dilu.runtime.minimal_factorial_calibration."
                    "require_complete_claim_campaign"
                ),
                mock.patch(
                    "dilu.runtime.minimal_factorial_calibration.require_frozen_checkout"
                ),
                mock.patch(
                    "dilu.runtime.minimal_factorial_calibration."
                    "resolve_simulation_env_bundle",
                    return_value={
                        "env_config_snapshot": environment,
                        "env_config_map": {"highway-fast-v0": environment},
                    },
                ),
                mock.patch(
                    "dilu.runtime.minimal_factorial_calibration.run_baseline_episode",
                    return_value=failed_episode,
                ),
                self.assertRaisesRegex(RuntimeError, "evaluator failed"),
            ):
                run_baseline_campaign(claim_path, output_root=output)

            self.assertFalse(any(output.iterdir()))

    def test_completion_gate_rejects_ledger_trace_join_failure(self) -> None:
        claim, _cases = _claim_bundle()
        schedule = claim["schedule"]
        statuses = {
            row["episode_attempt_id"]: AttemptStatus.COMPLETED for row in schedule
        }
        summaries = [
            {"episode_attempt_id": row["episode_attempt_id"]} for row in schedule
        ]
        durable_ledger = mock.Mock()
        durable_ledger.validate_trace_evidence.side_effect = (
            ScientificAttemptWriteError("trace references drifted")
        )
        with (
            mock.patch(
                "dilu.runtime._minimal_factorial_calibration_completion."
                "read_validated_attempt_snapshot",
                return_value=AttemptLedgerSnapshot(statuses, frozenset()),
            ),
            mock.patch(
                "dilu.runtime._minimal_factorial_calibration_completion."
                "load_summary_records",
                return_value=tuple(summaries),
            ),
            mock.patch(
                "dilu.runtime._minimal_factorial_calibration_completion."
                "ScientificAttemptLedger",
                return_value=durable_ledger,
            ),
            mock.patch(
                "dilu.runtime._minimal_factorial_calibration_completion."
                "ScientificTraceWriter"
            ),
            self.assertRaisesRegex(ValueError, "ledger-to-trace"),
        ):
            require_complete_claim_campaign(
                Path("claim") / "campaign_manifest.json",
                claim,
            )

    def test_checkout_provenance_rejects_revision_and_dirty_drift(self) -> None:
        claim, cases = _claim_bundle()
        snapshot = claim["runtime_snapshot"]
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            scoring = repo / "dilu" / "runtime" / "dilu_scoring.py"
            scoring.parent.mkdir(parents=True)
            scoring.write_text("SCORING_VERSION = 1\n", encoding="utf-8")
            snapshot["scoring_fingerprint"] = hashlib.sha256(
                scoring.read_bytes()
            ).hexdigest()
            snapshot["predicate_fingerprint"] = canonical_sha256(
                [case["success_criteria"] for case in cases["cases"]]
            )
            wrong_revision = subprocess.CompletedProcess(
                args=["git"], returncode=0, stdout="b" * 40 + "\n", stderr=""
            )
            with (
                mock.patch(
                    "dilu.runtime._minimal_factorial_calibration_provenance._git",
                    return_value=wrong_revision,
                ),
                self.assertRaisesRegex(ValueError, "differs from the frozen"),
            ):
                require_frozen_checkout(repo, snapshot, cases["cases"])

            frozen_revision = subprocess.CompletedProcess(
                args=["git"], returncode=0, stdout="a" * 40 + "\n", stderr=""
            )
            dirty = subprocess.CompletedProcess(
                args=["git"], returncode=0, stdout="?? local.py\n", stderr=""
            )
            with (
                mock.patch(
                    "dilu.runtime._minimal_factorial_calibration_provenance._git",
                    side_effect=(frozen_revision, dirty),
                ),
                self.assertRaisesRegex(ValueError, "clean frozen checkout"),
            ):
                require_frozen_checkout(repo, snapshot, cases["cases"])


if __name__ == "__main__":
    unittest.main()
