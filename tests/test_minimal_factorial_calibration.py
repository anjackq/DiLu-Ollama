from __future__ import annotations

import copy
import csv
import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime._minimal_factorial_schedule_support import canonical_sha256
from dilu.runtime.harness_config import ShieldConfig
from dilu.runtime.minimal_factorial_calibration import (
    build_calibration_contract,
    run_baseline_campaign,
    validate_baseline_bundle,
)


POLICIES = ("always_left", "speed_hold_25", "idm_mobil")


def _claim_bundle() -> tuple[dict[str, object], dict[str, object]]:
    shield = json.loads(
        json.dumps(dataclasses.asdict(ShieldConfig.implementation_defaults()))
    )
    cases = {
        "benchmark_name": "stress-v2",
        "cases": [
            {
                "case_id": f"case-{index:03d}",
                "seed": 1000 + index,
                "success_criteria": {"kind": "complete"},
            }
            for index in range(120)
        ],
    }
    benchmark_fingerprint = "sha256:" + canonical_sha256(cases)
    c111 = [
        {
            "episode_attempt_id": f"c111-{model}-{index:03d}",
            "campaign_id": "campaign-001",
            "model_slot": model,
            "condition_id": "c111",
            "condition": {
                "condition": {"execution_mode": "shielded"},
                "shield": shield,
            },
            "case_id": case["case_id"],
            "simulator_seed": case["seed"],
            "benchmark_fingerprint": benchmark_fingerprint,
            "code_revision": "a" * 40,
        }
        for model in ("qwen", "llama")
        for index, case in enumerate(cases["cases"])
    ]
    filler = [
        {
            "episode_attempt_id": f"filler-{index:03d}",
            "campaign_id": "campaign-001",
            "condition_id": "c000",
        }
        for index in range(600)
    ]
    runtime = {"simulation_duration": 30, "OPENAI_API_TYPE": "ollama"}
    environment = {"duration": 30, "vehicles_count": 10}
    snapshot = {
        "code_revision": "a" * 40,
        "runtime_config": runtime,
        "environment_config": environment,
        "shield_config": shield,
        "predicate_fingerprint": "c" * 64,
        "scoring_fingerprint": "d" * 64,
        "case_set_fingerprint": benchmark_fingerprint,
        "primary_metric_spec": {
            "metric": "driving_score_balanced_v1",
            "version": "balanced_driving_score_policy_v1",
        },
    }
    claim = {
        "manifest": {
            "case_path": "cases.json",
            "simulation": {"target_env_id": "highway-fast-v0"},
            "outputs": {
                "root": "results/iclr2027",
                "llm_campaign": "llm_campaign",
            },
        },
        "runtime_snapshot": snapshot,
        "runtime_snapshot_sha256": canonical_sha256(snapshot),
        "schedule": c111 + filler,
    }
    return claim, cases


def _contract_rows(contract: object) -> list[dict[str, object]]:
    return [
        {
            **dict(contract.provenance),
            "baseline_policy": policy,
            "case_id": case_id,
            "simulator_seed": seed,
            "safety_shields_enabled": True,
            "driving_score_balanced_v1": 0.5,
            "balanced_driving_score_policy_version": contract.scoring_version,
        }
        for policy in contract.policies
        for case_id, seed in contract.case_seeds
    ]


def _write_bundle(
    root: Path,
    rows: list[dict[str, object]],
    contract: object,
) -> tuple[Path, Path]:
    report = root / "non_llm_baseline_report.json"
    episodes = root / "episode_metrics.csv"
    report.write_text(
        json.dumps(
            {
                "artifact_type": "matched_non_llm_calibration_report_v1",
                "baselines": list(POLICIES),
                "episode_count": contract.expected_rows,
                "safety_shields_enabled": True,
                "provenance": dict(contract.provenance),
                "episodes": rows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with episodes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report, episodes


class MinimalFactorialCalibrationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.claim, self.cases = _claim_bundle()
        self.c111 = [
            row for row in self.claim["schedule"] if row["condition_id"] == "c111"
        ]
        self.contract = build_calibration_contract(self.claim, self.c111)

    def test_contract_freezes_three_policies_cases_shield_and_provenance(self) -> None:
        self.assertEqual(self.contract.policies, POLICIES)
        self.assertEqual(len(self.contract.case_seeds), 120)
        self.assertEqual(self.contract.expected_rows, 360)
        self.assertEqual(
            dict(self.contract.shield_config),
            json.loads(
                json.dumps(dataclasses.asdict(ShieldConfig.implementation_defaults()))
            ),
        )
        self.assertEqual(
            set(self.contract.provenance),
            {
                "benchmark_fingerprint",
                "case_set_sha256",
                "environment_config_sha256",
                "predicate_sha256",
                "resolved_runtime_config_sha256",
                "scoring_sha256",
                "primary_metric_spec_sha256",
                "scoring_policy_version",
                "shield_config_sha256",
                "source_revision",
            },
        )

    def test_contract_rejects_same_ids_with_altered_c111_payload(self) -> None:
        altered = copy.deepcopy(self.c111)
        altered[0]["simulator_seed"] = 999999

        with self.assertRaisesRegex(ValueError, "exact 240-row c111"):
            build_calibration_contract(self.claim, altered)

    def test_invalid_row_sets_block_calibration(self) -> None:
        valid_rows = _contract_rows(self.contract)
        variants: dict[str, list[dict[str, object]]] = {
            "missing": valid_rows[:-1],
            "extra_policy": valid_rows
            + [{**valid_rows[0], "baseline_policy": "idle_always"}],
            "seed_drift": copy.deepcopy(valid_rows),
            "unshielded": copy.deepcopy(valid_rows),
            "scoring_drift": copy.deepcopy(valid_rows),
            "scoring_version_drift": copy.deepcopy(valid_rows),
            "evaluator_error": copy.deepcopy(valid_rows),
        }
        variants["seed_drift"][0]["simulator_seed"] = 999999
        variants["unshielded"][0]["safety_shields_enabled"] = False
        variants["scoring_drift"][0]["scoring_sha256"] = "sha256:" + "0" * 64
        variants["scoring_version_drift"][0][
            "balanced_driving_score_policy_version"
        ] = "drifted"
        variants["evaluator_error"][0]["error"] = "simulator unavailable"
        variants["evaluator_error"][0]["episode_stop_reason"] = "error"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, rows in variants.items():
                case_root = root / name
                case_root.mkdir()
                report, episodes = _write_bundle(case_root, rows, self.contract)
                validation = validate_baseline_bundle(
                    report,
                    episodes,
                    self.contract,
                )
                self.assertFalse(validation.valid, name)
                self.assertTrue(validation.errors, name)

    def test_exact_rows_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report, episodes = _write_bundle(
                Path(tmp),
                _contract_rows(self.contract),
                self.contract,
            )

            validation = validate_baseline_bundle(
                report,
                episodes,
                self.contract,
            )

        self.assertTrue(validation.valid)
        self.assertEqual(validation.observed_rows, 360)

    def test_report_and_csv_metric_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report, episodes = _write_bundle(
                Path(tmp),
                _contract_rows(self.contract),
                self.contract,
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["episodes"][0]["driving_score_balanced_v1"] = 0.75
            report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            validation = validate_baseline_bundle(report, episodes, self.contract)

        self.assertFalse(validation.valid)
        self.assertIn("report and CSV episode contents differ", validation.errors)


class MinimalFactorialCalibrationRunnerTests(unittest.TestCase):
    def test_cli_help_imports_from_repository_checkout(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_iclr2027_minimal_factorial.py",
                "--help",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("baselines", result.stdout)

    def test_runner_delegates_all_360_shielded_episodes(self) -> None:
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
                json.dumps(cases, sort_keys=True),
                encoding="utf-8",
            )
            output = repo / "results" / "iclr2027" / "baselines"
            environment = claim["runtime_snapshot"]["environment_config"]

            def episode(**kwargs: object) -> dict[str, object]:
                policy = kwargs["policy"]
                case = kwargs["case"]
                return {
                    "baseline_policy": policy.name,
                    "baseline_safety_shields_enabled": True,
                    "case_id": case["case_id"],
                    "simulator_seed": case["seed"],
                    "safety_shields_enabled": kwargs["safety_shields_enabled"],
                    "balanced_driving_score_policy_version": (
                        claim["runtime_snapshot"]["primary_metric_spec"]["version"]
                    ),
                }

            with (
                mock.patch(
                    "dilu.runtime.minimal_factorial_calibration."
                    "require_complete_claim_campaign"
                ) as require_complete,
                mock.patch(
                    "dilu.runtime.minimal_factorial_calibration.require_frozen_checkout"
                ) as require_checkout,
                mock.patch(
                    "dilu.runtime.minimal_factorial_calibration.resolve_simulation_env_bundle",
                    return_value={
                        "env_config_snapshot": environment,
                        "env_config_map": {"highway-fast-v0": environment},
                    },
                ),
                mock.patch(
                    "dilu.runtime.minimal_factorial_calibration.run_baseline_episode",
                    side_effect=episode,
                ) as run_episode,
                mock.patch(
                    "dilu.runtime.minimal_factorial_calibration.aggregate_results",
                    side_effect=lambda name, rows, **_kwargs: {
                        "baseline_policy": name,
                        "episode_count": len(rows),
                    },
                ),
            ):
                validation_path = run_baseline_campaign(
                    claim_path,
                    output_root=output,
                )

            self.assertEqual(run_episode.call_count, 360)
            require_complete.assert_called_once()
            require_checkout.assert_called_once()
            self.assertEqual(
                {call.kwargs["policy"].name for call in run_episode.call_args_list},
                set(POLICIES),
            )
            self.assertTrue(
                all(
                    call.kwargs["safety_shields_enabled"]
                    for call in run_episode.call_args_list
                )
            )
            self.assertEqual(validation_path, output / "calibration_validation.json")
            for name in (
                "baseline_manifest.json",
                "non_llm_baseline_report.json",
                "episode_metrics.csv",
                "calibration_validation.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertTrue(validation["valid"])

    def test_runner_blocks_when_claim_completion_evidence_is_missing(self) -> None:
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
                json.dumps(cases, sort_keys=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "840/840 completed"):
                run_baseline_campaign(
                    claim_path,
                    output_root=repo / "results" / "iclr2027" / "baselines",
                )


if __name__ == "__main__":
    unittest.main()
