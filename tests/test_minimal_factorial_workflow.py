from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from statistics import fmean
from unittest import mock

from dilu.driver_agent.prompt_modules import build_prompt_artifact
from dilu.runtime._minimal_factorial_analysis_artifacts import (
    EXACT_SUCCESS_FILES,
    publish_analysis_bundle,
)
from dilu.runtime._minimal_factorial_analysis_bootstrap import BootstrapInterval
from dilu.runtime._minimal_factorial_analysis_tables import compute_registered_tables
from dilu.runtime._minimal_factorial_analysis_validation import validate_joined_rows
from dilu.runtime._minimal_factorial_calibration_validation import (
    validate_baseline_bundle,
)
from dilu.runtime._minimal_factorial_manifest import RuntimeSnapshot
from dilu.runtime._minimal_factorial_schedule_support import canonical_sha256
from dilu.runtime._scientific_runtime_binding import (
    RuntimeLockBinding,
    load_verified_runtime_lock_binding,
)
from dilu.runtime.campaign_attempts import (
    AttemptStatus,
    ScientificAttemptLedger,
)
from dilu.runtime.minimal_factorial_calibration import build_calibration_contract
from dilu.runtime.minimal_factorial_schedule import (
    build_smoke_schedule,
    load_experiment_manifest,
)
from dilu.runtime.minimal_factorial_runner import _select_pending_rows
from dilu.runtime.scientific_trace import (
    TRACE_SCHEMA_VERSION,
    TraceReference,
    trace_schema_sha256,
)
from tests.minimal_factorial_analysis_support import synthetic_analysis_bundle

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "minimal_factorial"
MODEL_DIGESTS = {
    "qwen_06b": "sha256:" + "a" * 64,
    "llama_1b": "sha256:" + "b" * 64,
}


class MinimalFactorialOfflineWorkflowTests(unittest.TestCase):
    def test_complete_offline_workflow_is_byte_stable(self) -> None:
        manifest = load_experiment_manifest(
            ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"
        )
        case_set = json.loads((FIXTURE_ROOT / "cases.json").read_text("utf-8"))
        fingerprint = "sha256:" + canonical_sha256(case_set)
        snapshot = RuntimeSnapshot.create(
            {
                "case_set_fingerprint": fingerprint,
                "code_revision": "1" * 40,
            }
        )
        with mock.patch(
            "dilu.runtime.minimal_factorial_schedule.case_fingerprint",
            return_value=fingerprint,
        ):
            schedule = build_smoke_schedule(
                manifest,
                case_set,
                MODEL_DIGESTS,
                runtime_snapshot=snapshot,
            )
        self.assertEqual(len(schedule), 16)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded_locks = _write_and_load_locks(root / "locks", manifest, schedule)
            self.assertEqual(len(loaded_locks), 16)
            self.assertEqual(
                {binding.model_digest for binding in loaded_locks},
                set(MODEL_DIGESTS.values()),
            )

            ledger_path = root / "campaign_attempts.jsonl"
            ledger = ScientificAttemptLedger(
                ledger_path,
                campaign_id=manifest.smoke_campaign_id,
            )
            for line_number, row in enumerate(schedule, start=1):
                ledger.append_started(row.episode_attempt_id)
                ledger.append_terminal(
                    row.episode_attempt_id,
                    status=AttemptStatus.COMPLETED,
                    decision_count=1,
                    trace_references=(_trace_reference(line_number),),
                )
            resumed = ScientificAttemptLedger(
                ledger_path,
                campaign_id=manifest.smoke_campaign_id,
                resume=True,
            )
            self.assertEqual(
                _select_pending_rows(
                    schedule,
                    resumed.attempt_statuses(),
                    resume=True,
                ),
                (),
            )

            claim, episodes, baselines = synthetic_analysis_bundle()
            validation = validate_joined_rows(claim, episodes, baselines)
            self.assertEqual(validation.status, "complete")
            contract = build_calibration_contract(
                claim,
                [row for row in claim["schedule"] if row["condition_id"] == "c111"],
            )
            report_path, csv_path = _write_baseline_bundle(
                root / "baseline",
                baselines,
                contract,
            )
            self.assertTrue(
                validate_baseline_bundle(report_path, csv_path, contract).valid
            )

            with mock.patch(
                "dilu.runtime._minimal_factorial_analysis_tables.stratified_bootstrap",
                side_effect=_fast_interval,
            ):
                first_tables = compute_registered_tables(
                    claim,
                    episodes,
                    baselines,
                    manifest_sha256="sha256:" + "2" * 64,
                )
                second_tables = compute_registered_tables(
                    claim,
                    episodes,
                    baselines,
                    manifest_sha256="sha256:" + "2" * 64,
                )
            first = root / "analysis-a"
            second = root / "analysis-b"
            publish_analysis_bundle(first, validation, first_tables)
            publish_analysis_bundle(second, validation, second_tables)
            self.assertEqual(set(_bundle_bytes(first)), EXACT_SUCCESS_FILES)
            self.assertEqual(_bundle_bytes(first), _bundle_bytes(second))


def _write_and_load_locks(
    root: Path,
    manifest: object,
    schedule: tuple[object, ...],
) -> tuple[object, ...]:
    loaded = []
    for row in schedule:
        directory = root / row.model_slot / row.condition_id
        directory.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt_artifact(
            row.condition.condition.policy_content,
            output_enforcement=row.condition.condition.output_enforcement,
            few_shot_num=0,
        )
        binding = RuntimeLockBinding(
            condition_id=row.condition_id,
            config_sha256="sha256:" + row.condition.config_hash(),
            prompt_sha256="sha256:" + prompt.prompt_hash(),
            model_tag=row.model_tag,
            model_digest=row.model_digest,
            native_endpoint=manifest.transport.native_endpoint,
            think_mode=row.condition.transport.think_mode,
            capability_artifact_sha256=_sha(row.model_slot + "|artifact"),
            capability_snapshot_sha256=_sha(row.model_slot + "|snapshot"),
            trace_schema_sha256=trace_schema_sha256(),
            benchmark_fingerprint=row.benchmark_fingerprint,
            code_revision=row.code_revision,
        )
        runtime_bytes = json.dumps(
            binding.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        runtime_path = directory / "RUNTIME_PROTOCOL_LOCK.json"
        runtime_path.write_bytes(runtime_bytes)
        authorization_path = directory / "PROTOCOL_FROZEN.json"
        authorization_path.write_text(
            json.dumps(
                {
                    "artifact_type": "runtime_lock_authorization_v1",
                    "runtime_lock_sha256": _bytes_sha(runtime_bytes),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        loaded.append(
            load_verified_runtime_lock_binding(
                runtime_lock_path=runtime_path,
                authorization_path=authorization_path,
            )
        )
    return tuple(loaded)


def _write_baseline_bundle(
    root: Path,
    rows: list[dict[str, object]],
    contract: object,
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    report = root / "non_llm_baseline_report.json"
    episodes = root / "episode_metrics.csv"
    report.write_text(
        json.dumps(
            {
                "artifact_type": "matched_non_llm_calibration_report_v1",
                "baselines": list(contract.policies),
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


def _trace_reference(line_number: int) -> TraceReference:
    return TraceReference(
        "traces/decision_traces.jsonl",
        line_number,
        _sha(f"trace|{line_number}"),
        TRACE_SCHEMA_VERSION,
        trace_schema_sha256(),
    )


def _fast_interval(
    values: object,
    *,
    draws: int,
    seed: int,
    **_kwargs: object,
) -> BootstrapInterval:
    flattened = [value for category in values.values() for value in category]
    effect = fmean(flattened)
    return BootstrapInterval(
        effect,
        effect,
        effect,
        draws,
        seed,
        "fixed-suite sensitivity interval",
    )


def _bundle_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _bytes_sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    unittest.main()
