from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dilu.runtime import _minimal_factorial_runner_status as runner_status
from dilu.runtime._campaign_attempt_state import AttemptLedgerSnapshot
from dilu.runtime._minimal_factorial_manifest import RuntimeSnapshot
from dilu.runtime.campaign_attempts import AttemptStatus


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "iclr2027" / "model_breadth_factorial_v7.yaml"


class V7FullFactorialTests(unittest.TestCase):
    def test_v7_builds_exact_three_model_full_factorial(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_smoke_schedule,
            build_union_schedule,
            load_experiment_manifest,
        )

        manifest = load_experiment_manifest(MANIFEST)
        cases = json.loads((ROOT / manifest.case_path).read_text(encoding="utf-8"))
        snapshot = RuntimeSnapshot.create(
            {
                "case_set_fingerprint": (
                    "sha256:bd6d65d694a1452e0770e9854e478bb463be8302168e8c17396e86786401fd33"
                ),
                "code_revision": "a" * 40,
            }
        )
        digests = {
            "llama_3b": "sha256:" + "a" * 64,
            "gemma_4b": "sha256:" + "b" * 64,
            "qwen_8b": "sha256:" + "c" * 64,
        }

        claim = build_union_schedule(
            manifest, cases, digests, runtime_snapshot=snapshot
        )
        smoke = build_smoke_schedule(
            manifest, cases, digests, runtime_snapshot=snapshot
        )

        self.assertEqual(len(claim), 720)
        self.assertEqual(len(smoke), 24)
        self.assertEqual({row.stage for row in claim}, {"stage1"})
        self.assertEqual(
            {row.condition_id for row in claim},
            {f"c{index:03b}" for index in range(8)},
        )
        self.assertEqual(len({row.case_id for row in claim}), 30)
        self.assertEqual(len({row.episode_attempt_id for row in claim}), 720)
        for model in digests:
            for condition in {f"c{index:03b}" for index in range(8)}:
                self.assertEqual(
                    sum(
                        row.model_slot == model and row.condition_id == condition
                        for row in claim
                    ),
                    30,
                )

    def test_complete_720_row_frozen_claim_is_promotion_eligible(self) -> None:
        count = 720
        rows = tuple(self._row(index) for index in range(count))
        prepared = SimpleNamespace(
            schedule=rows,
            output_root=Path("claim"),
            snapshot=SimpleNamespace(sha256="a" * 64),
        )
        statuses = {
            row.episode_attempt_id: AttemptStatus.COMPLETED for row in rows
        }
        summaries = tuple(
            {
                "episode_attempt_id": row.episode_attempt_id,
                "scientific_trace_references": [self._reference(index + 1)],
            }
            for index, row in enumerate(rows)
        )
        traces = {
            (row.campaign_id, row.episode_attempt_id): (self._reference(index + 1),)
            for index, row in enumerate(rows)
        }
        with (
            mock.patch.object(
                runner_status,
                "_read_attempt_snapshot",
                return_value=AttemptLedgerSnapshot(statuses, frozenset()),
            ),
            mock.patch.object(
                runner_status,
                "load_summary_records",
                return_value=summaries,
            ),
            mock.patch.object(
                runner_status,
                "_read_validated_trace_references",
                return_value=traces,
            ),
        ):
            report = runner_status.campaign_status((prepared,))

        self.assertTrue(report["artifact_validation"]["valid"])
        self.assertTrue(
            report["artifact_validation"]["claim_promotion_allowed"]
        )
        self.assertEqual(report["totals"]["completed"], 720)

    @staticmethod
    def _row(index: int) -> SimpleNamespace:
        payload = {
            "stage": "stage1",
            "campaign_id": "iclr2027-model-breadth-factorial-v7",
            "episode_attempt_id": f"episode-{index:03d}",
            "model_slot": "model",
            "condition_id": f"c{index % 8:03b}",
        }
        return SimpleNamespace(**payload, to_payload=lambda payload=payload: payload)

    @staticmethod
    def _reference(line_number: int) -> dict[str, object]:
        return {
            "relative_path": "traces/decision_traces.jsonl",
            "line_number": line_number,
            "record_sha256": "sha256:" + f"{line_number:064x}",
            "schema_version": "iclr2027.scientific_trace.v1",
            "schema_sha256": "sha256:" + "e" * 64,
        }


if __name__ == "__main__":
    unittest.main()
