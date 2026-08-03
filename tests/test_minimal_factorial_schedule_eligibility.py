from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dilu.runtime.ollama_transport import OllamaModelIdentity

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"


class MinimalFactorialScheduleEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            build_runtime_snapshot,
            build_union_schedule,
            load_experiment_manifest,
        )

        self.manifest = load_experiment_manifest(MANIFEST)
        self.cases = json.loads((ROOT / self.manifest.case_path).read_text())
        self.digests = {
            "qwen_06b": "sha256:" + "a" * 64,
            "llama_1b": "sha256:" + "b" * 64,
        }
        self.bindings = {
            model.slot: OllamaModelIdentity(model.tag, self.digests[model.slot])
            for model in self.manifest.models
        }
        with patch(
            "dilu.runtime._minimal_factorial_manifest.subprocess.run", self._git
        ):
            self.snapshot = build_runtime_snapshot(self.manifest, self.cases)
        self.schedule = build_union_schedule(
            self.manifest,
            self.cases,
            self.digests,
            runtime_snapshot=self.snapshot,
        )

    @staticmethod
    def _git(command, **_kwargs):
        action = command[1]
        stdout = (
            "46367dae59ddf349bacf39b36efb1bf464befcb5\n"
            if action == "rev-parse"
            else command[-1] + "\n"
            if action == "ls-files"
            else ""
        )
        return type("Completed", (), {"stdout": stdout, "returncode": 0})()

    @staticmethod
    def _rehash(row, **changes):
        candidate = replace(row, **changes)
        pair_id = (
            "pair-"
            + hashlib.sha256(
                (
                    f"{candidate.campaign_id}|{candidate.case_id}|"
                    f"{candidate.simulator_seed}"
                ).encode()
            ).hexdigest()
        )
        template_id = (
            "stress-v2-"
            + hashlib.sha256(
                f"{candidate.benchmark_fingerprint}|{candidate.case_id}".encode()
            ).hexdigest()
        )
        primary_snapshot_id = (
            "snapshot-"
            + hashlib.sha256(
                (
                    f"{candidate.benchmark_fingerprint}|{candidate.case_id}|"
                    f"{candidate.simulator_seed}"
                ).encode()
            ).hexdigest()
        )
        episode_attempt_id = (
            "episode-"
            + hashlib.sha256(
                (
                    f"{candidate.campaign_id}|{candidate.model_tag}|"
                    f"{candidate.model_digest}|{candidate.condition_id}|"
                    f"{candidate.case_id}|{candidate.simulator_seed}|0"
                ).encode()
            ).hexdigest()
        )
        return replace(
            candidate,
            pair_id=pair_id,
            template_id=template_id,
            primary_snapshot_id=primary_snapshot_id,
            episode_attempt_id=episode_attempt_id,
        )

    def _assert_rejected(self, rows, name: str) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            write_frozen_campaign_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            try:
                with self.assertRaises(ValueError):
                    write_frozen_campaign_manifest(
                        Path(directory) / f"{name}.json",
                        self.manifest,
                        self.snapshot,
                        rows,
                        case_set=self.cases,
                        model_bindings=self.bindings,
                    )
            except TypeError:
                self.fail("writer must require explicit trusted model_bindings")

    def test_writer_rejects_unknown_and_mixed_stage_vocabularies(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import build_smoke_schedule

        unknown = (replace(self.schedule[0], stage="stage3"), *self.schedule[1:])
        self._assert_rejected(unknown, "unknown-stage")
        smoke = build_smoke_schedule(
            self.manifest,
            self.cases,
            self.digests,
            runtime_snapshot=self.snapshot,
        )
        self._assert_rejected((*self.schedule, smoke[0]), "mixed-stage")

    def test_writer_rejects_nonexistent_rehashed_case(self) -> None:
        tampered = self._rehash(
            self.schedule[0],
            case_id="nonexistent_case_999",
            simulator_seed=999_999,
        )
        self._assert_rejected((tampered, *self.schedule[1:]), "unknown-case")

    def test_writer_rejects_wrong_stage_eligibility(self) -> None:
        wrong = replace(self.schedule[0], stage="stage2_additional")
        self._assert_rejected((wrong, *self.schedule[1:]), "wrong-stage")

    def test_writer_rejects_duplicate_missing_and_extra_rows(self) -> None:
        self._assert_rejected((*self.schedule[:-1], self.schedule[0]), "duplicate")
        self._assert_rejected(self.schedule[:-1], "missing")
        extra = replace(self.schedule[0], stage="stage2_additional")
        self._assert_rejected((*self.schedule, extra), "extra")

    def test_writer_rejects_wrong_case_seed(self) -> None:
        wrong = replace(
            self.schedule[0],
            simulator_seed=self.schedule[0].simulator_seed + 1,
        )
        self._assert_rejected((wrong, *self.schedule[1:]), "wrong-seed")

    def test_writer_rejects_coordinated_rehashed_case_and_stage_tamper(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import select_stage1_cases

        selected_ids = {
            case["case_id"]
            for case in select_stage1_cases(
                self.cases, self.manifest.selection.stage1_hash_prefix
            )
        }
        remaining = next(
            case for case in self.cases["cases"] if case["case_id"] not in selected_ids
        )
        index = next(
            index
            for index, row in enumerate(self.schedule)
            if row.condition_id == "c001"
        )
        wrong = self._rehash(
            self.schedule[index],
            case_id=remaining["case_id"],
            simulator_seed=remaining["seed"],
        )
        mutated = (*self.schedule[:index], wrong, *self.schedule[index + 1 :])
        self._assert_rejected(mutated, "coordinated")

    def test_writer_rejects_multiple_digests_for_one_model_slot(self) -> None:
        wrong = self._rehash(
            self.schedule[0],
            model_digest="sha256:" + "c" * 64,
        )
        self._assert_rejected((wrong, *self.schedule[1:]), "multiple-digests")

    def test_writer_rejects_coordinated_digest_replacement_for_entire_slot(
        self,
    ) -> None:
        tampered = tuple(
            self._rehash(row, model_digest="sha256:" + "c" * 64)
            if row.model_slot == "qwen_06b"
            else row
            for row in self.schedule
        )
        self._assert_rejected(tampered, "coordinated-slot-digest")

    def test_writer_serializes_permuted_rows_in_canonical_order(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            write_frozen_campaign_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            for path, rows in (
                (first, self.schedule),
                (second, tuple(reversed(self.schedule))),
            ):
                write_frozen_campaign_manifest(
                    path,
                    self.manifest,
                    self.snapshot,
                    rows,
                    case_set=self.cases,
                    model_bindings=self.bindings,
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            original = first.read_bytes()
            write_frozen_campaign_manifest(
                first,
                self.manifest,
                self.snapshot,
                tuple(reversed(self.schedule)),
                case_set=self.cases,
                model_bindings=self.bindings,
            )
            self.assertEqual(first.read_bytes(), original)

    def test_writer_rejects_untrusted_binding_slots_tags_and_digests(self) -> None:
        from dilu.runtime.minimal_factorial_schedule import (
            write_frozen_campaign_manifest,
        )

        qwen = self.bindings["qwen_06b"]
        invalid_bindings = {
            "missing_slot": {"qwen_06b": qwen},
            "extra_slot": {
                **self.bindings,
                "extra": OllamaModelIdentity("extra:model", "sha256:" + "d" * 64),
            },
            "wrong_tag": {
                **self.bindings,
                "qwen_06b": OllamaModelIdentity(
                    "wrong:model",
                    qwen.model_digest,
                ),
            },
            "wrong_digest": {
                **self.bindings,
                "qwen_06b": OllamaModelIdentity(
                    qwen.model_tag,
                    "sha256:" + "c" * 64,
                ),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            for label, bindings in invalid_bindings.items():
                with self.subTest(label=label), self.assertRaises(ValueError):
                    write_frozen_campaign_manifest(
                        Path(directory) / f"{label}.json",
                        self.manifest,
                        self.snapshot,
                        self.schedule,
                        case_set=self.cases,
                        model_bindings=bindings,
                    )


if __name__ == "__main__":
    unittest.main()
