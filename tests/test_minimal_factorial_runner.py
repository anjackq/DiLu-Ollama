from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Literal, get_type_hints
from unittest import mock

from dilu.runtime import minimal_factorial_runner as runner


def _rows(stage: str, count: int) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(stage=stage, episode_attempt_id=f"{stage}-{index}")
        for index in range(count)
    )


class MinimalFactorialPublicApiTests(unittest.TestCase):
    def test_public_runner_signatures_are_narrow(self) -> None:
        self.assertEqual(
            str(inspect.signature(runner.run_probe_lock)),
            "(manifest_path: 'Path') -> 'Path'",
        )
        self.assertEqual(
            str(inspect.signature(runner.run_smoke)),
            "(manifest_path: 'Path', *, resume: 'bool') -> 'RunSummary'",
        )
        claim_signature = inspect.signature(runner.run_claim_stage)
        self.assertEqual(
            tuple(claim_signature.parameters),
            ("manifest_path", "stage", "resume", "max_episodes"),
        )
        self.assertEqual(
            get_type_hints(runner.run_claim_stage)["stage"],
            Literal["stage1", "stage2"],
        )

    def test_probe_lock_delegates_to_s1_authoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "configs" / "iclr2027" / "minimal_factorial.yaml"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("placeholder", encoding="utf-8")
            authored = SimpleNamespace(
                preflight_path=root / "results" / "s1" / "model_preflight.json"
            )
            with mock.patch.object(
                runner,
                "author_verified_runtime_locks",
                return_value=authored,
            ) as author:
                result = runner.run_probe_lock(manifest_path)

        self.assertEqual(result, authored.preflight_path)
        author.assert_called_once_with(root, manifest_path=manifest_path.resolve())


class MinimalFactorialSchedulingTests(unittest.TestCase):
    def test_smoke_executes_exactly_16_rows_in_smoke_root(self) -> None:
        smoke_rows = _rows("smoke", 16)
        prepared = SimpleNamespace(
            schedule=smoke_rows,
            output_root=Path("results") / "smoke",
        )
        expected = runner.RunSummary(
            stage="smoke",
            output_root=prepared.output_root,
            scheduled=16,
            completed=16,
            blocked=0,
            failed=0,
            ambiguous=0,
            resumable=0,
            pending=0,
        )
        with (
            mock.patch.object(runner, "_prepare_campaign", return_value=prepared),
            mock.patch.object(
                runner, "_execute_campaign", return_value=expected
            ) as execute,
        ):
            actual = runner.run_smoke(Path("manifest.yaml"), resume=False)

        self.assertEqual(actual, expected)
        execute.assert_called_once_with(
            prepared,
            scheduled_rows=smoke_rows,
            denominator_rows=smoke_rows,
            resume=False,
            stage="smoke",
        )

    def test_stage1_executes_exactly_480_claim_rows(self) -> None:
        union = _rows("stage1", 480) + _rows("stage2_additional", 360)
        prepared = SimpleNamespace(schedule=union, output_root=Path("llm_campaign"))
        with (
            mock.patch.object(runner, "_prepare_campaign", return_value=prepared),
            mock.patch.object(
                runner,
                "_execute_campaign",
                return_value=mock.sentinel.summary,
            ) as execute,
        ):
            result = runner.run_claim_stage(
                Path("manifest.yaml"),
                stage="stage1",
                resume=False,
            )

        self.assertIs(result, mock.sentinel.summary)
        call = execute.call_args
        self.assertEqual(len(call.kwargs["scheduled_rows"]), 480)
        self.assertEqual(len(call.kwargs["denominator_rows"]), 480)
        self.assertEqual(call.kwargs["stage"], "stage1")

    def test_stage2_executes_only_360_additional_rows_with_full_denominator(
        self,
    ) -> None:
        stage1 = _rows("stage1", 480)
        additional = _rows("stage2_additional", 360)
        prepared = SimpleNamespace(
            schedule=stage1 + additional,
            output_root=Path("llm_campaign"),
        )
        with (
            mock.patch.object(runner, "_prepare_campaign", return_value=prepared),
            mock.patch.object(
                runner,
                "_execute_campaign",
                return_value=mock.sentinel.summary,
            ) as execute,
        ):
            result = runner.run_claim_stage(
                Path("manifest.yaml"),
                stage="stage2",
                resume=True,
                max_episodes=20,
            )

        self.assertIs(result, mock.sentinel.summary)
        call = execute.call_args
        self.assertEqual(call.kwargs["scheduled_rows"], additional)
        self.assertEqual(call.kwargs["denominator_rows"], stage1 + additional)
        self.assertTrue(call.kwargs["resume"])
        self.assertEqual(call.kwargs["stage"], "stage2")
        self.assertEqual(call.kwargs["max_episodes"], 20)

    def test_completed_schedule_has_zero_pending_on_second_resume(self) -> None:
        rows = _rows("smoke", 16)
        statuses = {
            row.episode_attempt_id: runner.AttemptStatus.COMPLETED for row in rows
        }

        selected = runner._select_pending_rows(rows, statuses, resume=True)

        self.assertEqual(selected, ())

    def test_resume_accepts_only_unseen_or_ledger_resumable_rows(self) -> None:
        rows = _rows("smoke", 4)
        statuses = {
            rows[0].episode_attempt_id: runner.AttemptStatus.COMPLETED,
            rows[1].episode_attempt_id: runner.AttemptStatus.STARTED,
            rows[2].episode_attempt_id: runner.AttemptStatus.FAILED,
        }

        selected = runner._select_pending_rows(rows, statuses, resume=True)

        self.assertEqual(selected, (rows[1], rows[3]))
        with self.assertRaisesRegex(ValueError, "resume=True"):
            runner._select_pending_rows(rows, statuses, resume=False)


class MinimalFactorialPreparationTests(unittest.TestCase):
    def test_snapshot_is_revalidated_before_campaign_artifacts_open(self) -> None:
        calls: list[str] = []
        validated = SimpleNamespace()
        prepared = SimpleNamespace()
        with (
            mock.patch.object(
                runner,
                "_validate_live_snapshot",
                side_effect=lambda path: calls.append("validate") or validated,
            ),
            mock.patch.object(
                runner,
                "_open_frozen_campaign",
                side_effect=lambda state, campaign: calls.append("open") or prepared,
            ),
        ):
            actual = runner._prepare_campaign(
                Path("manifest.yaml"),
                campaign="smoke",
            )

        self.assertIs(actual, prepared)
        self.assertEqual(calls, ["validate", "open"])

    def test_live_validation_loads_case_set_and_snapshot_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "configs" / "iclr2027" / "minimal_factorial.yaml"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("registered", encoding="utf-8")
            manifest = SimpleNamespace(case_path="benchmarks/cases.json")
            case_set = {"cases": mock.sentinel.cases}
            snapshot = mock.sentinel.snapshot
            with (
                mock.patch.object(
                    runner,
                    "load_experiment_manifest",
                    return_value=manifest,
                ) as load_manifest,
                mock.patch.object(
                    runner,
                    "_load_checked_case_set",
                    return_value=case_set,
                ) as load_cases,
                mock.patch.object(
                    runner,
                    "build_runtime_snapshot",
                    return_value=snapshot,
                ) as build_snapshot,
            ):
                validated = runner._validate_live_snapshot(manifest_path)

        self.assertEqual(validated.repo_root, root)
        self.assertIs(validated.manifest, manifest)
        self.assertIs(validated.case_set, case_set)
        self.assertIs(validated.snapshot, snapshot)
        load_manifest.assert_called_once_with(manifest_path.resolve())
        load_cases.assert_called_once_with(root, manifest)
        build_snapshot.assert_called_once_with(manifest, case_set)

    def test_frozen_campaign_uses_distinct_smoke_and_claim_roots(self) -> None:
        manifest = SimpleNamespace(
            outputs=SimpleNamespace(
                root="results/factorial",
                smoke="smoke",
                llm_campaign="llm_campaign",
            ),
            simulation=SimpleNamespace(target_env_id="highway-fast-v0"),
            fixed_harness=SimpleNamespace(),
        )
        snapshot = SimpleNamespace(
            payload=MappingProxyType(
                {
                    "runtime_config": MappingProxyType({"simulation_duration": 30}),
                    "environment_config": MappingProxyType(
                        {"duration": 30, "lanes": (1, 2)}
                    ),
                }
            )
        )
        validated = SimpleNamespace(
            repo_root=Path("repo"),
            manifest=manifest,
            case_set={"cases": [{"case_id": "case-1"}]},
            snapshot=snapshot,
        )
        s1 = SimpleNamespace(
            model_bindings={"qwen": SimpleNamespace(model_digest="sha256:" + "a" * 64)},
            capabilities=mock.sentinel.capabilities,
        )
        smoke_schedule = _rows("smoke", 16)
        claim_schedule = _rows("stage1", 480) + _rows(
            "stage2_additional",
            360,
        )
        with (
            mock.patch.object(
                runner,
                "_load_frozen_s1",
                return_value=s1,
            ),
            mock.patch.object(
                runner,
                "build_smoke_schedule",
                return_value=smoke_schedule,
            ),
            mock.patch.object(
                runner,
                "build_union_schedule",
                return_value=claim_schedule,
            ),
            mock.patch.object(
                runner,
                "_verify_frozen_campaign",
            ) as verify,
        ):
            smoke = runner._open_frozen_campaign(validated, "smoke")
            claim = runner._open_frozen_campaign(validated, "claim")

        self.assertEqual(smoke.output_root, Path("repo/results/factorial/smoke"))
        self.assertEqual(
            claim.output_root,
            Path("repo/results/factorial/llm_campaign"),
        )
        self.assertEqual(smoke.schedule, smoke_schedule)
        self.assertEqual(claim.schedule, claim_schedule)
        self.assertIs(type(smoke.runtime_config), dict)
        self.assertIs(type(smoke.environment_config["highway-fast-v0"]), dict)
        self.assertEqual(
            smoke.environment_config["highway-fast-v0"]["lanes"],
            [1, 2],
        )
        self.assertEqual(verify.call_count, 2)


if __name__ == "__main__":
    unittest.main()
