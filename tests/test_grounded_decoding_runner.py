from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.grounded_decoding_schedule_support import (
    MANIFEST_PATH,
    ROOT,
    fake_git,
    frozen_bindings,
    matching_snapshot,
)


class GroundedCampaignArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        from dilu.runtime.grounded_decoding_schedule import (
            build_runtime_snapshot,
            load_grounded_decoding_manifest,
        )

        self.manifest = load_grounded_decoding_manifest(MANIFEST_PATH)
        self.case_set = json.loads((ROOT / self.manifest.case_path).read_text())
        with mock.patch(
            "dilu.runtime._minimal_factorial_manifest.subprocess.run", fake_git
        ):
            self.snapshot = matching_snapshot(
                build_runtime_snapshot(self.manifest, self.case_set)
            )

    def test_frozen_payload_contains_v8_conditions_and_comparators(self) -> None:
        from dilu.runtime._minimal_factorial_manifest import serialize_frozen_campaign
        from dilu.runtime.grounded_decoding_schedule import build_v8_smoke_schedule

        schedule = build_v8_smoke_schedule(
            self.manifest,
            self.case_set,
            frozen_bindings(),
            runtime_snapshot=self.snapshot,
        )
        payload = json.loads(
            serialize_frozen_campaign(
                self.manifest, self.snapshot, schedule, self.case_set
            )
        )

        self.assertEqual(
            payload["manifest"]["conditions"]["output_enforcement"],
            "backend_schema_grounded",
        )
        self.assertEqual(
            payload["manifest"]["comparators"]["v5_manifest"],
            self.manifest.comparators.v5_manifest,
        )

    def test_freeze_writes_exact_10_and_480_row_campaigns(self) -> None:
        from dilu.runtime.grounded_decoding_campaign import (
            freeze_v8_campaign_artifacts,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "v8"
            artifacts = freeze_v8_campaign_artifacts(
                repo_root=ROOT,
                manifest=self.manifest,
                case_set=self.case_set,
                snapshot=self.snapshot,
                model_bindings=frozen_bindings(),
                output_root=output,
            )

            smoke = json.loads(artifacts.smoke_manifest.read_text())
            claim = json.loads(artifacts.claim_manifest.read_text())
            union = json.loads(artifacts.union_schedule.read_text())

        self.assertEqual(len(smoke["schedule"]), 10)
        self.assertEqual(len(claim["schedule"]), 480)
        self.assertEqual(len(union), 480)
        self.assertEqual(
            {row["stage"] for row in claim["schedule"]},
            {"stage1", "stage2_additional"},
        )


class GroundedRunnerTests(unittest.TestCase):
    def test_freeze_uses_the_authored_model_bindings_contract(self) -> None:
        from dilu.runtime import grounded_decoding_runner as runner

        authored = SimpleNamespace(model_bindings=frozen_bindings())
        manifest = SimpleNamespace(repo_root=lambda: ROOT)
        case_set = {"cases": []}
        snapshot = mock.sentinel.snapshot
        with (
            mock.patch.object(
                runner, "load_grounded_decoding_manifest", return_value=manifest
            ),
            mock.patch.object(
                runner, "load_checked_case_set", return_value=case_set
            ),
            mock.patch.object(
                runner, "build_runtime_snapshot", return_value=snapshot
            ),
            mock.patch.object(runner, "freeze_v8_campaign_artifacts") as freeze,
        ):
            runner._freeze_from_authored_s1(MANIFEST_PATH, authored)

        self.assertEqual(freeze.call_args.kwargs["model_bindings"], authored.model_bindings)

    def test_probe_lock_freezes_campaigns_only_when_comparator_digests_match(
        self,
    ) -> None:
        from dilu.runtime import grounded_decoding_runner as runner

        authored = SimpleNamespace(
            preflight_path=Path("results/v8/s1/model_preflight.json"),
            comparator_digest_match={"qwen_06b": True, "llama_1b": True},
        )
        with (
            mock.patch.object(
                runner, "author_verified_grounded_decoding_locks", return_value=authored
            ),
            mock.patch.object(runner, "_freeze_from_authored_s1") as freeze,
        ):
            result = runner.run_probe_lock(Path("configs/iclr2027/grounded.yaml"))

        self.assertEqual(result, authored.preflight_path)
        freeze.assert_called_once()

    def test_probe_lock_leaves_only_s1_when_a_comparator_digest_drifted(self) -> None:
        from dilu.runtime import grounded_decoding_runner as runner

        authored = SimpleNamespace(
            preflight_path=Path("results/v8/s1/model_preflight.json"),
            comparator_digest_match={"qwen_06b": False, "llama_1b": True},
        )
        with (
            mock.patch.object(
                runner, "author_verified_grounded_decoding_locks", return_value=authored
            ),
            mock.patch.object(runner, "_freeze_from_authored_s1") as freeze,
        ):
            result = runner.run_probe_lock(Path("configs/iclr2027/grounded.yaml"))

        self.assertEqual(result, authored.preflight_path)
        freeze.assert_not_called()

    def test_smoke_dispatches_exactly_ten_rows_with_resume_unchanged(self) -> None:
        from dilu.runtime import grounded_decoding_runner as runner

        rows = tuple(
            SimpleNamespace(stage="smoke", episode_attempt_id=f"smoke-{index}")
            for index in range(10)
        )
        prepared = SimpleNamespace(schedule=rows, output_root=Path("results/v8/smoke"))
        with (
            mock.patch.object(runner, "_prepare_campaign", return_value=prepared),
            mock.patch.object(
                runner, "_execute_campaign", return_value=mock.sentinel.summary
            ) as execute,
        ):
            result = runner.run_smoke(Path("grounded.yaml"), resume=True)

        self.assertIs(result, mock.sentinel.summary)
        execute.assert_called_once_with(
            prepared,
            scheduled_rows=rows,
            denominator_rows=rows,
            resume=True,
            stage="smoke",
        )

    def test_claim_dispatch_preserves_300_then_180_stage_split(self) -> None:
        from dilu.runtime import grounded_decoding_runner as runner

        stage1 = tuple(SimpleNamespace(stage="stage1") for _ in range(300))
        stage2 = tuple(SimpleNamespace(stage="stage2_additional") for _ in range(180))
        prepared = SimpleNamespace(
            schedule=stage1 + stage2,
            output_root=Path("results/v8/llm_campaign"),
        )
        with (
            mock.patch.object(runner, "_prepare_campaign", return_value=prepared),
            mock.patch.object(
                runner, "_execute_campaign", return_value=mock.sentinel.summary
            ) as execute,
        ):
            result = runner.run_claim_stage(
                Path("grounded.yaml"), stage="stage2", resume=True
            )

        self.assertIs(result, mock.sentinel.summary)
        call = execute.call_args
        self.assertEqual(len(call.kwargs["scheduled_rows"]), 180)
        self.assertEqual(len(call.kwargs["denominator_rows"]), 480)
        self.assertTrue(call.kwargs["resume"])
        self.assertEqual(call.kwargs["stage"], "stage2")


if __name__ == "__main__":
    unittest.main()
