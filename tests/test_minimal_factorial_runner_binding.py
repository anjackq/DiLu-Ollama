from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import evaluate_models_ollama

from dilu.runtime import minimal_factorial_runner as runner


class MinimalFactorialEpisodeBindingTests(unittest.TestCase):
    def test_episode_binds_actual_run_episode_signature_and_frozen_inputs(
        self,
    ) -> None:
        row = SimpleNamespace(
            model_slot="qwen_06b",
            model_tag="qwen3:0.6b",
            condition_id="c101",
            simulator_seed=17,
            case_id="case-017",
            condition=SimpleNamespace(
                condition=SimpleNamespace(execution_mode=mock.sentinel.execution),
                shield=mock.sentinel.shield,
            ),
            identity=lambda: mock.sentinel.identity,
        )
        prepared = SimpleNamespace(
            runtime_config=mock.sentinel.runtime_config,
            environment_config=mock.sentinel.environment_config,
            target_env_id="highway-fast-v0",
            default_max_steps=30,
            thresholds=runner.FrozenThresholds(
                2.0,
                15.0,
                2.5,
                12.0,
                8.5,
                25.0,
                4.0,
                0.1,
                1.0,
                5.0,
            ),
            case_by_id={"case-017": {"case_id": "case-017"}},
            lock_root=Path("campaign") / "s1" / "locks",
        )
        captured: dict[str, object] = {}
        runtime_lock = SimpleNamespace(
            authorization_artifact_sha256="sha256:" + "1" * 64,
            binding_sha256="sha256:" + "2" * 64,
            prompt_sha256="sha256:" + "3" * 64,
            capability_artifact_sha256="sha256:" + "4" * 64,
            capability_snapshot_sha256="sha256:" + "5" * 64,
            trace_schema_sha256="sha256:" + "6" * 64,
        )

        def fake_run_episode(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {"episode_attempt_id": "episode-017"}

        with (
            mock.patch.object(
                runner,
                "load_verified_runtime_lock_binding",
                return_value=runtime_lock,
            ) as load_lock,
            mock.patch.object(
                runner,
                "build_scientific_episode_runtime",
                return_value=mock.sentinel.scientific_runtime,
            ) as build_runtime,
            mock.patch.object(
                runner,
                "build_case_env_config",
                return_value=(
                    {"highway-fast-v0": {"duration": 30}},
                    {"duration": 30},
                ),
            ),
            mock.patch.object(
                runner,
                "build_benchmark_instruction",
                return_value="instruction",
            ),
            mock.patch.object(
                runner,
                "benchmark_max_steps",
                return_value=41,
            ),
            mock.patch.object(
                runner,
                "run_episode",
                side_effect=fake_run_episode,
            ),
        ):
            result = runner._run_scheduled_episode(
                prepared,
                row,
                ledger=mock.sentinel.ledger,
                trace_writer=mock.sentinel.trace_writer,
                client=mock.sentinel.client,
                episode_temp_dir=Path("temp") / "episode-017",
            )

        self.assertEqual(result["episode_attempt_id"], "episode-017")
        self.assertEqual(
            result["runtime_lock_authorization_artifact_sha256"],
            runtime_lock.authorization_artifact_sha256,
        )
        self.assertEqual(
            result["runtime_lock_binding_sha256"],
            runtime_lock.binding_sha256,
        )
        self.assertEqual(result["prompt_sha256"], runtime_lock.prompt_sha256)
        self.assertEqual(
            result["capability_artifact_sha256"],
            runtime_lock.capability_artifact_sha256,
        )
        self.assertEqual(
            result["capability_snapshot_sha256"],
            runtime_lock.capability_snapshot_sha256,
        )
        self.assertEqual(
            result["trace_schema_sha256"],
            runtime_lock.trace_schema_sha256,
        )
        inspect.signature(evaluate_models_ollama.run_episode).bind(**captured)
        self.assertIs(captured["config"], prepared.runtime_config)
        self.assertEqual(captured["few_shot_num"], 0)
        self.assertIs(captured["benchmark_case"], prepared.case_by_id["case-017"])
        self.assertIs(captured["execution_mode"], mock.sentinel.execution)
        self.assertIs(captured["shield_config"], mock.sentinel.shield)
        self.assertIs(
            captured["scientific_runtime"],
            mock.sentinel.scientific_runtime,
        )
        self.assertEqual(captured["max_steps_override"], 41)
        self.assertIsNone(captured["scientific_trace_writer"])
        self.assertIsNone(captured["scientific_trace_record_factory"])
        load_lock.assert_called_once_with(
            runtime_lock_path=(
                prepared.lock_root / "qwen_06b" / "c101" / "RUNTIME_PROTOCOL_LOCK.json"
            ),
            authorization_path=(
                prepared.lock_root / "qwen_06b" / "c101" / "PROTOCOL_FROZEN.json"
            ),
        )
        build_runtime.assert_called_once_with(
            harness_config=row.condition,
            identity=mock.sentinel.identity,
            runtime_lock=runtime_lock,
            transport_client=mock.sentinel.client,
            trace_writer=mock.sentinel.trace_writer,
            attempt_ledger=mock.sentinel.ledger,
        )


if __name__ == "__main__":
    unittest.main()
