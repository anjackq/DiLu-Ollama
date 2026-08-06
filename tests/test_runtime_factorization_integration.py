from __future__ import annotations

import dataclasses
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evaluate_models_ollama as evaluator

import dilu.runtime as runtime_api
from dilu.driver_agent.prompt_modules import build_prompt_artifact
from dilu.runtime.action_resolution import resolve_action
from dilu.runtime.harness_config import OutputEnforcement, resolve_main_conditions
from dilu.runtime.ollama_scientific_client import (
    OllamaScientificClient,
    ScientificGenerationAbort,
)
from dilu.runtime.runtime_failures import RuntimeProtocolError
from dilu.runtime.scientific_trace import (
    ScientificSimulatorAbort,
    ScientificTraceWriteError,
    TraceDisposition,
)
from dilu.runtime.scientific_runtime import (
    RuntimeLockBinding,
    build_scientific_episode_runtime,
)
from dilu.runtime.shield_stack import execute_shield_stack
from tests.scientific_transport_support import (
    FakeResponse,
    identity_inspector_for,
    make_capabilities,
    make_retry_policy,
    make_request,
)
from tests.test_scientific_driver_action_resolution import _scientific_config
from tests.runtime_factorization_support import (
    AmbiguousTraceWriter,
    FakeEnvironment,
    identity,
    run_episode,
    runtime,
    terminal_attempt_payload,
    verified_runtime_lock,
)


class ScientificEpisodeRuntimeTests(unittest.TestCase):
    def test_scientific_runtime_exports_are_stable_public_types(self) -> None:
        expected_names = (
            "AttemptStatus",
            "HarnessConfig",
            "RuntimeLockBinding",
            "ScientificAttemptLedger",
            "ScientificEpisodeIdentity",
            "ScientificEpisodeRuntime",
            "ScientificTraceWriter",
            "VerifiedRuntimeLockBinding",
            "build_scientific_episode_runtime",
            "load_verified_runtime_lock_binding",
            "resolve_main_conditions",
        )

        for name in expected_names:
            with self.subTest(name=name):
                self.assertIn(name, runtime_api.__all__)
                self.assertIsNotNone(getattr(runtime_api, name))

    def test_factory_rejects_external_runtime_lock_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bound_runtime = runtime(Path(tmp))
            mapping = bound_runtime.runtime_lock.to_dict()
            mapping["capability_artifact_sha256"] = "sha256:" + "f" * 64
            external_lock = verified_runtime_lock(
                Path(tmp) / "drifted-lock",
                mapping,
            )

            with self.assertRaises(RuntimeProtocolError):
                build_scientific_episode_runtime(
                    harness_config=bound_runtime.harness_config,
                    identity=bound_runtime.identity,
                    runtime_lock=external_lock,
                    transport_client=bound_runtime.transport_client,
                    trace_writer=bound_runtime.trace_writer,
                    attempt_ledger=bound_runtime.attempt_ledger,
                )

    def test_factory_rejects_unverified_live_runtime_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bound_runtime = runtime(Path(tmp))
            live_binding = RuntimeLockBinding.from_runtime(
                harness_config=bound_runtime.harness_config,
                identity=bound_runtime.identity,
                capabilities=bound_runtime.transport_client.capabilities,
            )

            with self.assertRaises(ValueError):
                build_scientific_episode_runtime(
                    harness_config=bound_runtime.harness_config,
                    identity=bound_runtime.identity,
                    runtime_lock=live_binding,
                    transport_client=bound_runtime.transport_client,
                    trace_writer=bound_runtime.trace_writer,
                    attempt_ledger=bound_runtime.attempt_ledger,
                )

    def test_verified_binding_requires_expected_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mapping = runtime(Path(tmp)).runtime_lock.to_dict()

            with self.assertRaisesRegex(
                ValueError,
                "source artifact hash did not verify",
            ):
                verified_runtime_lock(
                    Path(tmp) / "mismatched-lock",
                    mapping,
                    expected_source_artifact_sha256="sha256:" + "b" * 64,
                )

    def test_identity_and_runtime_lock_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bound_runtime = runtime(Path(tmp))
            with self.assertRaises(dataclasses.FrozenInstanceError):
                bound_runtime.identity.case_id = "changed"  # type: ignore[misc]
            with self.assertRaises(dataclasses.FrozenInstanceError):
                bound_runtime.runtime_lock.model_tag = "changed"  # type: ignore[misc]

    def test_all_eight_conditions_bind_only_declared_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conditions = resolve_main_conditions(_scientific_config())
            ledgers = []
            for index, config in enumerate(conditions):
                bound_runtime = runtime(root / str(index), attempt=0, config=config)
                bound_runtime.validate_binding()
                ledgers.append(
                    (
                        bound_runtime.harness_config.condition_id(),
                        bound_runtime.runtime_lock.condition_id,
                        bound_runtime.runtime_lock.config_sha256,
                    )
                )

            self.assertEqual(
                [item[0] for item in ledgers],
                ["c000", "c001", "c010", "c011", "c100", "c101", "c110", "c111"],
            )
            self.assertTrue(all(actual == locked for actual, locked, _ in ledgers))
            self.assertEqual(len({digest for _, _, digest in ledgers}), 8)

    def test_generation_context_is_deterministic_and_campaign_owned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bound_runtime = runtime(Path(tmp))
            bound_runtime.begin_attempt()

            primary = bound_runtime.generation_context(0)
            later = bound_runtime.generation_context(1)

            self.assertNotEqual(primary.request_id, later.request_id)
            self.assertNotEqual(primary.generation_seed, later.generation_seed)
            with self.assertRaises(Exception):
                bound_runtime.generation_context(0)

    def test_runtime_builds_typed_trace_from_bound_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bound_runtime = runtime(Path(tmp))
            bound_runtime.begin_attempt()
            context = bound_runtime.generation_context(0)
            config = bound_runtime.harness_config
            prompt = build_prompt_artifact(
                config.condition.policy_content,
                output_enforcement=config.condition.output_enforcement,
            )
            request = dataclasses.replace(
                make_request(),
                request_id=context.request_id,
                model_digest=context.model_digest,
                messages=(("system", prompt.system_prompt()), ("user", "scenario")),
                options=dataclasses.replace(
                    make_request().options,
                    seed=context.generation_seed,
                ),
            )
            generation = bound_runtime.transport_client.generate(request)
            resolution = resolve_action(
                generation.contract_text or "",
                available_action_ids=(0, 1, 2, 3, 4),
            )
            stack = execute_shield_stack(
                scenario=object(),
                proposed_action_id=resolution.strict_action,
                fallback_modified_action_id=resolution.final_resolved_action,
                execution_mode=config.condition.execution_mode,
                shield_config=config.shield,
            )

            record = bound_runtime.build_trace_record(
                decision_index=0,
                env_step_index=0,
                available_action_ids=(0, 1, 2, 3, 4),
                prompt_artifact=prompt,
                generation=generation,
                resolution=resolution,
                shield_stack=stack,
                disposition=TraceDisposition.READY_FOR_ENV_STEP,
                decision_latency_ms=generation.latency_ms,
                benchmark_event_meta={"benchmark_event_ids": []},
            )

            self.assertEqual(record.context.key.condition_id, config.condition_id())
            self.assertEqual(
                record.context.key.episode_attempt_id, identity().episode_attempt_id
            )
            self.assertEqual(record.generation.request.request_id, context.request_id)

    def test_runtime_lock_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bound_runtime = runtime(Path(tmp))
            mapping = bound_runtime.runtime_lock.to_dict()
            mapping["model_digest"] = "sha256:" + "f" * 64
            drifted_lock = verified_runtime_lock(
                Path(tmp) / "drifted-lock",
                mapping,
            )
            drifted = dataclasses.replace(
                bound_runtime,
                runtime_lock=drifted_lock,
            )

            with self.assertRaises(RuntimeProtocolError):
                drifted.validate_binding()


class MockedEightCellCampaignTests(unittest.TestCase):
    def test_all_eight_cells_use_mandatory_runtime_and_ignore_legacy_env(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(
                os.environ,
                {
                    "DILU_PROMPT_PROFILE": "invalid-legacy-value",
                    "OLLAMA_THINK_MODE": "auto",
                    "OPENAI_API_TYPE": "invalid-provider",
                },
            ),
        ):
            root = Path(tmp)
            conditions = resolve_main_conditions(_scientific_config())
            for index, config in enumerate(conditions):
                cell_root = root / config.condition_id()
                bound_runtime = runtime(cell_root, attempt=index, config=config)
                timeout_state = {
                    "enabled": True,
                    "stage": 3,
                    "effective_decision_timeout_sec": 1.0,
                }
                before = copy.deepcopy(timeout_state)

                result = run_episode(
                    cell_root,
                    bound_runtime,
                    timeout_state=timeout_state,
                )

                self.assertEqual(timeout_state, before)
                self.assertEqual(len(result["scientific_trace_references"]), 1)
                self.assertFalse(
                    bound_runtime.attempt_ledger.can_resume(
                        bound_runtime.identity.episode_attempt_id
                    )
                )

    def test_runtime_lock_drift_is_zero_decision_blocked_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bound_runtime = runtime(root)
            mapping = bound_runtime.runtime_lock.to_dict()
            mapping["model_digest"] = "sha256:" + "f" * 64
            drifted_lock = verified_runtime_lock(
                root / "drifted-lock",
                mapping,
            )
            bound_runtime = dataclasses.replace(
                bound_runtime,
                runtime_lock=drifted_lock,
            )

            with self.assertRaises(RuntimeProtocolError):
                run_episode(root, bound_runtime)

            self.assertFalse(
                bound_runtime.attempt_ledger.can_resume(
                    bound_runtime.identity.episode_attempt_id
                )
            )

    def test_setup_and_postprocessing_failures_write_terminal_attempts(self) -> None:
        cases = ("setup", "postprocessing")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bound_runtime = runtime(root)
                if case == "setup":
                    with mock.patch.object(
                        evaluator,
                        "_resolve_simulation_duration",
                        side_effect=RuntimeError("setup failed"),
                    ):
                        with self.assertRaises(RuntimeError):
                            run_episode(
                                root,
                                bound_runtime,
                                max_steps_override=None,
                            )
                else:
                    with self.assertRaises(RuntimeError):
                        run_episode(
                            root,
                            bound_runtime,
                            score_side_effect=RuntimeError("scoring failed"),
                        )

                lifecycle = terminal_attempt_payload(root)
                self.assertEqual(lifecycle["status"], "failed")
                self.assertFalse(
                    bound_runtime.attempt_ledger.can_resume(
                        bound_runtime.identity.episode_attempt_id
                    )
                )

    def test_generation_abort_and_simulator_abort_persist_trace_evidence(self) -> None:
        cases = ("generation", "simulator")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = _scientific_config()
                client = None
                environment = FakeEnvironment(fail_step=case == "simulator")
                expected_error = ScientificSimulatorAbort
                if case == "generation":
                    config = next(
                        item
                        for item in resolve_main_conditions(config)
                        if item.condition.output_enforcement
                        is OutputEnforcement.BACKEND_SCHEMA
                    )
                    client = OllamaScientificClient(
                        capabilities=make_capabilities(),
                        retry_policy=make_retry_policy(),
                        identity_inspector=identity_inspector_for(),
                        post=lambda *args, **kwargs: FakeResponse(
                            {"error": "invalid format schema"},
                            status_code=400,
                        ),
                        sleep=lambda _: None,
                    )
                    expected_error = ScientificGenerationAbort
                bound_runtime = runtime(
                    root,
                    config=config,
                    transport_client=client,
                )

                with self.assertRaises(expected_error):
                    run_episode(root, bound_runtime, environment=environment)

                lifecycle = terminal_attempt_payload(root)
                self.assertIn(lifecycle["status"], {"blocked", "failed"})
                self.assertEqual(lifecycle["decision_count"], 1)
                self.assertEqual(len(lifecycle["trace_references"]), 1)

    def test_trace_write_ambiguity_poisoned_attempt_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = AmbiguousTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            bound_runtime = runtime(root, writer=writer)

            with self.assertRaises(ScientificTraceWriteError):
                run_episode(root, bound_runtime)

            lifecycle = terminal_attempt_payload(root)
            self.assertEqual(lifecycle["status"], "write_ambiguous")
            self.assertEqual(
                lifecycle["trace_absence_reason"],
                "trace_commit_ambiguous",
            )
            self.assertFalse(
                bound_runtime.attempt_ledger.can_resume(
                    bound_runtime.identity.episode_attempt_id
                )
            )
