from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evaluate_models_ollama as evaluator
from dilu.runtime.harness_config import ExecutionMode
from dilu.runtime.ollama_scientific_client import ScientificGenerationAbort
from dilu.runtime.runtime_failures import (
    ProtocolInvariantCode,
    ProtocolInvariantViolation,
    RuntimeFailureClass,
    RuntimeProtocolError,
)
from dilu.runtime.scientific_trace import (
    ScientificSimulatorAbort,
    ScientificTraceWriteError,
    ScientificTraceWriter,
)
from tests.test_scientific_trace import _record
from tests.scientific_trace_support import blocked_record


class _FakeEnvironment:
    def __init__(self, events: list[str]) -> None:
        self.unwrapped = self
        self.events = events
        self.closed = False

    def configure(self, config: dict) -> None:
        del config

    def reset(self, *, seed: int):
        del seed
        return [0.0], {}

    def step(self, action_id: int):
        self.events.append(f"step:{action_id}")
        return [0.0], 1.0, True, False, {"crashed": False}

    def close(self) -> None:
        self.closed = True


class _FakeScenario:
    def describe(self, frame_id: int) -> str:
        del frame_id
        return "deterministic scenario"

    def availableActionsDescription(self) -> str:
        return "0 lane-left; 1 idle; 2 lane-right; 3 faster; 4 slower"


class _FakeAgent:
    def __init__(self, template) -> None:
        self.last_prompt_artifact = template.prompt_artifact
        self.last_generation_result = template.generation
        self.last_action_resolution = template.resolution
        self.last_decision_meta = {
            "decision_elapsed_sec": template.generation.latency_ms / 1000.0,
            "original_selected_action": (
                None
                if template.resolution is None
                else template.resolution.strict_action
            ),
            "timed_out": False,
            "used_fallback": False,
        }

    def few_shot_decision(self, **kwargs):
        del kwargs
        return 3, "Response to user:#### 3", "question", "answer"

    def set_decision_timeout_sec(self, timeout_sec: float) -> None:
        del timeout_sec


class _BlockedAgent(_FakeAgent):
    def few_shot_decision(self, **kwargs):
        del kwargs
        raise ScientificGenerationAbort(self.last_generation_result)


class _ProtocolFailingAgent(_FakeAgent):
    def few_shot_decision(self, **kwargs):
        del kwargs
        violation = ProtocolInvariantViolation.from_mapping(
            ProtocolInvariantCode.FIXED_FALLBACK_UNAVAILABLE,
            "fixed IDLE unavailable",
            {"available_action_ids": (0, 2, 3, 4)},
        )
        raise RuntimeProtocolError(violation)


class _FailingEnvironment(_FakeEnvironment):
    def step(self, action_id: int):
        self.events.append(f"step:{action_id}")
        raise RuntimeError("simulator failed")


def _traffic_metrics() -> dict[str, object]:
    return {
        "ego_speed_mps": 25.0,
        "front_gap_m": None,
        "relative_speed_mps": None,
        "ttc_sec": None,
        "ttc_danger": False,
        "headway_violation": False,
        "rear_gap_m": None,
        "rear_closing_speed_mps": None,
        "rear_ttc_sec": None,
        "rear_ttc_danger": False,
        "rear_headway_violation": False,
        "low_speed_blocking": False,
        "stopped": False,
        "near_stop": False,
    }


def _trace_factory(template):
    def build(**kwargs):
        return dataclasses.replace(
            template,
            prompt_artifact=kwargs["prompt_artifact"],
            generation=kwargs["generation"],
            resolution=kwargs["resolution"],
            shield_stack=kwargs["shield_stack"],
            disposition=kwargs["disposition"],
            decision_latency_ms=kwargs["decision_latency_ms"],
        )

    return build


def _run_one_episode(
    *,
    root: Path,
    writer: ScientificTraceWriter,
    environment: _FakeEnvironment,
    template,
    agent=None,
):
    scenario = _FakeScenario()
    agent = agent or _FakeAgent(template)
    with (
        mock.patch.object(evaluator.gym, "make", return_value=environment),
        mock.patch.object(evaluator, "EnvScenario", return_value=scenario),
        mock.patch.object(evaluator, "DriverAgent", return_value=agent),
        mock.patch.object(
            evaluator,
            "extract_step_traffic_metrics",
            side_effect=lambda *args, **kwargs: _traffic_metrics(),
        ),
        mock.patch.object(
            evaluator,
            "compute_split_scores_for_episode",
            side_effect=lambda result: result,
        ),
    ):
        return evaluator.run_episode(
            config={},
            env_config={"fake-env": {}},
            env_type="fake-env",
            agent_memory=None,
            seed=7,
            few_shot_num=0,
            temp_dir=str(root),
            ttc_threshold_sec=2.0,
            headway_threshold_m=8.0,
            rear_ttc_threshold_sec=2.0,
            rear_headway_threshold_m=8.0,
            low_speed_blocking_threshold_mps=5.0,
            blocking_front_gap_safe_m=20.0,
            blocking_front_ttc_safe_sec=4.0,
            stop_threshold_mps=0.5,
            near_stop_threshold_mps=2.0,
            alignment_sample_rate=0.0,
            alignment_max_samples=0,
            slow_decision_threshold_sec=1.0,
            save_artifacts=False,
            record_video=False,
            quiet_mode=True,
            enable_db_logging=False,
            on_decision=None,
            max_steps_override=1,
            execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
            shield_config=template.harness_config.shield,
            scientific_trace_writer=writer,
            scientific_trace_record_factory=_trace_factory(template),
        )


class ScientificTraceIntegrationTests(unittest.TestCase):
    def test_quiet_episode_writes_trace_before_step_without_optional_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[str] = []
            environment = _FakeEnvironment(events)
            template = _record()
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            with mock.patch.object(
                os,
                "fsync",
                side_effect=lambda descriptor: events.append("fsync"),
            ):
                result = _run_one_episode(
                    root=root,
                    writer=writer,
                    environment=environment,
                    template=template,
                )

            # The append intent and trace data must both be durable before env.step.
            self.assertEqual(events, ["fsync", "fsync", "step:3"])
            self.assertIsNone(result["action_trace"])
            self.assertEqual(len(result["scientific_trace_references"]), 1)
            self.assertEqual(
                result["scientific_trace_references"][0]["relative_path"],
                "decision_traces.jsonl",
            )
            self.assertTrue(environment.closed)

    def test_trace_fsync_failure_escapes_episode_and_prevents_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[str] = []
            environment = _FakeEnvironment(events)
            template = _record()
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            with mock.patch.object(os, "fsync", side_effect=OSError("disk failure")):
                with self.assertRaises(ScientificTraceWriteError):
                    _run_one_episode(
                        root=root,
                        writer=writer,
                        environment=environment,
                        template=template,
                    )

            self.assertEqual(events, [])
            self.assertTrue(environment.closed)

    def test_blocked_generation_is_traced_then_aborts_without_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[str] = []
            environment = _FakeEnvironment(events)
            template = blocked_record()
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            with self.assertRaises(ScientificGenerationAbort) as raised:
                _run_one_episode(
                    root=root,
                    writer=writer,
                    environment=environment,
                    template=template,
                    agent=_BlockedAgent(template),
                )

            payload = json.loads(
                (root / "decision_traces.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["disposition"], "blocked_before_execution")
            self.assertEqual(
                raised.exception.trace_reference.relative_path,
                "decision_traces.jsonl",
            )
            self.assertEqual(raised.exception.trace_reference.line_number, 1)
            self.assertEqual(events, [])
            self.assertTrue(environment.closed)

    def test_protocol_failure_escapes_without_generation_or_environment_step(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[str] = []
            environment = _FakeEnvironment(events)
            template = _record()
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )

            with self.assertRaises(RuntimeProtocolError):
                _run_one_episode(
                    root=root,
                    writer=writer,
                    environment=environment,
                    template=template,
                    agent=_ProtocolFailingAgent(template),
                )

            self.assertEqual(events, [])
            self.assertFalse((root / "decision_traces.jsonl").exists())
            self.assertTrue(environment.closed)

    def test_simulator_failure_aborts_with_committed_trace_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[str] = []
            environment = _FailingEnvironment(events)
            template = _record()
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )

            with self.assertRaises(ScientificSimulatorAbort) as raised:
                _run_one_episode(
                    root=root,
                    writer=writer,
                    environment=environment,
                    template=template,
                )

            abort = raised.exception
            self.assertEqual(abort.failure_class, RuntimeFailureClass.SIMULATOR_FAILURE)
            self.assertEqual(
                abort.trace_reference.relative_path, "decision_traces.jsonl"
            )
            self.assertEqual(abort.trace_reference.line_number, 1)
            self.assertEqual(events, ["step:3"])
            self.assertEqual(
                len((root / "decision_traces.jsonl").read_text().splitlines()),
                1,
            )
            self.assertTrue(environment.closed)


if __name__ == "__main__":
    unittest.main()
