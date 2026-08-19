from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dilu.runtime import _scientific_trace_state as trace_state
from dilu.runtime import _scientific_trace_store as trace_store
from dilu.runtime._campaign_attempt_io import lock_path_for
from dilu.runtime.scientific_trace import (
    DecisionTraceRecord,
    ScientificTraceWriteError,
    ScientificTraceWriter,
    append_trace_before_step,
)
from tests.test_scientific_trace import _record
from tests.scientific_trace_support import (
    action_resolution_failure_record,
    blocked_record,
)


class _TamperedRecord(DecisionTraceRecord):
    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["generation"]["request"]["options"]["seed"] += 1
        return payload


class ScientificTraceStoreTests(unittest.TestCase):
    def test_read_validated_snapshot_rejects_marker_after_final_file_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            ScientificTraceWriter(path, artifact_root=root).append(_record())
            original_file_state = trace_state._file_state
            file_state_calls = 0

            def create_marker_after_final_state(target: Path) -> object:
                nonlocal file_state_calls
                file_state_calls += 1
                state = original_file_state(target)
                if file_state_calls == 3:
                    lock_path_for(path).write_text("late owner", encoding="utf-8")
                return state

            with (
                mock.patch.object(
                    trace_state,
                    "_file_state",
                    side_effect=create_marker_after_final_state,
                ),
                self.assertRaisesRegex(
                    ScientificTraceWriteError,
                    "busy or has ambiguous",
                ),
            ):
                trace_store.read_validated_trace_snapshot(
                    path,
                    artifact_root=root,
                )

    def test_read_validated_snapshot_is_read_only_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            record = _record()
            reference = ScientificTraceWriter(path, artifact_root=root).append(record)
            before = (
                tuple(item.relative_to(root) for item in root.rglob("*")),
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )

            snapshot = trace_store.read_validated_trace_snapshot(
                path,
                artifact_root=root,
            )

            after = (
                tuple(item.relative_to(root) for item in root.rglob("*")),
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
        episode = (
            record.context.key.campaign_id,
            record.context.key.episode_attempt_id,
        )
        self.assertEqual(snapshot.references_by_attempt[episode], (reference,))
        self.assertEqual(before, after)

    def test_writer_rejects_polymorphic_record_serialization(self) -> None:
        record = _record()
        tampered = _TamperedRecord(
            **{
                field.name: getattr(record, field.name)
                for field in dataclasses.fields(record)
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            writer = ScientificTraceWriter(
                Path(tmp) / "decision_traces.jsonl",
                artifact_root=Path(tmp),
            )
            with self.assertRaises(ScientificTraceWriteError):
                writer.append(tampered)

    def test_reference_binds_exact_canonical_record_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "decision_traces.jsonl"
            record = _record()
            reference = ScientificTraceWriter(path, artifact_root=Path(tmp)).append(
                record
            )
            encoded = path.read_bytes()
            canonical = json.dumps(
                record.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")

            self.assertEqual(encoded, canonical + b"\n")
            self.assertEqual(reference.relative_path, "nested/decision_traces.jsonl")
            self.assertEqual(
                reference.record_sha256,
                "sha256:" + hashlib.sha256(canonical).hexdigest(),
            )

    def test_existing_file_and_outside_root_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            ScientificTraceWriter(path, artifact_root=root).append(_record())
            with self.assertRaises(ScientificTraceWriteError):
                ScientificTraceWriter(path, artifact_root=root)
            with self.assertRaises(ValueError):
                ScientificTraceWriter(root.parent / "outside.jsonl", artifact_root=root)

    def test_resume_rejects_schema_tampering_and_allows_new_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            writer = ScientificTraceWriter(path, artifact_root=root)
            writer.append(_record())
            other = _record()
            other = dataclasses.replace(
                other,
                context=dataclasses.replace(
                    other.context,
                    key=dataclasses.replace(
                        other.context.key,
                        episode_attempt_id="episode-attempt-002",
                    ),
                ),
            )
            request_id = "req-case-001-attempt-002-step-000"
            request = dataclasses.replace(
                other.generation.request,
                request_id=request_id,
            )
            attempts = tuple(
                dataclasses.replace(
                    attempt,
                    request_id=request_id,
                    attempt_id=f"{request_id}:a{attempt.attempt_index}",
                )
                for attempt in other.generation.attempts
            )
            other = dataclasses.replace(
                other,
                generation=dataclasses.replace(
                    other.generation,
                    request=request,
                    attempts=attempts,
                ),
            )
            writer.append(other)
            self.assertEqual(
                ScientificTraceWriter(
                    path, artifact_root=root, resume=True
                ).next_line_number,
                3,
            )

            payloads = [json.loads(line) for line in path.read_text().splitlines()]
            payloads[1]["schema_sha256"] = "sha256:" + "0" * 64
            path.write_text(
                "\n".join(
                    json.dumps(item, sort_keys=True, separators=(",", ":"))
                    for item in payloads
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ScientificTraceWriteError):
                ScientificTraceWriter(path, artifact_root=root, resume=True)

    def test_resume_scans_trace_streamingly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            ScientificTraceWriter(path, artifact_root=root).append(_record())

            with mock.patch.object(
                type(path),
                "read_bytes",
                side_effect=AssertionError("whole-file read is prohibited"),
            ):
                resumed = ScientificTraceWriter(
                    path,
                    artifact_root=root,
                    resume=True,
                )

            self.assertEqual(resumed.next_line_number, 2)

    def test_request_id_is_unique_across_episode_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            writer.append(_record())
            duplicate_request = _record()
            duplicate_request = dataclasses.replace(
                duplicate_request,
                context=dataclasses.replace(
                    duplicate_request.context,
                    key=dataclasses.replace(
                        duplicate_request.context.key,
                        episode_attempt_id="episode-attempt-002",
                    ),
                ),
            )

            with self.assertRaises(ScientificTraceWriteError):
                writer.append(duplicate_request)

    def test_resume_rejects_semantic_tampering_noncanonical_json_and_nan(self) -> None:
        mutations = (
            lambda payload: payload["generation"]["request"]["options"].__setitem__(
                "seed", payload["generation"]["request"]["options"]["seed"] + 1
            ),
            lambda payload: payload["shield_stack"].__setitem__(
                "executed_action_id", 2
            ),
            lambda payload: payload.__setitem__("decision_latency_ms", float("nan")),
            lambda payload: payload["generation"]["attempts"][-1].__setitem__(
                "accepted_by_server", None
            ),
            lambda payload: payload["generation"]["attempts"][-1].__setitem__(
                "http_status", 500
            ),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "decision_traces.jsonl"
                payload = _record().to_dict()
                mutate(payload)
                path.write_text(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ScientificTraceWriteError):
                    ScientificTraceWriter(path, artifact_root=root, resume=True)

    def test_resume_rejects_failure_message_and_capability_snapshot_tampering(
        self,
    ) -> None:
        mutations = (
            lambda payload: payload["failure"].__setitem__(
                "message", "unrelated failure"
            ),
            lambda payload: payload["generation"]["transport_evidence"].__setitem__(
                "capability_artifact_sha256", "sha256:" + "f" * 64
            ),
            lambda payload: payload["generation"]["transport_evidence"][
                "capability_supported_think_modes"
            ].append("think"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "decision_traces.jsonl"
                payload = blocked_record().to_dict()
                mutate(payload)
                path.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ScientificTraceWriteError):
                    ScientificTraceWriter(path, artifact_root=root, resume=True)

    def test_resume_binds_operational_and_resolution_failure_messages(self) -> None:
        records = (
            _record(timeout=True),
            action_resolution_failure_record(),
        )
        for index, record in enumerate(records):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "decision_traces.jsonl"
                payload = record.to_dict()
                payload["failure"]["message"] = "unrelated failure"
                path.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ScientificTraceWriteError):
                    ScientificTraceWriter(path, artifact_root=root, resume=True)

    def test_resume_rejects_noncanonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            path.write_text(json.dumps(_record().to_dict()) + "\n", encoding="utf-8")
            with self.assertRaises(ScientificTraceWriteError):
                ScientificTraceWriter(path, artifact_root=root, resume=True)

    def test_resume_rejects_non_generation_failure_in_generation_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            payload = _record().to_dict()
            generation = payload["generation"]
            generation["error_class"] = "simulator_failure"
            generation["error_message"] = "simulator failed"
            final_attempt = generation["attempts"][-1]
            final_attempt["error_class"] = "simulator_failure"
            final_attempt["error_message"] = "simulator failed"
            payload["action_resolution"] = None
            payload["shield_stack"] = None
            payload["failure"] = {
                "failure_class": "simulator_failure",
                "message": "simulator failed",
            }
            payload["disposition"] = "blocked_before_execution"
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ScientificTraceWriteError):
                ScientificTraceWriter(path, artifact_root=root, resume=True)

    def test_episode_identity_is_stable_and_blocked_trace_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            writer.append(_record(0))
            drifted = _record(1)
            drifted = dataclasses.replace(
                drifted,
                context=dataclasses.replace(
                    drifted.context,
                    key=dataclasses.replace(
                        drifted.context.key,
                        template_id="template-drifted",
                    ),
                ),
            )
            with self.assertRaises(ScientificTraceWriteError):
                writer.append(drifted)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ScientificTraceWriter(
                root / "decision_traces.jsonl",
                artifact_root=root,
            )
            writer.append(blocked_record())
            with self.assertRaises(ScientificTraceWriteError):
                writer.append(_record(1))

    def test_short_write_poisoning_prevents_step_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ScientificTraceWriter(
                Path(tmp) / "decision_traces.jsonl",
                artifact_root=Path(tmp),
            )
            step = mock.Mock(return_value="stepped")
            with mock.patch.object(os, "write", return_value=1):
                with self.assertRaises(ScientificTraceWriteError):
                    append_trace_before_step(writer, _record(), step)
            step.assert_not_called()
            with self.assertRaises(ScientificTraceWriteError):
                writer.append(_record())


if __name__ == "__main__":
    unittest.main()
