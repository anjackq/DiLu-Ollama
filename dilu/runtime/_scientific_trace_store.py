from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._append_intent_io import (
    AppendCommitAmbiguousError,
    AppendIntentWriteError,
    durable_append_with_intent,
)
from ._campaign_attempt_io import exclusive_append_lock
from ._scientific_trace_artifacts import (
    ScientificSimulatorAbort,
    ScientificTraceCommitAmbiguousError,
    ScientificTraceValidationError,
    ScientificTraceWriteError,
    TraceReference,
)
from ._scientific_trace_execution import append_trace_before_step
from ._scientific_trace_records import DecisionTraceRecord
from ._scientific_trace_serialization import (
    TRACE_SCHEMA_VERSION,
    canonical_json_bytes,
    reject_json_constant,
    serialize_trace_record,
    trace_schema_sha256,
    validate_trace_payload,
)
from ._scientific_trace_state import (
    ScientificTraceSnapshot,
    initialize_scientific_trace_state,
)
from ._scientific_trace_state import (
    read_validated_trace_snapshot as _read_validated_trace_snapshot,
)


class ScientificTraceWriter:
    def __init__(
        self,
        path: Path,
        *,
        artifact_root: Path,
        resume: bool = False,
    ) -> None:
        initialize_scientific_trace_state(self, path, artifact_root)
        if self._poison_path.exists() or self._pending_path.exists():
            raise ScientificTraceCommitAmbiguousError(
                "Scientific trace is poisoned by an ambiguous durable append."
            )
        if self.path.exists() and self.path.stat().st_size:
            if not resume:
                raise ScientificTraceWriteError(
                    "Refusing to append to an existing trace without resume=True."
                )
            try:
                with exclusive_append_lock(self._lock_path):
                    self._scan_existing()
            except FileExistsError as exc:
                raise ScientificTraceWriteError(
                    "Scientific trace is busy during resume validation."
                ) from exc
        elif resume and self.path.exists():
            self._line_count = 0

    @property
    def next_line_number(self) -> int:
        return self._line_count + 1

    def append(self, record: DecisionTraceRecord) -> TraceReference:
        try:
            with exclusive_append_lock(self._lock_path):
                if self._storage_poisoned():
                    raise ScientificTraceCommitAmbiguousError(
                        "Scientific trace writer is poisoned after a storage failure."
                    )
                self._refresh()
                if type(record) is not DecisionTraceRecord:
                    raise ValueError("record must be a DecisionTraceRecord.")
                payload = serialize_trace_record(record)
                validate_trace_payload(payload)
                self._validate_sequence(record.context.key.identity(), payload)
                encoded = canonical_json_bytes(payload)
                line = encoded + b"\n"
                record_sha256 = "sha256:" + hashlib.sha256(encoded).hexdigest()
                try:
                    durable_append_with_intent(
                        self.path,
                        line,
                        artifact_kind="scientific_trace",
                        episode_attempt_id=record.context.key.episode_attempt_id,
                        expected_offset=self._byte_offset,
                        record_sha256=record_sha256,
                    )
                except AppendIntentWriteError as exc:
                    self._poisoned = True
                    raise ScientificTraceWriteError(
                        "Scientific trace append intent was not durably committed."
                    ) from exc
                except AppendCommitAmbiguousError as exc:
                    self._poisoned = True
                    raise ScientificTraceCommitAmbiguousError(
                        "Scientific trace append was not durably committed."
                    ) from exc
                self._line_count += 1
                self._byte_offset += len(line)
                reference = self._reference(encoded, self._line_count)
                self._remember(payload, reference)
                return reference
        except ScientificTraceWriteError:
            raise
        except FileExistsError as exc:
            raise ScientificTraceWriteError(
                "Another process owns the scientific trace append lock."
            ) from exc
        except Exception as exc:
            raise ScientificTraceValidationError(
                "Scientific trace validation or serialization failed."
            ) from exc

    def references_for_attempt(
        self,
        campaign_id: str,
        episode_attempt_id: str,
    ) -> tuple[TraceReference, ...]:
        try:
            with exclusive_append_lock(self._lock_path):
                if self._storage_poisoned():
                    # Only expose references remembered after a successful durable
                    # append.  A poisoned writer must reject new writes and global
                    # snapshots, but lifecycle terminalization still needs the
                    # known-good prefix committed before the ambiguous append.
                    return tuple(
                        self._references_by_episode.get(
                            (campaign_id, episode_attempt_id),
                            (),
                        )
                    )
                self._refresh()
                return tuple(
                    self._references_by_episode.get(
                        (campaign_id, episode_attempt_id),
                        (),
                    )
                )
        except FileExistsError as exc:
            raise ScientificTraceWriteError(
                "Another process owns the scientific trace append lock."
            ) from exc

    def cached_references_for_attempt(
        self,
        campaign_id: str,
        episode_attempt_id: str,
    ) -> tuple[TraceReference, ...]:
        """Return only references this writer durably committed and remembered."""
        return tuple(
            self._references_by_episode.get(
                (campaign_id, episode_attempt_id),
                (),
            )
        )

    def reference_snapshot(self) -> frozenset[TraceReference]:
        try:
            with exclusive_append_lock(self._lock_path):
                if self._storage_poisoned():
                    raise ScientificTraceCommitAmbiguousError(
                        "Scientific trace is poisoned by an ambiguous append."
                    )
                self._refresh()
                return frozenset(self._reference_index)
        except ScientificTraceWriteError:
            raise
        except FileExistsError as exc:
            raise ScientificTraceWriteError(
                "Another process owns the scientific trace append lock."
            ) from exc

    @contextmanager
    def locked_reference_snapshot_by_attempt(
        self,
    ) -> Iterator[Mapping[tuple[str, str], tuple[TraceReference, ...]]]:
        try:
            with exclusive_append_lock(self._lock_path):
                if self._storage_poisoned():
                    raise ScientificTraceCommitAmbiguousError(
                        "Scientific trace is poisoned by an ambiguous append."
                    )
                self._refresh()
                yield MappingProxyType(
                    {
                        key: tuple(references)
                        for key, references in self._references_by_episode.items()
                    }
                )
        except FileExistsError as exc:
            raise ScientificTraceWriteError(
                "Another process owns the scientific trace append lock."
            ) from exc

    def _storage_poisoned(self) -> bool:
        return (
            self._poisoned or self._poison_path.exists() or self._pending_path.exists()
        )

    def _refresh(self) -> None:
        size = self.path.stat().st_size if self.path.exists() else 0
        if size < self._byte_offset:
            raise ScientificTraceValidationError("Scientific trace was truncated.")
        if size > self._byte_offset:
            self._scan_existing()

    def _scan_existing(self) -> None:
        try:
            with self.path.open("rb") as handle:
                handle.seek(self._byte_offset)
                for line_number, raw_line in enumerate(
                    handle,
                    start=self._line_count + 1,
                ):
                    if not raw_line.endswith(b"\n"):
                        raise ScientificTraceWriteError(
                            "Existing trace has a truncated tail."
                        )
                    line = raw_line[:-1]
                    payload = json.loads(
                        line.decode("utf-8"),
                        parse_constant=reject_json_constant,
                    )
                    if not isinstance(payload, dict):
                        raise ValueError("Trace line is not a JSON object.")
                    validate_trace_payload(payload)
                    if line != canonical_json_bytes(payload):
                        raise ValueError("Existing trace line is not canonical JSON.")
                    key_tuple = self._key_tuple(payload["trace_key"])
                    self._validate_sequence(key_tuple, payload)
                    reference = self._reference(line, line_number)
                    self._remember(payload, reference)
                    self._line_count = line_number
                self._byte_offset = handle.tell()
        except ScientificTraceWriteError:
            raise
        except Exception as exc:
            raise ScientificTraceValidationError(
                "Existing scientific trace failed resume validation."
            ) from exc

    def _validate_sequence(
        self,
        key_tuple: tuple[object, ...],
        payload: dict[str, Any],
    ) -> None:
        key_payload = payload["trace_key"]
        if key_tuple in self._keys:
            raise ValueError("Duplicate scientific trace key.")
        episode = (
            str(key_payload["campaign_id"]),
            str(key_payload["episode_attempt_id"]),
        )
        request_id = str(payload["generation"]["request"]["request_id"])
        if request_id in self._request_owners:
            raise ValueError("Scientific request_id must be campaign-wide unique.")
        decision_index = int(key_payload["decision_index"])
        env_step_index = int(key_payload["env_step_index"])
        if episode in self._terminal_episodes:
            raise ValueError("A blocked episode attempt cannot receive more decisions.")
        previous = self._last_by_episode.get(episode)
        if previous is None:
            if decision_index != 0:
                raise ValueError(
                    "Each episode trace must begin at decision index zero."
                )
            return
        if self._signature_by_episode[episode] != self._episode_signature(payload):
            raise ValueError("Scientific episode identity drifted within one attempt.")
        previous_decision, previous_env_step = previous
        if decision_index != previous_decision + 1:
            raise ValueError("Scientific decision indices must be contiguous.")
        if env_step_index <= previous_env_step:
            raise ValueError("Scientific environment-step indices must increase.")

    def _remember(
        self,
        payload: dict[str, Any],
        reference: TraceReference,
    ) -> None:
        key_payload = payload["trace_key"]
        key_tuple = self._key_tuple(key_payload)
        self._keys.add(key_tuple)
        episode = (
            str(key_payload["campaign_id"]),
            str(key_payload["episode_attempt_id"]),
        )
        self._last_by_episode[episode] = (
            int(key_payload["decision_index"]),
            int(key_payload["env_step_index"]),
        )
        self._signature_by_episode.setdefault(
            episode,
            self._episode_signature(payload),
        )
        request_id = str(payload["generation"]["request"]["request_id"])
        self._request_owners[request_id] = episode
        self._references_by_episode.setdefault(episode, []).append(reference)
        self._reference_index.add(reference)
        if payload["disposition"] == "blocked_before_execution":
            self._terminal_episodes.add(episode)

    @staticmethod
    def _episode_signature(payload: dict[str, Any]) -> tuple[object, ...]:
        key = payload["trace_key"]
        context = payload["context"]
        request = payload["generation"]["request"]
        return (
            key["condition_id"],
            key["case_id"],
            key["pair_id"],
            key["template_id"],
            key["replicate_id"],
            context["benchmark_fingerprint"],
            context["code_revision"],
            context["simulator_seed"],
            context["generation_seed_master"],
            payload["config_sha256"],
            payload["prompt"]["prompt_sha256"],
            request["model_tag"],
            request["model_digest"],
            request["native_endpoint"],
        )

    @staticmethod
    def _key_tuple(key_payload: dict[str, Any]) -> tuple[object, ...]:
        names = (
            "campaign_id",
            "episode_attempt_id",
            "condition_id",
            "case_id",
            "pair_id",
            "template_id",
            "replicate_id",
            "decision_index",
            "env_step_index",
        )
        return tuple(key_payload[name] for name in names)

    def _reference(self, encoded: bytes, line_number: int) -> TraceReference:
        return TraceReference(
            relative_path=self.relative_path,
            line_number=line_number,
            record_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
            schema_version=TRACE_SCHEMA_VERSION,
            schema_sha256=trace_schema_sha256(),
        )


def read_validated_trace_snapshot(
    path: Path,
    *,
    artifact_root: Path,
) -> ScientificTraceSnapshot:
    return _read_validated_trace_snapshot(
        ScientificTraceWriter,
        path,
        artifact_root=artifact_root,
    )


__all__ = [
    "ScientificSimulatorAbort",
    "ScientificTraceCommitAmbiguousError",
    "ScientificTraceSnapshot",
    "ScientificTraceValidationError",
    "ScientificTraceWriteError",
    "ScientificTraceWriter",
    "TraceReference",
    "append_trace_before_step",
    "read_validated_trace_snapshot",
]
