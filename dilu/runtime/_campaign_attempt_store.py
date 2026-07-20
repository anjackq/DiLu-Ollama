from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ._append_intent_io import (
    AppendCommitAmbiguousError,
    AppendIntentWriteError,
    append_intent_path_for,
    durable_append_with_intent,
)
from ._campaign_attempt_io import exclusive_append_lock, lock_path_for, poison_path_for
from ._campaign_attempt_serialization import (
    ATTEMPT_EVENT_FIELDS,
    REQUEST_EVENT_FIELDS,
    canonical_bytes,
    hash_payload,
    record_from_dict,
    reject_json_constant,
    require_keys,
)
from ._scientific_trace_store import (
    ScientificTraceWriteError,
    ScientificTraceWriter,
    TraceReference,
)
from .campaign_attempts import (
    ATTEMPT_SCHEMA_VERSION,
    AttemptReference,
    AttemptStatus,
    ScientificAttemptRecord,
    ScientificAttemptWriteError,
    _is_sha256,
    _require_text,
)


class ScientificAttemptLedger:
    def __init__(
        self,
        path: Path,
        *,
        campaign_id: str,
        resume: bool = False,
    ) -> None:
        _require_text("campaign_id", campaign_id)
        self.path = Path(path).resolve()
        self._lock_path = lock_path_for(self.path)
        self._poison_path = poison_path_for(self.path)
        self._pending_path = append_intent_path_for(self.path)
        self.campaign_id = campaign_id
        self._attempt_status: dict[str, AttemptStatus] = {}
        self._request_owner: dict[str, str] = {}
        self._line_count = 0
        self._byte_offset = 0
        self._last_hash: str | None = None
        self._poisoned = False
        self._poisoned_attempts: set[str] = set()
        self._terminal_records: list[ScientificAttemptRecord] = []
        if self._poison_path.exists() or self._pending_path.exists():
            raise ScientificAttemptWriteError(
                "Attempt ledger is poisoned by an ambiguous durable append."
            )
        if self.path.exists() and self.path.stat().st_size:
            if not resume:
                raise ScientificAttemptWriteError(
                    "Existing ledger requires resume=True."
                )
            try:
                with exclusive_append_lock(self._lock_path):
                    self._scan_existing()
            except FileExistsError as exc:
                raise ScientificAttemptWriteError(
                    "Attempt ledger is busy during resume validation."
                ) from exc

    def append_started(self, episode_attempt_id: str) -> AttemptReference:
        with self._append_guard():
            record = ScientificAttemptRecord(
                self.campaign_id,
                episode_attempt_id,
                AttemptStatus.STARTED,
                0,
            )
            return self._append_attempt(record)

    def append_terminal(
        self,
        episode_attempt_id: str,
        *,
        status: AttemptStatus,
        decision_count: int,
        trace_references: tuple[TraceReference, ...] = (),
        failure_class: str | None = None,
        failure_message: str | None = None,
        trace_absence_reason: str | None = None,
    ) -> AttemptReference:
        with self._append_guard():
            if status is AttemptStatus.STARTED:
                raise ValueError("append_terminal requires a terminal status.")
            record = ScientificAttemptRecord(
                self.campaign_id,
                episode_attempt_id,
                status,
                decision_count,
                trace_references,
                failure_class,
                failure_message,
                trace_absence_reason,
            )
            return self._append_attempt(record)

    def register_request_id(
        self,
        request_id: str,
        episode_attempt_id: str,
    ) -> AttemptReference:
        with self._append_guard():
            _require_text("request_id", request_id)
            _require_text("episode_attempt_id", episode_attempt_id)
            self._ensure_writable(episode_attempt_id)
            if (
                self._attempt_status.get(episode_attempt_id)
                is not AttemptStatus.STARTED
            ):
                raise ScientificAttemptWriteError("Request has no live attempt owner.")
            if request_id in self._request_owner:
                raise ScientificAttemptWriteError(
                    "request_id must be campaign-wide unique."
                )
            reference = self._append_event(
                {
                    "event_type": "request_registered",
                    "campaign_id": self.campaign_id,
                    "request_id": request_id,
                    "episode_attempt_id": episode_attempt_id,
                },
                episode_attempt_id,
            )
            self._request_owner[request_id] = episode_attempt_id
            return reference

    def can_resume(self, episode_attempt_id: str) -> bool:
        _require_text("episode_attempt_id", episode_attempt_id)
        if self._poisoned or self._poison_path.exists() or self._pending_path.exists():
            return False
        try:
            with self._append_guard():
                return (
                    episode_attempt_id not in self._poisoned_attempts
                    and self._attempt_status.get(episode_attempt_id)
                    is AttemptStatus.STARTED
                )
        except ScientificAttemptWriteError:
            if (
                self._poisoned
                or self._poison_path.exists()
                or self._pending_path.exists()
            ):
                return False
            raise

    def validate_trace_evidence(self, writer: ScientificTraceWriter) -> None:
        if not isinstance(writer, ScientificTraceWriter):
            raise ValueError("writer must be ScientificTraceWriter.")
        try:
            with writer.locked_reference_snapshot_by_attempt() as available:
                # The fixed trace -> ledger lock order gives this join one stable
                # cross-artifact view. Runtime writes do not nest these locks.
                with self._append_guard():
                    for record in self._terminal_records:
                        episode = (
                            record.campaign_id,
                            record.episode_attempt_id,
                        )
                        if record.trace_references != available.get(episode, ()):
                            raise ScientificAttemptWriteError(
                                "Attempt ledger does not match ordered trace evidence."
                            )
        except ScientificTraceWriteError as exc:
            raise ScientificAttemptWriteError(
                "Scientific trace artifact failed campaign evidence validation."
            ) from exc

    @contextmanager
    def _append_guard(self) -> Iterator[None]:
        try:
            with exclusive_append_lock(self._lock_path):
                if self._poison_path.exists() or self._pending_path.exists():
                    raise ScientificAttemptWriteError(
                        "Attempt ledger has an ambiguous durable append."
                    )
                self._reload()
                yield
        except ScientificAttemptWriteError:
            raise
        except FileExistsError as exc:
            raise ScientificAttemptWriteError(
                "Another process owns the campaign attempt append lock."
            ) from exc

    def _reload(self) -> None:
        size = self.path.stat().st_size if self.path.exists() else 0
        if size < self._byte_offset:
            raise ScientificAttemptWriteError("Attempt ledger was truncated.")
        if size > self._byte_offset:
            self._scan_existing()

    def _append_attempt(self, record: ScientificAttemptRecord) -> AttemptReference:
        attempt_id = record.episode_attempt_id
        self._ensure_writable(attempt_id)
        current = self._attempt_status.get(attempt_id)
        if record.status is AttemptStatus.STARTED and current is not None:
            raise ScientificAttemptWriteError("Attempt may be started only once.")
        if (
            record.status is not AttemptStatus.STARTED
            and current is not AttemptStatus.STARTED
        ):
            raise ScientificAttemptWriteError("Attempt must have one live start.")
        reference = self._append_event(
            {"event_type": "attempt_lifecycle", "attempt": record.to_dict()},
            attempt_id,
        )
        self._attempt_status[attempt_id] = record.status
        if record.status is AttemptStatus.WRITE_AMBIGUOUS:
            self._poisoned_attempts.add(attempt_id)
        if record.status is not AttemptStatus.STARTED:
            self._terminal_records.append(record)
        return reference

    def _append_event(
        self,
        event: dict[str, object],
        episode_attempt_id: str,
    ) -> AttemptReference:
        self._ensure_writable(episode_attempt_id)
        payload = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "sequence": self._line_count + 1,
            "previous_record_sha256": self._last_hash,
            **event,
        }
        digest = hash_payload(payload)
        payload["record_sha256"] = digest
        line = canonical_bytes(payload) + b"\n"
        self._durable_append(line, episode_attempt_id, digest)
        previous = self._last_hash
        self._line_count += 1
        self._byte_offset += len(line)
        self._last_hash = digest
        return AttemptReference(self._line_count, digest, previous)

    def _scan_existing(self) -> None:
        try:
            with self.path.open("rb") as handle:
                handle.seek(self._byte_offset)
                for line_number, raw_line in enumerate(
                    handle,
                    start=self._line_count + 1,
                ):
                    if not raw_line.endswith(b"\n"):
                        raise ValueError("Attempt ledger has a truncated tail.")
                    encoded = raw_line[:-1]
                    payload = json.loads(
                        encoded.decode("utf-8"),
                        parse_constant=reject_json_constant,
                    )
                    if not isinstance(payload, dict):
                        raise ValueError("Attempt ledger line must be an object.")
                    if encoded != canonical_bytes(payload):
                        raise ValueError("Attempt ledger line is not canonical JSON.")
                    self._validate_chain(payload, line_number)
                    self._replay(payload)
                self._byte_offset = handle.tell()
        except Exception as exc:
            raise ScientificAttemptWriteError(
                "Existing attempt ledger failed resume validation."
            ) from exc

    def _validate_chain(self, payload: dict[str, Any], line_number: int) -> None:
        sequence = payload.get("sequence")
        if payload.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
            raise ValueError("Attempt ledger schema version drifted.")
        if type(sequence) is not int or sequence != line_number:
            raise ValueError("Attempt ledger sequence is not contiguous.")
        if payload.get("previous_record_sha256") != self._last_hash:
            raise ValueError("Attempt ledger hash chain is broken.")
        claimed = payload.get("record_sha256")
        unhashed = dict(payload)
        unhashed.pop("record_sha256", None)
        if not _is_sha256(claimed) or claimed != hash_payload(unhashed):
            raise ValueError("Attempt ledger integrity hash is invalid.")

    def _replay(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("event_type")
        if event_type == "attempt_lifecycle":
            self._replay_attempt(payload)
        elif event_type == "request_registered":
            self._replay_request(payload)
        else:
            raise ValueError("Unknown campaign attempt event type.")
        self._line_count += 1
        self._last_hash = payload["record_sha256"]

    def _replay_attempt(self, payload: dict[str, Any]) -> None:
        require_keys(payload, ATTEMPT_EVENT_FIELDS)
        record = record_from_dict(payload.get("attempt"))
        if record.campaign_id != self.campaign_id:
            raise ValueError("Attempt belongs to another campaign.")
        current = self._attempt_status.get(record.episode_attempt_id)
        if record.status is AttemptStatus.STARTED and current is not None:
            raise ValueError("Duplicate attempt start.")
        if (
            record.status is not AttemptStatus.STARTED
            and current is not AttemptStatus.STARTED
        ):
            raise ValueError("Terminal attempt has no unique start.")
        self._attempt_status[record.episode_attempt_id] = record.status
        if record.status is AttemptStatus.WRITE_AMBIGUOUS:
            self._poisoned_attempts.add(record.episode_attempt_id)
        if record.status is not AttemptStatus.STARTED:
            self._terminal_records.append(record)

    def _replay_request(self, payload: dict[str, Any]) -> None:
        require_keys(payload, REQUEST_EVENT_FIELDS)
        campaign_id = _require_text("campaign_id", payload.get("campaign_id"))
        request_id = _require_text("request_id", payload.get("request_id"))
        attempt_id = _require_text(
            "episode_attempt_id",
            payload.get("episode_attempt_id"),
        )
        if campaign_id != self.campaign_id:
            raise ValueError("Request belongs to another campaign.")
        if self._attempt_status.get(attempt_id) is not AttemptStatus.STARTED:
            raise ValueError("Request has no live attempt owner.")
        if request_id in self._request_owner:
            raise ValueError("Duplicate campaign request_id.")
        self._request_owner[request_id] = attempt_id

    def _ensure_writable(self, episode_attempt_id: str) -> None:
        if self._poisoned:
            raise ScientificAttemptWriteError("Attempt ledger is poisoned.")
        if episode_attempt_id in self._poisoned_attempts:
            raise ScientificAttemptWriteError("Episode attempt is poisoned.")

    def _durable_append(
        self,
        line: bytes,
        episode_attempt_id: str,
        record_sha256: str,
    ) -> None:
        try:
            durable_append_with_intent(
                self.path,
                line,
                artifact_kind="attempt_ledger",
                episode_attempt_id=episode_attempt_id,
                expected_offset=self._byte_offset,
                record_sha256=record_sha256,
            )
        except (AppendIntentWriteError, AppendCommitAmbiguousError) as exc:
            self._poisoned = True
            self._poisoned_attempts.add(episode_attempt_id)
            raise ScientificAttemptWriteError(
                "Attempt append was not durably committed."
            ) from exc


__all__ = ["ScientificAttemptLedger"]
