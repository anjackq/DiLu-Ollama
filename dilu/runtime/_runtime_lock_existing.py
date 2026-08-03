"""Strict zero-generation loading for completed S1 authoring destinations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._minimal_factorial_manifest import serialize_frozen_campaign
from ._runtime_lock_authoring_support import (
    build_probe_requests,
    build_request_evidence,
    bytes_sha256,
    canonical_bytes,
    derive_response_evidence,
    OLLAMA_NATIVE_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE,
)
from .action_resolution import ActionResolutionResult
from ._runtime_lock_authoring_workflow import (
    RuntimeLockArtifact,
    artifact_paths,
    build_capabilities,
    build_lock_plans,
    build_schedules,
    verify_lock_plan,
)
from ._runtime_lock_tree_validation import (
    validate_exact_lock_tree,
    validate_unredirected_artifact_paths,
)
from .harness_config import ThinkMode
from .minimal_factorial_schedule import ExperimentManifest, RuntimeSnapshot
from .ollama_transport import OllamaModelIdentity
from .scientific_transport_types import (
    GenerationRequest,
    ScientificTransportCapabilities,
    build_native_chat_payload,
)

_RECORD_FIELDS = {
    "model_slot",
    "request",
    "payload",
    "payload_sha256",
    "request_body",
    "http_status",
    "response_body",
    "raw_response",
    "contract_text",
    "action_resolution",
    "stop_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "backend_timing",
    "identity_before",
    "identity_after",
}
_REQUEST_FIELDS = {
    "model_tag",
    "model_digest",
    "request_id",
    "messages",
    "native_endpoint",
    "options",
    "output_enforcement",
    "think_mode",
    "timeout_sec",
}


@dataclass(frozen=True)
class ExistingCampaign:
    preflight_sha256: str
    bindings: Mapping[str, OllamaModelIdentity]
    capabilities: Mapping[str, ScientificTransportCapabilities]
    lock_artifacts: tuple[RuntimeLockArtifact, ...]


def load_existing_campaign(
    *,
    destination: Path,
    manifest: ExperimentManifest,
    case_set: Mapping[str, Any],
    snapshot: RuntimeSnapshot,
    canonical_schema_bytes: bytes,
) -> ExistingCampaign:
    relative_paths = _expected_relative_paths(manifest)
    absolute_paths = tuple(destination / path for path in relative_paths)
    validate_unredirected_artifact_paths(absolute_paths)
    validate_exact_lock_tree(destination, relative_paths)

    preflight_path = destination / "s1" / "model_preflight.json"
    preflight_bytes, preflight = _load_canonical_object(preflight_path)
    if set(preflight) != {
        "artifact_type",
        "runtime_snapshot_sha256",
        "records",
    }:
        raise ValueError("Completed preflight fields drifted.")
    if preflight["artifact_type"] != OLLAMA_NATIVE_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE:
        raise ValueError("Completed preflight artifact type drifted.")
    if preflight["runtime_snapshot_sha256"] != "sha256:" + snapshot.sha256:
        raise ValueError("Completed preflight runtime snapshot drifted.")
    records = preflight["records"]
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("Completed preflight must contain six records.")
    bindings = _load_bindings(
        manifest,
        records,
        canonical_schema_bytes=canonical_schema_bytes,
    )
    preflight_sha256 = bytes_sha256(preflight_bytes)
    capabilities = build_capabilities(manifest, bindings, preflight_sha256)
    smoke, union = build_schedules(manifest, case_set, snapshot, bindings)
    locks = build_lock_plans(destination, manifest, smoke, capabilities)
    if artifact_paths(destination, locks) != absolute_paths:
        raise ValueError("Completed artifact path plan drifted.")
    _verify_campaign_bytes(
        destination=destination,
        manifest=manifest,
        case_set=case_set,
        snapshot=snapshot,
        smoke=smoke,
        union=union,
    )
    lock_artifacts: list[RuntimeLockArtifact] = []
    for lock in locks:
        if (
            lock.runtime_path.read_bytes() != lock.runtime_bytes
            or lock.authorization_path.read_bytes() != lock.authorization_bytes
        ):
            raise ValueError("Completed runtime-lock bytes drifted.")
        lock_artifacts.append(verify_lock_plan(lock))
    return ExistingCampaign(
        preflight_sha256,
        MappingProxyType(bindings),
        MappingProxyType(capabilities),
        tuple(lock_artifacts),
    )


def _expected_relative_paths(manifest: ExperimentManifest) -> tuple[Path, ...]:
    locks = tuple(
        Path("s1") / "locks" / model.slot / f"c{index:03b}" / artifact
        for model in manifest.models
        for index in range(8)
        for artifact in ("RUNTIME_PROTOCOL_LOCK.json", "PROTOCOL_FROZEN.json")
    )
    return (
        Path("s1") / "model_preflight.json",
        Path("smoke") / "campaign_manifest.json",
        Path("llm_campaign") / "campaign_manifest.json",
        Path("llm_campaign") / "union_schedule.json",
        *locks,
    )


def _load_bindings(
    manifest: ExperimentManifest,
    records: list[Any],
    *,
    canonical_schema_bytes: bytes,
) -> dict[str, OllamaModelIdentity]:
    expected_slots = [
        model.slot for model in manifest.models for _probe_index in range(3)
    ]
    if [
        record.get("model_slot") if isinstance(record, Mapping) else None
        for record in records
    ] != expected_slots:
        raise ValueError("Completed preflight record order drifted.")
    bindings: dict[str, OllamaModelIdentity] = {}
    for model_index, model in enumerate(manifest.models):
        model_records = [
            record
            for record in records
            if isinstance(record, Mapping) and record.get("model_slot") == model.slot
        ]
        if len(model_records) != 3:
            raise ValueError("Completed preflight model record count drifted.")
        identity = _identity(model_records[0]["identity_before"], model.tag)
        expected_requests = build_probe_requests(
            model_slot=model.slot,
            identity=identity,
            native_endpoint=manifest.transport.native_endpoint,
            seed=manifest.transport.generation_seed_master + model_index,
            temperature=manifest.transport.temperature,
            context_tokens=manifest.transport.context_tokens,
            max_output_tokens=manifest.transport.max_output_tokens,
            timeout_sec=manifest.transport.timeout_sec,
            think_mode=ThinkMode(manifest.transport.think_mode),
        )
        validated = [
            _validate_record(
                record,
                model_slot=model.slot,
                model_tag=model.tag,
                canonical_schema_bytes=canonical_schema_bytes,
                expected_request=expected_request,
            )
            for record, expected_request in zip(
                model_records,
                expected_requests,
                strict=True,
            )
        ]
        first, repeat, _schema = validated
        if (
            first[1] != repeat[1]
            or first[2].final_resolved_action != repeat[2].final_resolved_action
        ):
            raise ValueError("Completed prompt-only repeat evidence drifted.")
        bindings[model.slot] = first[0]
    if len(bindings) != len(manifest.models):
        raise ValueError("Completed model bindings drifted.")
    return bindings


def _validate_record(
    record: Mapping[str, Any],
    *,
    model_slot: str,
    model_tag: str,
    canonical_schema_bytes: bytes,
    expected_request: GenerationRequest,
) -> tuple[OllamaModelIdentity, str, ActionResolutionResult]:
    if set(record) != _RECORD_FIELDS or record["model_slot"] != model_slot:
        raise ValueError("Completed preflight record fields drifted.")
    before = _identity(record["identity_before"], model_tag)
    after = _identity(record["identity_after"], model_tag)
    if before != after:
        raise ValueError("Completed preflight model identity drifted.")
    request = record["request"]
    payload = record["payload"]
    if (
        not isinstance(request, Mapping)
        or set(request) != _REQUEST_FIELDS
        or before.model_tag != expected_request.model_tag
        or before.model_digest != expected_request.model_digest
        or dict(request) != build_request_evidence(expected_request)
        or not isinstance(payload, Mapping)
        or dict(payload) != build_native_chat_payload(expected_request)
    ):
        raise ValueError("Completed preflight request evidence drifted.")
    payload_bytes = canonical_bytes(payload)
    if record["request_body"] != payload_bytes.decode("utf-8") or record[
        "payload_sha256"
    ] != bytes_sha256(payload_bytes):
        raise ValueError("Completed preflight payload bytes drifted.")
    if expected_request.output_enforcement.value == "backend_schema":
        if canonical_bytes(payload.get("format")) != canonical_schema_bytes:
            raise ValueError("Completed backend schema evidence drifted.")
    elif "format" in payload:
        raise ValueError("Completed prompt-only payload contains schema.")
    response_body = record["response_body"]
    status = record["http_status"]
    if not isinstance(response_body, str) or not _direct_status(status):
        raise ValueError("Completed response evidence drifted.")
    try:
        response_payload = json.loads(response_body)
    except (TypeError, ValueError) as exc:
        raise ValueError("Completed response body is malformed JSON.") from exc
    expected_response, resolution = derive_response_evidence(
        expected_request,
        status,
        response_payload,
        response_body,
    )
    if any(record[field] != value for field, value in expected_response.items()):
        raise ValueError("Completed response evidence drifted.")
    return before, record["request_body"], resolution


def _identity(value: object, model_tag: str) -> OllamaModelIdentity:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"model_tag", "model_digest"}
        or value["model_tag"] != model_tag
    ):
        raise ValueError("Completed identity evidence drifted.")
    return OllamaModelIdentity(value["model_tag"], value["model_digest"])


def _verify_campaign_bytes(
    *,
    destination: Path,
    manifest: ExperimentManifest,
    case_set: Mapping[str, Any],
    snapshot: RuntimeSnapshot,
    smoke: tuple[Any, ...],
    union: tuple[Any, ...],
) -> None:
    expected = (
        (
            destination / "smoke" / "campaign_manifest.json",
            serialize_frozen_campaign(manifest, snapshot, smoke, case_set),
        ),
        (
            destination / "llm_campaign" / "campaign_manifest.json",
            serialize_frozen_campaign(manifest, snapshot, union, case_set),
        ),
        (
            destination / "llm_campaign" / "union_schedule.json",
            canonical_bytes([row.to_payload() for row in union]),
        ),
    )
    if any(path.read_bytes() != content for path, content in expected):
        raise ValueError("Completed campaign artifact bytes drifted.")


def _load_canonical_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    content = path.read_bytes()
    try:
        decoded = json.loads(content.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Completed artifact is malformed JSON.") from exc
    if not isinstance(decoded, dict) or canonical_bytes(decoded) != content:
        raise ValueError("Completed artifact is not canonical JSON.")
    return content, decoded


def _direct_status(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 200 <= value < 300
