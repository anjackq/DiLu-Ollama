from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn

from dilu.driver_agent.prompt_modules import build_prompt_artifact

from ._scientific_trace_hashing import capability_snapshot_sha256
from .harness_config import HarnessConfig, ThinkMode
from .scientific_trace import trace_schema_sha256
from .scientific_transport_types import ScientificTransportCapabilities


_SHA256_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_RUNTIME_LOCK_FIELDS = {
    "condition_id",
    "config_sha256",
    "prompt_sha256",
    "model_tag",
    "model_digest",
    "native_endpoint",
    "think_mode",
    "capability_artifact_sha256",
    "capability_snapshot_sha256",
    "trace_schema_sha256",
    "benchmark_fingerprint",
    "code_revision",
}
_AUTHORIZATION_FIELDS = {
    "artifact_type",
    "runtime_lock_sha256",
}
_AUTHORIZATION_ARTIFACT_TYPE = "runtime_lock_authorization_v1"
_RUNTIME_LOCK_LOADER_TOKEN = object()


@dataclass(frozen=True)
class ScientificEpisodeIdentity:
    campaign_id: str
    episode_attempt_id: str
    case_id: str
    pair_id: str
    template_id: str
    replicate_id: int
    simulator_seed: int
    primary_snapshot_id: str
    benchmark_fingerprint: str
    code_revision: str

    def __post_init__(self) -> None:
        for name in (
            "campaign_id",
            "episode_attempt_id",
            "case_id",
            "pair_id",
            "template_id",
            "primary_snapshot_id",
            "code_revision",
        ):
            _require_text(name, getattr(self, name))
        _require_nonnegative_int("replicate_id", self.replicate_id)
        _require_uint32("simulator_seed", self.simulator_seed)
        _require_sha256("benchmark_fingerprint", self.benchmark_fingerprint)


@dataclass(frozen=True)
class RuntimeLockBinding:
    condition_id: str
    config_sha256: str
    prompt_sha256: str
    model_tag: str
    model_digest: str
    native_endpoint: str
    think_mode: ThinkMode
    capability_artifact_sha256: str
    capability_snapshot_sha256: str
    trace_schema_sha256: str
    benchmark_fingerprint: str
    code_revision: str

    def __post_init__(self) -> None:
        for name in ("condition_id", "model_tag", "native_endpoint", "code_revision"):
            _require_text(name, getattr(self, name))
        for name in (
            "config_sha256",
            "prompt_sha256",
            "model_digest",
            "capability_artifact_sha256",
            "capability_snapshot_sha256",
            "trace_schema_sha256",
            "benchmark_fingerprint",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            not isinstance(self.think_mode, ThinkMode)
            or self.think_mode is ThinkMode.AUTO
        ):
            raise ValueError("think_mode must be an explicit ThinkMode.")

    def to_dict(self) -> dict[str, str]:
        return {
            "condition_id": self.condition_id,
            "config_sha256": self.config_sha256,
            "prompt_sha256": self.prompt_sha256,
            "model_tag": self.model_tag,
            "model_digest": self.model_digest,
            "native_endpoint": self.native_endpoint,
            "think_mode": self.think_mode.value,
            "capability_artifact_sha256": self.capability_artifact_sha256,
            "capability_snapshot_sha256": self.capability_snapshot_sha256,
            "trace_schema_sha256": self.trace_schema_sha256,
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "code_revision": self.code_revision,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RuntimeLockBinding":
        if not isinstance(value, Mapping) or set(value) != _RUNTIME_LOCK_FIELDS:
            raise ValueError("Runtime lock mapping fields drifted.")
        return cls(
            condition_id=value["condition_id"],
            config_sha256=value["config_sha256"],
            prompt_sha256=value["prompt_sha256"],
            model_tag=value["model_tag"],
            model_digest=value["model_digest"],
            native_endpoint=value["native_endpoint"],
            think_mode=ThinkMode(value["think_mode"]),
            capability_artifact_sha256=value["capability_artifact_sha256"],
            capability_snapshot_sha256=value["capability_snapshot_sha256"],
            trace_schema_sha256=value["trace_schema_sha256"],
            benchmark_fingerprint=value["benchmark_fingerprint"],
            code_revision=value["code_revision"],
        )

    @classmethod
    def from_runtime(
        cls,
        *,
        harness_config: HarnessConfig,
        identity: ScientificEpisodeIdentity,
        capabilities: ScientificTransportCapabilities,
    ) -> "RuntimeLockBinding":
        """Capture preflight values; claim runs use the authorized-file loader."""
        harness_config.validate_scientific()
        prompt = build_prompt_artifact(
            harness_config.condition.policy_content,
            output_enforcement=harness_config.condition.output_enforcement,
            few_shot_num=0,
        )
        evidence = _capability_evidence(capabilities)
        return cls(
            condition_id=harness_config.condition_id(),
            config_sha256="sha256:" + harness_config.config_hash(),
            prompt_sha256="sha256:" + prompt.prompt_hash(),
            model_tag=capabilities.model_tag,
            model_digest=capabilities.model_digest,
            native_endpoint=capabilities.native_endpoint,
            think_mode=harness_config.transport.think_mode,
            capability_artifact_sha256=capabilities.capability_artifact_hash,
            capability_snapshot_sha256=capability_snapshot_sha256(evidence),
            trace_schema_sha256=trace_schema_sha256(),
            benchmark_fingerprint=identity.benchmark_fingerprint,
            code_revision=identity.code_revision,
        )


@dataclass(frozen=True)
class _RuntimeLockLoadProof:
    source_artifact_sha256: str
    authorization_artifact_sha256: str
    binding_sha256: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _RUNTIME_LOCK_LOADER_TOKEN:
            raise TypeError("Runtime-lock load proofs are loader-owned.")


@dataclass(frozen=True)
class VerifiedRuntimeLockBinding(RuntimeLockBinding):
    source_artifact_sha256: str
    authorization_artifact_sha256: str
    binding_sha256: str
    _load_proof: _RuntimeLockLoadProof = field(repr=False, compare=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> NoReturn:
        del cls, value
        raise TypeError(
            "Verified runtime bindings must be loaded from authorized files."
        )

    @classmethod
    def from_runtime(
        cls,
        *,
        harness_config: HarnessConfig,
        identity: ScientificEpisodeIdentity,
        capabilities: ScientificTransportCapabilities,
    ) -> NoReturn:
        del cls, harness_config, identity, capabilities
        raise TypeError(
            "Verified runtime bindings must be loaded from authorized files."
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_sha256("source_artifact_sha256", self.source_artifact_sha256)
        _require_sha256(
            "authorization_artifact_sha256",
            self.authorization_artifact_sha256,
        )
        _require_sha256("binding_sha256", self.binding_sha256)
        if self.binding_sha256 != _mapping_sha256(self.to_dict()):
            raise ValueError("Verified runtime binding content hash drifted.")
        if not isinstance(self._load_proof, _RuntimeLockLoadProof):
            raise TypeError("Verified runtime bindings require loader proof.")
        if (
            self._load_proof.source_artifact_sha256 != self.source_artifact_sha256
            or self._load_proof.authorization_artifact_sha256
            != self.authorization_artifact_sha256
            or self._load_proof.binding_sha256 != self.binding_sha256
        ):
            raise ValueError("Runtime-lock load proof drifted.")


def load_verified_runtime_lock_binding(
    *,
    runtime_lock_path: Path,
    authorization_path: Path,
) -> VerifiedRuntimeLockBinding:
    """Load a runtime binding authorized by a separate trusted artifact."""
    resolved_runtime_path = _resolve_artifact_file(
        "runtime_lock_path",
        runtime_lock_path,
    )
    resolved_authorization_path = _resolve_artifact_file(
        "authorization_path",
        authorization_path,
    )
    if resolved_runtime_path == resolved_authorization_path:
        raise ValueError("Runtime-lock and authorization paths must be distinct.")

    runtime_bytes = resolved_runtime_path.read_bytes()
    authorization_bytes = resolved_authorization_path.read_bytes()
    source_artifact_sha256 = _bytes_sha256(runtime_bytes)
    authorization = _decode_json_object(
        authorization_bytes,
        artifact_name="authorization artifact",
    )
    if set(authorization) != _AUTHORIZATION_FIELDS:
        raise ValueError("Authorization artifact fields drifted.")
    if authorization["artifact_type"] != _AUTHORIZATION_ARTIFACT_TYPE:
        raise ValueError("Authorization artifact type drifted.")
    expected_source_artifact_sha256 = authorization["runtime_lock_sha256"]
    _require_sha256(
        "runtime_lock_sha256",
        expected_source_artifact_sha256,
    )
    if source_artifact_sha256 != expected_source_artifact_sha256:
        raise ValueError("Runtime-lock source artifact hash did not verify.")

    runtime_mapping = _decode_json_object(
        runtime_bytes,
        artifact_name="runtime-lock artifact",
    )
    binding = RuntimeLockBinding.from_mapping(runtime_mapping)
    authorization_artifact_sha256 = _bytes_sha256(authorization_bytes)
    binding_sha256 = _mapping_sha256(binding.to_dict())
    proof = _RuntimeLockLoadProof(
        source_artifact_sha256=source_artifact_sha256,
        authorization_artifact_sha256=authorization_artifact_sha256,
        binding_sha256=binding_sha256,
        _token=_RUNTIME_LOCK_LOADER_TOKEN,
    )
    return VerifiedRuntimeLockBinding(
        **binding.__dict__,
        source_artifact_sha256=source_artifact_sha256,
        authorization_artifact_sha256=authorization_artifact_sha256,
        binding_sha256=binding_sha256,
        _load_proof=proof,
    )


def _capability_evidence(
    capabilities: ScientificTransportCapabilities,
) -> dict[str, Any]:
    return {
        "capability_model_tag": capabilities.model_tag,
        "capability_model_digest": capabilities.model_digest,
        "capability_native_endpoint": capabilities.native_endpoint,
        "capability_supported_think_modes": [
            item.value for item in capabilities.supported_think_modes
        ],
        "seed_verified": capabilities.seed_verified,
        "schema_verified": capabilities.schema_verified,
        "capability_probe_id": capabilities.capability_probe_id,
        "capability_artifact_sha256": capabilities.capability_artifact_hash,
        "schema_mechanism": capabilities.schema_mechanism,
    }


def _mapping_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _resolve_artifact_file(name: str, value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path.")
    resolved = value.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must identify a regular file.")
    return resolved


def _decode_json_object(value: bytes, *, artifact_name: str) -> dict[str, object]:
    def reject_duplicate_fields(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, item in pairs:
            if key in decoded:
                raise ValueError(
                    f"{artifact_name} contains duplicate JSON field {key!r}."
                )
            decoded[key] = item
        return decoded

    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(
            f"{artifact_name} contains non-finite JSON constant {value!r}."
        )

    try:
        decoded = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=reject_nonfinite_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{artifact_name} must be UTF-8 JSON.") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{artifact_name} must be valid JSON.") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{artifact_name} must be a JSON object.")
    return decoded


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text.")


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _require_uint32(name: str, value: object) -> None:
    _require_nonnegative_int(name, value)
    if int(value) > (1 << 32) - 1:
        raise ValueError(f"{name} must fit in uint32.")


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a full sha256 digest.")


__all__ = [
    "RuntimeLockBinding",
    "ScientificEpisodeIdentity",
    "VerifiedRuntimeLockBinding",
    "load_verified_runtime_lock_binding",
]
