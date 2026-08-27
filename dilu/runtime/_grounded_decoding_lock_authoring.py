"""S1 capability probing and lock-plan support for the ICLR 2027 V8 campaign.

Extends the existing S1 machinery (``_runtime_lock_authoring_support.py`` /
``_runtime_lock_authoring_workflow.py``) to the grounded (O2) mechanism
rather than forking it: model probing reuses ``probe_model`` (given one
extra grounded-schema request), lock-plan construction reuses
``build_lock_plans``/``verify_lock_plan`` unmodified, and the deterministic
smoke identities come from ``grounded_decoding_schedule.build_v8_smoke_schedule``.
The only genuinely new logic here is (a) building the grounded probe
request itself, (b) assembling capability records that record
``SCHEMA_MECHANISM_GROUNDED``, and (c) comparing each live model digest
against the digest recorded in the frozen V5/V7 comparator campaign.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._grounded_decoding_manifest_support import GroundedDecodingManifest
from ._minimal_factorial_manifest import ExperimentManifest
from ._runtime_lock_authoring_support import (
    GetCallable,
    PostCallable,
    canonical_bytes,
    probe_model,
)
from ._runtime_lock_authoring_workflow import LockPlan
from .action_resolution import FIXED_IDLE_ACTION_ID
from .harness_config import OutputEnforcement, ThinkMode
from .ollama_transport import OllamaModelIdentity
from .scientific_transport_types import (
    SCHEMA_MECHANISM_GROUNDED,
    GenerationRequest,
    NativeGenerationOptions,
    ScientificTransportCapabilities,
    canonical_action_text_schema,
)

GROUNDED_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE = (
    "ollama_native_capability_preflight_grounded_v1"
)
# Fixed per the task-5 brief: the grounded probe restricts the backend
# schema's enum to exactly these two action ids and verifies the returned
# action text resolves to one of them.
GROUNDED_PROBE_ACTION_IDS: tuple[int, ...] = (1, 3)
_GROUNDED_PROBE_INSTRUCTION_TEXT = f"Response to user:#### {FIXED_IDLE_ACTION_ID}"


def build_grounded_probe_request(
    *,
    model_slot: str,
    identity: OllamaModelIdentity,
    native_endpoint: str,
    seed: int,
    temperature: float,
    context_tokens: int,
    max_output_tokens: int,
    timeout_sec: float,
    think_mode: ThinkMode,
) -> GenerationRequest:
    """The fourth S1 probe request: grounded schema, enum restricted to (1, 3).

    Reuses the same instruction wording as the existing (O1) schema probe
    -- action id ``FIXED_IDLE_ACTION_ID`` (1) is legal under the restricted
    enum too, so a real backend that honors the narrowed schema can still
    comply with the instruction; a backend that ignores the restriction and
    returns an action outside {1, 3} fails the probe's evidence check.
    """
    return GenerationRequest(
        model_tag=identity.model_tag,
        model_digest=identity.model_digest,
        request_id=f"s1-{model_slot}-grounded-schema",
        messages=(
            (
                "system",
                f"Return exactly this text and nothing else: {_GROUNDED_PROBE_INSTRUCTION_TEXT}",
            ),
            ("user", "Perform the grounded response-format capability check."),
        ),
        native_endpoint=native_endpoint,
        options=NativeGenerationOptions(
            seed=seed,
            temperature=temperature,
            num_ctx=context_tokens,
            num_predict=max_output_tokens,
        ),
        output_enforcement=OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
        think_mode=think_mode,
        timeout_sec=timeout_sec,
        available_action_ids=GROUNDED_PROBE_ACTION_IDS,
    )


def probe_v8_models(
    manifest: GroundedDecodingManifest,
    *,
    get: GetCallable,
    post: PostCallable,
) -> tuple[dict[str, OllamaModelIdentity], list[dict[str, Any]]]:
    """Probe every V8 model with the trusted 3 calls plus 1 grounded call."""
    bindings: dict[str, OllamaModelIdentity] = {}
    records: list[dict[str, Any]] = []
    think_mode = ThinkMode(manifest.transport.think_mode)
    canonical_schema_bytes = canonical_bytes(canonical_action_text_schema())
    for index, model in enumerate(manifest.models):
        seed = manifest.transport.generation_seed_master + index

        def extra_requests(
            identity: OllamaModelIdentity,
            _slot: str = model.slot,
            _seed: int = seed,
        ) -> tuple[GenerationRequest, ...]:
            return (
                build_grounded_probe_request(
                    model_slot=_slot,
                    identity=identity,
                    native_endpoint=manifest.transport.native_endpoint,
                    seed=_seed,
                    temperature=manifest.transport.temperature,
                    context_tokens=manifest.transport.context_tokens,
                    max_output_tokens=manifest.transport.max_output_tokens,
                    timeout_sec=manifest.transport.timeout_sec,
                    think_mode=think_mode,
                ),
            )

        identity, model_records = probe_model(
            model_slot=model.slot,
            model_tag=model.tag,
            native_endpoint=manifest.transport.native_endpoint,
            seed=seed,
            temperature=manifest.transport.temperature,
            context_tokens=manifest.transport.context_tokens,
            max_output_tokens=manifest.transport.max_output_tokens,
            timeout_sec=manifest.transport.timeout_sec,
            think_mode=think_mode,
            canonical_schema_bytes=canonical_schema_bytes,
            get=get,
            post=post,
            extra_requests=extra_requests,
        )
        bindings[model.slot] = identity
        records.extend({"model_slot": model.slot, **record} for record in model_records)
    expected_records = 4 * len(manifest.models)
    if len(records) != expected_records:
        raise ValueError(
            "Grounded capability preflight must contain exactly four direct calls per model."
        )
    return bindings, records


def build_v8_capabilities(
    manifest: GroundedDecodingManifest,
    bindings: Mapping[str, OllamaModelIdentity],
    artifact_hash: str,
) -> dict[str, ScientificTransportCapabilities]:
    """Capabilities for V8's O2 cells: identical shape to V5/V7's, grounded mechanism."""
    think_mode = ThinkMode(manifest.transport.think_mode)
    return {
        slot: ScientificTransportCapabilities(
            model_tag=identity.model_tag,
            model_digest=identity.model_digest,
            native_endpoint=manifest.transport.native_endpoint,
            supported_think_modes=(think_mode,),
            seed_verified=True,
            schema_verified=True,
            capability_probe_id=f"s1-grounded-native-probe-{slot}",
            capability_artifact_hash=artifact_hash,
            schema_mechanism=SCHEMA_MECHANISM_GROUNDED,
        )
        for slot, identity in bindings.items()
    }


def resolve_comparator_digest_matches(
    manifest: GroundedDecodingManifest,
    bindings: Mapping[str, OllamaModelIdentity],
    v5_manifest: ExperimentManifest,
    v7_manifest: ExperimentManifest,
) -> dict[str, bool]:
    """Compare each live V8 model digest to the frozen V5/V7 comparator digest.

    A mismatch is recorded, not raised -- the caller writes the result into
    ``model_preflight.json`` as ``comparator_digest_match`` and lock
    authoring proceeds regardless; a later task feeds a ``False`` entry
    into the registered rerun contingency.
    """
    root = manifest.repo_root()
    v5_digests = _load_frozen_model_digests(
        root / manifest.comparators.v5_manifest, expected_campaign_id=v5_manifest.campaign_id
    )
    v7_digests = _load_frozen_model_digests(
        root / manifest.comparators.v7_manifest, expected_campaign_id=v7_manifest.campaign_id
    )
    overlap = set(v5_digests) & set(v7_digests)
    if overlap:
        raise ValueError(f"V5 and V7 comparator campaigns share model tags: {sorted(overlap)}.")
    frozen_digests = {**v5_digests, **v7_digests}
    matches: dict[str, bool] = {}
    for model in manifest.models:
        identity = bindings.get(model.slot)
        if identity is None or identity.model_tag != model.tag:
            raise ValueError(f"Missing or mismatched live binding for {model.slot}.")
        frozen_digest = frozen_digests.get(model.tag)
        if frozen_digest is None:
            raise ValueError(f"No frozen comparator digest recorded for {model.tag}.")
        matches[model.slot] = identity.model_digest == frozen_digest
    return matches


def _load_frozen_model_digests(path: Path, *, expected_campaign_id: str) -> dict[str, str]:
    """Read the one-digest-per-model map out of a frozen ``campaign_manifest.json``."""
    try:
        decoded = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{path} could not be read as a frozen campaign manifest.") from exc
    if not isinstance(decoded, Mapping) or not {"manifest", "schedule"} <= set(decoded):
        raise ValueError(f"{path} is not a valid frozen campaign manifest.")
    manifest_section = decoded["manifest"]
    if (
        not isinstance(manifest_section, Mapping)
        or manifest_section.get("campaign_id") != expected_campaign_id
    ):
        raise ValueError(f"{path} campaign_id does not match the trusted frozen manifest.")
    schedule = decoded["schedule"]
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(f"{path} has an empty or invalid schedule.")
    digests: dict[str, str] = {}
    for row in schedule:
        if not isinstance(row, Mapping) or row.get("campaign_id") != expected_campaign_id:
            raise ValueError(f"{path} contains a row with a mismatched campaign_id.")
        model_tag = row.get("model_tag")
        model_digest = row.get("model_digest")
        if not isinstance(model_tag, str) or not isinstance(model_digest, str):
            raise ValueError(f"{path} row is missing model_tag/model_digest.")
        if model_tag in digests and digests[model_tag] != model_digest:
            raise ValueError(f"{path} records inconsistent digests for {model_tag}.")
        digests[model_tag] = model_digest
    return digests


def v8_artifact_paths(destination: Path, locks: tuple[LockPlan, ...]) -> tuple[Path, ...]:
    """The exact V8 S1 artifact tree: preflight plus the lock/authorization pairs.

    Deliberately narrower than V5/V7's ``artifact_paths``: Task 5 scopes S1
    to the capability probe and lock authoring only, so no
    ``smoke``/``llm_campaign`` frozen schedule files are written here.
    """
    return (
        destination / "s1" / "model_preflight.json",
        *(
            path
            for plan in locks
            for path in (plan.runtime_path, plan.authorization_path)
        ),
    )


__all__ = [
    "GROUNDED_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE",
    "GROUNDED_PROBE_ACTION_IDS",
    "build_grounded_probe_request",
    "build_v8_capabilities",
    "probe_v8_models",
    "resolve_comparator_digest_matches",
    "v8_artifact_paths",
]
