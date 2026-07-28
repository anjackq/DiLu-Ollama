"""Central validation for frozen minimal-factorial episode provenance."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from ._scientific_transport_validation import require_model_digest

_REVISION_RE = re.compile(r"\A[0-9a-fA-F]{40}\Z")


def validate_schedule_rows(
    manifest: Any,
    snapshot: Any,
    schedule: Sequence[Any],
    case_set: Mapping[str, Any] | None = None,
) -> None:
    fingerprint = snapshot.payload.get("case_set_fingerprint")
    revision = snapshot.payload.get("code_revision")
    if not isinstance(fingerprint, str) or not isinstance(revision, str):
        raise ValueError("Frozen snapshot fingerprint or revision is invalid.")
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError("Frozen snapshot revision is invalid.")
    cases = {case["case_id"]: case for case in (case_set or {}).get("cases", [])}
    models = {model.slot: model.tag for model in manifest.models}
    for episode in schedule:
        validate_episode(episode, manifest, models, fingerprint, revision, cases)


def validate_episode(
    episode: Any,
    manifest: Any,
    models: Mapping[str, str],
    fingerprint: str,
    revision: str,
    cases: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_campaign = (
        manifest.smoke_campaign_id if episode.stage == "smoke" else manifest.campaign_id
    )
    if episode.campaign_id != expected_campaign:
        raise ValueError("Scheduled episode campaign drifted.")
    if models.get(episode.model_slot) != episode.model_tag:
        raise ValueError("Scheduled episode model binding drifted.")
    require_model_digest("scheduled model_digest", episode.model_digest)
    if episode.condition_id != episode.condition.condition_id():
        raise ValueError("Scheduled episode condition id drifted.")
    if episode.condition_id not in {f"c{index:03b}" for index in range(8)}:
        raise ValueError("Scheduled episode condition is not frozen.")
    case = cases.get(episode.case_id)
    if case is not None and case.get("seed") != episode.simulator_seed:
        raise ValueError("Scheduled episode case seed drifted.")
    if episode.replicate_id != 0:
        raise ValueError("Scheduled episode replicate drifted.")
    case_id = episode.case_id
    seed = episode.simulator_seed
    campaign = episode.campaign_id
    _expect(episode.pair_id, "pair-", f"{campaign}|{case_id}|{seed}")
    _expect(episode.template_id, "stress-v2-", f"{fingerprint}|{case_id}")
    _expect(episode.primary_snapshot_id, "snapshot-", f"{fingerprint}|{case_id}|{seed}")
    _expect(
        episode.episode_attempt_id,
        "episode-",
        f"{campaign}|{episode.model_tag}|{episode.model_digest}|"
        f"{episode.condition_id}|{case_id}|{seed}|0",
    )
    if (
        episode.code_revision != revision
        or episode.benchmark_fingerprint != fingerprint
    ):
        raise ValueError("Scheduled episode snapshot binding drifted.")
    episode.identity()


def _expect(actual: str, prefix: str, payload: str) -> None:
    expected = prefix + hashlib.sha256(payload.encode()).hexdigest()
    if actual != expected:
        raise ValueError("Scheduled episode deterministic identity drifted.")
