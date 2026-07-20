from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


_CAPABILITY_SNAPSHOT_FIELDS = (
    "capability_model_tag",
    "capability_model_digest",
    "capability_native_endpoint",
    "capability_supported_think_modes",
    "seed_verified",
    "schema_verified",
    "capability_probe_id",
    "capability_artifact_sha256",
    "schema_mechanism",
)


def capability_snapshot_sha256(evidence: Mapping[str, Any]) -> str:
    snapshot = {name: evidence[name] for name in _CAPABILITY_SNAPSHOT_FIELDS}
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = ["capability_snapshot_sha256"]
