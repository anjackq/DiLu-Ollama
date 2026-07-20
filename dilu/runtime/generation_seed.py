from __future__ import annotations

import hashlib

from ._scientific_transport_validation import require_model_digest


UINT32_MAX = (1 << 32) - 1


def primary_snapshot_generation_seed(
    master_seed: int,
    model_digest: str,
    pair_id: str,
    decision_snapshot_id: str,
    replicate_id: int,
) -> int:
    """Return the pair-shared seed before counterfactual trajectories diverge."""
    return _sha256_uint32(
        _validate_master_seed(master_seed),
        _validate_model_digest(model_digest),
        _validate_text("pair_id", pair_id),
        _validate_text("decision_snapshot_id", decision_snapshot_id),
        _validate_nonnegative_int("replicate_id", replicate_id),
    )


def post_divergence_generation_seed(
    master_seed: int,
    model_digest: str,
    case_id: str,
    decision_index: int,
    replicate_id: int,
) -> int:
    """Return a case-scoped seed after trajectories can differ."""
    return _sha256_uint32(
        _validate_master_seed(master_seed),
        _validate_model_digest(model_digest),
        _validate_text("case_id", case_id),
        _validate_nonnegative_int("decision_index", decision_index),
        _validate_nonnegative_int("replicate_id", replicate_id),
    )


def _sha256_uint32(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def _validate_master_seed(value: int) -> int:
    seed = _validate_nonnegative_int("master_seed", value)
    if seed > UINT32_MAX:
        raise ValueError("master_seed must fit in uint32.")
    return seed


def _validate_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _validate_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string.")
    if "|" in value:
        raise ValueError(f"{name} cannot contain the seed field delimiter '|'.")
    return value


def _validate_model_digest(value: str) -> str:
    require_model_digest("model_digest", value)
    return value


__all__ = [
    "UINT32_MAX",
    "post_divergence_generation_seed",
    "primary_snapshot_generation_seed",
]
