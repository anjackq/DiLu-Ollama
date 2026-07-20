from __future__ import annotations

from typing import Any

from ._scientific_trace_hashing import capability_snapshot_sha256


def validate_transport_evidence(
    evidence: dict[str, Any],
    generation: dict[str, Any],
    transport: dict[str, Any],
    request: dict[str, Any],
) -> None:
    requested_expected = (
        transport["profile"],
        transport["think_mode"],
    )
    requested_observed = (
        evidence["requested_profile"],
        evidence["requested_think_mode"],
    )
    if requested_observed != requested_expected:
        raise ValueError("Serialized requested transport evidence drifted.")
    supported_modes = evidence["capability_supported_think_modes"]
    if supported_modes != sorted(set(supported_modes)):
        raise ValueError("Serialized think-mode capability evidence is invalid.")
    if evidence["capability_snapshot_sha256"] != capability_snapshot_sha256(evidence):
        raise ValueError("Serialized capability snapshot hash is invalid.")

    profile_effective = any(
        attempt["accepted_by_server"] is True for attempt in generation["attempts"]
    )
    think_mode_effective = profile_effective and generation["error_class"] in {
        None,
        "model_empty_output",
    }
    capabilities_expected = (
        request["model_tag"],
        request["model_digest"],
        request["native_endpoint"],
    )
    capabilities_observed = (
        evidence["capability_model_tag"],
        evidence["capability_model_digest"],
        evidence["capability_native_endpoint"],
    )
    capability_preflight_failed = (
        capabilities_observed != capabilities_expected
        or request["think_mode"] not in supported_modes
        or not evidence["seed_verified"]
        or (
            request["output_enforcement"] == "backend_schema"
            and not evidence["schema_verified"]
        )
    )
    if generation["attempts"] and capability_preflight_failed:
        raise ValueError("Attempted generation lacks verified capabilities.")

    expected_profile = transport["profile"] if profile_effective else None
    if evidence["effective_profile"] != expected_profile:
        raise ValueError("Serialized effective transport profile drifted.")
    expected_think_mode = request["think_mode"] if think_mode_effective else None
    if evidence["effective_think_mode"] != expected_think_mode:
        raise ValueError("Serialized effective think-mode evidence drifted.")

    if not generation["attempts"]:
        checks = generation["identity_checks"]
        if capability_preflight_failed and checks:
            raise ValueError(
                "Capability-preflight failure contains identity-check evidence."
            )
        if not capability_preflight_failed and not checks:
            raise ValueError(
                "Preflight block lacks capability or identity failure evidence."
            )
    if not generation["attempts"] and generation["error_class"] != "transport_drift":
        raise ValueError("Blocked preflight has an invalid failure class.")


__all__ = ["validate_transport_evidence"]
