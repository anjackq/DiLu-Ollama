"""Cross-campaign comparator contract for the ICLR 2027 grounded-decoding V8 run.

Pairs every V8 grounded-decoding (P1 O2) row against the matching frozen
"O1 in place of O2" row already recorded in the frozen V5 or V7
``episodes.jsonl`` (the same model, case, and simulator seed, on the O1
output-enforcement level). All four registered failure modes fail closed
with a dedicated exception type: a missing frozen row, a live/frozen model
digest mismatch, case-set fingerprint drift, and scoring-policy-version
drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ._harness_config_support import OutputEnforcement
from ._minimal_factorial_manifest import CASE_FINGERPRINT, ExperimentManifest, case_fingerprint
from ._grounded_decoding_manifest_support import GroundedDecodingManifest
from .dilu_scoring import BALANCED_DRIVING_SCORE_POLICY_VERSION, SPLIT_SCORING_POLICY_VERSION
from .minimal_factorial_schedule import ScheduledEpisode

_REQUIRED_FIELDS = (
    "campaign_id",
    "model_tag",
    "model_digest",
    "condition_id",
    "case_id",
    "simulator_seed",
    "benchmark_fingerprint",
    "split_scoring_policy_version",
    "balanced_driving_score_policy_version",
)


class ComparatorContractError(ValueError):
    """Base type for a comparator contract that cannot be trusted."""


class MissingComparatorRowError(ComparatorContractError):
    """No frozen V5/V7 row exists for a required (model, condition, case, seed)."""


class ComparatorDigestMismatchError(ComparatorContractError):
    """The live V8 model digest does not match the frozen comparator digest."""


class CaseSetFingerprintDriftError(ComparatorContractError):
    """A frozen comparator row's benchmark fingerprint no longer matches V8's case set."""


class ScoringVersionDriftError(ComparatorContractError):
    """A frozen comparator row's scoring policy version has drifted."""


@dataclass(frozen=True)
class ComparatorRow:
    campaign_id: str
    model_tag: str
    model_digest: str
    condition_id: str
    case_id: str
    simulator_seed: int
    benchmark_fingerprint: str
    split_scoring_policy_version: str
    balanced_driving_score_policy_version: str


@dataclass(frozen=True, init=False)
class ComparatorContract:
    """The frozen comparator row index, keyed by the full and natural keys.

    ``index`` is keyed by ``(model_tag, model_digest, condition_id, case_id,
    simulator_seed)`` as specified by the task brief. ``resolve`` additionally
    looks the row up by the *natural* key (without digest) so that a digest
    drift is reported as :class:`ComparatorDigestMismatchError` rather than
    an indistinguishable :class:`MissingComparatorRowError`.
    """

    index: Mapping[tuple[str, str, str, str, int], ComparatorRow]
    _by_natural_key: Mapping[tuple[str, str, str, int], ComparatorRow]

    def __init__(self, rows: Sequence[ComparatorRow]) -> None:
        full_index: dict[tuple[str, str, str, str, int], ComparatorRow] = {}
        natural_index: dict[tuple[str, str, str, int], ComparatorRow] = {}
        for row in rows:
            full_key = (
                row.model_tag,
                row.model_digest,
                row.condition_id,
                row.case_id,
                row.simulator_seed,
            )
            natural_key = (row.model_tag, row.condition_id, row.case_id, row.simulator_seed)
            if full_key in full_index:
                raise ComparatorContractError(f"Duplicate frozen comparator row: {full_key}.")
            if natural_key in natural_index:
                raise ComparatorContractError(
                    f"Ambiguous frozen comparator row (digest conflict): {natural_key}."
                )
            full_index[full_key] = row
            natural_index[natural_key] = row
        object.__setattr__(self, "index", MappingProxyType(full_index))
        object.__setattr__(self, "_by_natural_key", MappingProxyType(natural_index))

    def resolve(
        self,
        *,
        model_tag: str,
        model_digest: str,
        condition_id: str,
        case_id: str,
        simulator_seed: int,
    ) -> ComparatorRow:
        natural_key = (model_tag, condition_id, case_id, simulator_seed)
        row = self._by_natural_key.get(natural_key)
        if row is None:
            raise MissingComparatorRowError(
                f"No frozen comparator row for model_tag={model_tag!r} "
                f"condition_id={condition_id!r} case_id={case_id!r} "
                f"simulator_seed={simulator_seed!r}."
            )
        if row.model_digest != model_digest:
            raise ComparatorDigestMismatchError(
                f"Live digest {model_digest!r} for {model_tag!r} does not match the "
                f"frozen comparator digest {row.model_digest!r} "
                f"(condition_id={condition_id!r}, case_id={case_id!r}); register this "
                "model slot in rerun_comparators_for instead of cross-campaign pairing."
            )
        return row


def pair_v8_row(contract: ComparatorContract, row: ScheduledEpisode) -> ComparatorRow:
    """Resolve a V8 grounded-decoding (O2) row to its frozen O1 comparator row."""
    spec = row.condition.condition
    if spec.output_enforcement is not OutputEnforcement.BACKEND_SCHEMA_GROUNDED:
        raise ValueError("pair_v8_row expects a grounded-decoding (P1 O2) V8 row.")
    paired_condition_id = replace(
        spec, output_enforcement=OutputEnforcement.BACKEND_SCHEMA
    ).condition_id()
    return contract.resolve(
        model_tag=row.model_tag,
        model_digest=row.model_digest,
        condition_id=paired_condition_id,
        case_id=row.case_id,
        simulator_seed=row.simulator_seed,
    )


def build_comparator_contract(
    manifest: GroundedDecodingManifest,
    case_set: Mapping[str, Any],
    v5_manifest: ExperimentManifest,
    v7_manifest: ExperimentManifest,
) -> ComparatorContract:
    """Load and validate the frozen V5+V7 comparator rows V8 will pair against."""
    fingerprint = case_fingerprint(case_set)
    if fingerprint != CASE_FINGERPRINT:
        raise CaseSetFingerprintDriftError("V8 case set fingerprint drifted from the frozen value.")
    root = manifest.repo_root()
    rows: list[ComparatorRow] = []
    for source_manifest, episodes_relpath in (
        (v5_manifest, manifest.comparators.v5_episodes),
        (v7_manifest, manifest.comparators.v7_episodes),
    ):
        rows.extend(
            _load_comparator_rows(
                root / episodes_relpath,
                expected_campaign_id=source_manifest.campaign_id,
                expected_fingerprint=fingerprint,
            )
        )
    return ComparatorContract(rows)


def _load_comparator_rows(
    path: Path, *, expected_campaign_id: str, expected_fingerprint: str
) -> list[ComparatorRow]:
    rows: list[ComparatorRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping) or any(
                field not in record for field in _REQUIRED_FIELDS
            ):
                raise ComparatorContractError(
                    f"{path}:{line_number} is missing required comparator fields."
                )
            if record["campaign_id"] != expected_campaign_id:
                raise ComparatorContractError(
                    f"{path}:{line_number} campaign_id does not match the trusted "
                    "frozen manifest."
                )
            if record["benchmark_fingerprint"] != expected_fingerprint:
                raise CaseSetFingerprintDriftError(
                    f"{path}:{line_number} benchmark_fingerprint drifted from the "
                    "current V8 case set."
                )
            if (
                record["split_scoring_policy_version"] != SPLIT_SCORING_POLICY_VERSION
                or record["balanced_driving_score_policy_version"]
                != BALANCED_DRIVING_SCORE_POLICY_VERSION
            ):
                raise ScoringVersionDriftError(
                    f"{path}:{line_number} scoring policy version drifted from the "
                    "registered version."
                )
            rows.append(
                ComparatorRow(
                    campaign_id=record["campaign_id"],
                    model_tag=record["model_tag"],
                    model_digest=record["model_digest"],
                    condition_id=record["condition_id"],
                    case_id=record["case_id"],
                    simulator_seed=record["simulator_seed"],
                    benchmark_fingerprint=record["benchmark_fingerprint"],
                    split_scoring_policy_version=record["split_scoring_policy_version"],
                    balanced_driving_score_policy_version=record[
                        "balanced_driving_score_policy_version"
                    ],
                )
            )
    return rows


__all__ = [
    "CaseSetFingerprintDriftError",
    "ComparatorContract",
    "ComparatorContractError",
    "ComparatorDigestMismatchError",
    "ComparatorRow",
    "MissingComparatorRowError",
    "ScoringVersionDriftError",
    "build_comparator_contract",
    "pair_v8_row",
]
