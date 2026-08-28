"""Synthetic V8 + frozen-comparator fixtures for the grounded-decoding analysis tests.

Not a ``test_*.py`` module itself, so ``unittest discover`` does not collect
it directly. Builds a fully valid 480-row V8 schedule (300 Stage 1 + 180
Stage 2) plus matching frozen O1 comparator rows, with every score a
controlled constant offset from a fixed frozen baseline so contrast tests
can assert exact paired-mean equality rather than approximate values.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from dilu.runtime._minimal_factorial_manifest import CASE_FINGERPRINT
from dilu.runtime.dilu_scoring import (
    BALANCED_DRIVING_SCORE_POLICY_VERSION,
    SPLIT_SCORING_POLICY_VERSION,
)

MODEL_SLOTS = ("qwen_06b", "llama_1b", "llama_3b", "gemma_4b", "qwen_8b")
SMALL_MODEL_SLOTS = ("qwen_06b", "llama_1b")
MODEL_TAGS = {
    "qwen_06b": "qwen3:0.6b",
    "llama_1b": "llama3.2:1b",
    "llama_3b": "llama3.2:3b",
    "gemma_4b": "gemma3:4b",
    "qwen_8b": "qwen3:8b",
}
MODEL_DIGESTS = {slot: f"sha256:{'0' * 63}{index}" for index, slot in enumerate(MODEL_SLOTS)}
BASE_SCORE = 0.5
V8_CAMPAIGN_ID = "iclr2027-grounded-decoding-v8"
V5_CAMPAIGN_ID = "iclr2027-minimal-factorial-v5"
V7_CAMPAIGN_ID = "iclr2027-model-breadth-factorial-v7"

_CATEGORIES = tuple(f"category-{index:02d}" for index in range(10))


def _stage1_case_ids() -> list[str]:
    return [f"case1_{category}_{i}" for category in _CATEGORIES for i in range(3)]


def _stage2_case_ids() -> list[str]:
    return [f"case2_{category}_{i}" for category in _CATEGORIES for i in range(9)]


def _category_of(case_id: str) -> str:
    return case_id.split("_")[1]


def _seed_of(case_id: str) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    return 1_000 + int.from_bytes(digest[:4], "big") % 100_000


def _episode_row(
    *,
    campaign_id: str,
    stage: str,
    model_slot: str,
    condition_id: str,
    case_id: str,
    score: float,
    action_unavailable: int = 0,
) -> dict[str, Any]:
    seed = _seed_of(case_id)
    return {
        "campaign_id": campaign_id,
        "stage": stage,
        "model_slot": model_slot,
        "model_tag": MODEL_TAGS[model_slot],
        "model_digest": MODEL_DIGESTS[model_slot],
        "condition_id": condition_id,
        "case_id": case_id,
        "category": _category_of(case_id),
        "simulator_seed": seed,
        "episode_attempt_id": f"episode-{campaign_id}-{model_slot}-{condition_id}-{case_id}",
        "benchmark_fingerprint": CASE_FINGERPRINT,
        "split_scoring_policy_version": SPLIT_SCORING_POLICY_VERSION,
        "balanced_driving_score_policy_version": BALANCED_DRIVING_SCORE_POLICY_VERSION,
        "driving_score_balanced_v1": score,
        "task_completed": True,
        "crashed": False,
        "fallback_action_rate": 0.1,
        "shield_intervention_rate": 0.05,
        "decision_latency_ms_avg": 200.0,
        "analysis_action_unavailable_count": action_unavailable,
    }


def build_v8_fixture(
    *,
    family_a_offsets: Mapping[str, float] | None = None,
    family_b_offsets: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the 480 V8 rows plus their frozen O1 comparator rows.

    ``family_a_offsets``/``family_b_offsets`` map ``model_slot`` to the
    constant ``driving_score_balanced_v1`` V8 adds over the frozen
    ``BASE_SCORE`` on c121/c120 respectively (default 0.0, i.e. no effect).
    Because every frozen row scores exactly ``BASE_SCORE``, the paired
    Family A/B/C/D contrasts equal these offsets (and their difference)
    exactly, with zero case-to-case variance.
    """
    a_offsets = dict(family_a_offsets or {})
    b_offsets = dict(family_b_offsets or {})
    stage1_cases = _stage1_case_ids()
    stage2_cases = _stage2_case_ids()

    v8_rows: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    for model in MODEL_SLOTS:
        delta_a = a_offsets.get(model, 0.0)
        delta_b = b_offsets.get(model, 0.0)
        for case_id in stage1_cases:
            v8_rows.append(
                _episode_row(
                    campaign_id=V8_CAMPAIGN_ID,
                    stage="stage1",
                    model_slot=model,
                    condition_id="c121",
                    case_id=case_id,
                    score=BASE_SCORE + delta_a,
                )
            )
            v8_rows.append(
                _episode_row(
                    campaign_id=V8_CAMPAIGN_ID,
                    stage="stage1",
                    model_slot=model,
                    condition_id="c120",
                    case_id=case_id,
                    score=BASE_SCORE + delta_b,
                )
            )
            frozen_campaign = V5_CAMPAIGN_ID if model in SMALL_MODEL_SLOTS else V7_CAMPAIGN_ID
            for condition_id in ("c110", "c111"):
                frozen_rows.append(
                    _episode_row(
                        campaign_id=frozen_campaign,
                        stage="stage1",
                        model_slot=model,
                        condition_id=condition_id,
                        case_id=case_id,
                        score=BASE_SCORE,
                    )
                )
        if model in SMALL_MODEL_SLOTS:
            for case_id in stage2_cases:
                v8_rows.append(
                    _episode_row(
                        campaign_id=V8_CAMPAIGN_ID,
                        stage="stage2_additional",
                        model_slot=model,
                        condition_id="c121",
                        case_id=case_id,
                        score=BASE_SCORE + delta_a,
                    )
                )
                frozen_rows.append(
                    _episode_row(
                        campaign_id=V5_CAMPAIGN_ID,
                        stage="stage2_additional",
                        model_slot=model,
                        condition_id="c111",
                        case_id=case_id,
                        score=BASE_SCORE,
                    )
                )
    return v8_rows, frozen_rows


def duplicate_a_row(v8_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mutated = copy.deepcopy(v8_rows)
    mutated.append(copy.deepcopy(mutated[0]))
    return mutated


def drop_a_row(v8_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(v8_rows[:-1])


def mismatch_a_digest(v8_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mutated = copy.deepcopy(v8_rows)
    mutated[0]["model_digest"] = "sha256:" + "f" * 64
    return mutated


def drift_a_fingerprint(frozen_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mutated = copy.deepcopy(frozen_rows)
    mutated[0]["benchmark_fingerprint"] = "sha256:" + "e" * 64
    return mutated


def trigger_family_m(v8_rows: list[dict[str, Any]], model_slot: str) -> list[dict[str, Any]]:
    mutated = copy.deepcopy(v8_rows)
    for row in mutated:
        if row["model_slot"] == model_slot and row["condition_id"] == "c121":
            row["analysis_action_unavailable_count"] = 1
            break
    return mutated


__all__ = [
    "BASE_SCORE",
    "MODEL_DIGESTS",
    "MODEL_SLOTS",
    "MODEL_TAGS",
    "SMALL_MODEL_SLOTS",
    "V5_CAMPAIGN_ID",
    "V7_CAMPAIGN_ID",
    "V8_CAMPAIGN_ID",
    "build_v8_fixture",
    "drift_a_fingerprint",
    "drop_a_row",
    "duplicate_a_row",
    "mismatch_a_digest",
    "trigger_family_m",
]
