import argparse
import math
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fine_tuning.pipeline import (
    action_distribution,
    duplicate_rate,
    extract_action_id,
    read_jsonl,
    save_json,
    validate_canonical_row,
    write_jsonl,
)

ACTIONS = ["0", "1", "2", "3", "4"]
BUCKETS = ["critical", "caution", "clear"]

DEFAULT_BUCKET_ACTION_MIX = {
    "critical": {"0": 0.05, "1": 0.25, "2": 0.10, "3": 0.05, "4": 0.55},
    "caution": {"0": 0.10, "1": 0.30, "2": 0.10, "3": 0.20, "4": 0.30},
    "clear": {"0": 0.12, "1": 0.34, "2": 0.12, "3": 0.34, "4": 0.08},
}

BUCKET_FALLBACK_PRIORITY = {
    "critical": ["4", "1", "2", "0", "3"],
    "caution": ["1", "4", "3", "0", "2"],
    "clear": ["1", "3", "2", "0", "4"],
}

EGO_LANE_POS_RE = re.compile(
    r"Your current position.*?lane position is\s*([-+]?\d+(?:\.\d+)?)\s*m",
    re.IGNORECASE | re.DOTALL,
)
SAME_LANE_AHEAD_RE = re.compile(
    r"is driving on the same lane as you and is ahead of you\..*?lane position is\s*([-+]?\d+(?:\.\d+)?)\s*m",
    re.IGNORECASE | re.DOTALL,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebalance canonical fine-tuning data with context-aware action balancing."
    )
    parser.add_argument(
        "--input",
        default="data/gold_standard_data_clean.jsonl",
        help="Input canonical dataset path.",
    )
    parser.add_argument(
        "--output",
        default="data/gold_standard_data_rebalanced.jsonl",
        help="Output rebalanced canonical dataset path.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional audit report path. Default: <output>_rebalance_report.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic rebalance sampling.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=0,
        help="Target output row count (0 -> keep input size).",
    )
    parser.add_argument(
        "--max-dup-ratio",
        type=float,
        default=3.0,
        help="Maximum duplication ratio per source row inside each bucket/action pool.",
    )
    parser.add_argument(
        "--critical-gap-m",
        type=float,
        default=20.0,
        help="Lead-gap threshold (meters) for critical bucket.",
    )
    parser.add_argument(
        "--caution-gap-m",
        type=float,
        default=35.0,
        help="Lead-gap threshold (meters) for caution bucket.",
    )
    return parser.parse_args()


def _default_report_path(output_path: str) -> str:
    root, _ext = os.path.splitext(output_path)
    return f"{root}_rebalance_report.json"


def _extract_min_same_lane_ahead_gap(input_text: str) -> float:
    if not isinstance(input_text, str):
        return float("inf")

    ego_match = EGO_LANE_POS_RE.search(input_text)
    if not ego_match:
        return float("inf")
    ego_lane_pos = float(ego_match.group(1))

    min_gap = float("inf")
    for match in SAME_LANE_AHEAD_RE.finditer(input_text):
        vehicle_lane_pos = float(match.group(1))
        gap = vehicle_lane_pos - ego_lane_pos
        if gap > 0:
            min_gap = min(min_gap, gap)
    return min_gap


def _bucket_from_gap(gap_m: float, critical_gap_m: float, caution_gap_m: float) -> str:
    if gap_m < critical_gap_m:
        return "critical"
    if gap_m <= caution_gap_m:
        return "caution"
    return "clear"


def _largest_remainder_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    if total <= 0:
        return {k: 0 for k in weights}
    weight_sum = sum(max(0.0, float(v)) for v in weights.values())
    if weight_sum <= 0:
        uniform = 1.0 / max(len(weights), 1)
        weights = {k: uniform for k in weights}
        weight_sum = 1.0

    raw = {k: (float(v) / weight_sum) * total for k, v in weights.items()}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(base.values())
    if remainder > 0:
        ranked = sorted(weights.keys(), key=lambda k: (raw[k] - base[k], raw[k]), reverse=True)
        for k in ranked[:remainder]:
            base[k] += 1
    return base


def _bucket_action_distribution(rows: List[Dict[str, Any]], bucket_of_row: Dict[int, str]) -> Dict[str, Dict[str, int]]:
    out = {bucket: {action: 0 for action in ACTIONS} for bucket in BUCKETS}
    for idx, row in enumerate(rows):
        bucket = bucket_of_row[idx]
        action = extract_action_id(row.get("output", ""))
        if action is None:
            continue
        out[bucket][str(action)] += 1
    return out


def _rebalance_bucket(
    rows: List[Dict[str, Any]],
    bucket_name: str,
    target_rows: int,
    rng: random.Random,
    max_dup_ratio: float,
) -> List[Dict[str, Any]]:
    if target_rows <= 0:
        return []
    if not rows:
        raise ValueError(f"Bucket '{bucket_name}' is empty but target_rows={target_rows}.")

    rows_by_action: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        action = extract_action_id(row.get("output", ""))
        if action is None:
            continue
        rows_by_action[str(action)].append(row)

    target_by_action = _largest_remainder_counts(target_rows, DEFAULT_BUCKET_ACTION_MIX[bucket_name])
    selected: List[Dict[str, Any]] = []
    selected_count = {action: 0 for action in ACTIONS}
    unresolved = 0

    for action in ACTIONS:
        need = target_by_action[action]
        pool = rows_by_action.get(action, [])
        if need <= 0:
            continue
        if not pool:
            unresolved += need
            continue

        max_allowed = max(1, int(math.floor(len(pool) * max_dup_ratio)))
        take = min(need, max_allowed)
        if take <= len(pool):
            picked = rng.sample(pool, k=take)
        else:
            picked = list(pool)
            picked.extend(rng.choices(pool, k=take - len(pool)))
        selected.extend(picked)
        selected_count[action] += take
        unresolved += (need - take)

    if unresolved > 0:
        for action in BUCKET_FALLBACK_PRIORITY[bucket_name]:
            if unresolved <= 0:
                break
            pool = rows_by_action.get(action, [])
            if not pool:
                continue
            max_allowed = max(1, int(math.floor(len(pool) * max_dup_ratio)))
            spare = max_allowed - selected_count[action]
            if spare <= 0:
                continue
            add = min(spare, unresolved)
            selected.extend(rng.choices(pool, k=add))
            selected_count[action] += add
            unresolved -= add

    if unresolved > 0:
        raise ValueError(
            f"Bucket '{bucket_name}' could not satisfy target_rows={target_rows} with max_dup_ratio={max_dup_ratio}. "
            f"Increase --max-dup-ratio or recollect more diverse data."
        )

    rng.shuffle(selected)
    return selected


def main() -> None:
    args = _parse_args()

    report_path = args.report.strip() or _default_report_path(args.output)
    rng = random.Random(int(args.seed))

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit("Input dataset is empty; nothing to rebalance.")

    invalid_rows = 0
    for row in rows:
        ok, _msg = validate_canonical_row(row)
        if not ok:
            invalid_rows += 1
    if invalid_rows > 0:
        raise SystemExit(f"Input dataset has {invalid_rows} invalid canonical rows. Run conversion/validation first.")

    before_action_dist = action_distribution(rows)
    missing_actions = [a for a in ACTIONS if int(before_action_dist.get(a, 0)) == 0]
    if missing_actions:
        report = {
            "success": False,
            "reason": "missing_action_classes",
            "missing_actions": missing_actions,
            "input_path": args.input,
            "output_path": args.output,
            "input_rows": len(rows),
            "action_distribution_before": before_action_dist,
        }
        save_json(report_path, report)
        print(f"[FAIL] Missing action classes in input: {missing_actions}")
        print(f"[INFO] Audit report saved: {report_path}")
        raise SystemExit(2)

    bucket_of_row: Dict[int, str] = {}
    rows_by_bucket: Dict[str, List[Dict[str, Any]]] = {bucket: [] for bucket in BUCKETS}
    for idx, row in enumerate(rows):
        gap_m = _extract_min_same_lane_ahead_gap(row.get("input", ""))
        bucket = _bucket_from_gap(
            gap_m=gap_m,
            critical_gap_m=float(args.critical_gap_m),
            caution_gap_m=float(args.caution_gap_m),
        )
        bucket_of_row[idx] = bucket
        rows_by_bucket[bucket].append(row)

    target_size = int(args.target_size) if int(args.target_size) > 0 else len(rows)
    nonempty_bucket_weights = {
        bucket: len(bucket_rows)
        for bucket, bucket_rows in rows_by_bucket.items()
        if len(bucket_rows) > 0
    }
    target_by_bucket_partial = _largest_remainder_counts(target_size, nonempty_bucket_weights)
    target_by_bucket = {bucket: int(target_by_bucket_partial.get(bucket, 0)) for bucket in BUCKETS}

    rebalanced_rows: List[Dict[str, Any]] = []
    for bucket in BUCKETS:
        bucket_rows = _rebalance_bucket(
            rows=rows_by_bucket[bucket],
            bucket_name=bucket,
            target_rows=target_by_bucket[bucket],
            rng=rng,
            max_dup_ratio=float(args.max_dup_ratio),
        )
        rebalanced_rows.extend(bucket_rows)

    rng.shuffle(rebalanced_rows)
    write_jsonl(args.output, rebalanced_rows)

    after_bucket_of_row = {}
    for idx, row in enumerate(rebalanced_rows):
        gap_m = _extract_min_same_lane_ahead_gap(row.get("input", ""))
        after_bucket_of_row[idx] = _bucket_from_gap(
            gap_m=gap_m,
            critical_gap_m=float(args.critical_gap_m),
            caution_gap_m=float(args.caution_gap_m),
        )

    report = {
        "success": True,
        "input_path": args.input,
        "output_path": args.output,
        "seed": int(args.seed),
        "target_size": target_size,
        "max_dup_ratio": float(args.max_dup_ratio),
        "bucket_thresholds_m": {
            "critical_lt": float(args.critical_gap_m),
            "caution_lte": float(args.caution_gap_m),
        },
        "input_rows": len(rows),
        "output_rows": len(rebalanced_rows),
        "action_distribution_before": action_distribution(rows),
        "action_distribution_after": action_distribution(rebalanced_rows),
        "bucket_action_distribution_before": _bucket_action_distribution(rows, bucket_of_row),
        "bucket_action_distribution_after": _bucket_action_distribution(rebalanced_rows, after_bucket_of_row),
        "duplicate_rate_before": duplicate_rate(rows),
        "duplicate_rate_after": duplicate_rate(rebalanced_rows),
        "target_rows_by_bucket": target_by_bucket,
        "input_rows_by_bucket": {bucket: len(bucket_rows) for bucket, bucket_rows in rows_by_bucket.items()},
    }
    save_json(report_path, report)

    print(f"[OK] Rebalanced dataset saved: {args.output}")
    print(f"[OK] Rebalance report saved: {report_path}")
    print(f"[INFO] Action distribution before: {report['action_distribution_before']}")
    print(f"[INFO] Action distribution after:  {report['action_distribution_after']}")


if __name__ == "__main__":
    main()
