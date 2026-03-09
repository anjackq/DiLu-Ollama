from collections import Counter
from typing import Dict, List, Any

from .schema import extract_action_id, validate_canonical_row


def action_distribution(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        action = extract_action_id(row.get("output", ""))
        key = "unknown" if action is None else str(action)
        counts[key] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def duplicate_rate(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    seen = set()
    dup = 0
    for row in rows:
        key = (row.get("instruction", ""), row.get("input", ""), row.get("output", ""))
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    return dup / len(rows)


def profile_dataset_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    invalid_count = 0
    errors = Counter()
    for row in rows:
        ok, message = validate_canonical_row(row)
        if not ok:
            invalid_count += 1
            errors[message] += 1

    return {
        "rows": len(rows),
        "invalid_rows": invalid_count,
        "valid_rows": len(rows) - invalid_count,
        "invalid_breakdown": dict(errors),
        "action_distribution": action_distribution(rows),
        "duplicate_rate": duplicate_rate(rows),
    }

