import re
from typing import Dict, Any, Tuple, Optional


STRICT_ACTION_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*([0-4])\s*$", re.IGNORECASE)


def extract_action_id(output_text: str) -> Optional[int]:
    if not isinstance(output_text, str):
        return None
    match = STRICT_ACTION_PATTERN.search(output_text.strip())
    if not match:
        return None
    return int(match.group(1))


def validate_canonical_row(row: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(row, dict):
        return False, "row must be a JSON object"

    for key in ("instruction", "input", "output"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            return False, f"missing/empty required field: {key}"

    output = row["output"].strip()
    if "Reasoning:" not in output:
        return False, "output missing 'Reasoning:'"
    if extract_action_id(output) is None:
        return False, "output missing final 'Response to user:#### <0-4>'"

    return True, "ok"

