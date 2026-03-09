import argparse
import json
import re
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fine_tuning.pipeline import validate_canonical_row, read_jsonl, write_jsonl


NEW_INSTRUCTION = """You are an autonomous driving decision engine.
Analyze the scenario and output the single best Action_id integer (0-4).

Strict Output Format:
Reasoning: <one sentence>
Response to user:#### <integer>"""

ACTION_PATTERN = re.compile(r"(?:Action_id\s*:\s*|Response to user:\s*####\s*)([0-4])", re.IGNORECASE)
REASONING_PATTERN = re.compile(r"Reasoning\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_output_payload(raw_output):
    reasoning = ""
    action_id = None

    if isinstance(raw_output, str):
        try:
            output_data = json.loads(raw_output)
            reasoning = str(output_data.get("reasoning", "")).strip()
            action_id = output_data.get("action_id")
            return reasoning, action_id
        except json.JSONDecodeError:
            reasoning_match = REASONING_PATTERN.search(raw_output)
            action_match = ACTION_PATTERN.search(raw_output)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                reasoning = re.split(r"\n\s*(?:Action_id|Response to user)\s*:", reasoning, maxsplit=1)[0].strip()
            if action_match:
                action_id = action_match.group(1)
            return reasoning, action_id

    if isinstance(raw_output, dict):
        reasoning = str(raw_output.get("reasoning", "")).strip()
        action_id = raw_output.get("action_id")
        return reasoning, action_id

    return reasoning, action_id


def convert_rows(rows):
    converted = []
    skipped = 0

    for row in rows:
        original_input = row.get("input", "")
        if "Decision:" not in original_input:
            original_input += "\nDecision:"

        raw_output = row.get("output", row.get("checkpoints", ""))
        reasoning, action_id = _parse_output_payload(raw_output)
        try:
            action_id = int(action_id)
        except (ValueError, TypeError):
            skipped += 1
            continue
        if action_id < 0 or action_id > 4:
            skipped += 1
            continue

        reasoning = reasoning or "Scenario analyzed and best action selected."
        candidate = {
            "instruction": NEW_INSTRUCTION,
            "input": original_input,
            "output": f"Reasoning: {reasoning}\nResponse to user:#### {action_id}",
        }
        ok, _ = validate_canonical_row(candidate)
        if not ok:
            skipped += 1
            continue
        converted.append(candidate)

    return converted, skipped


def parse_args():
    parser = argparse.ArgumentParser(description="Convert collected expert dataset into strict fine-tuning format.")
    parser.add_argument("--input", default="data/gold_standard_data.jsonl", help="Source JSONL path.")
    parser.add_argument("--output", default="data/gold_standard_data_clean.jsonl", help="Converted JSONL path.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(args.input)
    converted, skipped = convert_rows(rows)
    write_jsonl(args.output, converted)

    print(f"Done. Converted {len(converted)} rows, skipped {skipped} invalid rows.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
