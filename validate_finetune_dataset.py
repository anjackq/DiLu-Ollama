import argparse

from fine_tuning.pipeline import profile_dataset_rows, read_jsonl, validate_canonical_row


def validate_dataset(path: str, max_examples: int = 5) -> int:
    rows = read_jsonl(path)
    profile = profile_dataset_rows(rows)
    examples = []

    for idx, row in enumerate(rows, start=1):
        ok, _ = validate_canonical_row(row)
        if ok:
            examples.append(
                {
                    "line": idx,
                    "output": row["output"].strip().splitlines()[-1],
                }
            )
        if len(examples) >= max_examples:
            break

    print(f"Rows checked: {profile['rows']}")
    print(f"Invalid rows: {profile['invalid_rows']}")
    print(f"Valid rows: {profile['valid_rows']}")
    print(f"Action distribution: {profile['action_distribution']}")
    print(f"Duplicate rate: {profile['duplicate_rate']:.4f}")
    if profile["invalid_breakdown"]:
        print(f"Invalid breakdown: {profile['invalid_breakdown']}")

    if examples:
        print("Sample valid outputs:")
        for ex in examples:
            print(f"- line {ex['line']} | {ex['output']}")

    return 1 if profile["invalid_rows"] else 0


def main():
    parser = argparse.ArgumentParser(description="Validate fine-tuning dataset schema and output format.")
    parser.add_argument("path", nargs="?", default="data/gold_standard_data_clean.jsonl")
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(validate_dataset(args.path, args.max_examples))


if __name__ == "__main__":
    main()

