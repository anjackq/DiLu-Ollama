import argparse
import subprocess
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end fine-tuning pipeline runner.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--collect", action="store_true", help="Run data collection stage.")
    parser.add_argument("--convert", action="store_true", help="Run data conversion stage.")
    parser.add_argument("--validate", action="store_true", help="Run dataset validation stage.")
    parser.add_argument("--train", action="store_true", help="Run training stage.")
    parser.add_argument("--all", action="store_true", help="Run all stages.")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes for collect stage.")
    parser.add_argument("--raw-output", default="data/gold_standard_data.jsonl", help="Raw JSONL output.")
    parser.add_argument("--clean-output", default="data/gold_standard_data_clean.jsonl", help="Clean JSONL output.")
    parser.add_argument("--model-name", default="unsloth/Llama-3.1-8B-Instruct", help="Training base model.")
    parser.add_argument("--merged-model-dir", default="fine_tuning/merged_models/dilu-llama3_1-8b-v1", help="Merged model output.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_all = args.all or not any([args.collect, args.convert, args.validate, args.train])

    if args.collect or run_all:
        _run(
            [
                args.python,
                os.path.join(ROOT_DIR, "collect_data.py"),
                "--episodes",
                str(args.episodes),
                "--output",
                args.raw_output,
            ]
        )

    if args.convert or run_all:
        _run(
            [
                args.python,
                os.path.join(ROOT_DIR, "data/convert_data.py"),
                "--input",
                args.raw_output,
                "--output",
                args.clean_output,
            ]
        )

    if args.validate or run_all:
        _run([args.python, os.path.join(ROOT_DIR, "validate_finetune_dataset.py"), args.clean_output])

    if args.train or run_all:
        _run(
            [
                args.python,
                os.path.join(ROOT_DIR, "fine_tuning/train_dilu_updated.py"),
                "--data-file",
                args.clean_output,
                "--model-name",
                args.model_name,
                "--merged-model-dir",
                args.merged_model_dir,
            ]
        )


if __name__ == "__main__":
    main()
