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
    parser.add_argument("--config", default="config.yaml", help="Runtime config path for collect/RL stages.")
    parser.add_argument("--collect", action="store_true", help="Run data collection stage.")
    parser.add_argument("--train-rl", action="store_true", help="Run RL training stage (DQN repo profile).")
    parser.add_argument("--convert", action="store_true", help="Run data conversion stage.")
    parser.add_argument("--rebalance", action="store_true", help="Run dataset rebalance stage.")
    parser.add_argument("--validate", action="store_true", help="Run dataset validation stage.")
    parser.add_argument("--train", action="store_true", help="Run training stage.")
    parser.add_argument("--gguf", action="store_true", help="Run GGUF conversion stage after training.")
    parser.add_argument("--all", action="store_true", help="Run all stages.")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes for collect stage.")
    parser.add_argument(
        "--collect-simulation-duration",
        type=int,
        default=300,
        help="Collection-only simulation horizon per episode.",
    )
    parser.add_argument(
        "--collect-vehicle-count",
        type=int,
        default=40,
        help="Collection-only number of vehicles in scenario observations.",
    )
    parser.add_argument(
        "--collect-vehicles-density",
        type=float,
        default=3.5,
        help="Collection-only traffic density.",
    )
    parser.add_argument(
        "--collect-controller",
        default="rule",
        choices=["rule", "rl"],
        help="Controller used for collection actions.",
    )
    parser.add_argument(
        "--rl-model-path",
        default="",
        help="Path to trained SB3 model zip for RL-based collection.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Max collection attempts (0 -> episodes * 20).",
    )
    parser.add_argument(
        "--no-collect-curriculum",
        action="store_true",
        help="Disable staged collection curriculum.",
    )
    parser.add_argument("--rl-timesteps", type=int, default=1_000_000, help="RL training timesteps.")
    parser.add_argument("--rl-save-path", default="fine_tuning/rl_models/highway_dqn", help="RL model save path.")
    parser.add_argument("--rl-seed", type=int, default=42, help="RL training seed.")
    parser.add_argument("--rl-eval-episodes", type=int, default=20, help="RL deterministic eval episodes.")
    parser.add_argument("--rl-device", default="auto", help="RL device: auto|cpu|cuda|cuda:0")
    parser.add_argument("--rl-train-profile", default="dqn_repo", choices=["dqn_repo"], help="RL training profile.")
    parser.add_argument("--rl-env-id", default="highway-fast-v0", help="RL training env id.")
    parser.add_argument("--rl-n-envs", type=int, default=6, help="Deprecated (ignored in dqn_repo).")
    parser.add_argument("--rl-vec-env", default="subproc", choices=["subproc", "dummy"], help="Deprecated (ignored in dqn_repo).")
    parser.add_argument("--rl-batch-size", type=int, default=32, help="RL DQN batch size.")
    parser.add_argument("--rl-n-epochs", type=int, default=10, help="Deprecated (ignored in dqn_repo).")
    parser.add_argument("--rl-learning-rate", type=float, default=5e-4, help="RL DQN learning rate.")
    parser.add_argument("--rl-gamma", type=float, default=0.8, help="RL DQN gamma.")
    parser.add_argument("--rl-tensorboard-log", default="highway_ppo/", help="RL TensorBoard log dir.")
    parser.add_argument("--reward-env-weight", type=float, default=0.4, help="Deprecated in Safety-First Reward V2 (kept for compatibility).")
    parser.add_argument("--reward-speed-scale", type=float, default=0.8, help="Deprecated in Safety-First Reward V2 (kept for compatibility).")
    parser.add_argument("--reward-progress-scale", type=float, default=0.2, help="Deprecated in Safety-First Reward V2 (kept for compatibility).")
    parser.add_argument("--reward-crash-penalty", type=float, default=-2.5, help="Deprecated in Safety-First Reward V2 (kept for compatibility).")
    parser.add_argument("--reward-risk-brake-bonus", type=float, default=0.15, help="Deprecated in Safety-First Reward V2 (kept for compatibility).")
    parser.add_argument("--reward-early-brake-penalty", type=float, default=-0.35, help="Deprecated in Safety-First Reward V2 (kept for compatibility).")
    parser.add_argument("--reward-unjustified-brake-penalty", type=float, default=-0.25, help="Deprecated in Safety-First Reward V2 (kept for compatibility).")
    parser.add_argument("--risk-front-gap-m", type=float, default=22.0, help="RL reward shaping risk gap threshold.")
    parser.add_argument("--risk-ttc-sec", type=float, default=3.0, help="RL reward shaping risk TTC threshold.")
    parser.add_argument("--early-brake-steps", type=int, default=5, help="RL reward shaping early-step window.")
    parser.add_argument("--rl-max-early-brake-ratio", type=float, default=0.6, help="RL collection gate max early brake ratio.")
    parser.add_argument("--rl-min-avg-speed", type=float, default=18.0, help="RL collection gate min average speed.")
    parser.add_argument("--rl-min-progress", type=float, default=40.0, help="RL collection gate min progress floor.")
    parser.add_argument("--no-rl-quality-gate", action="store_true", help="Disable RL quality gate in collection.")
    parser.add_argument("--raw-output", default="data/gold_standard_data.jsonl", help="Raw JSONL output.")
    parser.add_argument("--clean-output", default="data/gold_standard_data_clean.jsonl", help="Clean JSONL output.")
    parser.add_argument(
        "--rebalanced-output",
        default="data/gold_standard_data_rebalanced.jsonl",
        help="Rebalanced JSONL output.",
    )
    parser.add_argument("--rebalance-seed", type=int, default=42, help="Rebalance sampling seed.")
    parser.add_argument(
        "--rebalance-target-size",
        type=int,
        default=0,
        help="Rebalance target output row count (0 -> keep input row count).",
    )
    parser.add_argument("--model-name", default="unsloth/Llama-3.1-8B-Instruct", help="Training base model.")
    parser.add_argument("--model-family", default="auto", choices=["auto", "llama3", "qwen", "mistral", "deepseek", "phi"], help="Model family preset for training.")
    parser.add_argument("--merged-model-dir", default="fine_tuning/merged_models/dilu-llama3_1-8b-v1", help="Merged model output.")
    parser.add_argument("--llama-cpp-dir", default="", help="Optional llama.cpp dir for GGUF conversion (auto-detected if empty).")
    parser.add_argument("--gguf-output-dir", default="fine_tuning/gguf", help="Output directory for GGUF artifacts.")
    parser.add_argument("--gguf-name", default="", help="Optional output model name for GGUF files.")
    parser.add_argument("--gguf-outtype", default="f16", choices=["f16", "bf16", "f32", "q8_0", "auto"], help="GGUF outtype.")
    parser.add_argument("--gguf-quantize", default="", help="Optional quantization type, e.g. Q4_K_M.")
    parser.add_argument("--gguf-create-ollama", action="store_true", help="Run ollama create after GGUF build.")
    parser.add_argument("--ollama-model", default="", help="Ollama model name for --gguf-create-ollama.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_all = args.all or not any([args.collect, args.convert, args.rebalance, args.validate, args.train, args.gguf])
    rebalance_enabled = args.rebalance or args.all
    training_data_file = args.clean_output

    if args.train_rl:
        _run(
            [
                args.python,
                os.path.join(ROOT_DIR, "fine_tuning/train_highway_rl.py"),
                "--config",
                args.config,
                "--timesteps",
                str(args.rl_timesteps),
                "--save-path",
                args.rl_save_path,
                "--seed",
                str(args.rl_seed),
                "--eval-episodes",
                str(args.rl_eval_episodes),
                "--device",
                str(args.rl_device),
                "--rl-train-profile",
                str(args.rl_train_profile),
                "--rl-env-id",
                str(args.rl_env_id),
                "--rl-n-envs",
                str(args.rl_n_envs),
                "--rl-vec-env",
                str(args.rl_vec_env),
                "--rl-batch-size",
                str(args.rl_batch_size),
                "--rl-n-epochs",
                str(args.rl_n_epochs),
                "--rl-learning-rate",
                str(args.rl_learning_rate),
                "--rl-gamma",
                str(args.rl_gamma),
                "--rl-tensorboard-log",
                str(args.rl_tensorboard_log),
                "--reward-env-weight",
                str(args.reward_env_weight),
                "--reward-speed-scale",
                str(args.reward_speed_scale),
                "--reward-progress-scale",
                str(args.reward_progress_scale),
                "--reward-crash-penalty",
                str(args.reward_crash_penalty),
                "--reward-risk-brake-bonus",
                str(args.reward_risk_brake_bonus),
                "--reward-early-brake-penalty",
                str(args.reward_early_brake_penalty),
                "--reward-unjustified-brake-penalty",
                str(args.reward_unjustified_brake_penalty),
                "--risk-front-gap-m",
                str(args.risk_front_gap_m),
                "--risk-ttc-sec",
                str(args.risk_ttc_sec),
                "--early-brake-steps",
                str(args.early_brake_steps),
            ]
        )

    if args.collect or run_all:
        collect_cmd = [
            args.python,
            os.path.join(ROOT_DIR, "fine_tuning/collect_data.py"),
            "--config",
            args.config,
            "--episodes",
            str(args.episodes),
            "--output",
            args.raw_output,
            "--collect-simulation-duration",
            str(args.collect_simulation_duration),
            "--collect-vehicle-count",
            str(args.collect_vehicle_count),
            "--collect-vehicles-density",
            str(args.collect_vehicles_density),
            "--collect-controller",
            args.collect_controller,
            "--max-attempts",
            str(args.max_attempts),
            "--rl-max-early-brake-ratio",
            str(args.rl_max_early_brake_ratio),
            "--rl-min-avg-speed",
            str(args.rl_min_avg_speed),
            "--rl-min-progress",
            str(args.rl_min_progress),
        ]
        if args.rl_model_path.strip():
            collect_cmd.extend(["--rl-model-path", args.rl_model_path])
        if args.no_collect_curriculum:
            collect_cmd.append("--no-collect-curriculum")
        if args.no_rl_quality_gate:
            collect_cmd.append("--no-rl-quality-gate")
        _run(collect_cmd)

    if args.convert or run_all:
        _run(
            [
                args.python,
                os.path.join(ROOT_DIR, "fine_tuning/convert_data.py"),
                "--input",
                args.raw_output,
                "--output",
                args.clean_output,
            ]
        )

    if rebalance_enabled:
        rebalance_cmd = [
            args.python,
            os.path.join(ROOT_DIR, "fine_tuning/rebalance_data.py"),
            "--input",
            args.clean_output,
            "--output",
            args.rebalanced_output,
            "--seed",
            str(args.rebalance_seed),
        ]
        if int(args.rebalance_target_size) > 0:
            rebalance_cmd.extend(["--target-size", str(args.rebalance_target_size)])
        _run(rebalance_cmd)
        training_data_file = args.rebalanced_output

    if args.validate or run_all:
        _run([args.python, os.path.join(ROOT_DIR, "fine_tuning/validate_finetune_dataset.py"), training_data_file])

    if args.train or run_all:
        _run(
            [
                args.python,
                os.path.join(ROOT_DIR, "fine_tuning/train_dilu_ollama.py"),
                "--data-file",
                training_data_file,
                "--model-name",
                args.model_name,
                "--model-family",
                args.model_family,
                "--merged-model-dir",
                args.merged_model_dir,
            ]
        )

    if args.gguf:
        gguf_cmd = [
            args.python,
            os.path.join(ROOT_DIR, "fine_tuning/build_gguf.py"),
            "--hf-model-dir",
            args.merged_model_dir,
            "--output-dir",
            args.gguf_output_dir,
            "--outtype",
            args.gguf_outtype,
        ]
        if args.llama_cpp_dir.strip():
            gguf_cmd.extend(["--llama-cpp-dir", args.llama_cpp_dir])
        if args.gguf_name.strip():
            gguf_cmd.extend(["--name", args.gguf_name])
        if args.gguf_quantize.strip():
            gguf_cmd.extend(["--quantize", args.gguf_quantize])
        if args.gguf_create_ollama:
            gguf_cmd.append("--create-ollama")
            if args.ollama_model.strip():
                gguf_cmd.extend(["--ollama-model", args.ollama_model])

        _run(gguf_cmd)


if __name__ == "__main__":
    main()
