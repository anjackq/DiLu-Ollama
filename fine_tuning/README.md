# Fine-Tuning Guide

This folder contains the fine-tuning pipeline for DiLu-Ollama models.

## Pipeline Overview

1. Collect expert-labeled driving data.
2. Convert to strict training format.
3. Rebalance action distribution with context-aware buckets.
4. Validate dataset schema and action format.
5. Fine-tune and export merged model weights.
6. (Recommended for local serving) Convert merged model to GGUF and create Ollama model from GGUF.

## Scripts

- `collect_data.py`: collect rule-based expert trajectories.
- `convert_data.py`: convert raw JSONL to strict `Reasoning + Response to user:#### <id>` format.
- `rebalance_data.py`: context-aware dataset rebalance for action distribution.
- `validate_finetune_dataset.py`: validate row schema and output format.
- `train_dilu_ollama.py`: main training/export script.
- `train_highway_rl.py`: optional PPO trainer for HighwayEnv ego policy.
- `evaluate_rl_policy.py`: evaluate trained PPO policy and capture per-episode videos.
- `run_pipeline.py`: orchestrates full pipeline.
- `build_gguf.py`: convert merged HF export to GGUF (+ optional quantization and Ollama create).
- `pipeline/`: shared helpers for config/io/schema/policy/quality/training.
- `modelfiles/`: Ollama Modelfile templates.

## Quick Start

### 1) Collect data

Collection now shows nested progress in interactive terminals:
- outer progress: successful episodes collected
- inner progress: steps in the current collection attempt
- default curriculum: A(120/20/1.8) -> B(200/30/2.6) -> C(300/40/3.5)

```bash
python fine_tuning/collect_data.py --episodes 50 --output data/gold_standard_data.jsonl
```

Fixed-configuration long-track collection (disable curriculum):

```bash
python fine_tuning/collect_data.py \
  --episodes 50 \
  --output data/gold_standard_data.jsonl \
  --no-collect-curriculum \
  --collect-simulation-duration 300 \
  --collect-vehicle-count 40 \
  --collect-vehicles-density 3.5
```

Collection safety controls:
- `--max-attempts` stops endless retries (`0` means `episodes * 20`)
- `--collect-controller rule|rl` chooses rule expert or RL policy controller

Train an optional PPO controller (1M steps baseline):

```bash
python fine_tuning/train_highway_rl.py \
  --timesteps 1000000 \
  --save-path fine_tuning/rl_models/highway_ppo \
  --eval-episodes 20
```

### DQN Repo Profile (default)

Default RL training now follows the HighwayEnv DQN-style fast-driving setup:
- `highway-fast-v0` training env
- `DQN(MlpPolicy)` with repo-style defaults
- defaults include: `learning_rate=5e-4`, `buffer_size=15000`, `learning_starts=200`, `batch_size=32`, `gamma=0.8`
- dual evaluation in metadata: `evaluation_fast` and `evaluation_v0`
- global env resolver defaults: `sim_env_id=highway-fast-v0`, `sim_use_native_env_defaults=true`

Canonical command:

```bash
python fine_tuning/train_highway_rl.py \
  --config config.yaml \
  --timesteps 1000000 \
  --rl-train-profile dqn_repo \
  --env-id highway-fast-v0 \
  --save-path fine_tuning/rl_models/highway_dqn_fast \
  --eval-episodes 20
```

Runtime notes:
- use `>=100k` timesteps for smoke comparisons
- use `>=1,000,000` timesteps for final-quality policy selection
- for DQN profile, training runs on a single env (repo-style); PPO profiles still support vec env settings.

### Legacy Safety-First Profile (optional)

If you need the previous Safety-First shaping path for reproducibility, use:

```bash
python fine_tuning/train_highway_rl.py \
  --config config.yaml \
  --timesteps 1000000 \
  --rl-train-profile legacy_v2 \
  --save-path fine_tuning/rl_models/highway_ppo_legacy_v2 \
  --eval-episodes 20 \
  --device auto
```

Notes:
- `legacy_v2` keeps the older single-env v0 training behavior.
- legacy reward scalar flags are accepted for compatibility but ignored by `dqn_repo` and `fast_repo`.

Collect with RL controller:

```bash
python fine_tuning/collect_data.py \
  --episodes 50 \
  --output data/gold_standard_data.jsonl \
  --collect-controller rl \
  --rl-model-path fine_tuning/rl_models/highway_ppo.zip
```

Evaluate RL policy and capture simulation videos:

```bash
python fine_tuning/evaluate_rl_policy.py \
  --config config.yaml \
  --rl-model-path fine_tuning/rl_models/highway_ppo_samecfg.zip \
  --env-id highway-fast-v0 \
  --native-env-defaults \
  --episodes 10 \
  --seed 42 \
  --video-dir results/diagnostics/rl_eval_videos \
  --output results/diagnostics/rl_eval_report.json
```

### 2) Convert format

```bash
python fine_tuning/convert_data.py --input data/gold_standard_data.jsonl --output data/gold_standard_data_clean.jsonl
```

### 3) Rebalance dataset (recommended)

```bash
python fine_tuning/rebalance_data.py \
  --input data/gold_standard_data_clean.jsonl \
  --output data/gold_standard_data_rebalanced.jsonl
```

This stage fails fast if any action class (`0..4`) is missing and writes an audit report next to the output path.

### 4) Validate dataset

```bash
python fine_tuning/validate_finetune_dataset.py data/gold_standard_data_rebalanced.jsonl
```

### 5) Train and export merged model

```bash
python fine_tuning/train_dilu_ollama.py \
  --data-file data/gold_standard_data_rebalanced.jsonl \
  --model-name unsloth/Llama-3.1-8B-Instruct \
  --model-family llama3 \
  --merged-model-dir fine_tuning/merged_models/dilu-llama3_1-8b-v1
```

### 6) Create Ollama model

Update `fine_tuning/modelfiles/dilu-llama3_1-8b-v1.Modelfile` so `FROM` points to your merged model folder, then run:

```bash
ollama create dilu-llama3_1-8b-v1 -f fine_tuning/modelfiles/dilu-llama3_1-8b-v1.Modelfile
```

### 7) GGUF-first packaging (recommended)

1. Build/clone `llama.cpp` locally and ensure `convert_hf_to_gguf.py` is available.
2. Convert merged model to GGUF (and optionally quantize), then generate an Ollama Modelfile:

```bash
python fine_tuning/build_gguf.py \
  --llama-cpp-dir C:/dev/llama.cpp \
  --hf-model-dir fine_tuning/merged_models/dilu-llama3_1-8b-v1 \
  --name dilu-llama3_1-8b-v1 \
  --outtype f16 \
  --quantize Q4_K_M
```

This creates:
- `fine_tuning/gguf/dilu-llama3_1-8b-v1.f16.gguf`
- `fine_tuning/gguf/dilu-llama3_1-8b-v1.Q4_K_M.gguf` (if quantized)
- `fine_tuning/gguf/dilu-llama3_1-8b-v1.gguf.Modelfile`

3. Create an Ollama model from generated GGUF Modelfile:

```bash
ollama create dilu-llama3_1-8b-v1-gguf -f fine_tuning/gguf/dilu-llama3_1-8b-v1.gguf.Modelfile
```

Or do step 3 automatically:

```bash
python fine_tuning/build_gguf.py \
  --llama-cpp-dir C:/dev/llama.cpp \
  --hf-model-dir fine_tuning/merged_models/dilu-llama3_1-8b-v1 \
  --name dilu-llama3_1-8b-v1 \
  --outtype f16 \
  --quantize Q4_K_M \
  --create-ollama \
  --ollama-model dilu-llama3_1-8b-v1-gguf
```

## Full Pipeline Command

Run all stages:

```bash
python fine_tuning/run_pipeline.py --all
```

Run convert + rebalance + validate + train with one command:

```bash
python fine_tuning/run_pipeline.py \
  --convert \
  --rebalance \
  --validate \
  --train \
  --raw-output data/gold_standard_data.jsonl \
  --clean-output data/gold_standard_data_clean.jsonl \
  --rebalanced-output data/gold_standard_data_rebalanced.jsonl
```

Run collection via pipeline with long-track medium-high traffic overrides:

```bash
python fine_tuning/run_pipeline.py --collect \
  --episodes 50 \
  --collect-simulation-duration 300 \
  --collect-vehicle-count 40 \
  --collect-vehicles-density 3.5 \
  --raw-output data/gold_standard_data.jsonl
```

Run RL training + RL-driven collection in pipeline:

```bash
python fine_tuning/run_pipeline.py \
  --train-rl \
  --collect \
  --rl-timesteps 1000000 \
  --rl-save-path fine_tuning/rl_models/highway_ppo \
  --collect-controller rl \
  --rl-model-path fine_tuning/rl_models/highway_ppo.zip \
  --episodes 50 \
  --raw-output data/gold_standard_data.jsonl
```

Run only training on an existing cleaned dataset:

```bash
python fine_tuning/run_pipeline.py --train --clean-output data/gold_standard_data_clean.jsonl
```

## Notes

- `config.yaml` controls simulation settings used during data collection.
- Keep training datasets in `data/` (`gold_standard_data.jsonl`, `gold_standard_data_clean.jsonl`, `gold_standard_data_rebalanced.jsonl`).
- For different base model families, set `--model-family` appropriately (`llama3`, `qwen`, `mistral`, or `auto`).
- Recommended deployment path in this repo: merged export -> GGUF -> `ollama create`.
