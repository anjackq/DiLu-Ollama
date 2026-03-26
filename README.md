# DiLu-Ollama

Local-first DiLu fork for autonomous driving simulation, model benchmarking, and fine-tuning with Ollama models.

Primary workflows in this repo:
- Simulation and benchmarking: `evaluate_models_ollama.py`
- Fine-tuning pipeline: `fine_tuning/run_pipeline.py`

## Quick Start

1. Create environment and install dependencies.

```bash
conda create -n DiLu_Ollama python=3.10 -y
conda activate DiLu_Ollama
pip install -r requirements.txt
```

2. Create local config.

```bash
cp config.example.yaml config.yaml
```

3. In `config.yaml`, set local runtime values at minimum:
- `OPENAI_API_TYPE: 'ollama'`
- `OLLAMA_CHAT_MODEL`
- `OLLAMA_REFLECTION_MODEL`
- `OLLAMA_EMBED_MODEL`

4. Make sure Ollama is running and models are available.

```bash
ollama list
```

## Simulation / Evaluation (Primary)

Environment alignment defaults (global across RL + LLM paths):
- `sim_env_id: "highway-fast-v0"`
- `sim_use_native_env_defaults: true`
- Deprecated aliases still read for transition: `rl_env_id`, `rl_use_native_env_defaults`.
- CLI overrides on simulation entrypoints: `--env-id`, `--native-env-defaults`, `--no-native-env-defaults`.

Main script:

```bash
python evaluate_models_ollama.py --help
```

### 1) Smoke test (fast sanity check)

```bash
python evaluate_models_ollama.py --models qwen3.5:0.8b --limit 1 --few-shot-num 0 --experiment-id smoke_test --quiet --progress
```

### 2) Multi-model benchmark

```bash
python evaluate_models_ollama.py --models llama3.2:1b llama3.2:3b qwen3.5:0.8b qwen3.5:2b deepseek-r1:1.5b --limit 5 --few-shot-num 0 --experiment-id tier1_lightweight_base_instruct
```

### 3) Progress bars + compact LLM reply previews

```bash
python evaluate_models_ollama.py --models qwen3.5:0.8b --limit 1 --few-shot-num 0 --experiment-id smoke_with_replies --progress --progress-replies compact
```

### 4) Timeout-guarded run (timeout-only policy)

```bash
python evaluate_models_ollama.py --models qwen3.5:0.8b qwen3.5:2b --limit 3 --few-shot-num 0 --experiment-id qwen_timeout_guard --decision-timeout-sec 10
```

### 5) High-limit evaluation with reduced overhead

```bash
python evaluate_models_ollama.py --models qwen3.5:0.8b qwen3.5:2b --limit 100 --few-shot-num 0 --experiment-id long_eval --progress --performance-mode fast
```

### 6) LaMPilot-style highway task benchmark

This is a highway-only, task-metric adaptation on top of the existing DiLu action-id loop. It does not switch the repo to code-generation policies.

```bash
python evaluate_models_ollama.py --models qwen3.5:0.8b deepseek-r1:1.5b --benchmark-case-set lampilot_highway_v1 --experiment-id lampilot_highway_smoke --progress
```

Plot task metrics:

```bash
python plot_eval_compare.py -i results/experiments/lampilot_highway_smoke/compare/eval_compare_<timestamp>.json --extended
```

Timeout-only policy (shared for eval + runtime):
- Policy controls only `decision_timeout_sec` and adaptive timeout penalty.
- Use `model_policy_overrides` (or legacy `eval_model_overrides`) with `decision_timeout_sec` entries.
- Output-affecting policy fields (`max tokens`, `streaming`, `checker`, `think/native`) are deprecated and ignored with warnings.
- Decision timeout can auto-shrink on slow/timeout decisions via:
  `adaptive_timeout_penalty_enabled`, `adaptive_timeout_halving_factor`,
  `adaptive_timeout_min_sec`, `adaptive_timeout_trigger_consecutive_slow`.

Optional single-model interactive runner:

```bash
python run_dilu_ollama.py --progress --progress-replies compact
```

Quiet mode notes:
- `--quiet` suppresses high-frequency step/decision logs only.
- Warnings/errors, run progress, and final summaries are still printed.
- Config defaults: `quiet_mode`, `eval_quiet_mode`, `runtime_quiet_mode`.

Progress bar notes:
- Nested bars are enabled by default on interactive terminals.
- Eval bars: model -> seed -> step.
- Runtime bars: episode -> step.
- CLI overrides: `--progress` / `--no-progress`.
- Config defaults: `progress_bar`, `eval_progress_bar`, `runtime_progress_bar`.
- Reply previews with bars: `--progress-replies off|compact|full`.
- Quiet mode wins: with `--quiet`, reply previews are forced off.
- Config defaults: `progress_reply_mode`, `eval_progress_reply_mode`, `runtime_progress_reply_mode`.

Default structured outputs are written under:

```text
results/experiments/<experiment_id>/
```

## Fine-Tuning (Primary)

Main script:

```bash
python fine_tuning/run_pipeline.py --help
```

### Most-used end-to-end pipeline

```bash
python fine_tuning/run_pipeline.py --all --model-name microsoft/Phi-4-mini-instruct --model-family phi --merged-model-dir fine_tuning/merged_models/dilu-phi4-mini-instruct-3_8b-v1
```

### Common variant: train + GGUF + optional Ollama create

```bash
python fine_tuning/run_pipeline.py --train --gguf --model-name microsoft/Phi-4-mini-instruct --model-family phi --merged-model-dir fine_tuning/merged_models/dilu-phi4-mini-instruct-3_8b-v1 --gguf-name dilu-phi4-mini-instruct-3_8b-v1 --gguf-outtype f16 --gguf-quantize Q4_K_M --gguf-create-ollama --ollama-model dilu-phi4-mini-instruct-3_8b-v1-gguf
```

Advanced low-level scripts are still available:
- `fine_tuning/train_dilu_ollama.py`
- `fine_tuning/build_gguf.py`

Detailed fine-tuning notes: `fine_tuning/README.md`.

## Merge + Plot Results

When models were evaluated separately under the same experiment id, merge latest per-model outputs without re-running all models.

List available models:

```bash
python merge_eval_reports.py --experiment-id tier1_lightweight_base_instruct --results-root results --list-models
```

Merge all available models in that experiment:

```bash
python merge_eval_reports.py --experiment-id tier1_lightweight_base_instruct --results-root results
```

Merge selected models only:

```bash
python merge_eval_reports.py --experiment-id tier1_lightweight_base_instruct --models llama3.2:1b llama3.2:3b qwen3.5:0.8b qwen3.5:2b deepseek-r1:1.5b --results-root results
```

Merge selected models plus all available models (union):

```bash
python merge_eval_reports.py --experiment-id tier1_lightweight_base_instruct --models qwen3.5:0.8b --include-available --results-root results
```

Plot merged report:

```bash
python plot_eval_compare.py -i results/tier1_lightweight_base_instruct/compare/eval_compare_<timestamp>.json
python plot_eval_compare.py -i results/tier1_lightweight_base_instruct/compare/eval_compare_<timestamp>.json --extended
python plot_eval_compare.py -i results/tier1_lightweight_base_instruct/compare/eval_compare_<timestamp>.json --all-metrics
```

## Troubleshooting (Short)

- Long waits on Qwen small models:
  - Lower timeout (for example `--decision-timeout-sec 8` or `10`)
  - Enable adaptive timeout penalty in config (`adaptive_timeout_*` keys)
  - If you still need output behavior changes, configure them outside policy mode.
- `Native Ollama chat failed ... Falling back to OpenAI-compatible path`:
  - Usually native `/api/chat` timeout or model-specific incompatibility.
  - Keep `OLLAMA_USE_NATIVE_CHAT: true` and tune timeout/think mode per model.
- GGUF conversion error about missing tokenizer files:
  - Ensure merged model directory includes tokenizer assets required by `convert_hf_to_gguf.py`.

## Background

Original project and paper:
- Paper: https://arxiv.org/abs/2309.16292
- Project page: https://pjlab-adg.github.io/DiLu
- Original repo: https://github.com/PJLab-ADG/DiLu

## Citation

```bibtex
@article{wen2023dilu,
  title={Dilu: A knowledge-driven approach to autonomous driving with large language models},
  author={Wen, Licheng and Fu, Daocheng and Li, Xin and Cai, Xinyu and Ma, Tao and Cai, Pinlong and Dou, Min and Shi, Botian and He, Liang and Qiao, Yu},
  journal={arXiv preprint arXiv:2309.16292},
  year={2023}
}
```

## License

Apache 2.0
