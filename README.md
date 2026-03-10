# DiLu🐴: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models

[![Custom badge](https://img.shields.io/badge/Paper-Arxiv-b31b1b?logo=arxiv&logoColor=white?style=flat-square)](https://arxiv.org/abs/2309.16292)
[![Custom badge](https://img.shields.io/badge/Project%20Page-white?logo=GitHub&color=green?style=flat-square)](https://pjlab-adg.github.io/DiLu)
[![Stars](https://img.shields.io/github/stars/pjlab-adg/DiLu?style=social)](https://github.com/pjlab-adg/DiLu/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/PJLab-ADG/DiLu?style=flat-square)](https://github.com/PJLab-ADG/DiLu/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/PJLab-ADG/DiLu/pulls)


> 堕檀溪水中，溺不得出。备急曰：‘**的卢**，今日厄矣，可努力！’**的卢**乃一踊三丈，遂得过，乘浮渡河. -- 三国志
> 
> In the face of adversity, the legendary horse DiLu  follows Liu Bei's words to leapt across the Tanxi River, embodying intelligence and strength. Inspired by this tale, our DiLu framework endeavors to navigate the challenging terrains of autonomous driving.



https://github.com/PJLab-ADG/DiLu/assets/18390668/cd48747f-f710-4a42-abb9-ca15e7ee68f2




## 🔍 Framework Overview

<img src="./assets/framework.png" width=80%>

DiLu is an innovative **closed-loop**, **self-evolving** framework, blending common-sense knowledge and memory components with the power of large language models. DiLu consists of four core modules: Environment, Reasoning, Reflection, and Memory.

DiLu is not just a framework, it's an exploration of [Knowledge-driven Autonomous Driving](https://github.com/PJLab-ADG/awesome-knowledge-driven-AD).


## 🌟 Highlights
- **`2024-01-22`** Codes are now release!
- **`2024-01-17`** Exciting news! **DiLu is accepted by ICLR 2024 🎉🎉!** 
- **`2023-10-12`** Explore our project page, now live [here](https://pjlab-adg.github.io/DiLu)🔗!
- **`2023-09-28`** Our paper is available on [Arxiv](https://arxiv.org/abs/2309.16292)📄!


## 🚀 Getting Started
### 1. Requirements 📦

For an optimal experience, we recommend using conda to set up a new environment for DiLu.

```bash
conda create -n dilu python=3.8 
conda activate dilu
pip install -r requirements.txt
```

📝 **Note:** This fork uses a newer local/Ollama-compatible dependency stack in `requirements.txt` (LangChain split packages + newer OpenAI/Chroma versions).

### 2. Configuration ⚙️ 

All configurable parameters are located in `config.yaml`.

Before running DiLu, set up your API keys. DiLu supports OpenAI, Azure OpenAI, Ollama, and Gemini.

Configure as below in `config.yaml`:
```yaml
OPENAI_API_TYPE: # 'openai' or 'azure' or 'ollama' or 'gemini'
# below are for Openai
OPENAI_KEY: # 'sk-xxxxxx' 
OPENAI_CHAT_MODEL: 'gpt-4-1106-preview' # Alternative models: 'gpt-3.5-turbo-16k-0613' (note: performance may vary)
# below are for Azure OAI service
AZURE_API_BASE: # https://xxxxxxx.openai.azure.com/
AZURE_API_VERSION: "2023-07-01-preview"
AZURE_API_KEY: #'xxxxxxx'
AZURE_CHAT_DEPLOY_NAME: # chat model deployment name
AZURE_EMBED_DEPLOY_NAME: # text embed model deployment name  
# below are for Gemini (AI Studio)
GEMINI_API_KEY: # 'AIza...'
GEMINI_CHAT_MODEL: 'gemini-2.0-flash'
GEMINI_REFLECTION_MODEL: null
```


### 3. Running DiLu 🐴

Running DiLu is straightforward:
```bash
python run_dilu_ollama.py
```
The default setting runs a 3-shot simulation with different seeds. You can modify this in `config.yaml`.

📝 **Fork Note:** This fork is maintained around the local-model/Ollama workflow. The original `run_dilu.py` has been moved to `archive/run_dilu.py` as a legacy script.

After completing the simulations, check the `results` folder. `log.txt` contains detailed steps and seeds for each simulation, and all simulation videos are saved here too.

📝 **Note:** During DiLu execution, the 'highway-env' pygame window might appear unresponsive. If the terminal is actively outputting, everything is running as expected.


#### Use reflection module:

To activate the reflection module, set `reflection_module` to True in `config.yaml`. New memory items will be saved to the updated memory module.

For unattended runs (for example VS Code debug runs, long batches, or eval scripts), you can disable reflection prompts:
- `reflection_interactive: False`
- `reflection_auto_add: True` (optional, auto-commit new memory items)
- `reflection_add_every_n: 5` (controls non-collision sampling cadence)

## 4. Visualizing Results 📊

We provide a visualization scripts for the simulation result.
```bash
python ./visualize_results.py -r results/highway_0.db -m memories/20_mem
```
Open `http://127.0.0.1:7860` to view each frame's prompts and decisions!

## 5. Fork Additions: Local Ollama + Fine-Tuning + Evaluation

This fork adds support for running DiLu with local open-source models via Ollama, plus a fine-tuning workflow and model comparison utilities.

### Local Ollama Runtime

Use the project requirements (this fork now uses a single `requirements.txt`):

```bash
pip install -r requirements.txt
```

Create a local config from the tracked template and customize it:

```bash
cp config.example.yaml config.yaml
```

Key settings for local runs:
- `OPENAI_API_TYPE: 'ollama'`
- `OLLAMA_CHAT_MODEL`
- `OLLAMA_REFLECTION_MODEL`
- `OLLAMA_EMBED_MODEL`
- `memory_path` (embedding dimensions differ across models, so use separate memory DBs)
- `results_root` / `experiment_id` / `run_id` (structured result layout)
- `result_folder_override` (optional legacy direct-output override)

Run DiLu with local Ollama:

```bash
python run_dilu_ollama.py
```

By default this now writes to:

```text
results/experiments/<experiment_id>/models/<model_slug>/runs/<run_id>/
```

including videos, episode DBs, `log.txt`, and `run_metrics_*.json`.

### Gemini Runtime

Set `config.yaml`:

- `OPENAI_API_TYPE: 'gemini'`
- `GEMINI_API_KEY`
- `GEMINI_CHAT_MODEL`
- `GEMINI_REFLECTION_MODEL` (optional, defaults to `GEMINI_CHAT_MODEL`)

Run:

```bash
python run_dilu_ollama.py
```

Important for memory embeddings in Gemini mode:
- Gemini chat is supported natively.
- Embeddings are still non-Gemini in this phase.
- Configure either:
  - OpenAI-compatible embedding backend via env vars `OPENAI_API_BASE` + `OPENAI_API_KEY` (optional `OPENAI_EMBED_MODEL`), or
  - Ollama embedding backend via `OLLAMA_API_BASE` + `OLLAMA_API_KEY` + `OLLAMA_EMBED_MODEL`.

### Experiment-Oriented Results Layout

```text
results/
  experiments/
    <experiment_id>/
      manifest.json
      compare/
        eval_compare_<timestamp>.json
      models/
        <model_slug>/
          runs/<run_id>/
          eval/
          plots/
```

### Fine-Tuning Workflow (Small/Local Models)

Generate training data (rule-based expert labels from DiLu scenarios):

```bash
python fine_tuning/collect_data.py
```
This writes `data/gold_standard_data.jsonl`.

Convert data to the strict instruction/output format used by the training scripts:

```bash
python fine_tuning/convert_data.py
```

Validate the cleaned dataset schema/format before training:

```bash
python fine_tuning/validate_finetune_dataset.py data/gold_standard_data_clean.jsonl
```

Train and export a merged model for Ollama (Unsloth + TRL):

```bash
python fine_tuning/train_dilu_ollama.py
```

Family-aware training examples:

```bash
python fine_tuning/train_dilu_ollama.py --model-name unsloth/Llama-3.1-8B-Instruct --model-family llama3
python fine_tuning/train_dilu_ollama.py --model-name unsloth/Qwen2.5-7B-Instruct --model-family qwen
```

Run the full pipeline (collect -> convert -> validate -> train):

```bash
python fine_tuning/run_pipeline.py --all
```

Example Ollama Modelfile template is tracked here:
- `fine_tuning/modelfiles/dilu-llama3_1-8b-v1.Modelfile`

### Compare Models on Fixed Seeds

Run a small smoke comparison:

```bash
python evaluate_models_ollama.py --models deepseek-r1:14b dilu-llama3_1-8b-v1 --limit 1 --few-shot-num 0
```

If a model hangs or responds very slowly, run with timeout guardrails:

```bash
python evaluate_models_ollama.py --models qwen3:0.6b --limit 1 --few-shot-num 0 --decision-timeout-sec 60 --disable-streaming --disable-checker-llm
```

Run a larger comparison:

```bash
python evaluate_models_ollama.py --models deepseek-r1:14b dilu-llama3_1-8b-v1 --limit 5
```

Structured comparison output options:

```bash
python evaluate_models_ollama.py --models deepseek-r1:14b dilu-llama3_1-8b-v1 --limit 5 --experiment-id exp_nonthinking_v1
python evaluate_models_ollama.py --models deepseek-r1:14b dilu-llama3_1-8b-v1 --limit 5 --results-root results/experiments
python evaluate_models_ollama.py --models deepseek-r1:14b --limit 3 --output results/my_eval_report.json
```

Optional reasoning-alignment sample collection (for manual or later judge-model scoring):

```bash
python evaluate_models_ollama.py --models deepseek-r1:14b dilu-llama3_1-8b-v1 --limit 5 --alignment-sample-rate 0.1 --alignment-max-samples 50
```

By default this saves:
- Global compare report: `results/experiments/<experiment_id>/compare/eval_compare_<timestamp>.json`
- Per-model summaries:
  - `.../models/<model_slug>/eval/eval_summary_<timestamp>.json`
  - `.../models/<model_slug>/eval/eval_episodes_<timestamp>.json`
- Experiment manifest: `results/experiments/<experiment_id>/manifest.json`

The benchmark report includes:
- Existing metrics: crash/no-collision rates, avg steps, runtime, response format rates
- New online safety metrics: TTC danger rate (`TTC < 2.0s`), headway violation rate (`front gap < 15m`)
- New efficiency/comfort metrics: reward summary, average ego speed, lane-change rate, accel/decel flapping rate (`3↔4`)
- New system metrics: format failure rate and decision-latency estimate (`ms/step`)
- Timeout reliability metrics: decision timeout rate, timeout episode rate, fallback action rate, timeout episode count

Timeout Reliability Metrics:
- `decision_timeout_rate_mean`: primary runtime reliability metric (`decision_timeouts_total / decision_calls_total`)
- `timeout_episode_rate`: episodes with at least one timeout over total episodes
- `fallback_action_rate_mean`: share of decisions that required safety fallback (`action 4` via timeout/parse fallback)
- Interpretation: lower timeout/fallback rates indicate a more stable inference pipeline

Plot the comparison report:

```bash
python plot_eval_compare.py -i results/experiments/<experiment_id>/compare/eval_compare_<timestamp>.json
```

This generates a PNG chart with crash rate / no-collision rate / average steps / runtime.

Plot extended runtime+safety metrics:

```bash
python plot_eval_compare.py -i results/experiments/<experiment_id>/compare/eval_compare_<timestamp>.json --extended
```

This generates a PNG chart for decision timeout rate / timeout episode rate / fallback action rate / TTC danger rate.

Emit per-model plots under each model folder:

```bash
python plot_eval_compare.py -i results/experiments/<experiment_id>/compare/eval_compare_<timestamp>.json --all-metrics --emit-per-model
```

### Example Runs

Quick smoke test (single model, 1 seed, zero-shot):

```bash
python evaluate_models_ollama.py --models deepseek-r1:14b --limit 1 --few-shot-num 0
```

Two-model comparison on fixed seeds:

```bash
python evaluate_models_ollama.py --models deepseek-r1:14b dilu-llama3_1-8b-v1 --limit 5
```

Two-model comparison with alignment sample collection:

```bash
python evaluate_models_ollama.py --models deepseek-r1:14b dilu-llama3_1-8b-v1 --limit 5 --alignment-sample-rate 0.1 --alignment-max-samples 50
```

Custom output path:

```bash
python evaluate_models_ollama.py --models deepseek-r1:14b --limit 3 --output results/my_eval_report.json
```

Plot default 2x2 metrics:

```bash
python plot_eval_compare.py -i results/experiments/<experiment_id>/compare/eval_compare_<timestamp>.json
```

Plot extended runtime+safety metrics:

```bash
python plot_eval_compare.py -i results/experiments/<experiment_id>/compare/eval_compare_<timestamp>.json --extended
```

Plot all metrics together:

```bash
python plot_eval_compare.py -i results/experiments/<experiment_id>/compare/eval_compare_<timestamp>.json --all-metrics
```

Emit per-model plots into each model folder:

```bash
python plot_eval_compare.py -i results/experiments/<experiment_id>/compare/eval_compare_<timestamp>.json --all-metrics --emit-per-model
```

### Git Hygiene (What Is Intentionally Not Tracked)

This fork ignores local/generated artifacts such as:
- `config.yaml` (local secrets/runtime config)
- `results/` (videos, DBs, eval reports, plots)
- `memories/` (Chroma vector stores)
- `temp/`, `outputs/`
- `fine_tuning/checkpoints/`, `fine_tuning/merged_models/`, `fine_tuning/adapters/`
- `unsloth_compiled_cache/`
- IDE files (e.g. `.idea/`)

Use `config.example.yaml` as the versioned template and keep your real `config.yaml` local.

### Troubleshooting (Gemini)

- `GEMINI_API_KEY must be set`: add `GEMINI_API_KEY` in `config.yaml`.
- `GEMINI_CHAT_MODEL must be set`: set a valid Gemini model (for example `gemini-2.0-flash`).
- Memory init error in Gemini mode: configure an embedding backend (OpenAI-compatible or Ollama) as described above.
- Provider/model not found errors: verify model name availability and API key project access.


## 🔖 Citation
If you find our paper and codes useful, please kindly cite us via:

```bibtex
@article{wen2023dilu,
  title={Dilu: A knowledge-driven approach to autonomous driving with large language models},
  author={Wen, Licheng and Fu, Daocheng and Li, Xin and Cai, Xinyu and Ma, Tao and Cai, Pinlong and Dou, Min and Shi, Botian and He, Liang and Qiao, Yu},
  journal={arXiv preprint arXiv:2309.16292},
  year={2023}
}
```

## 📝 License
DiLu is released under the Apache 2.0 license.
