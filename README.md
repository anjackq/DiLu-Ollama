# DiLu-Ollama

Local-first DiLu fork for autonomous driving simulation and model benchmarking with Ollama models.

## Overview

DiLu-Ollama is a local-first research framework for evaluating and adapting language-model driving agents in `highway-env` with Ollama-hosted models. The project started from a practical problem: local LLM driving experiments are easy to run informally, but hard to compare rigorously once runtime instability, timeout collapse, model scale, and benchmark reporting begin to interact. This repository turns that workflow into a reproducible benchmark and analysis pipeline.

The core evaluation path uses the DiLu decision loop with task-conditioned highway benchmarks, report merging, tiered model studies, and runtime diagnostics. Around that core, the repo supports discrete benchmark extensions for `highway-fast-v0`, `merge-v0`, and `intersection-v1`, while explicitly treating continuous-control settings such as `parking-v0` as out of scope for the current action interface.

The current evidence base is intentionally reported conservatively. In the three-tier highway study, the lightweight tier provides the strongest usable comparison set, the midclass tier remains screening-quality because several runs are still invalid, and the highclass tier currently functions mainly as a runtime-scalability result because all tested 14B models failed ranking eligibility under the present local timeout policy. The framework therefore supports not only model comparison, but also diagnosis of when benchmark validity breaks down as model size and latency increase.

In that form, DiLu-Ollama is both an experimental framework and a benchmark-oriented analysis workflow: it can run local driving simulations, benchmark base SLMs, summarize tier-level evidence, and extend evaluation across compatible discrete scenarios.

Primary workflows in this repo:
- Simulation and benchmarking: `evaluate_models_ollama.py`

Project status:
- the core benchmark and runtime-diagnostics pipeline is the supported path
- retired fine-tuning code and generated fine-tuning artifacts were moved under `archive/`
- compatibility shims remain available intentionally for transition use, but the entrypoints listed below are the supported surface

Supported entrypoints:
- `evaluate_models_ollama.py`
- `merge_eval_reports.py`
- `plot_eval_compare.py`

## Results Snapshot

The current benchmark evidence is kept under `results/benchmarks/`, with archived exploratory runs separated under `archive/`.

## Quick Start

1. Create environment and install dependencies.

```bash
conda create -n DiLu_Ollama python=3.10 -y
conda activate DiLu_Ollama
pip install -r requirements.txt
```

2. Use the checked-in lean `config.yaml` as your working Ollama config.
   `config.example.yaml` is the fuller option catalog with inactive providers,
   optional overrides, and legacy-compatible keys.

3. In `config.yaml`, set local runtime values at minimum:
- `OPENAI_API_TYPE: 'ollama'`
- `OLLAMA_CHAT_MODEL`
- `OLLAMA_EMBED_MODEL`
- optionally `OLLAMA_USE_NATIVE_CHAT`

4. Make sure Ollama is running and models are available.

```bash
ollama list
```

## Simulation / Evaluation (Primary)

Environment alignment defaults (global across RL + LLM paths):
- `sim_env_id: "highway-fast-v0"`
- `sim_use_native_env_defaults: true`
- `sim_action_target_speeds: [0, 5, 10, 15, 20, 25, 30]`
- Explicit env overrides are optional; the lean `config.yaml` relies on the native `highway-fast-v0` preset plus `sim_action_target_speeds`.
- Deprecated aliases still read for transition: `rl_env_id`, `rl_use_native_env_defaults`.
- CLI overrides on simulation entrypoints: `--env-id`, `--native-env-defaults`, `--no-native-env-defaults`, `--action-target-speeds`.

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

### 4) Laddered timeout report run (default eval policy)

```bash
python evaluate_models_ollama.py --models qwen3.5:0.8b qwen3.5:2b --limit 3 --few-shot-num 0 --experiment-id qwen_timeout_ladder --progress
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

Current built-in benchmark properties:
- `lampilot_highway_v1` now ships a fixed 40-case suite.
- 5 cases each for:
  - `speed_increase`
  - `speed_decrease`
  - `follow_gap_increase`
  - `follow_gap_decrease`
  - `lane_change_left`
  - `lane_change_right`
  - `overtake_left`
  - `overtake_right`
- all cases are prevalidated before model evaluation begins
- invalid case sets abort the run early instead of contaminating results

Benchmark score interpretation:
- `overall_score` is a safety/efficiency subscore, not task success
- `driving_score_v2` is the recommended task-conditioned headline score
- `driving_score` remains in reports as the legacy compatibility metric
- `driving_score = 0` on crash or on task non-completion
- `driving_score_v2` further penalizes stop-heavy blocking behavior and timeout/fallback-driven collapse
- timeout/fallback-dominated benchmark runs are marked invalid in the report and should not be compared against valid runs

Default benchmark speed profile:
- `config.yaml` now allows `SLOWER` to reach `0 m/s`
- this is the canonical LaMPilot benchmark profile
- it keeps the same benchmark cases and scoring, but removes the native `20 m/s` floor
- the report surfaces:
  - `stop_episode_rate`
  - `stop_rate_mean`
  - `near_stop_rate_mean`
  - `min_ego_speed_mps_mean`
- interpret high `no_collision_rate` + high stop metrics + low `driving_score_v2` as a degenerate conservative policy, not successful driving

Canonical task benchmark:

```bash
python evaluate_models_ollama.py --config config.yaml --models qwen3:1.7b dilu-qwen3-1_7b-v1 --benchmark-case-set lampilot_highway_v1 --experiment-id lampilot_default_benchmark --progress
```

Plot task metrics:

```bash
python plot_eval_compare.py -i results/lampilot_highway_smoke/compare/eval_compare_<timestamp>.json --extended
```

Plot output layout notes:
- the primary output is now a task-first summary figure
- companion figures are emitted automatically beside it:
  - `_behavior.png`
  - `_efficiency.png`
  - `_energy.png` when energy metrics are present
  - `_runtime.png` when `--extended` or `--all-metrics` is used
- dense compare plots automatically switch to horizontal bars once more than 8 models are shown

Eval timeout ladder:
- Eval and benchmark runs use an aggressive bounded timeout ladder: `10 -> 15 -> 20` seconds.
- Escalation happens only on actual timeout.
- Recovery steps down one level after `3` consecutive successful non-timeout decisions.
- Early-stop containment is also enabled:
  - eligible after `3` decisions
  - stop after `2` consecutive timeout+fallback decisions
  - quarantine a model after `2` timeout-collapsed episodes
- Treat this as a local benchmark compute budget, not a real highway-control latency claim.
- Reports now expose:
  - `timeout_level_10_rate_mean`
  - `timeout_level_15_rate_mean`
  - `timeout_level_20_rate_mean`
  - `timeout_escalation_count_mean`
  - `timeout_recovery_count_mean`

Canonical eval example under the laddered timeout policy:

```bash
python evaluate_models_ollama.py --models qwen3:1.7b dilu-qwen3-1_7b-v1 --benchmark-case-set lampilot_highway_v1 --experiment-id lampilot_timeout_ladder --progress
```

### 7) Benchmark + efficiency reporting

`evaluate_models_ollama.py` is now the canonical entrypoint for both standard evaluation and measurement mode.
LaMPilot benchmark runs are the canonical path for task quality, latency, and optional hardware energy reporting.
Measurement mode reuses the same closed-loop driving evaluator, but adds:
- end-to-end episode runtime
- decision latency
- response throughput
- optional Joulescope hardware energy capture
- idle-baseline subtraction

Latency-only smoke run:

```bash
python evaluate_models_ollama.py --config config.yaml --models qwen3:1.7b dilu-qwen3-1_7b-v1 --limit 3 --experiment-id energy_latency_smoke --results-root results/energy_benchmarks --energy-mode latency_only --progress
```

Canonical LaMPilot task benchmark with latency-only measurement:

```bash
python evaluate_models_ollama.py --config config.yaml --models qwen3:1.7b dilu-qwen3-1_7b-v1 --benchmark-case-set lampilot_highway_v1 --experiment-id energy_latency_task --results-root results/energy_benchmarks --energy-mode latency_only --progress
```

Canonical LaMPilot task benchmark with hardware energy measurement:

```bash
python evaluate_models_ollama.py --config config.yaml --models qwen3:1.7b dilu-qwen3-1_7b-v1 --benchmark-case-set lampilot_highway_v1 --experiment-id energy_latency_hw --results-root results/energy_benchmarks --energy-mode joulescope_hw --idle-calibration results/energy_benchmarks/idle_power.json --progress
```

Optional Joulescope setup:
- install the optional Python package separately: `pip install joulescope`
- place the Joulescope JS110 in series with the device power supply
- keep the benchmark run serial; do not run other heavy jobs on the same machine during measurement
- calibrate idle power before report runs

Idle calibration:

```bash
python evaluate_models_ollama.py --config config.yaml --results-root results/energy_benchmarks --energy-mode joulescope_hw --calibrate-idle --idle-duration-sec 120 --calibration-output results/energy_benchmarks/idle_power.json
```

Hardware energy run with baseline subtraction:

```bash
python evaluate_models_ollama.py --config config.yaml --models qwen3:1.7b dilu-qwen3-1_7b-v1 --limit 10 --experiment-id energy_latency_hw --results-root results/energy_benchmarks --energy-mode joulescope_hw --idle-calibration results/energy_benchmarks/idle_power.json --progress
```

Measurement protocol:
- keep the machine thermally stable before starting the report run
- run one workload at a time
- use the same `config.yaml`, default stop-capable benchmark profile, and few-shot settings across models
- keep the eval timeout ladder fixed across compared models unless you are running a separate timeout-policy ablation
- interpret `raw_energy_j` as measured total device energy over the episode window
- interpret `net_energy_j` as `raw_energy_j - idle_baseline_energy_j`
- keep `driving_score_v2` as the headline task metric and interpret latency/energy as separate deployment-efficiency metrics
- compare model quality only when timeout/fallback collapse is not dominating the run

Canonical energy/task benchmark example under the laddered timeout policy:

```bash
python evaluate_models_ollama.py --config config.yaml --models qwen3:1.7b dilu-qwen3-1_7b-v1 --benchmark-case-set lampilot_highway_v1 --experiment-id energy_latency_timeout_ladder --results-root results/energy_benchmarks --energy-mode latency_only --progress
```

Energy benchmark outputs:
- compare report: `results/energy_benchmarks/<experiment_id>/compare/energy_latency_compare_<timestamp>.json`
- per-model summary: `results/energy_benchmarks/<experiment_id>/models/<model>/energy/`
- plots: `python plot_eval_compare.py -i <compare_report>.json --extended`
- the primary compare plot is task-first; latency/token/energy/runtime metrics are emitted as companion figures

Timeout policy notes:
- Eval and benchmark use the laddered policy by default:
  - policy mode: `laddered`
  - ladder: `10 -> 15 -> 20`
  - recovery: `3` clean decisions
- The checked-in `config.yaml` now pins this aggressive local compute-budget policy explicitly; `config.example.yaml` keeps the fuller catalog of related options.
- Runtime remains on the legacy shrink-only adaptive policy unless you change it separately.
- Output-affecting policy fields (`max tokens`, `streaming`, `checker`, `think/native`) are deprecated and ignored with warnings in timeout-only model policy overrides.
- Legacy adaptive timeout knobs:
  - `adaptive_timeout_penalty_enabled`
  - `adaptive_timeout_halving_factor`
  - `adaptive_timeout_min_sec`
  - `adaptive_timeout_trigger_consecutive_slow`
  only apply to runtime mode, or to eval if you explicitly switch `eval_timeout_policy_mode` away from `laddered`.

Optional single-model interactive runner:

```bash
python run_dilu_ollama.py --progress --progress-replies compact
```

Quiet mode notes:
- `--quiet` suppresses high-frequency step/decision logs only.
- Warnings/errors, run progress, and final summaries are still printed.
- Optional config keys: `quiet_mode`, `eval_quiet_mode`, `runtime_quiet_mode`.

Progress bar notes:
- Nested bars are enabled by default on interactive terminals.
- Eval bars: model -> seed -> step.
- Runtime bars: episode -> step.
- CLI overrides: `--progress` / `--no-progress`.
- Optional config keys: `progress_bar`, `eval_progress_bar`, `runtime_progress_bar`.
- Reply previews with bars: `--progress-replies off|compact|full`.
- Quiet mode wins: with `--quiet`, reply previews are forced off.
- The lean `config.yaml` pins only `progress_reply_mode`; the fuller template keeps all related keys.

Structured outputs are written under:

```text
<results_root>/<experiment_id>/
```

With the checked-in lean `config.yaml`, that means:

```text
results/<experiment_id>/
```

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

For merged and benchmark compare reports:
- the requested output path remains the main summary plot
- themed companion plots are written alongside it with suffixes like `_behavior`, `_efficiency`, `_energy`, and `_runtime`

## Cross-Scenario Discrete Benchmarks

Built-in case sets now cover three discrete highway-env families:
- `lampilot_highway_v1` -> `highway-fast-v0`
- `lampilot_merge_v1` -> `merge-v0`
- `lampilot_intersection_v1` -> `intersection-v0`

Run one environment per benchmark job:

```bash
python evaluate_models_ollama.py --config config.yaml --env-id merge-v0 --benchmark-case-set lampilot_merge_v1 --models qwen3:4b --experiment-id merge_smoke --progress
python evaluate_models_ollama.py --config config.yaml --env-id intersection-v0 --benchmark-case-set lampilot_intersection_v1 --models qwen3:4b --experiment-id intersection_smoke --progress
```

Notes:
- the discrete framework now rejects unsupported continuous-control envs such as `parking-v0`
- `merge-v0` is exposed by highway-env as a fixed-layout merge setup, so `lampilot_merge_v1` uses protocol-sliced merge tasks rather than pretending the native env provides rich procedurally varied traffic families
- highway remains the only scenario using `driving_score_v2` as the headline metric; merge and intersection keep scenario-local task metrics without reusing the highway behavior-aware scalar

## Maintenance Notes

- Treat tracked code, configs, benchmarks, and active analysis files as source-of-truth project assets.
- Treat experiment roots under ignored folders as generated outputs that should be regenerated rather than hand-maintained.
- Fine-tuning code, RL-teacher data, adapter conversion utilities, and old base-vs-fine-tuned analysis scripts are retired under `archive/`.
- Compatibility shims, old rerun configs, local memory utilities, and trace replay helpers are retired under `archive/`.
- Before publishing new analysis conclusions, rerun the focused verification suite:
  - `tests.test_plot_eval_compare`
  - `tests.test_cli_merge`

## Troubleshooting (Short)

- Long waits on Qwen small models:
  - Lower timeout (for example `--decision-timeout-sec 8` or `10`)
  - Enable adaptive timeout penalty in config (`adaptive_timeout_*` keys)
  - If you still need output behavior changes, configure them outside policy mode.
- `Native Ollama chat failed ... Falling back to OpenAI-compatible path`:
  - Usually native `/api/chat` timeout or model-specific incompatibility.
  - The lean `config.yaml` already prefers `/v1` with `OLLAMA_USE_NATIVE_CHAT: false` for smaller local models.
  - If you intentionally switch native chat on, tune timeout and think mode per model.

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
