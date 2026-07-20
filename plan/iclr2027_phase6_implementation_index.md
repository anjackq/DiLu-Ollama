# ICLR 2027 Research Redesign Implementation Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Convert the approved Contract-Policy Decomposition and Counterfactual Stress V3 specification into an implementation-ready, test-first work program without starting Ollama or simulator experiments during planning.

**Architecture:** Add a typed scientific runtime beside the legacy evaluator, a separate `stress_v3` package, machine-verifiable protocol locks, and a resumable campaign/analysis pipeline. Preserve Stress V2 and current CLI behavior through adapters while claim-bearing ICLR runs use explicit immutable configuration and mandatory traces.

**Tech Stack:** Python 3, `unittest`/`pytest`, dataclasses, PyYAML, NumPy, Gymnasium/highway-env, Ollama native chat, JSON/JSONL/CSV, PowerShell.

---

## 1. Authority and Scope

These documents are the scientific source of truth:

- `plan/iclr2027_contract_policy_stress_v3_design.md`
- `plan/iclr2027_smoke_investigation_protocol.md`
- `plan/iclr2027_evaluation_analysis_protocol.md`
- `plan/iclr2027_spec_review_log.md`

This implementation index does not authorize model calls or simulation. It authorizes only test-driven code work through the smoke dry-run boundary. The 3,840-episode main campaign remains forbidden until all required locks and gates verify.

## 2. Plan Set

The plan files describe four workstreams. Execute them by the dependency milestones below, rather than finishing one entire file before opening the next:

1. Foundation milestone:
   - Runtime Tasks 1-6 from `plan/iclr2027_runtime_harness_implementation_plan.md`.
   - Protocol Tasks 1-2 from `plan/iclr2027_protocol_smoke_implementation_plan.md` so the scientific lock exists before benchmark validation.
2. Benchmark milestone:
   - Stress V3 Tasks 1-8 from `plan/iclr2027_stress_v3_implementation_plan.md`.
   - Real candidate generation/validation is allowed only after the science lock verifies; unit fixtures may be built earlier.
3. Integration milestone:
   - Runtime Task 7 and Stress V3 Tasks 9-10.
   - Resolve all cross-package schemas, traces, annotations, and regression adapters.
4. Smoke milestone:
   - Protocol/Smoke Tasks 3-10.
   - Stop at dry runs until explicit authorization for real S1-S5 execution.
5. Campaign/analysis milestone:
   - All tasks in `plan/iclr2027_campaign_analysis_implementation_plan.md`.
   - Code and synthetic tests may be implemented early; the real 3,840 run remains last.

The four workstreams cover:

1. `plan/iclr2027_runtime_harness_implementation_plan.md`
   - immutable factors and condition hashes;
   - provenance-locked prompt construction;
   - typed action resolution and native Ollama transport;
   - explicit execution modes and mandatory scientific traces.
2. `plan/iclr2027_stress_v3_implementation_plan.md`
   - paired case schema and deterministic candidate generation;
   - corrected temporal/directional predicates;
   - observable and privileged oracles;
   - independent validation, development split, sealed split, and annotation bundle.
3. `plan/iclr2027_protocol_smoke_implementation_plan.md`
   - lock schemas and hash binding;
   - S0-S5 staged smoke investigation;
   - fault injection and machine-recomputed go/no-go gates.
4. `plan/iclr2027_campaign_analysis_implementation_plan.md`
   - deterministic 3,840-episode schedule and resume semantics;
   - paired endpoints, 12 composite gates, Holm correction;
   - clean-room artifact regeneration and manuscript evidence bundle.

Do not cross a milestone gate until all tasks named for that milestone pass. In particular, never validate candidates before the science lock, never run behavioral smoke before the runtime lock, and never open the sealed main campaign before final protocol binding.

## 3. Repository Layout

Create focused modules instead of expanding the large legacy files:

```text
dilu/driver_agent/
  prompt_modules.py
  prompts/original_dilu_2024.txt
dilu/runtime/
  harness_config.py
  action_resolution.py
  generation_seed.py
  runtime_failures.py
  ollama_scientific_client.py
  shield_stack.py
  scientific_trace.py
  protocol_locks.py
  campaign_schedule.py
  campaign_attempts.py
  campaign.py
  stress_v3/
    schema.py
    snapshots.py
    generator.py
    predicates.py
    oracles.py
    replay_validator.py
    label_validator.py
    validation.py
    sealing.py
    intent_annotation.py
    metrics.py
    statistics.py
    reporting.py
configs/iclr2027/
protocol/iclr2027/
schemas/iclr2027/
scripts/
tests/
```

Generated artifacts use linked, stage-specific roots so the approved dated smoke path remains exact:

```text
results/analysis/stress_v3_protocol_<protocol_id>/
  protocol/
results/analysis/stress_v3_validation/<benchmark_version>/
  <redacted aggregate validation summaries only>
results/analysis/stress_v3_smoke_<YYYYMMDD>/
  <required smoke artifacts at this level>
results/analysis/stress_v3_main_<protocol_id>/
  campaign/episodes/
  campaign/traces/
  analysis/
```

The locks bind these roots by relative path and checksum. Candidate rows, trajectories, rejection rows, and private sealed cases belong only under ignored `results/secure/stress_v3/<version>/`; `results/analysis/stress_v3_validation/` may contain aggregate redacted summaries only. Public development cases and a content-hiding seal manifest belong under `benchmarks/dilu_highway_reactive_stress_v3/`.

## 4. Cross-Cutting Rules

### 4.1 Preserve Current State

- Treat all current modified, deleted, and untracked files as owner/project state.
- Do not revert the existing prompt-profile work, video work, paper deletions, or Stress V2 scripts.
- Before each implementation task, run `git status --short` and restrict edits to listed files.
- Do not read, copy, or commit `.env` or secret-bearing configuration.

### 4.2 Small-File Boundary

- New modules should target 200-400 lines.
- Split a module before 400 lines; never permit a new file above 800 lines.
- Keep `evaluate_models_ollama.py`, `driverAgent.py`, and `task_benchmark.py` as compatibility/adaptation surfaces, not homes for new research subsystems.

### 4.3 Scientific Invariants

- A transport failure is not a driving score of zero.
- A model-invalid response is preserved as model behavior and may trigger fixed operational fallback.
- Raw proposal, strict parse, availability, fallback, each shield stage, and executed action are separate fields.
- Scientific trace collection never depends on Rich progress, video, SQLite, or `save_artifacts`.
- Main-factor cells change only policy content, output enforcement, and execution mode.
- Parser mode remains `strict_only`; resolver and deterministic recovery remain disabled in confirmatory cells.
- Main fallback remains fixed `IDLE=1`; unavailable IDLE is a protocol failure.
- No transport downgrade, model substitution, silent think-mode change, or adaptive timeout is allowed.
- Stress V3 test content stays sealed until final protocol binding.

## 5. Owner Gates Before Behavioral Smoke

The user's instruction to continue authorizes implementation planning and test-driven implementation, but these scientific/runtime items remain explicit freeze points:

- provisional model panel: Qwen 0.6B, Llama 1B, Llama 3B, Qwen 8B;
- exact Ollama tags and immutable digests after S1 transport probe;
- backend schema mechanism after the native capability probe;
- two blinded intent annotators plus an adjudicator;
- energy measurement remains supplemental/out of the confirmatory endpoints unless amended;
- margins remain those in the approved evaluation protocol;
- TTC thresholds remain front `<2.0 s` and rear `<2.5 s`, each for two consecutive eligible decisions;
- `3,840` is the confirmatory LLM budget and `6,816` is the pre-amendment simulator cap.

The runner must refuse behavioral smoke if any required owner/runtime value is unresolved. It must never infer a favorable default after results are visible.

## 6. Global TDD Sequence

For every task:

1. Write the named failing unit test.
2. Run only that test and confirm the expected failure reason.
3. Implement the smallest production change.
4. Run the focused test until it passes.
5. Run the task's regression subset.
6. Inspect `git diff --check` and the scoped diff.
7. Record the command and result in the active task log.

Suggested commit boundaries are included in the child plans. Do not create commits unless the owner requests them; the messages document reviewable boundaries.

## 7. Global Verification Commands

### 7.1 Required-Test Mapping

The exact tests preregistered in the evaluation protocol map as follows. Implement these names verbatim so protocol verification can fingerprint them:

| Required test | Test file | Implementation task |
|---|---|---|
| `test_harness_factors_resolve_independently` | `tests/test_harness_config.py` | Runtime Task 1 |
| `test_original_dilu_prompt_hash` | `tests/test_prompt_modules.py` | Runtime Task 2 |
| `test_transport_drift_invalidates_run` | `tests/test_scientific_transport.py` | Runtime Task 4 |
| `test_claim_run_requires_action_trace` | `tests/test_scientific_trace.py` | Runtime Task 6 |
| `test_trace_action_stages_are_consistent` | `tests/test_scientific_trace.py` | Runtime Task 6 |
| `test_stress_v3_directional_mirror_balance` | `tests/test_stress_v3_generator.py` | Stress V3 Task 3 |
| `test_counterfactual_pair_changes_one_factor` | `tests/test_stress_v3_snapshots.py` | Stress V3 Task 2 |
| `test_opposite_direction_cannot_complete` | `tests/test_stress_v3_predicates.py` | Stress V3 Task 4 |
| `test_recovery_cannot_precede_hazard_event` | `tests/test_stress_v3_predicates.py` | Stress V3 Task 4 |
| `test_every_case_has_oracle_solution` | `tests/test_stress_v3_oracles.py` | Stress V3 Tasks 5-7 |
| `test_passive_trap_uses_executed_transition` | `tests/test_stress_v3_predicates.py` | Stress V3 Task 4 |
| `test_task_score_excludes_runtime_penalty` | `tests/test_stress_v3_metrics.py` | Analysis Task 4 |
| `test_missing_metric_is_not_perfect` | `tests/test_stress_v3_metrics.py` | Analysis Task 4 |
| `test_smoke_pass_requires_all_gates` | `tests/test_smoke_verifier.py` | Smoke Task 9 |
| `test_schema_mode_does_not_mask_unavailable_actions` | `tests/test_action_resolution.py` | Runtime Task 3 |
| `test_intent_pair_correct_uses_blinded_preintervention_response` | `tests/test_stress_v3_metrics.py` | Analysis Task 4 |
| `test_generation_seed_is_common_across_matched_cells` | `tests/test_generation_seed.py` | Runtime Task 4 |
| `test_single_generation_requires_determinism_gate` | `tests/test_smoke_stage_plans.py` | Smoke Task 7 |
| `test_sealed_validator_hides_case_content` | `tests/test_stress_v3_sealing.py` | Stress V3 Task 8 |
| `test_protocol_frozen_binds_smoke_and_seal` | `tests/test_protocol_freeze_integration.py` | Smoke Task 10 |

### 7.2 Commands

Run after each child plan:

```powershell
python -m pytest tests/test_reactive_guardrails.py tests/test_timeout_policy.py tests/test_task_benchmark.py tests/test_non_llm_baselines.py tests/test_scientific_reporting.py -q
git diff --check
```

Expected: all selected tests pass; no whitespace errors; no experiment calls are made by unit tests.

Use this staged command order; later commands are invalid until the preceding authorization exists:

```powershell
$protocolId = 'iclr2027_v1'
$benchmarkVersion = 'stress_v3_0'
$smokeDate = '20260713'
$protocolRoot = "results/analysis/stress_v3_protocol_$protocolId"
$validationRoot = "results/analysis/stress_v3_validation/$benchmarkVersion"
$privateValidationRoot = "results/secure/stress_v3/$benchmarkVersion"
$smokeRoot = "results/analysis/stress_v3_smoke_$smokeDate"
$mainRoot = "results/analysis/stress_v3_main_$protocolId"

# Pre-validation: provenance, sensitivity, and science lock.
python scripts/extract_original_dilu_prompt.py --revision 1eed4ed --verify
python scripts/stress_v3_protocol.py sensitivity --spec protocol/iclr2027/stress_v3_science.yaml --output "$protocolRoot/protocol/sensitivity_analysis.csv"
python scripts/stress_v3_protocol.py science-lock --spec protocol/iclr2027/stress_v3_science.yaml --sensitivity "$protocolRoot/protocol/sensitivity_analysis.csv" --output "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json"
python scripts/stress_v3_protocol.py verify --artifact "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json"

# Pre-S1: independently validate and verify the public content-hiding seal.
python scripts/validate_and_seal_stress_v3.py --science-lock "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json" --private-output-root "$privateValidationRoot" --audit-output-root "$validationRoot" --public-output benchmarks/dilu_highway_reactive_stress_v3 --verify

# Authorized S0/S1: produce static/preflight evidence without opening sealed test content.
python scripts/run_stress_v3_smoke.py --stage S0,S1 --science-lock "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json" --seal benchmarks/dilu_highway_reactive_stress_v3/STRESS_V3_SEAL.json --development-cases benchmarks/dilu_highway_reactive_stress_v3/development_cases.json --root "$smokeRoot" --resume

# Pre-S2: bind exact runtime/model values after S1.
python scripts/stress_v3_protocol.py runtime-lock --science-lock "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json" --preflight "$smokeRoot/model_preflight.json" --output "$smokeRoot/RUNTIME_PROTOCOL_LOCK.json"
python scripts/stress_v3_protocol.py verify --artifact "$smokeRoot/RUNTIME_PROTOCOL_LOCK.json"

# Behavioral smoke through S3R raw-response generation.
python scripts/run_stress_v3_smoke.py --stage S1F,S2,S3,S3R --science-lock "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json" --runtime-lock "$smokeRoot/RUNTIME_PROTOCOL_LOCK.json" --development-cases benchmarks/dilu_highway_reactive_stress_v3/development_cases.json --root "$smokeRoot" --resume

# Blinded S3R labeling: two independent files plus adjudication supplied by the annotation team.
python scripts/build_stress_v3_intent_annotations.py export --purpose smoke_determinism --responses "$smokeRoot/s3r_responses.jsonl" --output "$smokeRoot/annotations/s3r_blinded_batch.jsonl"
python scripts/build_stress_v3_intent_annotations.py ingest --purpose smoke_determinism --batch "$smokeRoot/annotations/s3r_blinded_batch.jsonl" --annotator-a "$smokeRoot/annotations/s3r_annotator_a.jsonl" --annotator-b "$smokeRoot/annotations/s3r_annotator_b.jsonl" --adjudication "$smokeRoot/annotations/s3r_adjudication.jsonl" --output "$smokeRoot/annotations/s3r_final_labels.jsonl"
python scripts/run_stress_v3_smoke.py --stage S3R --science-lock "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json" --runtime-lock "$smokeRoot/RUNTIME_PROTOCOL_LOCK.json" --development-cases benchmarks/dilu_highway_reactive_stress_v3/development_cases.json --s3r-annotations "$smokeRoot/annotations/s3r_final_labels.jsonl" --root "$smokeRoot" --resume

# S4 repeatability, followed by its separately blinded semantic labels.
python scripts/run_stress_v3_smoke.py --stage S4 --science-lock "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json" --runtime-lock "$smokeRoot/RUNTIME_PROTOCOL_LOCK.json" --development-cases benchmarks/dilu_highway_reactive_stress_v3/development_cases.json --root "$smokeRoot" --resume
python scripts/build_stress_v3_intent_annotations.py export --purpose smoke_repeatability --responses "$smokeRoot/s4_responses.jsonl" --output "$smokeRoot/annotations/s4_blinded_batch.jsonl"
python scripts/build_stress_v3_intent_annotations.py ingest --purpose smoke_repeatability --batch "$smokeRoot/annotations/s4_blinded_batch.jsonl" --annotator-a "$smokeRoot/annotations/s4_annotator_a.jsonl" --annotator-b "$smokeRoot/annotations/s4_annotator_b.jsonl" --adjudication "$smokeRoot/annotations/s4_adjudication.jsonl" --output "$smokeRoot/annotations/s4_final_labels.jsonl"
python scripts/run_stress_v3_smoke.py --stage S4,S5 --science-lock "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json" --runtime-lock "$smokeRoot/RUNTIME_PROTOCOL_LOCK.json" --development-cases benchmarks/dilu_highway_reactive_stress_v3/development_cases.json --s4-annotations "$smokeRoot/annotations/s4_final_labels.jsonl" --root "$smokeRoot" --resume

# Post-S5: recompute smoke gates, then bind final protocol.
python scripts/verify_smoke.py --bundle "$smokeRoot"
python scripts/stress_v3_protocol.py freeze --science-lock "$protocolRoot/protocol/SCIENTIFIC_PROTOCOL_LOCK.json" --runtime-lock "$smokeRoot/RUNTIME_PROTOCOL_LOCK.json" --seal benchmarks/dilu_highway_reactive_stress_v3/STRESS_V3_SEAL.json --smoke-pass "$smokeRoot/SMOKE_PASS.json" --output "$mainRoot/protocol/PROTOCOL_FROZEN.json"

# Pre-main: independent authorization and zero-execution schedule audit.
python scripts/verify_stress_v3_campaign.py --protocol "$mainRoot/protocol/PROTOCOL_FROZEN.json"
python scripts/run_stress_v3_factorial.py --protocol "$mainRoot/protocol/PROTOCOL_FROZEN.json" --dry-run
```

Expected final dry-run: exactly 3,840 unique planned episode keys, `4 models x 8 conditions x 120 cases`, zero model calls, zero simulator episodes.

## 8. Authorization Chain

The only valid progression is:

```text
SCIENTIFIC_PROTOCOL_LOCK.json
  -> candidate validation
  -> STRESS_V3_SEAL.json
  -> transport-only S1
  -> RUNTIME_PROTOCOL_LOCK.json
  -> S1F-S5 behavioral smoke
  -> verified SMOKE_PASS.json
  -> PROTOCOL_FROZEN.json
  -> sealed main campaign
```

Any hash drift invalidates downstream authorization. `SMOKE_BLOCKED.json` stops progression and cannot coexist with `SMOKE_PASS.json` for the same campaign version.

## 9. Phase Exit Criteria

Phase 6 is complete when:

- all four child plans exist and pass independent plan review;
- every required test in the scientific specification maps to a test file and task;
- create/modify boundaries are explicit for all large legacy files;
- artifact paths, CLIs, locks, and failure stop rules are unambiguous;
- no Ollama request or simulation episode has been run during planning.

Phase 7 may then begin with the runtime plan, not with Stress V3 generation or smoke execution.
