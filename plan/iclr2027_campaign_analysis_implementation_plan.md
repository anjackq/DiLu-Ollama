# ICLR 2027 Main Campaign and Confirmatory Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Schedule, execute, resume, audit, and analyze the authorized 3,840-episode factorial without denominator inflation, artifact drift, or post-hoc endpoint changes.

**Architecture:** A deterministic campaign runner consumes only a verified `PROTOCOL_FROZEN.json`, checkpoints immutable episode attempts, and preserves complete traces. A separate clean-room analysis package joins traces, benchmark metadata, and blinded annotations to produce paired effects, composite gates, diagnostics, tables, and figures from scratch.

**Tech Stack:** Python dataclasses, NumPy, deterministic resampling, CSV/JSON/JSONL, Matplotlib, existing runtime/Stress V3 packages, `unittest`/`pytest`.

---

## Task 1: Deterministic Main Schedule

**Files:**
- Modify: `dilu/runtime/campaign_schedule.py`
- Modify: `dilu/runtime/campaign.py`
- Create: `scripts/run_stress_v3_factorial.py`
- Create: `tests/test_factorial_schedule.py`
- Create: `tests/fixtures/iclr2027/protocol_frozen_minimal.json`

1. Write tests for exactly `4 models x 8 conditions x 120 cases = 3,840` unique episode keys.
2. Interleave conditions within model/pair, counterbalance A/B versus B/A, and serialize a frozen run-order fingerprint.
3. Label cold/warm state and apply the frozen warm-state reset policy.
4. Test matched primary snapshots share generation seeds while post-divergence decisions use case-scoped seeds.
5. Add `--protocol`, `--dry-run`, `--resume`, `--models`, and `--max-episodes`; scientific mode rejects filters that would produce claim-bearing partial contrasts.
6. `--dry-run` verifies locks/seal/smoke, prints counts and hashes, and never opens a model connection or simulator.

```powershell
python -m pytest tests/test_factorial_schedule.py -q
python scripts/run_stress_v3_factorial.py --protocol tests/fixtures/iclr2027/protocol_frozen_minimal.json --dry-run
```

Expected: 3,840 unique keys and 480 episodes per condition across models/cases.

Suggested commit: `feat(campaign): add frozen Stress V3 factorial schedule`

## Task 2: Immutable Episode Attempts and Resume

**Files:**
- Modify: `dilu/runtime/campaign_attempts.py`
- Modify: `dilu/runtime/campaign.py`
- Create: `tests/test_campaign_resume.py`
- Modify: `scripts/run_stress_v3_factorial.py`

1. Write tests for atomic per-episode result, trace, request, and checksum artifacts.
2. Preserve failed and retried attempts; never overwrite attempt 1 with attempt 2.
3. Resume only incomplete episode keys after verifying model digest, condition hash, prompt hash, trace schema, benchmark fingerprint, and code revision.
4. Abort a cell on drift and preserve completed evidence as partial/non-contrast data.
5. Test duplicate completion, missing trace, truncated JSONL, checksum mismatch, and changed run order.
6. Write campaign progress and compute ledger without treating partial cells as complete comparisons.

```powershell
python -m pytest tests/test_campaign_resume.py -q
```

Suggested commit: `feat(campaign): add tamper-evident resume semantics`

## Task 3: Runtime Budget and Fail-Fast Enforcement

**Files:**
- Create: `dilu/runtime/compute_ledger.py`
- Create: `tests/test_compute_ledger.py`
- Modify: `scripts/run_stress_v3_factorial.py`

1. Encode the frozen simulator ledger: 3,840 confirmatory LLM, 176 smoke LLM, 520 smoke non-LLM, 1,560 sealed non-LLM, and 720 candidate-validation episodes.
2. Record 148 direct smoke calls/replays separately from simulator episodes.
3. Test the scheduled simulator total and preregistered cap are both 6,816 (`3,840 + 176 + 520 + 1,560 + 720`); there is no implicit headroom for secondary work.
4. Test secondary/transfer/replication work has zero implicit authorization.
5. Refuse to start the next episode when a budget would be exceeded.
6. Use smoke throughput only to estimate wall-clock/energy; do not alter scientific scheduling.

```powershell
python -m pytest tests/test_compute_ledger.py -q
```

Suggested commit: `feat(campaign): enforce preregistered compute budgets`

## Task 4: Endpoint Extraction and Denominator Audit

**Files:**
- Create: `dilu/runtime/stress_v3/metrics.py`
- Create: `tests/test_stress_v3_metrics.py`

1. Extract strict syntax, unconditional blinded semantic intent, state availability, fixed fallback, execution/shield interventions, task completion, task-only score, and latency from source artifacts.
2. Implement `intent_pair_correct` on the first eligible pre-intervention response for both pair members, requiring disjoint correct labels and an actual label change.
3. Define task-only score exclusively from closed-loop task evidence. Add `test_task_score_excludes_runtime_penalty`: identical task trajectories with changed timeout, fallback, latency, or contract fields must have identical task score.
4. Implement `safety_failure`: crash, front TTC `<2.0 s` for two consecutive decision steps, or rear TTC `<2.5 s` for two consecutive lane-change decision steps, only under positive closing speed. Consecutive means adjacent decision indices; reset the streak on every intervening nonqualifying step, including one where TTC is unavailable/inapplicable.
5. Add adjacency tests proving dangerous steps separated by a safe or ineligible step do not trigger failure.
6. Build common all-snapshot denominators; invalid syntax can still have semantic intent, while `NONE`/`AMBIGUOUS` are incorrect.
7. Treat transport-blocked cells as missing/invalid, never zero. Treat missing metrics as `NA`, never perfect.
8. Produce a denominator audit with expected, observed, excluded, and reason counts for every endpoint/contrast.

```powershell
python -m pytest tests/test_stress_v3_metrics.py -q
```

Suggested commit: `feat(analysis): extract confirmatory Stress V3 endpoints`

## Task 5: Confirmatory Statistics Engine

**Files:**
- Create: `dilu/runtime/stress_v3/statistics.py`
- Create: `tests/test_stress_v3_statistics.py`
- Modify if required: `requirements.txt`

1. Write reference-fixture tests for frozen `cluster_signflip_v1`: matched risk differences, 99,999-draw one-sided margin-shifted sign flips with add-one correction, paired TOST, non-inferiority/superiority bounds, 20,000-draw template-cluster bootstrap, deterministic seeds, all-zero p=1, and diagnostic zero-discordance McNemar p=1.
2. Average nuisance cells within case/pair before inference; never treat repeated factors or decision steps as independent.
3. Implement 12 model-level IUT gates: each composite p-value is `max(component p-values)`.
4. Apply one global Holm correction over `H1-M1..M4`, `H2-M1..M4`, and `H3-M1..M4` at familywise alpha 0.05.
5. Test H1 syntax `+0.20` plus intent equivalence `[-0.05,+0.05]`; H2 responsiveness `+0.10` plus safety non-inferiority `+0.02`; H3 safety `-0.05` plus intervention `+0.10`.
6. Compute machine-readable paper-level claim gates: H1, H2, or H3 is supported only when at least three of four corresponding model gates pass both components and global Holm. This is a frozen counting rule over corrected model gates, not a second uncorrected test.
7. Add fixtures where 2/4 is false, 3/4 is true, and a raw-passing but Holm-failing model does not count. Preserve mixed outcomes independently for H1/H2/H3.
8. Report effects and intervals beside p-values. Four-model averages and interactions remain descriptive/exploratory.
9. Add a dependency only if the frozen statistical method requires it; pin and record the version. Do not rely on an undeclared transitive package.

```powershell
python -m pytest tests/test_stress_v3_statistics.py -q
```

Suggested commit: `feat(analysis): implement preregistered paired inference`

## Task 6: Degeneracy and Mechanism Diagnostics

**Files:**
- Create: `dilu/runtime/stress_v3/diagnostics.py`
- Create: `tests/test_stress_v3_diagnostics.py`

1. Compute agreement/gap to fixed LEFT/RIGHT/IDLE/FASTER/SLOWER policies.
2. Report action distributions and entropy by causal member, direction, category, and condition.
3. Build proposed/fallback/unshielded/shielded/executed action confusion matrices.
4. Compute passive-stop, lane-change-refusal, action-flapping, missed causal flip, fallback dependence, and shield dependence.
5. Test the same scalar score can mask different behavior and cannot be used as competence proof.
6. Keep all category and model-scaling comparisons descriptive unless separately preregistered.

```powershell
python -m pytest tests/test_stress_v3_diagnostics.py -q
```

Suggested commit: `feat(analysis): add shortcut and intervention diagnostics`

## Task 7: Clean-Room Reporting Pipeline

**Files:**
- Create: `dilu/runtime/stress_v3/reporting.py`
- Create: `scripts/analyze_stress_v3_factorial.py`
- Create: `tests/test_stress_v3_reporting.py`
- Leave unchanged: `dilu/runtime/scientific_reporting.py`

1. Read only frozen locks, sealed benchmark metadata, immutable campaign artifacts, and completed blinded annotations. Refuse H1/H2 analysis unless the batch hash verifies and coverage is exactly 3,840/3,840 with two annotator inputs plus adjudication.
2. Generate at minimum:
   - `denominator_audit.csv`;
   - `endpoint_summary.csv`;
   - `paired_effects.csv`;
   - `composite_gates.csv`;
   - `paper_claim_gates.json`;
   - `holm_family.csv`;
   - `category_diagnostics.csv`;
   - `fixed_policy_agreement.csv`;
   - `failure_taxonomy.csv`;
   - `runtime_summary.csv`;
   - `analysis-report.md`;
   - `stats-appendix.md`;
   - `MANIFEST.md`.
3. Generate manuscript tables/figures only from these machine-produced CSV files.
4. Add a `--check` mode that rebuilds in a temporary directory and compares hashes without mutating the evidence bundle.
5. Keep legacy descriptive/composite reporting untouched; V3 confirmatory analysis uses only the new package.

```powershell
python -m pytest tests/test_stress_v3_reporting.py -q
```

Suggested commit: `feat(analysis): regenerate Stress V3 evidence from raw traces`

## Task 8: Tiny Synthetic Campaign Integration

**Files:**
- Create: `tests/test_stress_v3_campaign_pipeline.py`

1. Build two mock models, eight conditions, four paired cases, deterministic raw responses, and both execution modes.
2. Run schedule, checkpoint, interrupt/resume, trace validation, annotation join, endpoint extraction, statistics, Holm, and reporting.
3. Inject one transport-blocked cell and verify it becomes invalid/missing rather than a zero-score observation.
4. Regenerate analysis twice and compare hashes.
5. Tamper with one trace and verify analysis refuses to run.

```powershell
python -m pytest tests/test_stress_v3_campaign_pipeline.py -q
```

Suggested commit: `test(campaign): verify frozen end-to-end analysis pipeline`

## Task 9: Pre-Run Verification Checklist

**Files:**
- Create: `scripts/verify_stress_v3_campaign.py`
- Create: `tests/test_campaign_verifier.py`

1. Verify science/runtime/final locks, smoke pass, seal, model digests, prompt/config/trace hashes, run order, budget, and private sealed access.
2. Verify all test modules and their source fingerprint match the final lock.
3. Verify no PASS/BLOCKED conflict and no stale partial campaign is mixed into the new version.
4. Print an explicit authorization verdict and reasons. The verifier never launches the campaign itself.

```powershell
python -m pytest tests/test_campaign_verifier.py -q
python scripts/verify_stress_v3_campaign.py --protocol tests/fixtures/iclr2027/protocol_frozen_minimal.json
```

Expected before real execution: `AUTHORIZED`, 3,840 planned episodes, all hashes matching. Any other result blocks execution.

Suggested commit: `feat(campaign): add independent main-run authorization verifier`

## Task 10: Real Execution Order After Explicit Authorization

No command in this task is run during implementation planning. Once the owner explicitly authorizes execution and the verifier says `AUTHORIZED`:

```powershell
$protocolId = 'iclr2027_v1'
$mainRoot = "results/analysis/stress_v3_main_$protocolId"

python scripts/run_stress_v3_factorial.py `
  --protocol "$mainRoot/protocol/PROTOCOL_FROZEN.json" `
  --resume

# Pause here until two independent annotator files and blinded adjudication are complete.
python scripts/build_stress_v3_intent_annotations.py export `
  --purpose main_endpoint `
  --responses "$mainRoot/campaign/traces/decision_traces.jsonl" `
  --output "$mainRoot/annotations/main_blinded_batch.jsonl"

python scripts/build_stress_v3_intent_annotations.py ingest `
  --purpose main_endpoint `
  --batch "$mainRoot/annotations/main_blinded_batch.jsonl" `
  --annotator-a "$mainRoot/annotations/main_annotator_a.jsonl" `
  --annotator-b "$mainRoot/annotations/main_annotator_b.jsonl" `
  --adjudication "$mainRoot/annotations/main_adjudication.jsonl" `
  --output "$mainRoot/annotations/main_final_labels.jsonl"

python scripts/build_stress_v3_intent_annotations.py verify `
  --purpose main_endpoint `
  --batch "$mainRoot/annotations/main_blinded_batch.jsonl" `
  --final-labels "$mainRoot/annotations/main_final_labels.jsonl" `
  --expected-count 3840 `
  --output "$mainRoot/annotations/main_annotation_coverage.json"

python scripts/analyze_stress_v3_factorial.py `
  --protocol "$mainRoot/protocol/PROTOCOL_FROZEN.json" `
  --campaign "$mainRoot/campaign" `
  --annotations "$mainRoot/annotations/main_final_labels.jsonl" `
  --output "$mainRoot/analysis"

python scripts/analyze_stress_v3_factorial.py `
  --protocol "$mainRoot/protocol/PROTOCOL_FROZEN.json" `
  --campaign "$mainRoot/campaign" `
  --annotations "$mainRoot/annotations/main_final_labels.jsonl" `
  --output "$mainRoot/analysis" `
  --check
```

Stop immediately on drift, trace failure, budget overrun, or a blocked cell. Preserve partial evidence and diagnose; do not substitute a model, relax a gate, or open a new endpoint.

## Definition of Done

- Dry-run contains exactly 3,840 unique claim-bearing episodes.
- Campaign attempts are immutable, resumable, and drift-checked.
- Endpoint denominators and missingness are auditable.
- Main blinded annotation coverage is exactly 3,840/3,840 before H1/H2 analysis.
- All 12 gates and Holm correction reproduce from fixtures.
- Reporting regenerates deterministically from raw evidence.
- Main execution occurs only after verified locks, seal, smoke pass, and explicit owner authorization.
