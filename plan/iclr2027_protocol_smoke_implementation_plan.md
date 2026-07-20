# ICLR 2027 Protocol Locks and Smoke Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Make scientific authorization machine-verifiable and execute a staged smoke investigation that separates infrastructure/wiring failures from meaningful poor model behavior before the sealed 3,840-episode run.

**Architecture:** Canonical lock documents bind source specifications, runtime constants, benchmark seal, smoke evidence, and run order. A stage runner writes immutable source artifacts; an independent verifier recomputes gates and emits exactly one PASS or BLOCKED result.

**Tech Stack:** Python dataclasses, JSON Schema, SHA-256, YAML/JSON/JSONL/CSV, mocked fault fixtures, existing evaluator/baseline adapters, PowerShell.

---

## Smoke Artifact Contract

Under the exact dated root `results/analysis/stress_v3_smoke_<YYYYMMDD>/`, the stage runner must produce these source artifacts before independent verification:

| Artifact | Producer | Contract / expected denominator | Independent check |
|---|---|---|---|
| `resolved_conditions.csv` | S0 / Task 4 | 8 unique factorial rows | Task 9 recomputes configs/diffs |
| `model_preflight.json` | S0-S1 / Tasks 4-5 | 4 models, 5 probes/model = 20 probe records | Task 9 verifies tags/digests/settings |
| `decision_traces.jsonl` | S2-S4 / Task 7 | all decisions from 176 LLM simulation episodes | trace schema plus Task 9 coverage join |
| `trace_integrity.csv` | S1F-S4 / Tasks 6-7 | one integrity row per LLM episode plus injected branch | Task 9 recomputes stage transitions |
| `fault_injection_audit.csv` | S1F / Task 6 | exact mandatory branch-ID set | Task 9 set equality, no missing branch |
| `counterfactual_pair_audit.csv` | S5 / Task 8 | 20 development pairs | Task 9 recomputes pair diffs/gates |
| `oracle_validation.csv` | S5 / Task 8 | 2 oracles x 40 cases = 80 rows | Task 9 recomputes oracle thresholds |
| `runtime_repeatability.csv` | S4 / Task 7 | 16 groups, each backed by 3 repeats/48 episodes | Task 9 recomputes agreement/CV |
| `generation_determinism.csv` | S3R / Task 7 | 64 request groups, each backed by S3 + 2 replays | Task 9 verifies 192 blinded final labels |
| `failure_triage.md` | all / Task 9 | every blocked/unclassified failure, or explicit none | Task 9 checks error-ledger coverage |
| `MANIFEST.md` | all / Tasks 3-9 | one manifest with every artifact checksum/command | Task 9 recomputes all checksums |
| exactly one PASS/BLOCKED JSON | Task 9 | one authorization verdict | schema and mutual-exclusion check |

Every table row must retain its source artifact/checksum. CSV/JSON field contracts live in `schemas/iclr2027/smoke_artifacts.schema.json`; the authorization gate schema remains at the specification-mandated path `schemas/smoke_gate.schema.json`. `MANIFEST.md` records commands, versions, hashes, budgets, and deviations with secrets/local identity paths redacted.

## Task 1: Canonical Lock and Manifest Core

**Files:**
- Create: `dilu/runtime/protocol_locks.py`
- Create: `schemas/iclr2027/scientific_protocol_lock.schema.json`
- Create: `schemas/iclr2027/runtime_protocol_lock.schema.json`
- Create: `schemas/iclr2027/protocol_frozen.schema.json`
- Create: `tests/test_protocol_locks.py`
- Modify: `dilu/runtime/path_utils.py`

1. Write tests for canonical serialization, stable hashes, atomic writes, relative artifact paths, and checksum verification.
2. Implement typed builders/verifiers for science, runtime, and final locks.
3. Test lock order: science before validation; runtime after S1 and before behavioral smoke; final only after seal plus verified smoke.
4. Test any prompt/config/model/transport/trace/benchmark/test-fingerprint drift invalidates dependent locks.
5. Test final freeze can only bind existing contrast definitions; it cannot add endpoints or alter margins.
6. Extend path utilities with generic JSON/JSONL atomic helpers without changing current call behavior.

```powershell
python -m pytest tests/test_protocol_locks.py -q
```

Suggested commit: `feat(protocol): add canonical scientific lock chain`

## Task 2: Science Specification and Sensitivity Freeze

**Files:**
- Create: `protocol/iclr2027/stress_v3_science.yaml`
- Create: `scripts/stress_v3_protocol.py`
- Create: `tests/test_scientific_protocol.py`

1. Encode the endpoint registry, 12 composite gates, denominators, directions, margins, alpha, Holm family, IUT `max(p)` rule, annotation protocol, factor semantics, candidate rules, and exact `3,840 + 176 + 520 + 1,560 + 720 = 6,816` simulator ledger.
2. Freeze analysis method `cluster_signflip_v1`: nuisance-averaged within-cluster differences; 99,999 one-sided margin-shifted sign flips with add-one correction; paired TOST as two one-sided sign-flip tests; 20,000 template-cluster bootstrap draws; 95% effect intervals and 90% equivalence interval; all-zero shifted differences return p=1; seeds derive from science-lock preimage, contrast ID, and method version. Exact McNemar/binomial with zero-discordance p=1 is diagnostic only.
3. Freeze annotation agreement as raw agreement plus Cohen's kappa over the seven nominal labels. If expected agreement is one, serialize kappa as `NA_DEGENERATE` and still report raw agreement/coverage; agreement is descriptive, while complete two-annotator/adjudication coverage is mandatory.
4. Resolve science slots as immutable `M1..M4` plus provisional panel identities only. Exact Ollama tags/digests are intentionally absent from the science lock and must be bound to those slots by `RUNTIME_PROTOCOL_LOCK.json` after S1.
5. Implement a `sensitivity` command that writes `sensitivity_analysis.csv` and `.md` from the science spec/parameter grid only: risk-difference MDEs, discordant-pair requirements, continuous MDEs, Holm/four-model power, and category uncertainty. Reject any development-case, episode, trace, score, or outcome input.
6. Test the science lock refuses missing model slots, annotation workflow/statistic, analysis constants, gate definitions, sensitivity hash, or inconsistent budget arithmetic.
7. Implement commands `sensitivity`, `science-lock`, `verify --artifact`, `runtime-lock`, and `freeze` with the exact inputs shown in the implementation index.
8. Record owner-confirmed values and specification hashes; do not put local identity paths or secrets into reviewer-facing manifests.

```powershell
python -m pytest tests/test_scientific_protocol.py -q
python scripts/stress_v3_protocol.py sensitivity --spec protocol/iclr2027/stress_v3_science.yaml --output temp/sensitivity_analysis.csv --dry-run
python scripts/stress_v3_protocol.py science-lock --spec protocol/iclr2027/stress_v3_science.yaml --dry-run
```

Expected: no artifact is written by `--dry-run`; all 12 gate IDs and six component-test definitions resolve; attempts to pass outcome data to `sensitivity` fail.

Suggested commit: `feat(protocol): encode confirmatory Stress V3 science lock`

## Task 3: Campaign Skeleton for Smoke Stages

**Files:**
- Create: `dilu/runtime/campaign_schedule.py`
- Create: `dilu/runtime/campaign_attempts.py`
- Create: `dilu/runtime/campaign.py`
- Create: `tests/test_campaign_schedule.py`
- Create: `tests/test_campaign_attempts.py`
- Create: `tests/test_campaign.py`
- Create: `scripts/run_stress_v3_smoke.py`

1. Put stage specs, deterministic episode/request keys, and interleaved ordering in `campaign_schedule.py`.
2. Put atomic checkpoints, append-only attempts, checksums, and versioned resume in `campaign_attempts.py`.
3. Keep `campaign.py` as a thin facade joining schedule and attempt store; each new module must remain below 400 lines.
4. Test duplicate keys, partial writes, interrupted stage resume, and artifact checksum drift.
5. Test a retry receives a new attempt ID and never overwrites the original attempt.
6. Add `--stage`, `--root`, `--dry-run`, `--resume`, and `--force-new-version`; prohibit destructive overwrite.
7. A dry-run prints exact direct-call and simulator budgets without importing or contacting Ollama.

```powershell
python -m pytest tests/test_campaign_schedule.py tests/test_campaign_attempts.py tests/test_campaign.py -q
python scripts/run_stress_v3_smoke.py --stage S0,S1,S1F,S2,S3,S3R,S4,S5 --root temp/stress_v3_smoke_plan --dry-run
```

Expected: `176` LLM simulator episodes, `520` non-LLM simulator episodes, and `148` direct calls/replays.

Suggested commit: `feat(smoke): add resumable staged campaign skeleton`

## Task 4: S0 Static Preflight

**Files:**
- Create: `dilu/runtime/smoke_gates.py`
- Create: `schemas/smoke_gate.schema.json`
- Create: `schemas/iclr2027/smoke_artifacts.schema.json`
- Create: `tests/test_smoke_gates.py`
- Modify: `scripts/run_stress_v3_smoke.py`

1. Define gate records with ID, inputs, denominator, exact predicate, expected/observed values, severity, pass flag, and reason.
2. Recompute eight condition specs and validate factor isolation, prompt/config hashes, trace mode, source revision, environment inventory, and science lock.
3. Record Python/packages/Ollama/OS/hardware/model metadata using redacted paths and no environment secrets.
4. Reject missing tags, aliases, unresolved factors, dirty hash inputs not recorded in the manifest, and secret-like keys.
5. Test `resolved_conditions.csv` has exactly eight unique rows and `model_preflight.json` reserves exactly four model records before S1 expands them to 20 probes.
6. Test S0 uses zero model calls and zero simulation episodes.

```powershell
python -m pytest tests/test_smoke_gates.py -q
```

Suggested commit: `feat(smoke): implement static preflight gates`

## Task 5: S1 Transport Probe and Runtime Lock

**Files:**
- Create: `dilu/runtime/transport_probe.py`
- Create: `tests/test_transport_probe.py`
- Modify: `scripts/run_stress_v3_smoke.py`
- Modify: `scripts/stress_v3_protocol.py`

1. Use the scientific native client to implement five probe types/model: health, think mode, free output, schema output, and warm latency.
2. Test exact tag/digest, native endpoint, request options, seed, effective think mode, stop reason, raw output, tokens, cold/warm latency, and error class are captured.
3. Test transport success and contract success are separate.
4. Test missing/unsupported schema blocks the cell rather than downgrading.
5. Generate `RUNTIME_PROTOCOL_LOCK.json` only after all four provisional models pass transport; freeze digests, schema mechanism, seeds, timeout/retry, fallback, shield constants, trace schema, and software versions.

Unit tests mock the backend. Real S1 execution is a later explicitly authorized operation.

```powershell
python -m pytest tests/test_transport_probe.py -q
```

Suggested commit: `feat(smoke): add native transport probe and runtime lock`

## Task 6: S1F Deterministic Fault Injection

**Files:**
- Create: `dilu/runtime/fault_injection.py`
- Create: `tests/test_fault_injection.py`
- Modify: `scripts/run_stress_v3_smoke.py`

1. Force strict-valid, recoverable, invalid, empty, timeout, unavailable action, fixed fallback, every shield override, transport drift, and trace-write failure.
2. Verify error classes, attempt preservation, action-stage order, availability, fallback, shield provenance, and fail-fast behavior.
3. Write `fault_injection_audit.csv` with every mandatory branch exactly represented.
4. Test this stage needs no model and no simulator.

```powershell
python -m pytest tests/test_fault_injection.py -q
```

Suggested commit: `test(smoke): cover all runtime branches by fault injection`

## Task 7: S2-S4 Behavioral Wiring and Repeatability

**Files:**
- Create: `tests/test_smoke_stage_plans.py`
- Modify: `scripts/run_stress_v3_smoke.py`

1. Encode S2 as Qwen 0.6B, eight cells, four development pairs/eight cases: 64 episodes.
2. Encode S3 as four models, eight cells, one complete development pair/two cases: 64 episodes.
3. Encode S3R as the archived S3 response plus two extra primary-snapshot replays for each of 64 model/cell/case requests: 128 direct calls; preserve 192 raw responses and automated syntax results.
4. Build a purpose-tagged, metadata-blinded S3R annotation batch containing 192 response items with randomized annotation IDs. Ingest two independent annotator files plus blinded adjudication through the Stress V3 annotation module.
5. Generate 64 `generation_determinism.csv` groups only after all 192 responses have final labels; compare syntax status and semantic label across all three responses. Missing labels, exposed model/condition metadata, or incomplete adjudication blocks smoke.
6. Encode S4 as Qwen 0.6B and Llama 1B, two endpoint conditions, four cases, and three campaigns: 48 simulator episodes and no additional direct replay budget.
7. Export the 48 S4 first-snapshot responses as a separate `smoke_repeatability` blinded batch; require two annotators, adjudication, 48/48 final labels, and batch-hash verification before computing semantic agreement.
8. Freeze S4 denominators: 48 source episodes; 16 model/endpoint/case triplicate groups for first-snapshot syntax/semantic agreement and executed-action-sequence/task-outcome agreement; four model/endpoint groups for latency stability across three campaigns. Cold starts are labeled only by explicit load boundary or positive backend load duration, stay in behavioral agreement, and are excluded only from latency CV/ratio.
9. Add fixture tests for the exact S4 gates: zero config/prompt/digest/transport/trace drift; zero unclassified errors; 16/16 first-snapshot syntax and semantic agreement; 16/16 executed-sequence and task-outcome agreement; latency median CV `<=0.25` in all four groups; within-group maximum/minimum campaign-median ratio `<=2.0` in all four groups.
10. Test model behavior may be invalid/passive/unsafe without infrastructure gate failure; missing traces, drift, transport faults, incomplete annotations, and non-repeatable generation block.
11. Test no stage can inspect sealed test content.

```powershell
python -m pytest tests/test_smoke_stage_plans.py -q
```

Suggested commit: `feat(smoke): encode factorial and repeatability stages`

## Task 8: S5 Development Benchmark and Oracle Smoke

**Files:**
- Create: `tests/test_smoke_benchmark_gates.py`
- Modify: `scripts/run_stress_v3_smoke.py`
- Modify: `evaluate_non_llm_baselines.py`

1. Encode the declared 520 non-LLM development episodes with explicit shield modes: two oracles, ten fixed execution variants, and one observation-only rule controller over 40 cases.
2. Recompute direction balance, event ordering, oracle solvability, mirrored consistency, and shortcut limits from source trajectories.
3. Test privileged trajectories pass independent replay for 40/40 development cases and observable oracle intent is correct for every one of 20 pairs.
4. Test observable-oracle task completion is `>=95%` over 40 cases and `>=85%` over each four-case category denominator.
5. Test each fixed variant has pair intent correctness zero, `success_A + success_B <=1` for every pair, completion `<=25%` over 40 cases, and completion `<=50%` over each category.
6. Test observable/privileged oracle labels remain distinct from competing-agent results and every threshold uses its exact declared denominator.
7. Any benchmark gate failure emits BLOCKED and requires a new suite version; it never lowers a threshold.

```powershell
python -m pytest tests/test_smoke_benchmark_gates.py -q
```

Suggested commit: `feat(smoke): add development benchmark validity gates`

## Task 9: Independent Smoke Verifier

**Files:**
- Create: `scripts/verify_smoke.py`
- Create: `tests/test_smoke_verifier.py`

1. Recompute every gate from CSV/JSONL/source manifests; do not trust stored pass flags.
2. Validate schema, artifact checksums, the artifact-matrix row counts, denominators, budgets, all core cells, trace integrity, repeatability, complete S3R annotations, generation determinism, and benchmark validity.
3. Emit exactly one of `SMOKE_PASS.json` or `SMOKE_BLOCKED.json` atomically.
4. Test tampering, missing rows, duplicate attempts, stale locks, and coexistence of PASS/BLOCKED fail verification.
5. Record blocking layer, evidence, repairs attempted, and required decision without converting failures into fallback driving outcomes.

```powershell
python -m pytest tests/test_smoke_verifier.py -q
```

Suggested commit: `feat(smoke): independently verify go-no-go authorization`

## Task 10: Final Protocol Binding Dry Run

**Files:**
- Create: `tests/test_protocol_freeze_integration.py`
- Modify: `scripts/stress_v3_protocol.py`

1. Build a synthetic complete bundle with science lock, seal, runtime lock, and smoke pass.
2. Freeze and verify hashes, run order, code/test fingerprint, and access policy.
3. Test any changed input invalidates `PROTOCOL_FROZEN.json`.
4. Test final binding cannot open sealed cases during `--dry-run`.

```powershell
python -m pytest tests/test_protocol_freeze_integration.py -q
git diff --check
```

## Definition of Done

- Lock ordering and hashes are machine-verifiable.
- Smoke stage budgets and stop rules are executable.
- Every runtime branch is covered by deterministic fault injection.
- PASS/BLOCKED is independently recomputed and mutually exclusive.
- A verified final freeze can be produced from synthetic fixtures.
- Real S1/S2-S5 execution has not begun without explicit owner authorization.
