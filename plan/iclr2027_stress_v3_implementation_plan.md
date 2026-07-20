# ICLR 2027 Counterfactual Stress V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Build a deterministic paired benchmark that rejects constant-action shortcuts, proves mechanical solvability, preserves causal pairing, and withholds the 120-case test split until protocol freeze.

**Architecture:** Add an isolated `dilu.runtime.stress_v3` package for schema, snapshots, generation, predicates, oracles, independent validation, sealing, annotation, and metrics. Reuse low-level highway scenario construction, but do not reuse the benchmark success function inside the independent checker.

**Tech Stack:** Python frozen dataclasses, JSON Schema, NumPy, Gymnasium/highway-env, existing scenario utilities, JSON/JSONL/CSV, `unittest`/`pytest`.

---

## Task 1: Pair and Case Schema

**Files:**
- Create: `dilu/runtime/stress_v3/__init__.py`
- Create: `dilu/runtime/stress_v3/schema.py`
- Create: `schemas/iclr2027/stress_v3_pair.schema.json`
- Create: `tests/test_stress_v3_schema.py`
- Modify: `requirements.txt`
- Modify later: `dilu/runtime/task_benchmark.py`

1. Write tests for frozen `IntentLabel`, `DecisionSnapshotKey`, `CounterfactualCase`, and `CounterfactualPair`.
2. Require pair/template/category IDs, named intervention and A/B values, nuisance/shared-prefix hashes, decision window, valid action/semantic sets, required event IDs, and endpoint predicates.
3. Reject duplicate IDs, overlapping semantic sets, identical A/B interventions, missing snapshot keys, ambiguous direction, and undeclared differences.
4. Implement canonical JSON serialization and stable benchmark/pair/case fingerprints.
5. Add explicit `jsonschema>=4.21,<5` to `requirements.txt`; do not rely on a transitive install.
6. Add JSON Schema parity tests between Python validation and the public schema.

```powershell
python -m pytest tests/test_stress_v3_schema.py -q
```

Suggested commit: `feat(benchmark): add Stress V3 paired schema`

## Task 2: Canonical Snapshots and Single-Intervention Audit

**Files:**
- Create: `dilu/runtime/stress_v3/snapshots.py`
- Create: `tests/test_stress_v3_snapshots.py`
- Modify: `dilu/runtime/highway_scenario_spec.py`

1. Write tests for `capture_observable_snapshot()`, `nuisance_state_hash()`, `shared_prefix_hash()`, and `canonical_scenario_diff()`.
2. Ensure stable vehicle and event identities survive scenario application/replay.
3. Test pair members share all nuisance fields and prefix state while exactly one named causal field changes.
4. Test mirrored cases preserve distance/speed magnitudes and swap only declared directional fields.
5. Keep privileged fields out of the observable snapshot API.

```powershell
python -m pytest tests/test_stress_v3_snapshots.py -q
```

Suggested commit: `feat(benchmark): add canonical counterfactual snapshots`

## Task 3: Deterministic 240-Case Candidate Generator

**Files:**
- Create: `dilu/runtime/stress_v3/generator.py`
- Create: `scripts/generate_dilu_highway_reactive_stress_v3.py`
- Create: `tests/test_stress_v3_generator.py`
- Read/reuse only: `scripts/generate_dilu_highway_reactive_stress_v2.py`

1. Write tests for exactly 10 categories, 12 pairs/category, two members/pair, and 240 candidate cases.
2. Add an explicit 10-row `CATEGORY_CONTRACTS` registry matching the approved category-to-pair table: causal transform, valid semantic contrast, required events, target identity/direction, closed-loop evidence, and the one candidate-specific adversarial shortcut policy.
3. Implement the six transform families: mirrored opening, maneuver/wait, faster/slower, true/false opening, mirrored escape, and recovery/control, and map every category to exactly one declared family/contract.
4. Assign the exact frozen order per category: `dev_1`, `dev_2`, `nuisance_1`, `nuisance_2`, `mirror_1`, `mirror_2`, `event_1`, `event_2`, followed by one reserve for each role. A role with more failures than its single reserve invalidates that category/version.
5. Test global and within-direction action balance, unique seeds, pair-member nuisance equality, category-contract completeness, and disjoint parameter partitions.
6. Test two regenerations are byte-identical under the same generator version/master seed.
7. Implement `--write`, `--check`, and `--output` modes. `--check` must not modify files.

```powershell
python -m pytest tests/test_stress_v3_generator.py -q
python scripts/generate_dilu_highway_reactive_stress_v3.py --check
```

Expected: candidate fingerprint is stable and no simulator is invoked by `--check`.

Suggested commit: `feat(benchmark): generate deterministic Stress V3 candidates`

## Task 4: Correct Temporal and Directional Predicates

**Files:**
- Create: `dilu/runtime/stress_v3/predicates.py`
- Create: `tests/test_stress_v3_predicates.py`
- Modify: `dilu/runtime/task_benchmark.py`

1. Write failing fixtures for wrong-direction lane change, non-target transition, wrong lead vehicle, early maneuver, late maneuver, recovery before hazard, recovery before clear, and unrelated event completion.
2. Implement `TemporalEvidenceLedger` and `StressV3EpisodeEvaluator`.
3. Require executed target-lane transitions, named lead passes, validated opportunity windows, and `hazard < response < clear < recovery` ordering.
4. Define passive-trap escape from executed transitions/progress only. Add `test_passive_trap_uses_executed_transition`: a proposed or parsed lane change that is unavailable, blocked, replaced by fallback, or overridden by a shield cannot satisfy the trap.
5. Count required events by identifier, not by any applied event.
6. Route only V3 cases through the new evaluator; keep Stress V1/V2 behavior unchanged.
7. Update case-set normalization/fingerprinting to preserve all V3 pair and snapshot fields.

```powershell
python -m pytest tests/test_stress_v3_predicates.py tests/test_task_benchmark.py -q
```

Suggested commit: `fix(benchmark): enforce Stress V3 causal predicates`

## Task 5: Observable and Privileged Oracles

**Files:**
- Create: `dilu/runtime/stress_v3/oracles.py`
- Create: `tests/test_stress_v3_oracles.py`
- Modify later: `evaluate_non_llm_baselines.py`

1. Define `ObservableStateOracle.decide(snapshot)` so its API cannot receive raw case metadata, category labels, future events, or hidden criteria.
2. Define `PrivilegedEventOracle.decide(snapshot, event_plan)` solely for reference trajectory generation.
3. Write tests for all semantic contrast families and mirrored decisions.
4. Test observable oracle snapshot intent is correct for both members of every accepted pair fixture.
5. Test privileged oracle emits a complete reference plan, but is never tagged as a competing agent.
6. Keep both policies deterministic under explicit seed/config.

```powershell
python -m pytest tests/test_stress_v3_oracles.py -q
```

Suggested commit: `feat(benchmark): add independent Stress V3 oracles`

## Task 6: Independent Replay and Label Validators

**Files:**
- Create: `dilu/runtime/stress_v3/replay_validator.py`
- Create: `dilu/runtime/stress_v3/label_validator.py`
- Create: `tests/test_stress_v3_validation.py`

1. Test `validate_replay()` consumes only trajectory/event records and cannot import or call `StressV3EpisodeEvaluator`.
2. Check target identity, direction, transition, pass, event order, decision window, terminal safety, and no success before required events.
3. Implement independent snapshot-label validation for one named intervention, disjoint semantic sets, and oracle `intent_pair_correct=1`.
4. Add mutation tests: corrupt target ID, swap direction, reorder events, alter nuisance state, and tamper with valid semantic sets; each must fail for the intended reason.

```powershell
python -m pytest tests/test_stress_v3_validation.py -q
```

Suggested commit: `test(benchmark): add independent replay and label checks`

## Task 7: Candidate Validation and Shortcut Matrix

**Files:**
- Create: `dilu/runtime/stress_v3/validation.py`
- Create: `tests/test_stress_v3_shortcuts.py`
- Modify: `evaluate_non_llm_baselines.py`

1. Implement an oracle-policy protocol adapter that preserves proposed/executed actions and complete event trajectories.
2. Evaluate observable oracle, privileged oracle, and one template-declared adversarial shortcut policy on every candidate: exactly 720 candidate-validation episodes (`3 x 240`).
3. Apply all mechanical/oracle/candidate-shortcut gates in frozen order. Fill the eight slots/category by the first passing same-role candidate: two development, two held-out nuisance, two mirrored, and two held-out event composition. Serialize every attempted/rejected pair and never choose across roles opportunistically.
4. Before emitting the public seal, run the full non-LLM shortcut matrix on the withheld provisional 120-case test split: 10 fixed execution variants x 120 = 1,200, plus seeded random, observation-only rule, and IDM/MOBIL-style controllers = 360. Reuse selected oracle trajectories. Total sealed diagnostics: exactly 1,560 episodes.
5. Test every fixed policy has `intent_pair_correct=0`, cannot complete both members of a pair, stays at or below 25% overall and 50% per category. Keep shielded/unshielded labels explicit.
6. Enforce observable-oracle completion at least 95% overall and 85% per category, with intent correctness on every accepted pair.
7. A full-matrix suite-level shortcut failure invalidates the complete benchmark version; it does not trigger pair replacement after provisional test outcomes are observed.

```powershell
python -m pytest tests/test_stress_v3_shortcuts.py tests/test_non_llm_baselines.py -q
```

Suggested commit: `feat(benchmark): validate solvability and reject shortcuts`

## Task 8: Development/Test Split and Seal

**Files:**
- Create: `dilu/runtime/stress_v3/sealing.py`
- Create: `scripts/validate_and_seal_stress_v3.py`
- Create: `schemas/iclr2027/stress_v3_seal.schema.json`
- Create: `tests/test_stress_v3_sealing.py`
- Create: `scripts/package_stress_v3_reviewer_artifacts.py`
- Verify: `.gitignore` (modify only if `results/secure/` is not already covered)
- Generate later: `benchmarks/dilu_highway_reactive_stress_v3/development_cases.json`
- Generate later: `benchmarks/dilu_highway_reactive_stress_v3/STRESS_V3_SEAL.json`

1. Within each role (`development`, `heldout_nuisance`, `mirrored`, `heldout_event`), select the first two passing candidates in that role's frozen order; never select across roles. The result is two development and six sealed pairs/category.
2. Verify 40 development cases and 120 sealed cases, with disjoint seeds/parameters. In every category require exactly two development pairs plus two held-out-nuisance, two mirrored, and two held-out-event-composition sealed pairs.
3. Keep sealed content as a withheld private artifact at `results/secure/stress_v3/<version>/sealed_cases.private.json`; verify the existing `results/` ignore rule with `git check-ignore`, and modify `.gitignore` only if that protection is absent.
4. Public seal exposes only hashes, counts, quotas, rejection summaries, checker hashes, and access policy. Test that it contains no case IDs, scenario parameters, seeds, instructions, valid semantic/action sets, or recoverable per-case fingerprints.
5. Add tamper detection and an append-only access/opening log. The authorized reader must durably append the access intent before reading private case bytes and must require a verified `PROTOCOL_FROZEN.json` for test opening.
6. Add reviewer packaging that allowlists development cases, public schemas, seal summary, and documentation; scan the package against all private case IDs/parameter values and fail on any match.
7. Make the CLI refuse sealing without a verified `SCIENTIFIC_PROTOCOL_LOCK.json`.
8. Give the CLI three required destinations: ignored `--private-output-root` for candidate trajectories/rejections/sealed bytes, `--audit-output-root` for aggregate redacted validation summaries, and `--public-output` for development cases plus the content-hiding seal. Never copy private validation rows into either visible destination.
9. Make `--verify` recompute artifacts rather than trust stored pass flags.

```powershell
python -m pytest tests/test_stress_v3_sealing.py -q
```

Suggested commit: `feat(benchmark): add role-separated Stress V3 sealing`

## Task 9: Blinded Intent Annotation Surface

**Files:**
- Create: `dilu/runtime/stress_v3/intent_annotation.py`
- Create: `scripts/build_stress_v3_intent_annotations.py`
- Create: `schemas/iclr2027/stress_v3_annotation.schema.json`
- Create: `tests/test_stress_v3_intent_annotation.py`

1. Export raw first-eligible snapshot responses without model, profile, case, outcome, score, condition, or replicate metadata. Support `main_endpoint`, `smoke_determinism`, and `smoke_repeatability` purpose tags without exposing the purpose to annotators.
2. Support two independent annotator files and a separate blinded adjudication file.
3. Validate labels `LEFT`, `IDLE`, `RIGHT`, `FASTER`, `SLOWER`, `AMBIGUOUS`, and `NONE`.
4. Test syntactically invalid output may still express a correct semantic intent; `AMBIGUOUS`/`NONE` remain incorrect.
5. Compute raw agreement and Cohen's kappa over the seven nominal labels before adjudication. Test the frozen degenerate rule: when expected agreement is one, emit `kappa=NA_DEGENERATE` while retaining raw agreement and coverage.
6. Require complete annotation/adjudication coverage before confirmatory H1/H2 analysis, S3R generation-determinism verification, and S4 semantic-repeatability verification. Expected final-label counts are 3,840, 192, and 48 respectively.
7. Implement exact `export`, `ingest`, and `verify` subcommands with immutable batch hashes; reject labels from a different batch/purpose.

```powershell
python -m pytest tests/test_stress_v3_intent_annotation.py -q
```

Suggested commit: `feat(benchmark): add blinded intent annotation workflow`

## Task 10: Tiny End-to-End Benchmark Test

**Files:**
- Create: `tests/test_stress_v3_pipeline.py`
- Modify: `dilu/runtime/stress_v3/__init__.py`

1. Build a tiny two-category synthetic candidate pool.
2. Validate schema, snapshot diff, predicates, oracle trajectories, replay, rejection ledger, split, seal, and blinded annotation export.
3. Regenerate the bundle twice and compare hashes.
4. Assert no LLM client is imported or called.

```powershell
python -m pytest tests/test_stress_v3_*.py -q
git diff --check
```

## Definition of Done

- Generator deterministically creates 240 candidates with balanced counterfactual structure.
- Independent checks reject shortcut-friendly, ambiguous, invalid, or unsolved pairs.
- Accepted suite is exactly 40 development plus 120 sealed cases.
- Sealed content is withheld and tamper-evident.
- Blinded annotation is complete and metadata-safe.
- Stress V1/V2 regressions remain green.
- No LLM evaluation has been run while implementing this plan.
