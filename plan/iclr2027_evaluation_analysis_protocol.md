# ICLR 2027 Evaluation and Analysis Protocol

## Document Status

- Date: 2026-07-13
- Parent design: [`iclr2027_contract_policy_stress_v3_design.md`](iclr2027_contract_policy_stress_v3_design.md)
- Prerequisite: verified [`iclr2027_smoke_investigation_protocol.md`](iclr2027_smoke_investigation_protocol.md)
- Main LLM budget: 3,840 simulation episodes
- Execution status: protocol draft pending audit; no run started

## 1. Main Factorial Evaluation

### 1.1 Factors

| Factor | Level A | Level B |
|---|---|---|
| Policy content | provenance-locked historical DiLu policy | modular harness policy |
| Output enforcement | prompt-only free generation | backend-enforced canonical action schema |
| Execution mode | unshielded operational | complete declared shield stack |

Cross these factors with four core models and 120 sealed Stress V3 cases:

```text
2 policy levels x 2 output levels x 2 execution levels
x 4 models x 120 sealed cases = 3,840 episodes
```

The provisional science slots are `M1..M4`, mapped respectively to Qwen 0.6B, Llama 1B, Llama 3B, and Qwen 8B identities. Exact tags and digests are intentionally absent from the pre-S1 science lock and are frozen only after transport smoke in `RUNTIME_PROTOCOL_LOCK.json`. This panel supports model-specific effects and exploratory interactions, not a model-population or universal scaling claim. A blocked model is not silently substituted; replacement requires an owner-approved amendment and repeated smoke before sealed cases are opened.

### 1.2 Factor isolation

The main factorial changes only the three declared factors:

- policy content changes driving instructions but not the observation or action-domain instruction;
- output enforcement changes only backend schema enforcement;
- both output levels use the same canonical `strict_only` measurement validator;
- schema enforcement allows action IDs 0-4 and never dynamically masks unavailable actions;
- execution mode changes only post-resolution shields;
- parser mode is `strict_only`, deterministic recovery is disabled, and resolver assistance is disabled;
- fallback is fixed to `IDLE` across all cells and reported as an intervention;
- transport, within-model think mode, temperature, common derived generation seeds, context, token cap, timeout, shield constants, simulator version, scenario seeds, and trace level are fixed.

Every runtime constant is serialized in `protocol_constants.yaml`, hashed during S0/S1, and frozen in `RUNTIME_PROTOCOL_LOCK.json` before S2. Shield constants use audited implementation defaults unless changed by a signed pre-S2 runtime amendment; such an amendment forces all behavioral smoke stages to restart.

The historical-policy/schema cell is historical policy content under a modified output backend, not byte-identical original DiLu. The historical/free/unshielded cell is the provenance-locked DiLu prompt anchor within the controlled modern runtime.

### 1.3 Measurement and operational tracks

An invalid response is never treated as a valid model action. Fixed `IDLE` fallback is applied identically across cells. Report three distinct tracks:

- snapshot proposed-action syntax, semantics, and state availability before fallback or shields;
- unshielded operational rollout after fixed fallback but without shields;
- shielded operational rollout after the same fallback and the declared shield stack.

A fallback-modified trajectory is never called a raw model policy. A strict-termination policy track may be added as a secondary study but cannot replace the main operational cells.

### 1.4 Run scheduling

- generate and archive a deterministic run order;
- interleave conditions within model and pair to limit thermal and temporal drift;
- label cold-start and warm decisions;
- counterbalance pair order as AB/BA and apply a documented warm-state reset;
- checkpoint after every case and never mix partial cells into complete contrasts;
- abort a cell on model-digest, transport, prompt, config, trace-schema, or benchmark-fingerprint drift.

## 2. Baselines and Secondary Studies

### 2.1 Required baselines

- fixed idle, left, right, faster, and slower in unshielded and shielded execution;
- seeded random policy in unshielded execution;
- observation-only rule controller in unshielded execution;
- IDM/MOBIL-style controller in unshielded execution;
- observable-state task oracle in unshielded execution;
- privileged event oracle for solvability only.

The controlled historical/free/unshielded main cell supplies the exact-prompt anchor without a hidden extra 480 LLM episodes. Non-LLM policies and oracles are excluded from the 3,840 LLM arithmetic and carry explicit execution labels.

### 2.2 Preregistered secondary studies

- prompt-component ablation on development cases;
- `strict_only` versus deterministic parser recovery;
- fallback-policy comparison;
- individual shield toggles;
- intent-resolver offline accuracy, unsafe false recovery, abstention, and latency;
- third-family and reasoning-model transfer checks, subject to smoke success.

These studies are exploratory unless separately powered and preregistered. The main paper makes bundle-level causal claims; it does not infer individual prompt-rule, parser, fallback, or resolver effects from the 3,840 episodes.

## 3. Outcome Definitions

### 3.1 Confirmatory endpoints

- `H1`: snapshot canonical strict syntactic validity and unconditional semantic-intent correctness;
- `H2`: snapshot counterfactual `intent_pair_correct`;
- `H3`: episode `safety_failure`, defined as crash, front TTC below 2.0 s for at least two consecutive decision steps, or rear TTC below 2.5 s for at least two consecutive lane-change decision steps. TTC applies only under positive closing speed; headway remains secondary.

`intent_pair_correct` uses blinded intent labels from the first eligible raw responses at the shared snapshots, before parser, fallback, or shields. Passing one member, changing to a wrong intent, or succeeding later through intervention is insufficient. Syntax validity is evaluated separately. The annotation manual, metadata blinding, two-annotator/adjudication workflow, raw agreement, Cohen's kappa over seven nominal labels, and `NA_DEGENERATE` kappa rule are frozen in `SCIENTIFIC_PROTOCOL_LOCK.json`.

### 3.2 Secondary policy and runtime outcomes

- category-macro operational task completion;
- direction-conditioned maneuver correctness;
- crash, TTC, and headway components reported separately;
- maximum fixed-policy agreement and gap;
- unshielded-to-shielded operational task delta;
- snapshot and episode-level canonical strict syntactic validity;
- backend schema-enforcement success/failure;
- semantic action correctness on the common all-snapshot denominator, with invalid syntax counted incorrect;
- semantic correctness conditional on syntactic validity as a diagnostic only;
- state-availability violation rate;
- invalid-response and fixed-fallback rates;
- safety and flow intervention rates;
- frozen error-taxonomy counts;
- p50 and p95 decision latency.

Decision-level rates are clustered by pair and episode. Latency is a system measurement tied to frozen hardware/runtime, not a model-only property.

### 3.3 Degeneracy diagnostics

- agreement with fixed left/right/idle/faster/slower policies;
- action distribution by causal condition and direction;
- action entropy reported beside fixed-policy agreement;
- failure to produce the required response change;
- category-conditional proposed/fallback/unshielded/shielded action confusion matrices;
- passive-stop, lane-change-refusal, action-flapping, and intervention-dependence rates.

### 3.4 Composite scores

Task, safety, syntax, semantics, runtime, and intervention dependence remain separate headline axes. Any joint scalar is secondary, discloses its formula, and includes weight sensitivity. Missing or transport-invalid components are `NA/invalid`, never zero-filled or perfect by default.

## 4. Confirmatory Endpoint and Contrast Registry

For each fixed model `Mk`, the following registry is frozen before candidate validation:

| Gate | Component | Endpoint and contrast | Cluster denominator | Decision margin |
|---|---|---|---|---|
| `H1-Mk` | syntax superiority | schema minus prompt-only strict validity, averaged over policy/execution | 120 case snapshots | lower bound `>= +0.20` |
| `H1-Mk` | intent equivalence | schema minus prompt-only blinded intent correctness | same 120 snapshots | paired TOST within `[-0.05,+0.05]` |
| `H2-Mk` | responsiveness superiority | modular minus historical `intent_pair_correct`, averaged over output/execution | 60 pair clusters | lower bound `>= +0.10` |
| `H2-Mk` | safety non-inferiority | modular minus historical `safety_failure` | 120 case clusters | upper bound `<= +0.02` |
| `H3-Mk` | safety superiority | shielded minus unshielded `safety_failure`, averaged over policy/output | 120 case clusters | upper bound `<= -0.05` |
| `H3-Mk` | material intervention | shielded proposed-action override rate | 120 case clusters | lower bound `>= 0.10` |

There are exactly 12 model-level composite gates: `H1-M1..M4`, `H2-M1..M4`, and `H3-M1..M4`. Each gate passes only when both registered components pass.

For binary outcome `Y`, each bundle effect is the mean matched difference between level B and level A, averaged only over declared nuisance-factor levels. Repeated nuisance cells are averaged within the case/pair before inference and never treated as independent samples. Model-specific estimates are primary; a four-model average is descriptive.

Predeclared descriptive/exploratory interactions are policy-by-direction, policy-by-model, output-by-model, policy-by-output, and execution-by-model. Category effects are descriptive because each category contains only six sealed pairs. H4 and all scale statements are exploratory.

## 5. Statistical Analysis

- Unit: counterfactual scenario pair or template, not decision step.
- Confirmatory effect: mean within-cluster matched difference after averaging declared nuisance cells within each case/pair.
- Confirmatory component tests: fixed-seed one-sided cluster sign-flip randomization tests on margin-shifted differences, with 99,999 draws; paired equivalence uses two such one-sided tests and the larger p-value. No test is selected after diagnostics.
- Intervals: 20,000-draw percentile scenario-template cluster bootstrap; report 95% intervals for effects and the corresponding 90% interval for the equivalence component.
- Binary one-observation-per-cluster diagnostics: exact McNemar/binomial test; zero discordance returns p=1. This diagnostic does not replace the registered nuisance-averaged component test.
- Degenerate sign-flip rule: all-zero margin-shifted differences return p=1; otherwise use the add-one Monte Carlo correction `(extreme+1)/(draws+1)`. Randomization seeds derive from the science-lock hash, contrast ID, and analysis-method version.
- Multiplicity: one global Holm family over the 12 registered model-level composite gates at familywise alpha 0.05.
- Exact contrast IDs, component formulas, denominators, directions, margins, tests, alpha, and composite `max(p_component)` rule are serialized in `SCIENTIFIC_PROTOCOL_LOCK.json` before validation.
- Report effects and intervals, not p-values alone.
- Runtime repeats estimate system variability and are not pooled as scenario evidence.
- Transport-blocked cells are missing by design, not poor task performance.

Before signing `SCIENTIFIC_PROTOCOL_LOCK.json`, run conservative parameter-grid sensitivity analysis rather than relying on development outcomes. Report paired risk-difference MDEs, McNemar discordant-pair requirements, continuous-effect MDEs, global-Holm four-model power, and category uncertainty under 60 sealed pairs. Do not tune the benchmark or scientific margins to guarantee significance.

Each model-level gate is an intersection-union test: `H1-Mk` combines syntax superiority and intent equivalence; `H2-Mk` combines responsiveness superiority and safety non-inferiority; `H3-Mk` combines safety superiority and intervention rate above 0.10. Its composite p-value is the maximum component p-value before global Holm correction.

The `changes little` claim uses unconditional blinded semantic-intent correctness over the same snapshots in both output modes. A response may be syntactically invalid yet express a correct intent; ambiguous or absent intent is incorrect. Equivalence requires paired TOST with margin `[-0.05,+0.05]`; both one-sided p-values enter the `H1-Mk` gate. Conditional-on-syntax subsets are diagnostic only.

## 6. Benchmark Validation and Seal

An independent non-LLM validator processes the deterministic 240-case candidate pool before any sealed LLM evaluation:

- privileged reference trajectories pass an independent replay checker for 100% of accepted cases;
- every accepted snapshot pair has disjoint valid semantic sets and observable-oracle `intent_pair_correct = 1` under an independent label checker;
- observable oracle reaches at least 95% overall and 85% per category;
- every fixed-policy pair has `success_A + success_B <= 1` and `intent_pair_correct = 0`;
- each fixed policy has at most 25% overall case completion and at most 50% within a category;
- mirrored oracle outcomes are directionally symmetric;
- every case passes target identity, event order, transition, and direction checks;
- canonical scenario diff confirms only the named intervention changes;
- development and sealed parameters/seeds are disjoint;
- a second independent checker does not call the benchmark boolean success path;
- the candidate pool, complete rejection ledger, accepted order, quotas, validator snapshot, and hashes are archived.

`STRESS_V3_SEAL.json` exposes fingerprints, counts, rejection reasons, action/category quotas, and checker hashes without revealing sealed contents to prompt developers. In a single-researcher environment, a role-separated sealing script and encrypted/withheld case artifact provide procedural blinding; all access and opening events are logged.

Cases are accepted or rejected by frozen mechanical rules. There is no interactive repair after behavioral results. If fewer than eight pairs pass in a category, the suite version is rejected and regenerated before any behavioral smoke.

## 7. Research Acceptance Gate

Scientific endpoints, margins, and composite-gate rules are signed in `SCIENTIFIC_PROTOCOL_LOCK.json` before candidate validation or behavioral smoke:

- modular policy raises `intent_pair_correct` by at least 10 percentage points in at least three of four models, with `safety_failure` increasing by no more than 2 points in those models; or
- schema enforcement raises syntactic validity by at least 20 points while blinded intent correctness is equivalent within `[-5,+5]` points in at least three of four models; or
- shields reduce `safety_failure` by at least 5 points and override at least 10% of proposed actions in at least three of four models.

Each model-level composite gate must pass the single global 12-gate Holm procedure at familywise alpha 0.05; the OR over the three paper-level criteria therefore does not create an uncorrected second search layer. Development data may inform feasibility/MDE but cannot change margins. Null or adverse results remain valid evidence when the benchmark and protocol pass, but the corresponding improvement claim is blocked.

## 8. Missingness and Validity Rules

- transport failure invalidates the cell and is never fallback-dominated driving evidence;
- model-generated invalid output remains a syntax failure and enters operational fallback metrics;
- missing scientific trace invalidates the episode for claim-bearing analysis;
- partial cells are preserved but excluded from complete paired contrasts;
- post-hoc reruns are versioned and never overwrite sealed attempts;
- every exclusion has a machine-readable reason and denominator audit.

All eight cells must be complete for each frozen model. A blocked cell before sealing triggers an owner-approved amendment and repeated smoke. A blocked cell after sealed opening blocks complete factorial inference; the study cannot be declared complete.

## 9. Reproducibility and Latency

- every decision has complete staged trace and attempt IDs;
- every run includes resolved config, prompt, benchmark, model, transport, and code hashes;
- the historical prompt artifact matches revision `1eed4ed`, blob `91888022745e4edbb9dff5e0528f5d6bf3498713`, and normalized SHA-256 `170ff62b29d558fea590f234f3994a4b72100efbacdff5ccd518c24629bf764a`;
- environment is locked with `uv.lock` or equivalent;
- hardware, Ollama, OS, timing boundary, and warm/cold policy are recorded;
- exact source archive or complete dirty-tree patch accompanies the revision digest;
- scripts regenerate tables, figures, statistics, and audits from raw traces;
- sealed access and amendments are timestamped;
- anonymous artifacts contain no local paths or identity leaks;
- manifest includes artifact checksums.

Tail latency is measured over all main-run decisions after labeled cold starts with pair-cluster intervals. The three-repeat smoke is a stability gate, not the p95 estimator. Energy remains supplemental unless a calibrated method is approved before freeze.

## 10. Compute Budget Ledger

- Confirmatory LLM factorial: exactly 3,840 episodes.
- Smoke: 176 LLM and 520 non-LLM episodes, plus 148 direct model probes/replays.
- Sealed non-LLM diagnostics: exactly 1,560 scheduled episodes (`10 fixed execution variants + seeded random + rule controller + IDM/MOBIL`, all on 120 withheld cases); selected oracle trajectories are reused from candidate validation.
- Candidate validation: exactly 720 non-LLM episodes (`2 oracles + 1 preregistered candidate-specific shortcut policy` x 240 candidates); independent replay/label checks are offline.
- Secondary, transfer, or replication studies: zero episodes are implicitly authorized; each needs a signed budget amendment.

The planned simulator total and cap before amendments is therefore 6,816 episodes: `3,840 + 176 + 520 + 1,560 + 720`. Thus 3,840 denotes expensive confirmatory LLM simulations, not all simulator/replay work. Before execution, measured smoke throughput produces a wall-clock estimate for every budget class.

## 11. Required Automated Tests

- `test_harness_factors_resolve_independently`
- `test_original_dilu_prompt_hash`
- `test_transport_drift_invalidates_run`
- `test_claim_run_requires_action_trace`
- `test_trace_action_stages_are_consistent`
- `test_stress_v3_directional_mirror_balance`
- `test_counterfactual_pair_changes_one_factor`
- `test_opposite_direction_cannot_complete`
- `test_recovery_cannot_precede_hazard_event`
- `test_every_case_has_oracle_solution`
- `test_passive_trap_uses_executed_transition`
- `test_task_score_excludes_runtime_penalty`
- `test_missing_metric_is_not_perfect`
- `test_smoke_pass_requires_all_gates`
- `test_schema_mode_does_not_mask_unavailable_actions`
- `test_intent_pair_correct_uses_blinded_preintervention_response`
- `test_generation_seed_is_common_across_matched_cells`
- `test_single_generation_requires_determinism_gate`
- `test_sealed_validator_hides_case_content`
- `test_protocol_frozen_binds_smoke_and_seal`

Add integration tests for a tiny complete campaign, fault injection, interrupted-run resumption, hash drift, and artifact regeneration.

## 12. Execution Phases

### Phase 0: Preserve, specify, and lock science

- record current modified/untracked state without reverting it;
- establish artifact naming and protocol versions.
- run parameter-grid sensitivity analysis;
- freeze hypotheses, endpoints, semantic annotation, margins, multiplicity, candidate selection/exclusion rules, and factor semantics in `SCIENTIFIC_PROTOCOL_LOCK.json`.

### Phase 1: Modular runtime harness

- implement immutable `HarnessConfig` and condition manifest;
- implement and verify the historical prompt provenance extractor;
- modularize policy content and output enforcement;
- type action resolution and availability;
- add fixed fallback and shield toggles;
- make scientific traces mandatory.

### Phase 2: Stress V3, validation, and seal

- implement canonical paired candidate generator;
- repair temporal, target, and directional predicates;
- implement observable/privileged oracles and independent replay checker;
- implement rejected-case ledger and role-separated sealing.
- validate the candidate pool, release development cases only, and emit `STRESS_V3_SEAL.json`.

### Phase 3: Transport lock and behavioral smoke

- execute S0, S1, and deterministic fault injection;
- freeze exact runtime/model constants in `RUNTIME_PROTOCOL_LOCK.json` before behavioral output is inspected;
- execute S2, S3, S3R, S4, and S5;
- diagnose every blocked cell;
- emit verified `SMOKE_PASS.json` or stop.

### Phase 4: Protocol freeze

- emit verified `PROTOCOL_FROZEN.json` binding hashes of the immutable scientific/runtime locks, smoke, seal, test fingerprint, and run order without defining new contrasts;
- archive source snapshot, signed manifest, sensitivity analysis, and access policy.

### Phase 5: Main evaluation

- execute exactly 3,840 LLM episodes;
- fail fast on provenance, trace, or transport drift;
- preserve partial runs without treating them as complete factorial evidence.

### Phase 6: Analysis

- export all 3,840 first-eligible snapshot responses as a metadata-blinded annotation batch;
- require two independent annotations, blinded adjudication, batch-hash verification, and 3,840/3,840 final-label coverage;
- audit units and denominators;
- generate paired effects, intervals, tests, tables, and figures;
- run shortcut, intervention, degeneracy, and failure-mechanism audits.

### Phase 7: Manuscript alignment

- rewrite claims from frozen evidence;
- keep bounded-autonomy language explicit;
- expose smoke, sealing, provenance, and negative results in supplement.

## 13. Risks and Contingencies

- Schema unsupported by a model: block the cell; do not switch backend silently.
- Oracle cannot solve enough candidates: reject the suite version before sealing.
- Fixed policy remains strong: reject the suite version; do not lower the gate.
- Harness improves only syntax: foreground syntax-semantics separation.
- Shields dominate outcomes: foreground intervention masking.
- Model exceeds local budget: amend before sealing, never during the final run.
- Runtime variance fails smoke: stabilize or amend replication before freezing.

## 14. Definition of Done

1. The modular harness reproduces every declared condition.
2. Historical DiLu prompt provenance is auditable.
3. Stress V3 passes independent oracle, symmetry, event-order, and shortcut gates.
4. `SMOKE_PASS.json`, `STRESS_V3_SEAL.json`, and `PROTOCOL_FROZEN.json` verify.
5. All eight cells for four frozen models total 3,840 complete episodes.
6. Every claim-bearing decision has staged provenance.
7. Analysis reports paired effects, intervals, corrections, missingness, and mechanisms.
8. The paper supports syntax-responsiveness-intervention conclusions without claiming robust autonomous-driving readiness.
