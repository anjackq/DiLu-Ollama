# ICLR 2027 Research Redesign Specification Review Log

## Scope

Reviewed artifacts:

- `iclr2027_contract_policy_stress_v3_design.md`
- `iclr2027_smoke_investigation_protocol.md`
- `iclr2027_evaluation_analysis_protocol.md`

Review perspectives:

- runtime factorization and smoke executability;
- benchmark validity, statistics, and shortcut resistance;
- holistic ICLR contribution and protocol consistency.

No reviewer edited the specification files.

## Round 1: Request Changes

Blocking findings:

- recovery claims exceeded the three-factor 3,840 design;
- output enforcement and validator/parser were confounded;
- fallback-modified rollouts were incorrectly described as raw policy;
- counterfactual action flip lacked a shared decision point;
- sealed solvability validation conflicted with test withholding;
- smoke gates lacked machine predicates and forced branch coverage;
- acceptance margins, equivalence, multiplicity, and total compute were incomplete.

Resolution:

- narrowed the main claim to three bundle-level effects;
- fixed canonical `strict_only` validation and `IDLE` fallback;
- separated snapshot, unshielded operational, and shielded operational tracks;
- added frozen shared snapshots, independent sealing, fault injection, gate schemas, fixed margins, global multiplicity, and a compute ledger.

## Round 2: Targeted Recheck

Runtime review: `PASS WITH MINOR`.

Benchmark review found one remaining metric defect: the initial pair formula checked whether expected labels differed but did not require the model's actual semantic intents to differ. It also requested per-pair label solvability, a common equivalence denominator, and a closed total budget.

Holistic review requested deterministic generation control, a closed confirmatory endpoint registry, strict scientific/runtime lock ordering, and auditable historical prompt provenance.

Resolution:

- replaced action-only scoring with blinded `intent_pair_correct` and disjoint valid semantic sets;
- required observable-oracle intent correctness for every accepted pair;
- added common pair-snapshot generation seeds and a 100% determinism gate;
- defined 12 model-level intersection-union gates under one global Holm family;
- froze exact contrasts in `SCIENTIFIC_PROTOCOL_LOCK.json` before validation;
- added revision/blob/extraction/normalized hash provenance for the historical prompt;
- closed the simulator cap at 6,816 episodes plus 148 direct model calls.

## Final Blocker Audit

- Runtime review: `PASS WITH MINOR`; no Critical/Major findings.
- Benchmark/statistics review: `PASS WITH MINOR`; no Critical/Major findings.
- Holistic ICLR review: `PASS WITH MINOR`; no Critical/Major findings.

The written design is ready for owner approval and a separate file-level implementation plan. Passing this document review does not authorize implementation experiments or the 3,840-episode run.

## Phase 6 Implementation Review and Scientific Amendment

Date: 2026-07-13

Initial implementation-plan verdict: `REQUEST CHANGES`.

Major findings resolved:

- reordered dependencies so `SCIENTIFIC_PROTOCOL_LOCK.json` exists before candidate validation;
- corrected S3R to 128 extra replays and 148 total direct smoke calls;
- replaced the inconsistent 480-candidate/1,800-sealed budget with 720 candidate-validation and 1,560 sealed-diagnostic episodes, preserving the exact 6,816 simulator total;
- froze role-specific candidate selection, 10 category contracts, and same-role reserve behavior;
- fixed smoke artifact paths, schemas, producers, denominators, and staged execution commands;
- moved all candidate/sealed rows and trajectories under ignored `results/secure/`, leaving only redacted validation summaries visible;
- made S3R, S4, and main-study semantic labels explicit blinded dual-annotation/adjudication workflows with 192/192, 48/48, and 3,840/3,840 coverage gates;
- froze `cluster_signflip_v1`, bootstrap/randomization counts, TOST, degenerate rules, Cohen's kappa handling, IUT/Holm, and paper-level three-of-four model claim gates;
- split campaign scheduling, immutable attempts, and the facade to preserve the repository small-file rule;
- converted all 20 preregistered test names into exact file/task mappings and filled missing task-only/passive-transition behavior.

Final focused recheck: `PASS`.

- Critical findings: none.
- Major findings: none.
- Minor findings: none.
- Review scope authorized test-driven implementation only; it did not authorize real Ollama calls, smoke simulation, sealed-case opening, or the 3,840-episode run.

Reviewed artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `iclr2027_contract_policy_stress_v3_design.md` | `4ad943ed6bd92cd707a5df897e80ccbfcebbc3da4dd80c6dc6dbfc33bdd89d60` |
| `iclr2027_smoke_investigation_protocol.md` | `c89e2a965bf47e425478dae9ab1120749481160075a046a7c2de57762325932e` |
| `iclr2027_evaluation_analysis_protocol.md` | `6fa2a8d89515a17484dfa8b643cffebbc40648852a1edf36c2a749103a7ad739` |
| `iclr2027_phase6_implementation_index.md` | `c7d530758a936e468ab27dbc316d3821d94cf0dd6382f74bf8d366135a7ce49b` |
| `iclr2027_runtime_harness_implementation_plan.md` | `ebe20bd9aeeff4a7ddf7c975e6e6afd53e4fe1fe960c774c6dc56e345035b281` |
| `iclr2027_stress_v3_implementation_plan.md` | `f70e67a4a299e02dc7ba435283b46aad69da5fffffd37980d72962184140d58c` |
| `iclr2027_protocol_smoke_implementation_plan.md` | `113d91a9ccb50c57b362c673bc31b835fcb0317a41298bcfa83830b9a06347b8` |
| `iclr2027_campaign_analysis_implementation_plan.md` | `67c65ae253922821a7bbb5c07bb347884a895e07af2581b5bae82137b555ffdf` |
