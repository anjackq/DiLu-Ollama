# ICLR 2027 Design: Contract-Policy Decomposition and Counterfactual Stress V3

## Document Status

- Date: 2026-07-13
- Target: ICLR 2027, provisionally following the latest available ICLR review standard
- Direction: approved; protocol draft pending audit
- Main evaluation budget: approved at 3,840 simulation episodes
- Implementation status: not started
- Required prerequisite: smoke-run investigation and explicit go/no-go artifact

## 1. Executive Summary

DiLu-Ollama will be redesigned as a causal evaluation platform for local language-model driver agents. The research contribution is not that a stronger prompt improves a composite driving score. The 3,840-episode confirmatory study estimates three bundle-level effects that current language-agent evaluations often conflate:

1. output-interface enforcement and syntactic action validity;
2. counterfactual policy responsiveness;
3. dependence on post-policy safety intervention.

The redesigned evaluation combines a modular runtime harness with Counterfactual Stress V3. Stress V3 uses paired cases that require an appropriate action change when one causal variable changes. This makes constant-action, passive, and direction-biased policies directly detectable.

The main experiment is a `2 x 2 x 2 x 4 x 120` factorial totaling 3,840 episodes. No full evaluation may begin until a staged smoke investigation passes transport, trace, configuration, predicate, oracle, and repeatability gates.

## 2. Problem Statement

The current system provides useful runtime diagnostics, but the existing evidence cannot identify which predeclared harness bundle causes a behavioral change:

- `legacy_dilu_like` and `harness_v2` primarily differ in system-prompt content;
- parsing, semantic recovery, fallback, shields, and timeout handling are shared;
- safety and flow shields are applied as a fixed sequence;
- full-120 action provenance was not saved;
- Stress V2 contains a direction imbalance and task predicates that can reward shortcuts;
- the headline balanced score mixes task, behavior, and runtime effects.

Consequently, the present evidence can show that a prompt profile changes outcomes, but it cannot establish a causal runtime-harness effect or general local-SLM driving competence.

## 3. Central Research Claim

> Under a controlled local runtime, syntactic action validity, counterfactual policy responsiveness, and shielded operational performance are distinct empirical properties. Interface and policy harnesses can repair one property without repairing the others, and may shift rather than remove degenerate closed-loop behavior.

This claim remains valuable whether the full harness improves driving outcomes or only reveals a compliance-competence separation.

## 4. Research Questions and Hypotheses

### RQ1: Contract versus competence

Does schema-constrained action generation enforce syntactic validity without necessarily improving semantic action correctness or counterfactual policy responsiveness?

- H1a: schema enforcement increases strict syntactic validity and reduces invalid-output fallback exposure;
- H1b: syntactic improvement can exceed the corresponding change in semantic action correctness and task completion;
- H1c: a schema-valid model can remain equivalent to a trivial policy.

### RQ2: Harness policy boundaries

Do anti-passive, flow, and lane-boundary instructions produce state-responsive policy changes rather than a new constant-action shortcut?

- H2a: modular harness policy improves snapshot-level counterfactual `intent_pair_correct`;
- H2b: gains persist across mirrored left/right cases;
- H2c: gains are not explained by increased agreement with any fixed-action baseline.

### RQ3: Shield intervention masking

How much do safety and flow shields change unshielded operational outcomes, and which proposed failures do they mask?

- H3a: shields improve executed safety but can conceal unsafe model-proposed actions;
- H3b: unshielded and shielded operational rankings need not agree;
- H3c: intervention rate identifies when apparent task performance depends on the runtime rather than the proposed policy.

Parser recovery, fallback alternatives, and resolver behavior remain preregistered secondary diagnostics. The 3,840-episode factorial does not make a causal claim about those components.

### RQ4: Model and runtime interactions

Are observed scale and model-family differences robust after transport, think mode, decoding budget, and timeout policy are controlled?

- H4: apparent inverse scaling may be a model-by-harness interaction rather than a universal scaling law. H4 is exploratory because the four-model panel is not a model-population sample.

## 5. Scope and Non-Goals

### In scope

- local Ollama-served SLM/LLM driver policies;
- discrete high-level actions in `highway-env`;
- prompt policy, action contract, parser, fallback, resolver, and shields;
- counterfactual closed-loop evaluation;
- runtime latency and failure provenance.

### Out of scope

- claiming production autonomous-driving readiness;
- continuous vehicle control;
- public-road validation;
- training or fine-tuning models in the main study;
- using a single scalar as proof of competence;
- treating logging as an experimental intervention.

## 6. Runtime-Harness Redesign

### 6.1 Immutable configuration

Introduce an immutable `HarnessConfig` with independently controlled fields:

```text
policy_prompt
output_contract
parser_mode
resolver_mode
fallback_policy
shield_mode
transport_profile
trace_level
```

Every run must serialize the resolved configuration, config hash, prompt hash, model digest, benchmark fingerprint, and code revision.

### 6.2 Modular prompt construction

Replace the two monolithic prompt profiles with composable modules:

- `historical_policy`: provenance-locked DiLu policy content;
- `minimal_policy`: concise neutral action-selection control;
- `anti_passive`: progress and no-unnecessary-stop boundary;
- `lane_boundary`: availability and target-gap constraints;
- `flow_policy`: speed recovery and lane-discipline boundary;
- `strict_contract`: exact action-response interface.

The provenance-locked profile is retained as `original_dilu_2024`, with the paper describing it as the historical project anchor unless external upstream identity is separately verified. Its mandatory provenance record is:

```text
revision: 1eed4ed
source: dilu/driver_agent/driverAgent.py
git_blob: 91888022745e4edbb9dff5e0528f5d6bf3498713
extraction: DriverAgent.few_shot_decision -> system_message
normalization: substitute delimiter ####; textwrap.dedent; LF; UTF-8; no trim
normalized_bytes: 836
sha256: 170ff62b29d558fea590f234f3994a4b72100efbacdff5ccd518c24629bf764a
```

Implementation creates `provenance/original_dilu_2024_prompt.yaml` and a normalized text artifact. For factorial cells that change output mode, report them as historical policy content with a modified output backend, not as byte-identical historical DiLu.

### 6.3 Typed action-resolution pipeline

Replace implicit parser fall-through with a typed result:

```text
raw_response
syntax_valid
strict_action
recovered_action
recovery_stage
violation_reason
action_available
availability_violation
fallback_action
final_resolved_action
```

Parser modes:

- `strict_only`;
- `deterministic_recovery`;
- `resolver_assisted` as a secondary condition.

Fallback modes:

- `invalid_terminate` for raw-policy measurement;
- `fixed_idle`;
- `fixed_slower` for historical compatibility;
- `state_aware` for deployed-system evaluation.

### 6.4 Execution modes

Provide independent lane, longitudinal, and low-speed recovery toggles. Main-paper execution levels are:

- `unshielded_operational`: execute the commonly resolved action, including the fixed main-study fallback, without safety/flow modification;
- `shielded`: execute the complete declared safety/flow stack.

The model-proposed action before parser fallback or shields is a snapshot-level policy measurement, not a closed-loop trajectory. Component-level shield ablations are secondary analyses. Unshielded and shielded conditions require separate rollouts because interventions change subsequent state trajectories.

### 6.5 Mandatory scientific trace

Every decision in a claim-bearing run records:

```text
raw response
model-proposed action
syntax validity and state availability
strict or recovered parsed action
fallback action, if any
lane-shield output
longitudinal-shield output
flow-recovery output
executed action
available actions
event phase
latency and token counts
generation seed
request and attempt ID
condition, case, pair, and step key
requested and effective transport/think mode
stop reason, timeout/error class, and model digest
```

Lightweight traces are mandatory even when video and SQLite logging are disabled. A run with missing traces is invalid for scientific claims.

### 6.6 Transport lock

Scientific comparisons fix:

- Ollama native chat transport;
- declared `think` or `no_think` mode;
- temperature;
- master and per-request generation seeds;
- context and output-token limits;
- timeout policy;
- model digest and Ollama version;
- hardware and timing boundary.

For master seed `S`, request seeds use two scopes:

```text
primary shared snapshot:
  uint32(SHA256(S | model_digest | pair_id | decision_snapshot_id | replicate_id))

post-snapshot closed loop:
  uint32(SHA256(S | model_digest | case_id | decision_index | replicate_id))
```

The condition ID is excluded so matched factor levels use common random numbers. Pair members also share the primary-snapshot seed, ensuring their intent contrast is not a sampling-draw contrast. After trajectories diverge, the case-scoped seed prevents collisions. Main runs use `replicate_id=0`; smoke replays reuse the identical seed. A backend that ignores or violates it fails the single-generation determinism gate.

Transport downgrade or model substitution invalidates the run instead of silently continuing.

## 7. Counterfactual Stress V3

### 7.1 Candidate pool, development split, and sealed split

The generator first emits a deterministic candidate pool of 240 cases:

```text
10 categories x 12 counterfactual pairs x 2 cases = 240 candidate cases
```

- An independent non-LLM validator applies fixed exclusion rules and preserves a complete rejected-case ledger.
- Within each category and frozen coverage role, the first two passing candidates form the suite: two development plus two held-out-nuisance, two mirrored, and two held-out-event-composition pairs.
- Development split: 2 accepted pairs/category, 40 cases.
- Sealed test split: 6 accepted pairs/category, 120 cases.
- Development and test use disjoint scenario parameters and seeds.
- Within each sealed category, two pairs test held-out nuisance ranges, two test mirrored transformations, and two test held-out event compositions.
- After sealing, the prompt/evaluation team receives the 40-case development split plus validation counts, quotas, and fingerprints. The 120-case sealed-test content remains withheld until `PROTOCOL_FROZEN.json` authorizes the independent runner.

### 7.2 Counterfactual pair families

Each pair changes one intended causal variable while holding nuisance variables fixed:

- safe-left versus safe-right opening;
- maneuver-now versus wait-for-gap;
- accelerate versus decelerate requirement;
- true opening versus false opening;
- escape direction mirror;
- recovery-required versus maintain-flow control.

Target actions are counterbalanced globally and within directional categories.

### 7.3 Pair schema and primary decision snapshot

Every pair serializes:

```text
pair_id, template_id, category
intervention_field, value_A, value_B
nuisance_state_hash
shared_prefix_hash, decision_snapshot_id
eligible_decision_window
valid_action_set_A, valid_action_set_B
valid_semantic_set_A, valid_semantic_set_B
required_events_A, required_events_B
endpoint_predicate_A, endpoint_predicate_B
```

The pair members share a deterministic pre-intervention prefix or frozen decision snapshot. Primary pairs require disjoint valid semantic sets; ambiguous pairs with a shared acceptable semantic response are excluded. A frozen annotation protocol maps each raw response, blind to model/profile/case/outcome metadata, to `LEFT`, `IDLE`, `RIGHT`, `FASTER`, `SLOWER`, `AMBIGUOUS`, or `NONE`. Two independent annotators label every primary snapshot; disagreements receive blinded adjudication. Before adjudication, report raw agreement and Cohen's kappa over the seven nominal labels; when expected agreement is one, report kappa as `NA_DEGENERATE` rather than inventing a numeric value.

The primary policy score uses semantic intent before parser recovery, fallback, or shields:

```text
intent_A = blinded_intent_label(raw_response_A)
intent_B = blinded_intent_label(raw_response_B)

intent_pair_correct =
  intent_A in valid_semantic_set_A
  and intent_B in valid_semantic_set_B
  and intent_A != intent_B
```

`strict_pair_correct` additionally requires canonical syntactic validity and valid action IDs, and is reported as an operational diagnostic. Multiple safe actions are permitted within each disjoint semantic set. Delayed maneuvers use a declared decision window and temporal predicate. Closed-loop success is separate and never substituted for `intent_pair_correct`.

### 7.4 Category-to-pair contract

| Category | Causal transform | Primary valid-response contrast | Required closed-loop evidence |
|---|---|---|---|
| mandatory overtake | mirror safe passing side | left versus right maneuver | target transition, pass lead, safe terminal state |
| timed-gap overtake | gap open versus not-yet-open | maneuver versus wait | transition inside opportunity window |
| traffic-jam escape | mirror feasible escape side | left versus right escape | specified target-lane transition |
| traffic-jam patience | true versus false opening | maneuver versus wait | no premature transition; progress after valid opening |
| route discipline | mirror target route lane | left versus right route action | reach specified route lane |
| bottleneck merge | mirror merge pressure | merge left versus right | target merge after trigger |
| cut-in recovery | hazard requires slow versus clear control | decelerate versus maintain/accelerate | ordered hazard-response-clear-recovery events |
| false-opening stability | true versus false opening | maneuver versus hold | no false-opening transition |
| dense flow | headway supports faster versus slower | accelerate versus decelerate/hold | safe headway and flow response |
| stop-go wave | recovery required versus maintain flow | accelerate versus hold | ordered wave-clear-recovery events |

Exact action sets and event identifiers are data fields, not inferred from the table text.

### 7.5 Predicate corrections

- Lane tasks require an executed transition to the specified target lane.
- Proposed or shield-blocked actions do not count as successful maneuvers.
- Overtake requires target-lane transition, passing the designated lead vehicle, and safe return/terminal state where applicable.
- Recovery ordering must satisfy `hazard event < response < clear event < recovered state`.
- Timed-gap success must occur inside the validated opportunity window.
- Event completion must reference the required event identifiers, not any applied event.

### 7.6 Independent solvability controls

Implement two validators:

- observable-state task oracle: a reproducible upper-bound policy using only evaluation-time observations;
- privileged event oracle: uses hidden event information only to generate a reference trajectory and is never a competing agent;
- independent replay checker: validates target identity, direction, event ordering, executed transitions, and safety invariants without calling the benchmark's boolean success function.

Case acceptance requires:

- privileged reference trajectory passes the independent replay checker;
- independent snapshot-label checker verifies disjoint valid semantic sets and one named causal intervention for every accepted pair;
- observable oracle achieves `intent_pair_correct = 1` on every accepted snapshot pair;
- observable oracle success under the declared threshold;
- one preregistered candidate-specific shortcut policy failure per case (opposite direction for directional tasks; the template-declared passive or wrong-longitudinal action for non-directional tasks);
- no success before required events;
- mirrored oracle outcome consistency.

The validator operates on all candidates without LLM access. It runs two oracle trajectories plus the single template-declared shortcut trajectory for every candidate, for 720 candidate-validation episodes. Within each category, candidate order and coverage-role replacement are frozen before execution as `dev_1`, `dev_2`, `nuisance_1`, `nuisance_2`, `mirror_1`, `mirror_2`, `event_1`, `event_2`, then one same-role reserve for each role. Within each role, select the first two passing candidates in frozen order; never replace across roles, and fewer than two passing candidates invalidates that category/version. Every tested and rejected candidate is retained. The full fixed/random/rule/IDM shortcut matrix is then run on the validator-withheld 120-case provisional test split; any suite-level shortcut failure invalidates the complete benchmark version rather than triggering post-result pair replacement. Only then does the validator emit `STRESS_V3_SEAL.json`, accepted/rejected counts, exclusion reasons, hashes, and category/action quotas while withholding sealed case content.

### 7.7 Baselines

- exact `original_dilu_2024` anchor;
- fixed idle/left/right/faster/slower, with unshielded and shielded results separated;
- seeded random policy;
- observation-only rule controller;
- IDM/MOBIL-style controller;
- observable-state task oracle;
- full DiLu-Ollama harness.

## 8. Companion Protocols and Authorization

This core design is implemented through two mandatory companion specifications:

- [`iclr2027_smoke_investigation_protocol.md`](iclr2027_smoke_investigation_protocol.md) defines transport, factor-isolation, trace, repeatability, and benchmark go/no-go gates.
- [`iclr2027_evaluation_analysis_protocol.md`](iclr2027_evaluation_analysis_protocol.md) defines the 3,840-episode factorial, outcomes, statistics, acceptance criteria, tests, and execution phases.

The authorization order is strict:

1. implement the factorized runtime, provenance extractor, candidate generator, validators, and sealing workflow;
2. freeze hypotheses, endpoint registry, exact contrast IDs/formulas/denominators, component margins, tests, alpha, global multiplicity family, shortcut/selection rules, annotation protocol, and factor semantics in `SCIENTIFIC_PROTOCOL_LOCK.json` before candidate validation;
3. independently validate the candidate pool, release only development cases, withhold test cases, and verify `STRESS_V3_SEAL.json`;
4. complete transport-only smoke, then freeze model digests, generation seeds, transport, timeout, fallback, shields, and trace schema in `RUNTIME_PROTOCOL_LOCK.json` before behavioral smoke;
5. execute development-only behavioral smoke and verify `SMOKE_PASS.json`;
6. bind hashes of both locks, seal, smoke, and run order in `PROTOCOL_FROZEN.json` without adding or changing any endpoint or contrast;
7. execute the sealed 3,840-episode main evaluation.

`SMOKE_BLOCKED.json`, a missing core cell, an invalid seal, or an undocumented factor/transport change forbids the main evaluation. Behavioral failure by a model does not itself block the study when the runtime and trace are valid; it remains a scientific outcome.
