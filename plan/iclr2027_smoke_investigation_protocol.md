# ICLR 2027 Mandatory Smoke-Run Investigation Protocol

## Document Status

- Date: 2026-07-13
- Role: hard go/no-go gate before the 3,840-episode evaluation
- Parent design: [`iclr2027_contract_policy_stress_v3_design.md`](iclr2027_contract_policy_stress_v3_design.md)
- Prerequisite benchmark artifact: independently verified `STRESS_V3_SEAL.json` with only the development split released
- Downstream protocol: [`iclr2027_evaluation_analysis_protocol.md`](iclr2027_evaluation_analysis_protocol.md)
- Execution status: protocol draft pending audit; no run started

## 1. Purpose

The smoke investigation determines whether failures come from infrastructure, experiment wiring, benchmark predicates, or model behavior. It must prevent invalid transport or trace failures from entering the evidence bundle without filtering out scientifically meaningful contract or policy failures.

The smoke is an investigation, not a miniature performance study. Its outputs authorize or block protocol freezing; they are not used to select favorable test conditions.

## 2. Artifact Contract

Use a dated folder:

```text
results/analysis/stress_v3_smoke_<YYYYMMDD>/
```

Required outputs:

- `resolved_conditions.csv`;
- `model_preflight.json`;
- `decision_traces.jsonl`;
- `trace_integrity.csv`;
- `fault_injection_audit.csv`;
- `counterfactual_pair_audit.csv`;
- `oracle_validation.csv`;
- `runtime_repeatability.csv`;
- `generation_determinism.csv`;
- `failure_triage.md`;
- `MANIFEST.md`;
- exactly one of `SMOKE_PASS.json` or `SMOKE_BLOCKED.json`.

The implementation must also provide `schemas/smoke_gate.schema.json` and `scripts/verify_smoke.py`. Each gate record contains `gate_id`, artifact inputs, denominator, exact predicate, expected value, observed value, severity, pass/fail, and machine-readable reason.

`MANIFEST.md` records code revision, working-tree state, environment, model digests, prompt/config/benchmark hashes, commands, artifact checksums, and any deviations. It must not contain secrets or identifying local paths in the reviewer package.

## 3. Stage S0: Static and Environment Preflight

No simulation episodes are run.

- resolve all eight main-factor combinations and assign unique condition IDs;
- verify every condition changes only its declared factors;
- record Python, package, Ollama, OS, CPU/GPU, RAM, model digest, and code revision;
- verify benchmark, prompt, config, and transport hashes;
- reject missing model tags, silent aliases, or unresolved config values;
- verify scientific trace mode is enabled for every claim-bearing condition;
- verify `SCIENTIFIC_PROTOCOL_LOCK.json` was signed before candidate validation and contains the complete 12-gate contrast registry;
- verify all eight conditions resolve before transport probes;
- verify no secret-bearing configuration is copied into artifacts.

Pass condition: the static manifest is complete and every planned factor transition is machine-auditable.

## 4. Stage S1: Model Transport Probe

Run five direct generation probes per core model before starting the simulator:

1. native-chat health and non-empty completion;
2. declared `think` or `no_think` behavior;
3. free-generation response;
4. constrained-action response;
5. repeated warm response for latency comparison.

For every probe, record model tag and digest, endpoint, request options, effective think mode, stop reason, raw response, token counts, first-token latency when available, total latency, and error class.

Required checks:

- no OpenAI-compatible or other transport downgrade;
- requested and effective transport settings agree;
- transport success is distinguished from contract success;
- cold-start and steady-state latency are labeled separately;
- empty or timed-out output is not converted into a synthetic policy action.

Budget: 20 direct model calls, not simulation episodes.

After S1 succeeds and before any behavioral model run, emit `RUNTIME_PROTOCOL_LOCK.json` containing exact model digests, native transport, within-model think mode, temperature, master/per-request generation seed rule, context/token limits, timeout/retry policy, canonical schema, `strict_only` validator, fixed `IDLE` fallback, shield constants, trace schema, and simulator/runtime versions. Any later change invalidates S2-S5.

## 5. Stage S1F: Deterministic Fault-Injection Contract Test

Before relying on natural model behavior, inject deterministic transport and policy fixtures that force every runtime branch:

- strict-valid response;
- syntactically recoverable response;
- invalid response;
- empty response;
- generation timeout;
- unavailable but syntactically valid action;
- fixed fallback execution;
- lane, longitudinal, and flow shield overrides;
- transport-profile drift;
- trace-write failure.

The fixtures verify error classification, attempt preservation, stage ordering, availability handling, fallback, shield provenance, and fail-fast behavior. This stage uses no real model calls or simulation episodes. Every branch must appear in `fault_injection_audit.csv`; absence is fatal.

## 6. Stage S2: Single-Model Factorial Micro-Smoke

Use Qwen 0.6B across all eight main factorial conditions on four counterfactual pairs, eight cases:

```text
1 model x 8 conditions x 8 cases = 64 episodes
```

The selected pairs must cover directional mirroring, maneuver/wait, longitudinal response, and false-opening stability.

This stage verifies:

- policy and output factors resolve independently;
- unshielded and shielded rollouts are distinct and correctly labeled;
- prompt-only and schema-enforced action paths are observable under the same canonical strict validator;
- fallback and shield interventions are staged in the trace;
- event ordering and available-action fields are present;
- each condition manifest matches the intended factorial cell.

Pass condition: all 64 episodes execute or produce explicit, correctly classified failure records, and no factor leaks into an undeclared component.

## 7. Stage S3: Cross-Family Full-Cell Smoke

Run all four core models on all eight factorial cells using one complete counterfactual pair:

```text
4 models x 8 conditions x 2 cases = 64 episodes
```

This stage detects model-specific policy-template, schema, execution, latency, or timeout incompatibilities in every cell before the full factorial.

Pass condition: every model/factorial cell has a valid transport record and staged trace. A genuine model contract or policy failure may remain; a transport failure explicitly blocks that model before protocol freezing.

## 8. Stage S3R: All-Cell Generation-Determinism Probe

Use the first eligible snapshots from the two S3 pair members. For each of the 64 model/cell/case combinations, replay the identical request twice using the same frozen generation seed; the S3 response is replicate one:

```text
4 models x 8 conditions x 2 cases x 2 extra replays
= 128 direct model calls
```

The three responses must agree exactly on syntax-validity status and blinded semantic-intent label. The 192 responses are exported as randomized annotation IDs without model, condition, case, outcome, or replicate metadata; two independent annotators label them and disagreements receive blinded adjudication under the frozen annotation protocol. Missing labels or incomplete adjudication blocks the gate. This is the prerequisite for treating one generation per main-study episode as an adequate deterministic measurement. Any disagreement blocks the 3,840 single-generation design and requires a signed replicated-design amendment.

## 9. Stage S4: Runtime Repeatability Smoke

Use Qwen 0.6B and Llama 1B, the historical/free/unshielded and modular/schema/shielded endpoint conditions, four representative cases, and three repeated campaigns:

```text
2 models x 2 endpoints x 4 cases x 3 repeats = 48 episodes
```

Fix cases, simulator seeds, prompt/config hashes, model digests, and runtime options across repeats. Record cold/warm state and invocation order. A cold start is labeled only by an explicit model-load boundary or positive backend-reported load duration, never by an observed latency threshold. Cold-start decisions remain in syntax, semantic, action-sequence, and task agreement; only the latency CV/ratio calculations exclude them.

The 48 first-snapshot responses form a separate blinded `smoke_repeatability` annotation batch. Two independent annotators label every response, disagreements receive blinded adjudication, and all 48 final labels must verify against the batch hash before semantic agreement is computed.

Report:

- output/action agreement across repeats;
- contract-path agreement;
- p50/p95 and dispersion of decision latency;
- timeout or empty-response recurrence;
- task-outcome agreement.

The repetitions estimate system variability and are not independent scenario samples. Go/no-go latency rules exclude mechanically labeled cold starts; behavioral agreement rules retain all repeats:

- zero config, prompt, model-digest, transport, or trace drift;
- zero unclassified errors;
- 100% first-snapshot syntax-status and semantic-intent agreement across triplicates;
- 100% executed-action-sequence and task-outcome agreement for deterministic simulator trajectories;
- per model/condition median-latency coefficient of variation at most 0.25;
- no campaign median latency more than twice another repeat under the same condition.

Failure blocks the single-repeat 3,840 design. The response is stabilization or a signed replication-design amendment, not removal of unfavorable observations.

## 10. Stage S5: Development Benchmark and Oracle Smoke

Run all 40 development cases with:

- privileged event oracle;
- observable-state oracle;
- fixed idle/left/right/faster/slower policies in both unshielded and shielded execution;
- observation-only rule controller.

These non-LLM runs validate predicates, direction balance, event ordering, solvability, and shortcut resistance. They are not counted in the 3,840 LLM main-study budget.

Budget:

```text
(2 oracles + 10 fixed-policy variants + 1 rule controller) x 40 cases
= 520 non-LLM simulation episodes
```

Required pair-level audit:

- exactly one intended causal variable changes within each pair;
- the oracle action or outcome flips as declared;
- success cannot occur before the required event sequence;
- for every fixed-policy variant and pair, `success_A + success_B <= 1` and `intent_pair_correct = 0`;
- unshielded and shielded baseline outcomes are labeled separately;
- all predicate decisions name the supporting event and transition.

## 11. Machine-Executable Go/No-Go Gates

### 11.1 Infrastructure gates

- all planned episodes execute or have an explicit failure record;
- no silent model skip, substitution, alias, or transport downgrade;
- every empty response is classified as model output, timeout, or transport failure under the frozen taxonomy;
- every condition has a resolved config, hashes, and model digest;
- every request has the frozen derived generation seed;
- decision latency is measured against one documented boundary.

### 11.2 Trace gates

- 100% of claim-bearing decisions have required staged action provenance;
- parser-path counts equal decision counts;
- every action-stage transition satisfies the enumerated transition table in the gate schema;
- syntactic validity and state availability are reported separately;
- event phase and available actions are present;
- factor toggles are visible in both config and trace.

### 11.3 Benchmark gates

- every smoke pair changes exactly one declared causal variable;
- expected oracle action or outcome flips are observed;
- a success predicate cannot fire before required events;
- every unshielded and shielded fixed policy has `success_A + success_B <= 1` and `intent_pair_correct = 0` for every pair;
- privileged oracle solves every development case;
- observable oracle reaches at least 95% overall and 85% per category;
- unshielded and shielded baseline results remain separated.

Behavioral failures such as low completion, malformed output, passive behavior, or constant-action agreement do not block the experiment when transport, factorization, traces, and predicates are valid. They remain evidence.

Any blocked core model/factorial cell makes global `SMOKE_PASS` impossible. Model behavior may be poor, invalid, passive, or unsafe without blocking when its transport, factor wiring, trace, and error classification are valid.

## 12. Error Taxonomy, Retry, and Non-Response Triage

Frozen top-level error classes are:

- `transport_unavailable_before_accept`;
- `transport_drift`;
- `generation_timeout`;
- `model_empty_output`;
- `schema_rejection`;
- `syntax_invalid`;
- `action_unavailable`;
- `trace_write_failure`;
- `simulator_failure`.

Only `transport_unavailable_before_accept` permits one retry after a fixed 10-second cooldown. The original attempt is immutable, the retry receives a new attempt ID, and the episode is marked `transport_retried`. Timeout, empty output, schema rejection, invalid syntax, and unavailable action are never retried. Transport drift, trace failure, or simulator failure aborts the cell. Resume creates a versioned campaign and does not overwrite attempts.

For a non-responding model, investigate in this order:

1. installed model name and digest;
2. Ollama health and native chat endpoint;
3. requested and effective think mode;
4. context and output-token limits;
5. cold-start versus steady-state latency;
6. timeout boundary;
7. prompt-template compatibility;
8. constrained-output support.

Do not convert a transport failure into fallback-dominated benchmark evidence. A blocked model/condition receives `SMOKE_BLOCKED.json` with the exact layer, observed evidence, attempted repairs, and decision required before resumption.

## 13. Authorization Artifact

`SMOKE_PASS.json` must contain:

- protocol version and timestamp;
- code revision and dirty-state digest;
- core model tags and digests;
- condition IDs and resolved hashes;
- development benchmark fingerprint;
- stage-level counts and gate results;
- links and checksums for required smoke artifacts;
- an explicit `protocol_freeze_eligible: true` field.

The verifier must validate the JSON schema and recompute every gate from source artifacts rather than trust a manually entered flag. Any blocked core cell forbids `SMOKE_PASS.json`. Any later change to a model digest, prompt, factor implementation, transport profile, timeout policy, trace schema, or benchmark predicate invalidates the authorization and requires the affected smoke stages to be repeated.

`SMOKE_PASS.json` authorizes protocol binding, not the main run by itself. The final authorization is `PROTOCOL_FROZEN.json`, which binds `SCIENTIFIC_PROTOCOL_LOCK.json`, `RUNTIME_PROTOCOL_LOCK.json`, the verified smoke artifact, `STRESS_V3_SEAL.json`, exact contrasts, model digests, and all frozen constants without redefining them.

## 14. Smoke Budget and Stop Rule

- 148 direct model calls: 20 transport probes plus 128 deterministic snapshot replays;
- 176 LLM simulation episodes;
- 520 development oracle/baseline simulation episodes;
- 696 total smoke simulation episodes, of which only 176 invoke an LLM;
- no sealed full-120 test execution.

Only the combination of verified `STRESS_V3_SEAL.json`, `SMOKE_PASS.json`, and `PROTOCOL_FROZEN.json` authorizes the 3,840-episode run. A failed gate stops progression; it does not authorize silently dropping a model, changing a condition, opening sealed test cases, or replacing missing evidence with fallback outcomes.
