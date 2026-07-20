# ICLR 2027 Typed Runtime Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Build an immutable, factorized, fully traced DiLu-Ollama runtime that reproduces all eight confirmatory conditions without confounding policy, output enforcement, parser, fallback, or shield behavior.

**Architecture:** New pure/typed runtime modules own experimental factors, prompts, action resolution, generation seeds, native transport, shields, failures, and traces. `DriverAgent` and `evaluate_models_ollama.py` become thin compatibility adapters for the scientific path while legacy commands remain supported.

**Tech Stack:** Python dataclasses/enums, hashlib, Ollama native `/api/chat`, JSONL, existing safety-shield primitives, `unittest`/`pytest`.

---

## Task 1: Immutable Harness Configuration

**Files:**
- Create: `dilu/runtime/harness_config.py`
- Create: `tests/test_harness_config.py`
- Create: `configs/iclr2027/protocol_constants.yaml`
- Modify later: `dilu/runtime/__init__.py`

1. Write failing tests for frozen `HarnessConfig`, `TransportConfig`, `ShieldConfig`, `RetryPolicy`, factor enums, canonical serialization, and stable SHA-256 hash.
2. Test resolution of exactly eight condition specs from `2 policy x 2 output x 2 execution`.
3. Test a factor-diff whitelist: paired condition specs may change only declared fields.
4. Test scientific validation rejects recovery/resolver enablement, non-IDLE fallback, adaptive timeout, transport fallback, missing trace, or unresolved factor values.
5. Implement `from_mapping()`, `to_canonical_dict()`, `config_hash()`, `condition_id()`, and `diff_conditions()` with frozen dataclasses.
6. Keep environment-variable parsing out of the new types. Add a later compatibility adapter in `llm_env.py` only.

Verify:

```powershell
python -m pytest tests/test_harness_config.py -q
```

Expected: eight unique condition IDs; repeated serialization has the same hash.

Suggested commit: `feat(runtime): add immutable scientific harness config`

## Task 2: Prompt Modules and Historical Provenance

**Files:**
- Create: `dilu/driver_agent/prompt_modules.py`
- Create: `dilu/driver_agent/prompts/original_dilu_2024.txt`
- Create: `provenance/original_dilu_2024_prompt.yaml`
- Create: `scripts/extract_original_dilu_prompt.py`
- Create: `tests/test_prompt_modules.py`
- Modify: `dilu/driver_agent/driverAgent.py`

1. Write `test_original_dilu_prompt_hash` using revision `1eed4ed`, source `dilu/driver_agent/driverAgent.py`, blob `91888022745e4edbb9dff5e0528f5d6bf3498713`, 836 normalized bytes, and SHA-256 `170ff62b29d558fea590f234f3994a4b72100efbacdff5ccd518c24629bf764a`.
2. Write tests proving historical and modular policies share observation/action-domain instructions and differ only in driving policy content.
3. Write tests for deterministic composition of `historical_policy`, `minimal_policy`, `anti_passive`, `lane_boundary`, `flow_policy`, and `strict_contract` modules.
4. Make the extractor read Git content, apply the frozen normalization, write text/YAML atomically, and fail on byte/hash mismatch.
5. Implement `PromptArtifact`, `build_policy_prompt()`, `build_system_prompt()`, and component hashes.
6. Set confirmatory `few_shot_num=0`. If later enabled in an exploratory study, retrieved examples must enter the prompt artifact and hash.
7. Delegate the scientific `DriverAgent` prompt path to the module while preserving the current two-profile adapter for legacy commands.

Verify:

```powershell
python -m pytest tests/test_prompt_modules.py tests/test_reactive_guardrails.py -q
python scripts/extract_original_dilu_prompt.py --revision 1eed4ed --verify
```

Suggested commit: `feat(prompt): add provenance-locked modular prompt artifacts`

## Task 3: Typed Action Resolution

**Files:**
- Create: `dilu/runtime/action_resolution.py`
- Create: `dilu/runtime/runtime_failures.py`
- Create: `tests/test_action_resolution.py`
- Modify: `dilu/scenario/envScenario.py`
- Modify: `dilu/driver_agent/driverAgent.py`

1. Write tests for the exact canonical grammar `Response to user:#### <0-4>` and rejection of extra text, JSON, Markdown, reasoning leakage, missing delimiter, multiple IDs, and out-of-range IDs.
2. Test that syntax validity is independent of current state availability. Backend schema allows all IDs `0..4`; it never dynamically masks unavailable actions.
3. Test `ActionResolutionResult` fields: raw response, syntax status, strict action, recovered action, recovery stage, violation, availability, fallback, and final resolved action.
4. Test confirmatory `strict_only` never invokes deterministic recovery or resolver assistance.
5. Test invalid/empty/timeout responses remain distinct and fixed `IDLE=1` is applied only as an explicit operational fallback.
6. Test IDLE unavailable raises a typed protocol failure instead of selecting a state-aware replacement.
7. Add a stable action token-to-ID lookup to `envScenario.py`; keep historical `preferred_fallback_action_id()` unchanged for legacy paths.
8. Replace measurement/execution parser duplication in the scientific path with the single typed result.

Verify:

```powershell
python -m pytest tests/test_action_resolution.py tests/test_reactive_guardrails.py -q
```

Suggested commit: `refactor(runtime): type action validation and fallback stages`

## Task 4: Deterministic Generation and Native Transport

**Files:**
- Create: `dilu/runtime/generation_seed.py`
- Create: `dilu/runtime/ollama_scientific_client.py`
- Create: `tests/test_generation_seed.py`
- Create: `tests/test_scientific_transport.py`
- Modify: `dilu/runtime/ollama_transport.py`
- Modify: `dilu/driver_agent/driverAgent.py`
- Modify: `evaluate_models_ollama.py`

1. Test the two frozen SHA-256 `uint32` seed scopes: pair-shared primary snapshot and case-scoped post-divergence decision.
2. Test matched condition IDs are excluded, pair members share the primary seed, later case/decision keys diverge, and `replicate_id` is stable.
3. Test immutable `GenerationRequest` and `GenerationResult` capture model tag/digest, request/attempt IDs, native endpoint, options, schema mode, think mode, raw response, stop reason, tokens, latency, and error class.
4. Test prompt-only and schema-enforced requests differ only by native schema enforcement.
5. Test exactly one retry is allowed only for `transport_unavailable_before_accept`, after the fixed 10-second policy; preserve both attempts.
6. Test timeout, empty output, schema rejection, syntax failure, and unavailable action are never retried.
7. Test scientific mode rejects OpenAI-compatible fallback, transport downgrade, `auto` think resolution, model digest drift, or ignored seed/schema settings.
8. Add immutable digest capture to model inspection and delegate scientific generation from `DriverAgent` to the client. Keep the legacy transport path intact.

All transport tests must use mocked HTTP responses; do not contact Ollama.

```powershell
python -m pytest tests/test_generation_seed.py tests/test_scientific_transport.py -q
```

Suggested commit: `feat(runtime): add deterministic native scientific transport`

## Task 5: Explicit Execution Modes

**Files:**
- Create: `dilu/runtime/shield_stack.py`
- Create: `tests/test_shield_stack.py`
- Modify: `evaluate_models_ollama.py`

1. Write tests for `unshielded_operational`: execute the commonly resolved action after fixed fallback, with every shield stage explicitly marked bypassed.
2. Write tests for `shielded`: lane, longitudinal, and low-speed/flow stages execute in frozen order using existing primitives.
3. Record input, output, applied/bypassed flag, and reason for every stage.
4. Test proposed, fallback-modified, unshielded, and shielded actions never overwrite one another.
5. Replace the unconditional three-shield call in `run_episode()` with the typed stack.

```powershell
python -m pytest tests/test_shield_stack.py tests/test_reactive_guardrails.py -q
```

Suggested commit: `feat(runtime): separate shielded and unshielded execution`

## Task 6: Mandatory Scientific Traces

**Files:**
- Create: `dilu/runtime/scientific_trace.py`
- Create: `schemas/iclr2027/decision_trace.schema.json`
- Create: `tests/test_scientific_trace.py`
- Modify: `evaluate_models_ollama.py`

1. Write tests for a frozen `DecisionTraceRecord` covering every field required by the design, including condition/case/pair/step keys, seeds, attempts, raw output, all action stages, availability, shields, latency/tokens, model digest, and failures.
2. Test action-stage transition validity and monotonic decision indices.
3. Test append-only JSONL writes occur when progress, video, SQLite, and `save_artifacts` are disabled.
4. Test the trace is durably appended before `env.step()` and a write failure aborts the cell.
5. Test synthetic timeout fallback text never replaces the true raw response.
6. Remove scientific trace dependence on `_on_decision`/Rich progress and return a trace reference in every episode record.

```powershell
python -m pytest tests/test_scientific_trace.py -q
```

Suggested commit: `fix(trace): make claim-bearing decision traces mandatory`

## Task 7: Integrate the Eight-Cell Runtime

**Files:**
- Create: `tests/test_runtime_factorization_integration.py`
- Modify: `dilu/driver_agent/driverAgent.py`
- Modify: `dilu/runtime/llm_env.py`
- Modify: `dilu/runtime/model_policy.py`
- Modify: `evaluate_models_ollama.py`
- Modify: `config.example.yaml`
- Modify: `dilu/runtime/__init__.py`

1. Add a fixture client that emits strict, recoverable, invalid, empty, timeout, unavailable, and schema-rejected outputs.
2. Test all eight conditions resolve and only declared factors change.
3. Test scientific runs bypass mutable environment factor selection and adaptive timeout ladders.
4. Update `run_episode()` to accept `HarnessConfig`, digest, seed context, condition/pair/replicate IDs, and mandatory trace writer.
5. Keep CLI compatibility, but mark `eval_prompt_profile` and related environment keys as legacy compatibility in `config.example.yaml`.
6. Export only stable public types from `dilu/runtime/__init__.py`.
7. Run a tiny mocked campaign through all eight conditions; no Ollama or Gym simulation is permitted in this integration test.

```powershell
python -m pytest tests/test_runtime_factorization_integration.py -q
python -m pytest tests/test_reactive_guardrails.py tests/test_timeout_policy.py tests/test_task_benchmark.py -q
git diff --check
```

Expected: factor isolation, fixed fallback, explicit shield modes, immutable attempts, and mandatory traces all pass; legacy regressions remain green.

Suggested commit: `feat(runtime): integrate factorized ICLR evaluation path`

## Definition of Done

- Eight conditions are reproducible from immutable config and have unique hashes.
- Historical prompt provenance verifies byte-for-byte.
- Output enforcement is isolated from the common strict validator.
- Native scientific transport cannot downgrade silently.
- Fixed IDLE fallback and shield modes are explicit.
- Quiet claim-bearing runs always emit complete traces.
- Existing Stress V2/legacy tests still pass.
- No real model or simulator call has occurred in this plan.
