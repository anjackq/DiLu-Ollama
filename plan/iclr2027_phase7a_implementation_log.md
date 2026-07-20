# ICLR 2027 Phase 7A Implementation Log

Date: 2026-07-13

## Scope Completed

Phase 7A Tasks 1-7 establish the non-experimental runtime foundation. This
checkpoint contains no Ollama request, model inference, Gym episode, Stress V3
case opening, or research simulation.

### Task 1: Immutable Harness Configuration

- Added frozen condition, transport, shield, retry, and trace configuration.
- Locked the eight factorial condition identities `c000` through `c111`.
- Separated condition identity from full configuration hashing.
- Added fail-closed validation for claim-bearing configurations.
- Added pre-S1 protocol constants with runtime-bound placeholders.

### Task 2: Prompt Modules and Historical Provenance

- Extracted the historical DiLu system message directly from Git commit
  `1eed4ed0bd2e483c2a604adc63f4a21c445dba06`.
- Locked the 836-byte artifact to SHA-256
  `170ff62b29d558fea590f234f3994a4b72100efbacdff5ccd518c24629bf764a`.
- Added deterministic historical-anchor and modular-policy prompt composition.
- Kept output-enforcement mode out of prompt identity.
- Preserved existing `harness_v2` and `legacy_dilu_like` behavior outside the
  scientific configuration branch.

Scientific wording boundary: this artifact is an exact historical system-message
anchor. The factorial condition is historical policy content under a controlled
modern interface, not an exact reproduction of the original DiLu runtime.

### Task 3: Typed Action Resolution

- Added the exact grammar `Response to user:#### <0-4>`.
- Separated syntax validity from state-dependent action availability.
- Added immutable typed resolution results and runtime/protocol failure classes.
- Enforced fixed `IDLE=1` as the only scientific operational fallback.
- Made unavailable IDLE, unresolved availability, and token-mapping drift typed
  protocol failures rather than state-aware substitutions.
- Routed scientific decisions directly through the typed resolver before every
  legacy parser, checker, semantic recovery, or intent resolver.
- Preserved verbatim raw output for invalid, empty, and timeout responses.
- Kept the legacy state-aware/SLOWER fallback path unchanged.

### Task 4: Deterministic Native Scientific Transport

- Added SHA-256-derived `uint32` generation seeds with separate primary-snapshot
  and post-divergence scopes.
- Added immutable native request, attempt, identity-check, backend-timing, and
  result evidence types.
- Enforced native `/api/chat`, explicit think mode, exact model tag/digest, and
  pre/post generation identity checks.
- Preserved model `raw_response`, complete HTTP `response_body`, decoded contract
  text, and transport error body as distinct evidence.
- Allowed one fixed 10-second retry only for proved pre-accept transport
  unavailability; timeout and empty output remain typed operational fallbacks.
- Made redirects, schema drift, model substitution, malformed metadata, and
  capability mismatch fail closed.

Scientific boundary: this is a typed/mock transport foundation. A verified S1
capability artifact, live seed/schema checks, and backend timing consistency are
still required before any result can be claim-bearing.

### Task 5: Explicit Execution Modes

- Added a frozen lane-change, longitudinal-safety, and low-speed-recovery stack.
- Made `unshielded_operational` bypass every primitive while recording all three
  stages; kept `shielded` in the legacy frozen order.
- Separated nullable model proposal, fallback-modified action, unshielded action,
  nullable shielded action, and executed action.
- Bound scientific shield configuration to the live primitive constants and
  rejected threshold or primitive-type drift.
- Retained legacy flat metadata through a compatibility adapter and added typed
  stage metadata to existing action traces.

### Task 6: Mandatory Scientific Traces

- Added a frozen `DecisionTraceRecord` and versioned JSON Schema covering
  campaign/episode/condition/case keys, seeds, prompt provenance, transport
  attempts, capability evidence, action resolution, shield stages, latency,
  tokens, disposition, and typed failures.
- Added canonical append-only JSONL persistence with schema/hash validation,
  duplicate and sequence guards, truncated-tail rejection, durable `fsync`, and
  fail-closed writer poisoning after storage ambiguity.
- Committed ready traces before `env.step()` and retained trace references for
  blocked generation and post-commit simulator failures.
- Bound requested versus effective transport to actual server acceptance;
  unaccepted and timeout attempts cannot claim an effective profile or think
  mode.
- Separated accepted transport profile from confirmed think-mode behavior;
  contract drift or reasoning leakage may retain an effective native profile
  while leaving the effective think mode unset.
- Added a canonical capability-snapshot hash for trace-internal integrity.
  External comparison with the frozen S1 artifact remains a Task 7 runtime-lock
  responsibility.
- Bound `prompt_only`, `backend_schema`, and `no_think` semantics in both typed
  runtime objects and JSONL resume validation. Observed no-think leakage is
  retained only as fail-closed `transport_drift` evidence.
- Bound successful HTTP response bodies to model identity, message content,
  stop reason, token counts, and backend timing; bound failure messages to their
  originating generation or resolution failure.
- Enforced the fixed retry as both a 10-second policy and a 10-second observed
  cooldown, while preserving true empty/timeout raw evidence without synthetic
  fallback text.

### Task 7: Complete Eight-Cell Runtime Integration

- Injected the frozen `HarnessConfig`, deterministic episode identity, verified
  model/runtime binding, fixed execution mode, mandatory trace writer, and
  attempt ledger through the scientific `run_episode()` path.
- Added a production factory that accepts only a disk-loaded verified runtime
  lock plus the native client, trace writer, and attempt ledger. Direct or live
  construction cannot create a verified binding.
- Added whole-episode attempt lifecycle handling for setup, generation,
  simulator, postprocessing, trace-finalization, and completion failures.
- Added campaign-wide deterministic request ownership and append-only,
  hash-chained attempt records with incremental resume validation.
- Added write-ahead `.append_pending` intent records before ledger or trace
  JSONL mutation. Intent, data, close, partial-tail, cleanup, and crash-point
  failures leave fresh processes fail closed.
- Preserved the known-good trace prefix when a later append is ambiguous and
  joined terminal evidence exactly by `(campaign_id, episode_attempt_id)` and
  original trace order.
- Verified the eight factorial cells and seven deterministic output fixtures
  while forbidding mutable legacy environment selection, recovery helpers, and
  adaptive timeout ladders in scientific mode.
- Kept the existing evaluator CLI as a legacy compatibility adapter. The later
  protocol smoke/campaign runner owns frozen-artifact loading and construction
  of each `ScientificEpisodeRuntime`.

## Verification

- Combined Phase 7A Tasks 1-6 mocked regression suite: `256 passed`.
- Independent Task 4 adversarial review: PASS (`170 passed`).
- Independent Task 5 adversarial review: PASS (`85 passed`).
- Historical prompt extractor `--verify`: passed.
- Black checks for new and split Task 1-5 Python files: passed.
- Python `compileall`: passed.
- Scoped `git diff --check`: passed; only existing line-ending warnings remain.
- Decision-trace schema JSON validation: passed.
- Final independent Task 6 code review: PASS (`72 passed` focused review).
- Final Task 6 adversarial recheck: PASS for forged preflight evidence and
  no-think effective-mode semantics.
- Independent Task 2 re-review: PASS.
- Independent Task 3 review: PASS.
- Combined Task 1-7 mocked scientific/runtime suite: `201 passed`.
- Required legacy regression suite: `114 passed`.
- Independent final Task 7 specification review: PASS.
- Independent final Task 7 durability/quality review: PASS (`68 passed`
  adversarial subset plus fresh-process rejection checks).
- Black checks for the new/split modules and scoped legacy edit ranges: passed.
- Ruff checks for the new/split Task 7 Python surface: passed.

The three pytest dependency warnings originate from the installed
`pygame/pkg_resources` stack and are not introduced by this work.

## Next Gated Work

Task 7 is complete. S1 remains a separate live capability gate and has not
started. It must produce verified model, seed, schema, think-mode, and backend-
timing evidence plus the real runtime-lock authorization artifact before any
behavioral smoke.

The future smoke runner must call `load_verified_runtime_lock_binding()` and
`build_scientific_episode_runtime()` for every claim-bearing episode. Mocked
authorization files used by Task 7 tests are fixtures, not scientific evidence.
The standalone campaign runner and complete protocol-freeze chain remain later
implementation tasks; they must not be routed through mutable legacy CLI state.

Append-intent behavior is verified for process crashes and fresh-process
rejection. Parent-directory `fsync` portability and sudden-power-loss behavior
remain a target-filesystem assumption that must be tested or explicitly frozen
before the main campaign. Until S1 and the later smoke gates pass, do not start
behavioral smoke, open sealed cases, or run the 3,840-episode campaign.
