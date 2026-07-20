# Task Plan: ICLR 2027 DiLu-Ollama Research Redesign

## Goal

Define and review an implementation-ready research design for a causal runtime-harness study and a counterfactual Stress V3 benchmark, with a mandatory smoke-run gate before the accepted 3,840-episode main evaluation.

## Phases

- [x] Phase 1: Inspect the implementation, Stress V2, full-120 evidence, and tests.
- [x] Phase 2: Select the research direction and main experiment scale.
- [x] Phase 3: Write the detailed design and smoke-run investigation protocol.
- [x] Phase 4: Complete independent specification review and resolve findings.
- [x] Phase 5: Obtain owner approval to proceed from the written specification into file-level planning.
- [x] Phase 6: Produce and independently review the file-level implementation plan.
- [ ] Phase 7: Implement with tests, then execute smoke investigation.
- [ ] Phase 8: Freeze Stress V3 and run the 3,840-episode evaluation.

### Phase 7A Runtime Foundation

- [x] Task 1: Immutable harness configuration and eight-cell factor identities.
- [x] Task 2: Provenance-locked historical prompt anchor and modular prompt bundle.
- [x] Task 3: Typed strict action resolution with fixed-IDLE operational fallback.
- [x] Task 4: Deterministic generation seeds and native Ollama transport.
- [x] Task 5: Explicit shielded and unshielded execution modes.
- [x] Task 6: Mandatory scientific traces and fail-closed trace validation.
- [x] Task 7: Integrate and verify the complete eight-cell runtime.

## Decisions Made

- Primary design: Contract-Policy Decomposition plus Counterfactual Stress V3.
- Main evaluation scale: 3,840 simulation episodes.
- Main factorial: 2 policy bundles x 2 output-enforcement modes x 2 execution modes x 4 models x 120 sealed cases.
- Smoke investigation is a hard prerequisite, not an optional sanity check.
- Action-contract compliance, policy competence, and runtime-intervention dependence remain separate outcome axes.
- A single balanced score will not be the headline result.
- Exact historical DiLu provenance will replace `legacy_dilu_like` as the main historical anchor.

## Owner Freeze Points Before Behavioral Smoke

- Confirm the final four core model tags/digests after the transport probe; the provisional panel is Qwen 0.6B, Llama 1B, Llama 3B, and Qwen 8B.
- Confirm whether energy measurement is supplemental or out of scope.
- Confirm the backend-enforced schema mechanism after the transport capability probe.
- Confirm availability of two blinded intent annotators plus an adjudicator.
- Confirm the preregistered effect margins and TTC safety thresholds.
- Confirm that 3,840 is the LLM main budget while the total simulator cap is 6,816 including smoke, validators, and non-LLM baselines.

## Errors Encountered

- The first monolithic draft exceeded the repository small-file guideline and was split into core, smoke, and evaluation protocols.
- One incremental patch and two validation commands failed without changing artifacts; each was rerun using atomic replacement or corrected PowerShell/PCRE2 syntax.
- The session aborted during Phase 6 review repair; recovery confirmed all plan edits were durable, no experiment process was active, and the final focused recheck passed after the remaining fixes.
- A second session interruption occurred during Task 7. Recovery found that the
  last ambiguity test patch had not landed; no model or simulator process had
  started. TDD and adversarial review then exposed and repaired lifecycle,
  runtime-lock trust, trace ownership, and append-durability defects before the
  task was closed.

## Status

**Phase 7A Tasks 1-7 complete; S1 is the next gated stage.** The complete
eight-cell runtime injection boundary, verified external runtime-lock loading,
immutable attempt ledger, mandatory trace lifecycle, and exact cross-artifact
evidence joins are mocked and regression-tested. The legacy evaluator CLI
remains a compatibility path; the later protocol smoke runner must load frozen
artifacts and call the scientific factory. S1 live transport probes, behavioral
smoke, sealed-case opening, and all research simulation remain unstarted and
separately owner- and lock-gated.
