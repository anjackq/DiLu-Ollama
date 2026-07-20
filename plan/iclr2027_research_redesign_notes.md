# Research Redesign Notes

## Live Truth Surfaces

- Implementation repository: `C:\Users\WiCon\Desktop\Dilu-Ollama`.
- Active evaluator: `evaluate_models_ollama.py`.
- Driver prompt and action resolution: `dilu/driver_agent/driverAgent.py`.
- Safety and flow interventions: `dilu/runtime/safety_shields.py`.
- Benchmark predicates: `dilu/runtime/task_benchmark.py`.
- Split scoring: `dilu/runtime/dilu_scoring.py`.
- Stress V2 generator: `scripts/generate_dilu_highway_reactive_stress_v2.py`.
- Full-120 ablation: `results/analysis/stress_v2_harness_ablation_full120_20260624`.

## Verified Current State

- The repository contains existing modified and untracked files. They are project state and must not be reverted.
- Focused runtime, benchmark, reporting, baseline, and timeout tests pass: `139 passed`.
- The current two-profile comparison changes the system prompt but shares parsing, recovery, fallback, safety shields, timeout handling, and most transport behavior.
- Exact historical DiLu prompt/runtime material is recoverable from Git commit `1eed4ed` and `driverAgent_backup.py`.
- Claim-bearing full-120 Qwen episodes have `action_trace=null` because scientific traces were not mandatory for those runs.

## Stress V2 Evidence Risks

- Directional composition is imbalanced: 48 explicitly left-directed cases, 12 right-directed cases, and 60 nominally neutral cases.
- Qwen harness and `always_left` have nearly identical aggregate balanced scores and the same four completed categories.
- `always_left` is not a pure executed constant-action policy because unavailable-action handling and safety/flow shields modify its actions.
- Some success predicates accept a lane transition without proving the required target direction.
- Recovery state can be recorded before the relevant hazard event.
- Current case validation checks structure and initial feasibility, not policy solvability.
- Some categories have no successful agent or baseline, so difficulty cannot be separated from invalid or unsolvable predicates.

## Metric Risks

- Task completion contributes to the efficiency component and then re-enters through the task-conditioned geometric mean.
- Runtime timeout/fallback penalties are embedded in the task score, preventing clean policy-runtime separation.
- Missing component metrics can be treated too favorably by weighted aggregation.
- Bootstrap intervals over designed deterministic cases are descriptive, not population-level inference.

## Research Opportunity

The strongest general claim is not that stronger prompting makes SLMs drive well. It is that local language-agent evaluation must distinguish:

1. action-contract compliance;
2. raw policy competence;
3. parser and fallback dependence;
4. safety-intervention dependence;
5. executed closed-loop system performance.

Counterfactual scenario pairs can test whether a policy responds to causal state changes instead of matching a constant-action shortcut.

## Final Design Resolution

- Main confirmatory claim separates syntax enforcement, blinded semantic-intent responsiveness, and shield dependence.
- Parser recovery, fallback alternatives, resolver behavior, component prompts, and scaling are secondary rather than causal main claims.
- Primary paired intent uses a shared snapshot and shared generation seed; closed-loop execution uses case-scoped seeds after divergence.
- Scientific contrasts are frozen before validation; runtime constants are frozen after transport-only smoke and before behavioral smoke.
- The 3,840 budget is the confirmatory LLM factorial. The complete preregistered simulator cap is 6,816 before secondary-study amendments.
- The closed 6,816 ledger is now `3,840 main + 176 LLM smoke + 520 development non-LLM smoke + 1,560 sealed non-LLM diagnostics + 720 candidate validation`; direct smoke calls remain 148 and are not simulator episodes.
- Candidate validation uses two oracles plus one template-declared shortcut policy per candidate; within each coverage role, the first two passing candidates are selected in frozen order.
