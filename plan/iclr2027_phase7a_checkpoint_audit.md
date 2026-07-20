# Phase 7A Checkpoint Audit

Date: 2026-07-20

## Purpose

Create a recoverable Git checkpoint for the completed ICLR 2027 scientific
runtime foundation without absorbing unrelated historical worktree changes.

## Included In The Phase 7A Checkpoint

- Frozen protocol constants, runtime configuration, prompt provenance, action
  resolution, deterministic generation, native transport, shield modes, trace
  persistence, attempt lifecycle, verified runtime-lock loading, and append
  intent durability under `configs/iclr2027`, `provenance`, `schemas/iclr2027`,
  `dilu/driver_agent`, and `dilu/runtime`.
- Scientific integration changes in `evaluate_models_ollama.py`,
  `dilu/scenario/envScenario.py`, and `config.example.yaml`.
- The `jsonschema` runtime dependency and the focused guardrail regression
  additions required by the scientific path.
- ICLR 2027 design, implementation, smoke, and analysis plans.
- Scientific/runtime tests and their deterministic fixture helpers.
- `scripts/extract_original_dilu_prompt.py`, which verifies the historical
  prompt artifact byte-for-byte.

## Explicitly Excluded

- The 36 tracked deletions under `paper/`; the manuscript was moved to a
  separate paper checkout and this repository cleanup needs its own decision.
- `evaluate_non_llm_baselines.py`; its video-recording change predates the
  Phase 7A scientific runtime checkpoint.
- `dilu/runtime/path_utils.py`; the Windows long-path change is an independent
  filesystem fix.
- The five untracked `scripts/build_stress_v2_*.py` analysis builders; these
  belong to the earlier Stress V2 evidence workflow.

## Verification Baseline

- Mocked scientific/runtime plus required legacy tests: `315 passed` on
  2026-07-20.
- The same `315` tests passed from a detached worktree materialized from the
  exact staged Git index, not from the surrounding dirty working tree.
- Historical prompt extractor `--verify`, Black, Ruff, `compileall`, and staged
  `git diff --check` passed in that clean-index worktree.
- No live Ollama request, Gym episode, Stress V3 case opening, or research
  simulation was executed.
- No sensitive file is included; `.env` and local result artifacts remain
  outside the checkpoint.

## Errors Found During Clean-Index Verification

- The first detached checkout converted the exact historical prompt from LF to
  CRLF, changing 836 bytes to 846. The prompt now uses a path-specific `-text`
  Git attribute so checkout preserves the provenance bytes exactly.
- One provenance test assumed the ignored repository `temp/` directory already
  existed. It now uses an isolated system temporary directory and therefore
  works in a clean clone.
- The CLI-level provenance verification also exposed CRLF conversion in the
  generated YAML record. The YAML provenance artifact is now byte-preserving,
  and clean-checkout verification includes the extractor's `--verify` mode.

## Remaining Worktree After Checkpoint

The excluded paths are expected to remain dirty. Their presence must not be
interpreted as an incomplete Phase 7A checkpoint or silently folded into later
protocol-smoke commits.
