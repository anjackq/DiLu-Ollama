# ICLR 2027 V4 Minimal Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Register a new V4 minimal-factorial campaign and add fail-closed host gating plus crash-contained batch execution without changing the approved scientific design.

**Architecture:** Preserve V3 as immutable interrupted evidence and version only campaign/output identities. Add a small host-gate module invoked before every model-accessing CLI command, and an optional episode limit applied only after ledger reconciliation and exact-once approval. The complete frozen denominator and all promotion/analysis gates remain unchanged.

**Tech Stack:** External CPython 3.12 environment managed by uv, argparse, frozen dataclasses, JSONL attempt/trace stores, Gymnasium/highway-env, Ollama native chat, pytest, Ruff, PowerShell.

**2026-08-19 execution override:** The full 48-hour host gate remains the
recommended promotion path, but the user explicitly accepted the documented
unresolved host risk and requested immediate implementation and simulation.
That override permits only a new V4 root, `-X faulthandler`, process batches of
at most five episodes initially, read-only artifact validation after every
boundary, and immediate stop on any new native/system failure. It never permits
resuming V3 or resending its ambiguous request.

---

## File map

- plan/iclr2027_v3_claim_interruption_recovery.md: immutable V3 evidence and V4 recovery contract.
- plan/iclr2027_v4_environment_and_runbook.md: timestamped environment, host-gate, run, monitoring, and promotion commands.
- configs/iclr2027/minimal_factorial.yaml: V4 campaign IDs and sibling output root only.
- dilu/runtime/_minimal_factorial_manifest.py: registered V4 manifest SHA.
- dilu/runtime/iclr2027_v4_host_gate.py: pure record validation, V3 inventory verification, and Windows event collection.
- dilu/runtime/minimal_factorial_runner.py: optional max_episodes forwarding.
- dilu/runtime/_minimal_factorial_runner_execution.py: post-ledger batch selection and invocation count.
- dilu/runtime/_minimal_factorial_runner_status.py: read-only incremental ledger/summary/trace validation.
- dilu/runtime/_minimal_factorial_analysis_validation.py: register V4 only for final analysis.
- scripts/run_iclr2027_minimal_factorial.py: host-gate enforcement, max-episodes CLI, and faulthandler.
- tests/test_iclr2027_v4_host_gate.py: fail-closed host-gate tests.
- tests/test_minimal_factorial_manifest.py: V4 constants and full scientific invariants.
- tests/test_minimal_factorial_schedule.py: V4/V3 schedule equivalence and identity disjointness.
- tests/test_minimal_factorial_runner.py: public batch forwarding.
- tests/test_minimal_factorial_runner_execution.py: exact-once-before-limit behavior.
- tests/test_minimal_factorial_runner_cli.py: CLI validation and gate ordering.
- tests/runtime_factorization_support.py: typed runtime helper accepts an explicit scientific identity.
- tests/test_minimal_factorial_analysis_validation.py: V4 accepted; V2/V3 rejected.
- tests/test_minimal_factorial_analysis_artifacts.py: V4 provenance in final artifacts.

## Fixed interpreter

Every repository Python command below uses:

    $v4Python = 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Scripts\python.exe'

Bare python, the Anaconda Python 3.13 interpreter, and the V3 artifact root are not permitted for V4 tests or runs.

### Task 0: Preserve the unresolved host boundary

**Files:**
- Read: Windows WER/System event evidence
- Read: administrator-copied BugCheck dumps
- Modify: plan/iclr2027_v3_claim_interruption_recovery.md

- [ ] Record the repeated FLTMGR 0x3B/C000001D, guard-icall 0x139, BFS 0x50, HAL 0x1E/C0000096, and hypervisor 0x20001 buckets without assigning an unsupported single root cause.
- [ ] Keep probe-lock, smoke, claim stages, and baselines blocked until Task 5's
      `approve` command produces a valid timestamped host-gate record. A
      `snapshot` is deliberately unapproved and cannot pass. Probe-lock counts
      as model access.
- [ ] Reverify V3: 42 files, 111,834,047 bytes, inventory SHA-256 00eca60ab74f66594dca7aab2d7179931f72bab7626fc0e971d9882a66e70f3d.

### Task 1: Build the external Python 3.12 environment

**Files:**
- Create outside Git: C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312
- Create: plan/iclr2027_v4_environment_and_runbook.md

- [ ] Create the external environment.

    uv venv --python 'C:\Users\WiCon\AppData\Local\Programs\Python\Python312\python.exe' 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312'
    $v4Python = 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Scripts\python.exe'

- [ ] Install tracked runtime dependencies plus test-only tools.

    uv pip install --python $v4Python -r requirements.txt pytest ruff
    uv pip check --python $v4Python

- [ ] Verify interpreter and native imports.

    & $v4Python -X faulthandler -c "import sys,numpy,gymnasium,highway_env,pygame; print(sys.executable); print(sys.version); print(numpy.__version__,numpy.__file__); print(gymnasium.__version__,gymnasium.__file__); print(highway_env.__file__); print(pygame.version.ver,pygame.__file__)"

- [ ] Capture the initial offline baseline.

    & $v4Python -X faulthandler -m pytest -q

Record exact pass/skip/warning totals in the runbook.

### Task 2: Register V4 and lock scientific invariants

**Files:**
- Modify: configs/iclr2027/minimal_factorial.yaml
- Modify: dilu/runtime/_minimal_factorial_manifest.py
- Modify: tests/test_minimal_factorial_manifest.py
- Modify: tests/test_minimal_factorial_schedule.py

- [ ] Write failing manifest assertions expecting claim V4, smoke V4, and results/iclr2027_minimal_factorial_v4.
- [ ] Retain explicit expected assertions for model slots/tags, endpoint, think mode, temperature, context/output limits, timeout, generation-seed master, runtime sources, fixed parser/resolver/fallback/retry/trace policy, simulation flags, scoring, bootstrap, V1 selection prefixes, and all non-root output subdirectories.
- [ ] Construct a V3 comparison manifest by replacing only the two campaign IDs and output root on the loaded V4 manifest. Build both schedules from the same runtime snapshot and bindings.
- [ ] Assert complete stage/case/simulator-seed equality:

    self.assertEqual(
        [(r.stage, r.case_id, r.simulator_seed) for r in v4_union],
        [(r.stage, r.case_id, r.simulator_seed) for r in v3_union],
    )

- [ ] Assert full condition payload equality:

    self.assertEqual(
        [r.condition.to_canonical_dict() for r in v4_union],
        [r.condition.to_canonical_dict() for r in v3_union],
    )

- [ ] Assert generation_seed_master remains 20270728 and pair/attempt identity sets are disjoint.
- [ ] Retain exact selection-hash assertions and prove the first V3/V4
      `generation_context(0).request_id` values differ through the typed-runtime
      procedure below.
- [ ] Extend `tests.runtime_factorization_support.runtime` with an optional
      `episode_identity: ScientificEpisodeIdentity | None = None`. When supplied,
      use that identity to create the real temporary `ScientificAttemptLedger`,
      `ScientificTraceWriter`, verified lock, and typed client. Do not use mocks
      for the runtime type boundary.
- [ ] For the request-ID test, convert the first V3/V4 scheduled rows to two
      `ScientificEpisodeIdentity` values, create each runtime in a separate
      `TemporaryDirectory` child through that helper, call `begin_attempt()`,
      and compare `generation_context(0).request_id`. This exercises the real
      ledger registration path and is executable under the production runtime
      type checks.
- [ ] Do not assert derived model seeds match: pair identity intentionally versions them. Simulator seeds and the seed master must match.
- [ ] Verify RED.

    & $v4Python -X faulthandler -m pytest -q tests/test_minimal_factorial_manifest.py tests/test_minimal_factorial_schedule.py

Expected: V4 assertions fail because the manifest is still V3.

- [ ] Change only campaign_id, smoke_campaign_id, and outputs.root in the YAML.
- [ ] Recompute the canonical manifest SHA.

    & $v4Python -c "from pathlib import Path; import yaml; from dilu.runtime._minimal_factorial_schedule_support import canonical_sha256; p=Path('configs/iclr2027/minimal_factorial.yaml'); print(canonical_sha256(yaml.safe_load(p.read_text(encoding='utf-8'))))"

- [ ] Replace only MANIFEST_SHA and verify GREEN with the same focused tests.

### Task 3: Add deterministic post-ledger batch limits

**Files:**
- Modify: dilu/runtime/minimal_factorial_runner.py
- Modify: dilu/runtime/_minimal_factorial_runner_execution.py
- Modify: scripts/run_iclr2027_minimal_factorial.py
- Modify: dilu/runtime/_minimal_factorial_runner_status.py
- Modify: tests/test_minimal_factorial_runner.py
- Modify: tests/test_minimal_factorial_runner_execution.py
- Modify: tests/test_minimal_factorial_runner_cli.py

- [ ] Extend the public forwarding test:

    result = runner.run_claim_stage(
        Path("manifest.yaml"), stage="stage2", resume=True, max_episodes=20
    )
    self.assertIs(result, mock.sentinel.summary)
    self.assertEqual(execute.call_args.kwargs["max_episodes"], 20)

- [ ] Add execution validation tests asserting True, False, 0, and -1 raise ValueError while None and 1 are accepted.
- [ ] Write an exact-once-before-limit test using five rows: completed, request-owned STARTED, ledger-resumable STARTED, unseen A, unseen B. Fake can_resume returns false only for request-owned. Invoke max_episodes=2 and assert:

    self.assertEqual(executed_ids, [resumable_id, unseen_a_id])
    self.assertNotIn(request_owned_id, executed_ids)
    self.assertEqual(result.selected_this_invocation, 2)
    self.assertEqual(result.scheduled, len(all_scheduled_rows))
    self.assertEqual(completion_checker.call_args.args[0], all_denominator_rows)

- [ ] Add a separate assertion that max_episodes=None executes every ledger-approved pending row.
- [ ] Add CLI tests forwarding positive 20 and rejecting zero, negative, boolean-equivalent invalid values, and non-integers before runner invocation. Mock a passing host gate to isolate forwarding.
- [ ] Add read-only status tests for `artifact_validation`. For each frozen
      campaign, require ledger syntax/evidence validation, unique summaries,
      exact equality between completed attempt IDs and summary IDs, and exact
      equality between each summary's ordered `scientific_trace_references` and
      the validated trace writer references for that attempt. Corrupted hashes,
      dangling references, duplicate summaries, and completed-without-summary
      must produce `valid: false` plus errors. Preserve the existing `groups`
      and `totals` fields. Add `claim_promotion_allowed`, which is true only
      when the claim schedule is exactly 840, every claim identity is uniquely
      completed, the canonical summary denominator is exactly 840, incremental
      trace joins validate, and no pending/failed/blocked/resumable/ambiguous
      attempt exists.
- [ ] Verify RED.

    & $v4Python -X faulthandler -m pytest -q tests/test_minimal_factorial_runner.py tests/test_minimal_factorial_runner_execution.py tests/test_minimal_factorial_runner_cli.py

- [ ] Append selected_this_invocation: int = 0 at the end of RunSummary, preserving positional fixtures.
- [ ] Change production RunSummary construction to keywords and set len(selected).
- [ ] Forward max_episodes: int | None = None through run_claim_stage, _execute_campaign, and execute_campaign. Defensively reject booleans, zero, and negatives in execution.
- [ ] Preserve the exact order:

    pending = pending_selector(scheduled_rows, statuses, resume=artifact_resume)
    pending = _ledger_approved_rows(pending, statuses, ledger)
    selected = pending if max_episodes is None else pending[:max_episodes]

- [ ] Iterate only selected; keep full scheduled_rows, denominator_rows, completion-checker inputs, and promotion gates.
- [ ] Enable faulthandler at main start when not already enabled; external commands also use -X faulthandler.
- [ ] Keep `status` gate-exempt and read-only, but append
      `artifact_validation: {valid, errors, claim_promotion_allowed}` from the incremental
      ledger/summary/trace checks. The final registered analysis remains the
      stricter 840-row scientific join gate.
- [ ] Verify GREEN with the same focused tests.

### Task 4: Register V4 analysis only

**Files:**
- Modify: dilu/runtime/_minimal_factorial_analysis_validation.py
- Modify: tests/test_minimal_factorial_analysis_validation.py
- Modify: tests/test_minimal_factorial_analysis_artifacts.py

- [ ] Write failing tests expecting the synthetic bundle and published tables to use V4.
- [ ] Add explicit tests that claims copied with V2 and V3 IDs both return blocked with a campaign-registration error.
- [ ] Verify RED.

    & $v4Python -X faulthandler -m pytest -q tests/test_minimal_factorial_analysis_validation.py tests/test_minimal_factorial_analysis_artifacts.py

- [ ] Replace only the registered analysis campaign ID with V4. Do not accept aliases or relax the 840/360 denominators or join checks.
- [ ] Verify GREEN with the same focused tests.

### Task 5: Add an enforceable pre-model host gate

**Files:**
- Create: dilu/runtime/iclr2027_v4_host_gate.py
- Modify: scripts/run_iclr2027_minimal_factorial.py
- Create: tests/test_iclr2027_v4_host_gate.py
- Modify: tests/test_minimal_factorial_runner_cli.py

- [ ] Write pure tests with temporary dump/report files and injected event tuples.
      A valid record has schema `iclr2027.v4_host_gate.v1`, `approved: true`,
      nonempty `approved_by`, `admin_reviewed_by`, `approved_at_utc`,
      `checkpoint_utc`, `minimum_stable_hours >= 48`, explicit
      `user_risk_acceptance: true`, the expected V3 inventory, SHA-256 hashes
      for every available recent dump plus archived WER evidence for any
      unavailable dump, and a separately hashed diagnostic-resolution report. The report
      schema is `iclr2027.v4_host_resolution.v1`, has
      `diagnostic_status: resolved_or_mitigated`, nonempty `repair_actions`,
      `bios_intel_defaults_confirmed: true`, nonempty analyzed dump-report hash
      mappings, and a nonempty residual-risk statement.
- [ ] Assert separate rejection for missing record, approval false, empty user
      or admin approver, missing risk acceptance, unresolved diagnostic report,
      empty repair evidence, malformed/future checkpoint, less than 48 elapsed
      hours, V3 mismatch, missing/hash-drifted dump or diagnostic evidence,
      post-checkpoint BugCheck/Kernel-Power/WHEA event, and Realtek tx-hang
      event. Assert a valid record returns no errors.
- [ ] For probe-lock, smoke, run, and baselines, mock gate validation to raise and assert the runner/model function was never called. Assert status remains read-only and gate-exempt.
- [ ] Verify RED.

    & $v4Python -X faulthandler -m pytest -q tests/test_iclr2027_v4_host_gate.py tests/test_minimal_factorial_runner_cli.py

- [ ] Implement frozen record types, type hints, a pure validator, the documented
      V3 inventory serialization, dump/report hashing, and injected Windows
      event collection. Collector errors fail closed.
- [ ] Implement a module CLI with three non-model commands:
      `snapshot --v3-root ... --dump ... --crash-report ... --output ...`
      creates an unapproved,
      fail-closed checkpoint record; `approve --draft ... --resolution-report
      ... --approved-by ... --admin-reviewed-by ... --i-accept-host-risk
      --output ...` recomputes all hashes/events and refuses approval before 48
      stable hours; `validate --record ...` independently revalidates current
      files and events. `approve` must not be invoked by an agent without fresh,
      explicit user approval after the diagnostic evidence exists.
- [ ] Add global --host-gate-record PATH. Validate before dispatching probe-lock, smoke, run, or baselines; missing/invalid/stale evidence exits non-zero. The gate record remains outside Git/results.
- [ ] Verify GREEN with the same focused tests.

### Task 6: Full verification and commit

- [ ] Run all focused tests.

    & $v4Python -X faulthandler -m pytest -q tests/test_iclr2027_v4_host_gate.py tests/test_minimal_factorial_manifest.py tests/test_minimal_factorial_schedule.py tests/test_minimal_factorial_runner.py tests/test_minimal_factorial_runner_execution.py tests/test_minimal_factorial_runner_cli.py tests/test_minimal_factorial_analysis_validation.py tests/test_minimal_factorial_analysis_artifacts.py

- [ ] Run Ruff checks over exactly the changed Python files.

    & $v4Python -m ruff check dilu/runtime/iclr2027_v4_host_gate.py dilu/runtime/_minimal_factorial_manifest.py dilu/runtime/minimal_factorial_runner.py dilu/runtime/_minimal_factorial_runner_execution.py dilu/runtime/_minimal_factorial_runner_status.py dilu/runtime/_minimal_factorial_analysis_validation.py scripts/run_iclr2027_minimal_factorial.py tests/runtime_factorization_support.py tests/test_iclr2027_v4_host_gate.py tests/test_minimal_factorial_manifest.py tests/test_minimal_factorial_schedule.py tests/test_minimal_factorial_runner.py tests/test_minimal_factorial_runner_execution.py tests/test_minimal_factorial_runner_cli.py tests/test_minimal_factorial_analysis_validation.py tests/test_minimal_factorial_analysis_artifacts.py

    & $v4Python -m ruff format --check dilu/runtime/iclr2027_v4_host_gate.py dilu/runtime/_minimal_factorial_manifest.py dilu/runtime/minimal_factorial_runner.py dilu/runtime/_minimal_factorial_runner_execution.py dilu/runtime/_minimal_factorial_runner_status.py dilu/runtime/_minimal_factorial_analysis_validation.py scripts/run_iclr2027_minimal_factorial.py tests/runtime_factorization_support.py tests/test_iclr2027_v4_host_gate.py tests/test_minimal_factorial_manifest.py tests/test_minimal_factorial_schedule.py tests/test_minimal_factorial_runner.py tests/test_minimal_factorial_runner_execution.py tests/test_minimal_factorial_runner_cli.py tests/test_minimal_factorial_analysis_validation.py tests/test_minimal_factorial_analysis_artifacts.py

- [ ] Run full suite and diff checks.

    & $v4Python -X faulthandler -m pytest -q
    git diff --check
    git status --short

- [ ] Recompute V3 inventory with the exact registered serialization and require
      the 42-file/count/size/hash boundary:

    $v3Root = (Resolve-Path 'results\iclr2027_minimal_factorial_v3').Path
    $inventoryRows = Get-ChildItem -LiteralPath $v3Root -Recurse -File |
        ForEach-Object {
            $relative = $_.FullName.Substring($v3Root.Length + 1)
            $sha = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            [pscustomobject]@{ Relative = $relative; Length = $_.Length; Row = "$relative|$sha|$($_.Length)" }
        } | Sort-Object Relative
    $inventoryText = ($inventoryRows.Row -join "`n")
    $inventoryHash = [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData(
            [Text.Encoding]::UTF8.GetBytes($inventoryText)
        )
    ).ToLowerInvariant()
    $inventory = [pscustomobject]@{
        FileCount = $inventoryRows.Count
        TotalBytes = ($inventoryRows | Measure-Object Length -Sum).Sum
        Sha256 = $inventoryHash
    }
    $inventory | ConvertTo-Json
    if ($inventory.FileCount -ne 42 -or
        $inventory.TotalBytes -ne 111834047 -or
        $inventory.Sha256 -ne '00eca60ab74f66594dca7aab2d7179931f72bab7626fc0e971d9882a66e70f3d') {
        throw 'Frozen V3 inventory drifted.'
    }
- [ ] Run the post-edit hook surrogate if present; otherwise record that no repo-local helper exists.
- [ ] Commit the tested implementation.

    git add -- configs/iclr2027/minimal_factorial.yaml dilu/runtime/_minimal_factorial_manifest.py dilu/runtime/minimal_factorial_runner.py dilu/runtime/_minimal_factorial_runner_execution.py dilu/runtime/_minimal_factorial_runner_status.py dilu/runtime/_minimal_factorial_analysis_validation.py dilu/runtime/iclr2027_v4_host_gate.py scripts/run_iclr2027_minimal_factorial.py tests/runtime_factorization_support.py tests/test_iclr2027_v4_host_gate.py tests/test_minimal_factorial_manifest.py tests/test_minimal_factorial_schedule.py tests/test_minimal_factorial_runner.py tests/test_minimal_factorial_runner_execution.py tests/test_minimal_factorial_runner_cli.py tests/test_minimal_factorial_analysis_validation.py tests/test_minimal_factorial_analysis_artifacts.py plan/iclr2027_v3_claim_interruption_recovery.md plan/iclr2027_v4_minimal_recovery_implementation_plan.md plan/iclr2027_v4_environment_and_runbook.md
    git commit -m "fix(experiment): register crash-contained factorial v4"
    git status --short

The runtime snapshot requires an exact committed clean revision before campaign artifacts are authored.

### Task 7: Execute gated V4 batches

**Files:**
- Create only after gate pass: results/iclr2027_minimal_factorial_v4/**
- External logs: C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-runlogs\**

- [ ] Validate the host gate independently and verify no active runner, exact Ollama tags/digests, sufficient disk, external Python 3.12, clean Git, and unchanged V3.
- [ ] Set:

    $gate = 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-runlogs\v4-host-gate.json'
    $runLogRoot = 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-runlogs'
    New-Item -ItemType Directory -Force -Path $runLogRoot | Out-Null

- [ ] First complete the actual diagnosis/mitigation and final reboot. An
      administrator copies every still-available dump and exports WER evidence
      for the two unavailable Aug-11 dumps into these exact external paths:

    $evidenceRoot = Join-Path $runLogRoot 'host-evidence'
    $dumpRoot = Join-Path $evidenceRoot 'dumps'
    $reportRoot = Join-Path $evidenceRoot 'reports'
    $availableDumps = @(
        (Join-Path $dumpRoot '081126-15109-01.dmp'),
        (Join-Path $dumpRoot '081126-7890-01.dmp'),
        (Join-Path $dumpRoot '081126-10625-01.dmp'),
        (Join-Path $dumpRoot '081726-12375-01.dmp'),
        (Join-Path $dumpRoot '081726-7781-01.dmp')
    )
    $crashReports = @(
        (Join-Path $reportRoot '081126-16625-WER.txt'),
        (Join-Path $reportRoot '081126-7687-WER.txt'),
        (Join-Path $reportRoot '081126-15109-WinDbg.txt'),
        (Join-Path $reportRoot '081126-7890-WinDbg.txt'),
        (Join-Path $reportRoot '081126-10625-WinDbg.txt'),
        (Join-Path $reportRoot '081726-12375-WinDbg.txt'),
        (Join-Path $reportRoot '081726-7781-WinDbg.txt')
    )
    if (@($availableDumps + $crashReports | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -ne 0) {
        throw 'The complete seven-crash evidence set has not been frozen.'
    }

- [ ] Create the exact external diagnostic-resolution JSON after the mitigation.
      It is not a checkbox-only waiver: replace the placeholder action and
      residual-risk text with observed facts.

    $dumpReports = $crashReports
    $dumpReportHashes = [ordered]@{}
    foreach ($report in $dumpReports) {
        $resolvedReport = (Resolve-Path -LiteralPath $report).Path
        $dumpReportHashes[$resolvedReport] = `
            (Get-FileHash -LiteralPath $resolvedReport -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $resolution = [ordered]@{
        schema = 'iclr2027.v4_host_resolution.v1'
        diagnostic_status = 'resolved_or_mitigated'
        repair_actions = @(
            '<concrete BIOS, CPU/RMA, RAM, hypervisor, or driver action actually completed>'
        )
        bios_intel_defaults_confirmed = $true
        analyzed_dump_reports = $dumpReportHashes
        residual_risk = '<specific residual risk after the completed mitigation>'
    }
    $resolutionPath = Join-Path $runLogRoot 'v4-host-resolution.json'
    $resolution | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath $resolutionPath -Encoding utf8NoBOM

- [ ] Only after the final repair/reboot, author the fail-closed checkpoint.
      This starts the stability window; pre-repair uptime never counts:

    $gateDraft = Join-Path $runLogRoot 'v4-host-gate.draft.json'
    $snapshotArgs = @(
        '-X', 'faulthandler', '-m', 'dilu.runtime.iclr2027_v4_host_gate',
        'snapshot', '--v3-root', (Resolve-Path 'results\iclr2027_minimal_factorial_v3').Path
    )
    foreach ($dump in $availableDumps) { $snapshotArgs += @('--dump', $dump) }
    foreach ($report in $crashReports) { $snapshotArgs += @('--crash-report', $report) }
    $snapshotArgs += @('--output', $gateDraft)
    & $v4Python @snapshotArgs
    if ($LASTEXITCODE -ne 0) { throw 'Post-repair host checkpoint snapshot failed.' }

- [ ] Observe at least 48 hours after that checkpoint with zero gated events.
      Then obtain fresh explicit user approval and administrator review and run
      `approve` exactly once:

    & $v4Python -X faulthandler -m dilu.runtime.iclr2027_v4_host_gate approve `
        --draft $gateDraft `
        --resolution-report $resolutionPath `
        --approved-by '<explicit-user-name>' `
        --admin-reviewed-by '<administrator-name>' `
        --i-accept-host-risk `
        --output $gate
    if ($LASTEXITCODE -ne 0) { throw 'Host gate approval failed.' }
    & $v4Python -X faulthandler -m dilu.runtime.iclr2027_v4_host_gate validate --record $gate
    if ($LASTEXITCODE -ne 0) { throw 'Host gate validation failed.' }

- [ ] Run probe-lock exactly once after the gate passes. Require exit zero and
      verify the authored preflight contains 16 lock pairs:

    & $v4Python -X faulthandler scripts\run_iclr2027_minimal_factorial.py `
        --manifest configs\iclr2027\minimal_factorial.yaml `
        --host-gate-record $gate probe-lock 2>&1 |
        Tee-Object -FilePath (Join-Path $runLogRoot 'v4-01-probe-lock.log')
    if ($LASTEXITCODE -ne 0) { throw 'V4 probe-lock failed.' }

- [ ] Run fresh smoke without resume. Require 16/16 completed and zero
      failed/blocked/ambiguous/resumable/pending:

    & $v4Python -X faulthandler scripts\run_iclr2027_minimal_factorial.py `
        --manifest configs\iclr2027\minimal_factorial.yaml `
        --host-gate-record $gate smoke 2>&1 |
        Tee-Object -FilePath (Join-Path $runLogRoot 'v4-02-smoke.log')
    if ($LASTEXITCODE -ne 0) { throw 'V4 smoke failed.' }

- [ ] Run the first Stage 1 batch without `--resume`:

    & $v4Python -X faulthandler scripts\run_iclr2027_minimal_factorial.py `
        --manifest configs\iclr2027\minimal_factorial.yaml `
        --host-gate-record $gate run --stage stage1 --max-episodes 20 2>&1 |
        Tee-Object -FilePath (Join-Path $runLogRoot 'v4-03-stage1-batch001.log')
    if ($LASTEXITCODE -ne 0) { throw 'V4 Stage 1 batch 001 failed.' }
- [ ] Run Stage 1 in batches of 20. The first invocation omits resume; all later invocations use:

    $stage1Batch = 2
    $stage1Log = Join-Path $runLogRoot ('v4-03-stage1-batch{0:d3}.log' -f $stage1Batch)
    & $v4Python -X faulthandler scripts\run_iclr2027_minimal_factorial.py `
        --manifest configs\iclr2027\minimal_factorial.yaml `
        --host-gate-record $gate run --stage stage1 --max-episodes 20 --resume 2>&1 |
        Tee-Object -FilePath $stage1Log
    if ($LASTEXITCODE -ne 0) { throw "V4 Stage 1 batch $stage1Batch failed." }

- [ ] After each process boundary, run status, validate ledger/summary/trace joins, recompute V3, and query new crash/WHEA/NIC events. Stop on failed, blocked, resumable, or ambiguous.
- [ ] The exact read-only status command after every process boundary is:

    $statusJson = & $v4Python -X faulthandler scripts\run_iclr2027_minimal_factorial.py `
        --manifest configs\iclr2027\minimal_factorial.yaml status
    if ($LASTEXITCODE -ne 0) { throw 'V4 status failed.' }
    $statusJson | Tee-Object -FilePath (Join-Path $runLogRoot 'v4-status-latest.json')
    $status = $statusJson | ConvertFrom-Json
    if (-not $status.artifact_validation.valid) { throw ($status.artifact_validation.errors -join '; ') }
    $claimGroups = @($status.groups | Where-Object { $_.stage -ne 'smoke' })
    foreach ($name in @('failed', 'blocked', 'resumable', 'ambiguous')) {
        $count = ($claimGroups | Measure-Object -Property $name -Sum).Sum
        if ($count -ne 0) { throw "V4 claim status has $count $name attempts." }
    }
    & $v4Python -X faulthandler -m dilu.runtime.iclr2027_v4_host_gate validate --record $gate
    if ($LASTEXITCODE -ne 0) { throw 'Host gate drifted after a process boundary.' }

      Re-run the independent host-gate `validate` command and the exact V3
      inventory block from Task 6. Any nonzero failed/blocked/resumable/ambiguous
      count, artifact-validation error, new host event, or V3 drift stops the run.
- [ ] Run Stage 2 batches:

    $stage2Batch = 1
    $stage2Log = Join-Path $runLogRoot ('v4-04-stage2-batch{0:d3}.log' -f $stage2Batch)
    & $v4Python -X faulthandler scripts\run_iclr2027_minimal_factorial.py `
        --manifest configs\iclr2027\minimal_factorial.yaml `
        --host-gate-record $gate run --stage stage2 --max-episodes 20 --resume 2>&1 |
        Tee-Object -FilePath $stage2Log
    if ($LASTEXITCODE -ne 0) { throw "V4 Stage 2 batch $stage2Batch failed." }

- [ ] Require the final claim gate with executable assertions before baselines:

    $statusJson = & $v4Python -X faulthandler scripts\run_iclr2027_minimal_factorial.py `
        --manifest configs\iclr2027\minimal_factorial.yaml status
    if ($LASTEXITCODE -ne 0) { throw 'Final V4 status failed.' }
    $status = $statusJson | ConvertFrom-Json
    $claimGroups = @($status.groups | Where-Object { $_.stage -ne 'smoke' })
    $claimScheduled = ($claimGroups | Measure-Object scheduled -Sum).Sum
    $claimCompleted = ($claimGroups | Measure-Object completed -Sum).Sum
    $claimPending = ($claimGroups | Measure-Object pending -Sum).Sum
    $claimBad = 0
    foreach ($name in @('failed', 'blocked', 'resumable', 'ambiguous')) {
        $claimBad += ($claimGroups | Measure-Object -Property $name -Sum).Sum
    }
    if ($claimScheduled -ne 840 -or $claimCompleted -ne 840 -or
        $claimPending -ne 0 -or $claimBad -ne 0 -or
        -not $status.artifact_validation.valid -or
        -not $status.artifact_validation.claim_promotion_allowed) {
        throw 'V4 final 840/840 promotion gate failed.'
    }
- [ ] Only then run baselines with the same gate/interpreter. Require 360 calibration rows and complete baselines/calibration_validation.json.
- [ ] Run baselines and capture the exact log:

    & $v4Python -X faulthandler scripts\run_iclr2027_minimal_factorial.py `
        --manifest configs\iclr2027\minimal_factorial.yaml `
        --host-gate-record $gate baselines 2>&1 |
        Tee-Object -FilePath (Join-Path $runLogRoot 'v4-90-baselines.log')
    if ($LASTEXITCODE -ne 0) { throw 'V4 baselines failed.' }

- [ ] Run registered analysis with exact input paths:

    $v4Root = 'results\iclr2027_minimal_factorial_v4'
    & $v4Python -X faulthandler scripts\analyze_iclr2027_minimal_factorial.py `
        --manifest "$v4Root\llm_campaign\campaign_manifest.json" `
        --episodes "$v4Root\llm_campaign\episodes.jsonl" `
        --baseline-report "$v4Root\baselines\non_llm_baseline_report.json" `
        --baseline-episodes "$v4Root\baselines\episode_metrics.csv" `
        --output-root "$v4Root\analysis" 2>&1 |
        Tee-Object -FilePath (Join-Path $runLogRoot 'v4-99-analysis.log')
    if ($LASTEXITCODE -ne 0) { throw 'V4 registered analysis failed.' }
    $analysisValidation = Get-Content -Raw "$v4Root\analysis\analysis_validation.json" | ConvertFrom-Json
    if ($analysisValidation.status -ne 'complete') { throw 'V4 analysis did not complete.' }
