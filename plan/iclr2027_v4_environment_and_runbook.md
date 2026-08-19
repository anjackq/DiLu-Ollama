# ICLR 2027 V4 Python 3.12 Environment and Runbook

## Status

- Task 1 outcome: **DONE_WITH_CONCERNS**.
- External Python environment: **READY for offline diagnostics only**.
- Full offline pytest baseline: **BLOCKED** by a native Windows access violation.
- Host gate: **BLOCKED**. There is no valid approved V4 host-gate record, and the
  full offline suite did not complete cleanly.
- Model access: **NOT PERFORMED**. No probe-lock, smoke, run, baselines, Ollama
  request, or other model request was executed in this task.
- V3 is immutable interrupted evidence. **Never run V3 with `--resume`, never
  resend its ambiguous attempt, and never use the V3 artifact root for V4.**

## Explicit risk override for bounded V4 execution

On 2026-08-19, after being told that the offline full-suite baseline and host
gate were blocked by the new `0xC0000005` process crash and prior heterogeneous
BSODs, the user explicitly replied: `接受风险，继续implementation并且尽快跑simulation出结果`.

This authorization permits a bounded V4 probe, smoke, and small-batch run. It
does not establish that the host is stable, does not promote the incomplete
full-suite baseline, and does not authorize V3 recovery. Until the formal host
gate is implemented and passed, the manual containment policy is:

1. use only the external CPython 3.12 interpreter with `-X faulthandler`;
2. commit a clean V4 revision before authoring runtime locks;
3. start Stage 1 with at most five selected episodes per process;
4. run read-only ledger/summary/trace status validation after each boundary;
5. recheck the frozen V3 inventory and Windows crash evidence after each batch;
6. stop immediately on a native crash, reboot, new BugCheck/WHEA/NIC-hang
   evidence, failed/blocked/resumable/ambiguous attempt, or artifact drift.

All timestamps below are Europe/Berlin local time (`CEST`, UTC+02:00) on
2026-08-19.

## Fixed paths and environment facts

```powershell
$repo = 'C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal'
$py312 = 'C:\Users\WiCon\AppData\Local\Programs\Python\Python312\python.exe'
$venv = 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312'
$v4Python = 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Scripts\python.exe'
```

All recorded relative commands in this document ran with `$repo` as the
current working directory. A future reproducible wrapper must enter that
directory with `Push-Location` and restore the caller's directory in a
`finally` block.

- Repository: `C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal`
- Branch at Task 1 start: `feature/iclr2027-minimal-factorial`
- The external venv did not exist before creation; it was not deleted or
  overwritten.
- `uv`: 0.10.9 (`f675560f3 2026-03-06`)
- Python: CPython 3.12.3, MSC v.1938, 64-bit AMD64
- Platform: `Windows-11-10.0.26200-SP0`
- `sys.prefix`: `C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312`
- `sys.base_prefix`:
  `C:\Users\WiCon\AppData\Local\Programs\Python\Python312`
- `requirements.txt` SHA-256:
  `78b748217ca793f589600f3c718649a5485794d638b269e6af7a96a5a5dd68b7`
- No repo-local `scripts/codex_hook_emulation.py` was present.

### External environment freeze artifact

The main agent subsequently created this external, non-repository freeze
artifact:

- Path:
  `C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-runlogs\v4-py312-freeze-20260819.txt`
- Freeze command: `uv pip freeze --python $v4Python`
- Lines: 166
- Bytes: 3299
- SHA-256:
  `22575d99453fc418e5e8b52d3b8f89b68483e07ae8ef07a160ab4434560cf8e5`

## Creation and installation record

### 1. Create the external venv

Exact command:

```powershell
uv venv --python 'C:\Users\WiCon\AppData\Local\Programs\Python\Python312\python.exe' 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312'
```

- Started: `2026-08-19T08:35:45.7060439+02:00`
- Finished: `2026-08-19T08:35:45.7690997+02:00`
- Exit code: 0
- Created with CPython 3.12.3 from the required base interpreter.

### 2. Install tracked requirements and test tools

Exact command:

```powershell
uv pip install --python 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Scripts\python.exe' -r requirements.txt pytest ruff
```

- Started: `2026-08-19T08:35:56.7682837+02:00`
- Finished: `2026-08-19T08:36:25.9843003+02:00`
- Exit code: 0
- Resolver/install totals: 165 packages resolved, 130 prepared, 165 installed.
- Test tools: pytest 9.1.1; Ruff 0.16.3.

### 3. Initial dependency check

Exact command:

```powershell
uv pip check --python 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Scripts\python.exe'
```

- Started: `2026-08-19T08:36:39.7883196+02:00`
- Finished: `2026-08-19T08:36:39.8367703+02:00`
- Exit code: 0
- Output: `Checked 165 packages`; `All installed packages are compatible`.

## Interpreter and native-import verification

Exact command:

```powershell
& $v4Python -X faulthandler -c "import sys,numpy,gymnasium,highway_env,pygame; print('sys.executable=' + sys.executable); print('sys.version=' + sys.version.replace(chr(10), ' ')); print('numpy=' + numpy.__version__ + '|' + numpy.__file__); print('gymnasium=' + gymnasium.__version__ + '|' + gymnasium.__file__); print('highway_env=' + getattr(highway_env, '__version__', '<no __version__>') + '|' + highway_env.__file__); print('pygame=' + pygame.version.ver + '|' + pygame.__file__)"
```

- Started: `2026-08-19T08:36:39.8375417+02:00`
- Finished: `2026-08-19T08:36:41.1133248+02:00`
- Exit code: 0

Observed facts:

```text
sys.executable=C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Scripts\python.exe
sys.version=3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)]
numpy=1.26.4|C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\numpy\__init__.py
gymnasium=0.29.1|C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\gymnasium\__init__.py
highway_env=<no __version__>|C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\highway_env\__init__.py
pygame=2.6.1|C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pygame\__init__.py
```

Package metadata independently reports `highway-env==1.8.2`. The following
direct `uv pip show` command was not executed or captured during Task 1, so its
exact output and timestamp are unavailable and are not reconstructed:

```powershell
uv pip show --python $v4Python highway-env
```

The recorded 1.8.2 value came from the successful uv resolution output and a
later `importlib.metadata` fact check, not from `uv pip show`.

## Offline pytest baseline record

The test processes used the fixed absolute interpreter and `-X faulthandler`.
The process-local network/model variables below redirected common proxy-aware
HTTP clients and the configured Ollama endpoint to an unavailable local port.
This was a defense-in-depth offline context, **not a strict network sandbox**:
clients that ignore proxy/environment configuration were not technically
blocked by these variables alone. No model request was observed or performed.

```powershell
$env:HTTP_PROXY = 'http://127.0.0.1:9'
$env:HTTPS_PROXY = 'http://127.0.0.1:9'
$env:ALL_PROXY = 'http://127.0.0.1:9'
$env:NO_PROXY = ''
$env:OLLAMA_HOST = 'http://127.0.0.1:9'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
& $v4Python -X faulthandler -m pytest -q
```

Use the following reusable wrapper for any future authorized offline pytest
capture. It supports either the full suite or the recorded isolated test,
always runs inside `$repo`, restores every touched environment variable and the
caller's working directory, tees the complete combined stream to an external
log, and hashes that log. Select one mode per authorized invocation; this
wrapper does not itself authorize a retry while the host gate is BLOCKED.

```powershell
function Invoke-V4OfflinePytest {
    param(
        [Parameter(Mandatory)]
        [ValidateSet('full', 'isolated')]
        [string] $Mode,

        [Parameter(Mandatory)]
        [string] $LogPath
    )

    $repo = 'C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal'
    $v4Python = 'C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Scripts\python.exe'
    $envNames = @(
        'HTTP_PROXY',
        'HTTPS_PROXY',
        'ALL_PROXY',
        'NO_PROXY',
        'OLLAMA_HOST',
        'HF_HUB_OFFLINE',
        'TRANSFORMERS_OFFLINE',
        'GRADIO_ANALYTICS_ENABLED'
    )
    $savedEnv = @{}

    foreach ($name in $envNames) {
        $entry = Get-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        $savedEnv[$name] = [pscustomobject]@{
            Exists = $null -ne $entry
            Value = if ($null -eq $entry) { $null } else { $entry.Value }
        }
    }

    Push-Location -LiteralPath $repo
    try {
        $env:HTTP_PROXY = 'http://127.0.0.1:9'
        $env:HTTPS_PROXY = 'http://127.0.0.1:9'
        $env:ALL_PROXY = 'http://127.0.0.1:9'
        $env:NO_PROXY = ''
        $env:OLLAMA_HOST = 'http://127.0.0.1:9'
        $env:HF_HUB_OFFLINE = '1'
        $env:TRANSFORMERS_OFFLINE = '1'
        $env:GRADIO_ANALYTICS_ENABLED = 'False'

        if ($Mode -eq 'full') {
            & $v4Python -X faulthandler -m pytest -q 2>&1 |
                Tee-Object -LiteralPath $LogPath
        }
        else {
            & $v4Python -X faulthandler -m pytest -q `
                'tests/test_runtime_lock_authoring_transaction.py::RuntimeLockAuthoringTransactionTests::test_failure_at_every_publication_boundary_leaves_no_destination' `
                2>&1 | Tee-Object -LiteralPath $LogPath
        }
        $pytestExit = $LASTEXITCODE
    }
    finally {
        Pop-Location
        foreach ($name in $envNames) {
            if ($savedEnv[$name].Exists) {
                Set-Item -LiteralPath "Env:$name" -Value $savedEnv[$name].Value
            }
            else {
                Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
            }
        }
    }

    $logHash = Get-FileHash -LiteralPath $LogPath -Algorithm SHA256
    [pscustomobject]@{
        Mode = $Mode
        ExitCode = $pytestExit
        LogPath = $logHash.Path
        LogSha256 = $logHash.Hash.ToLowerInvariant()
    }
}
```

### Attempt 1: collection failure caused by undeclared `psutil`

- Started: `2026-08-19T08:37:19.1929872+02:00`
- Finished: `2026-08-19T08:37:32.1240356+02:00`
- Exit code: 2
- Pytest runtime reported: 8.99 s
- Final totals: 61 collection errors, 1 warning; no tests executed, so 0
  passed and 0 skipped.
- Common error: `ModuleNotFoundError: No module named 'psutil'`, originating
  from `dilu/runtime/energy_monitor.py:14` during imports.

This exposed a dependency declaration gap: production code imports `psutil`,
but the tracked `requirements.txt` used for this environment does not declare
it. The repository file was not modified because Task 1 ownership permits only
this runbook.

### External-only supplement and recheck

Exact commands:

```powershell
uv pip install --python $v4Python psutil
uv pip check --python $v4Python
& $v4Python -X faulthandler -c "import psutil; print('psutil=' + psutil.__version__ + '|' + psutil.__file__)"
```

- Supplement started: `2026-08-19T08:38:07.2793507+02:00`
- Supplement finished: `2026-08-19T08:38:07.4577020+02:00`
- Install exit code: 0
- Installed: `psutil==7.2.2` in the external venv only.
- Separate start/finish timestamps for the subsequent `uv pip check` and
  `psutil` import are unavailable because they were not captured separately;
  no timestamps are inferred or reconstructed for those two commands.
- Post-supplement check exit code: 0
- Post-supplement output: `Checked 166 packages`; all compatible.
- Import exit code: 0.

### Attempt 2: native Windows access violation

The same exact offline environment-variable block and pytest command above were
used.

- Started: `2026-08-19T08:38:16.3458072+02:00`
- Process ended: `2026-08-19T08:39:40.2855926+02:00`
- Exit code: `-1073741819` (`0xC0000005`, Windows access violation)
- Last printed progress boundary: 61%, followed by additional progress markers.
- Exact emitted progress markers before process death: 382 pass markers (`.`)
  and 1 skip marker (`s`).
- Final pass/skip/warning totals are **unavailable**, because the process died
  before pytest emitted its final summary. The marker counts are observations,
  not a completed-suite result.
- Complete raw pytest stdout/stderr was **not persisted at crash time** and
  cannot be reconstructed. This runbook preserves the complete captured
  faulthandler tail, but the progress-marker count and surrounding prose are
  not a raw transcript. A future authorized invocation must use the wrapper
  above to `Tee-Object` the combined stream to an external log and record its
  SHA-256.

Complete captured faulthandler tail:

```text
Windows fatal exception: access violation

Current thread 0x000021d0 (most recent call first):
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\dilu\runtime\_minimal_factorial_provenance.py", line 82 in _expect
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\dilu\runtime\_minimal_factorial_provenance.py", line 67 in validate_episode
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\dilu\runtime\_minimal_factorial_provenance.py", line 29 in validate_schedule_rows
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\dilu\runtime\_minimal_factorial_manifest.py", line 267 in validate_schedule
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\dilu\runtime\_minimal_factorial_manifest.py", line 276 in serialize_frozen_campaign
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\dilu\runtime\_minimal_factorial_manifest.py", line 296 in write_frozen_campaign_manifest
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\dilu\runtime\minimal_factorial_schedule.py", line 186 in write_frozen_campaign_manifest
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\dilu\runtime\_runtime_lock_authoring_workflow.py", line 279 in publish_staged_campaign
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\dilu\runtime\runtime_lock_authoring.py", line 114 in author_verified_runtime_locks
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\tests\test_runtime_lock_authoring.py", line 181 in run_authoring
  File "C:\Users\WiCon\Desktop\DiLu-Ollama-iclr2027-minimal\tests\test_runtime_lock_authoring_transaction.py", line 193 in test_failure_at_every_publication_boundary_leaves_no_destination
  File "C:\Users\WiCon\AppData\Local\Programs\Python\Python312\Lib\unittest\case.py", line 589 in _callTestMethod
  File "C:\Users\WiCon\AppData\Local\Programs\Python\Python312\Lib\unittest\case.py", line 634 in run
  File "C:\Users\WiCon\AppData\Local\Programs\Python\Python312\Lib\unittest\case.py", line 690 in __call__
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\unittest.py", line 410 in runtest
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\runner.py", line 184 in pytest_runtest_call
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\runner.py", line 250 in <lambda>
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\runner.py", line 361 in from_call
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\runner.py", line 249 in call_and_report
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\runner.py", line 139 in runtestprotocol
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\runner.py", line 118 in pytest_runtest_protocol
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\main.py", line 408 in pytest_runtestloop
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\main.py", line 384 in _main
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\main.py", line 330 in wrap_session
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\main.py", line 377 in pytest_cmdline_main
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\config\__init__.py", line 229 in _main
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\_pytest\config\__init__.py", line 253 in _console_main
  File "C:\Users\WiCon\Desktop\Dilu-Ollama-iclr2027-envs\v4-py312\Lib\site-packages\pytest\__main__.py", line 9 in <module>
  File "<frozen runpy>", line 88 in _run_code
  File "<frozen runpy>", line 198 in _run_module_as_main
```

### Single permitted isolated check

After the native failure, exactly one isolated offline check was permitted and
run; no further retry or stress test followed. The isolated command inherited
the same recorded proxy/offline environment-variable context used by the
second full-suite attempt.

Exact command:

```powershell
& $v4Python -X faulthandler -m pytest -q 'tests/test_runtime_lock_authoring_transaction.py::RuntimeLockAuthoringTransactionTests::test_failure_at_every_publication_boundary_leaves_no_destination'
```

- Started: `2026-08-19T08:40:26.4298468+02:00`
- Finished: `2026-08-19T08:40:43.7616045+02:00`
- Exit code: 0
- Totals: 1 passed, 0 skipped, 0 warnings; pytest runtime 16.48 s.

The isolated pass does not clear or explain the full-suite native crash and
does not change the BLOCKED baseline or host-gate status.

## Mandatory gate and next-action policy

1. Do not run probe-lock, smoke, claim `run`, baselines, or any other
   model-accessing command while the V4 host gate is BLOCKED. Probe-lock counts
   as model access.
2. Do not run more full-suite retries or stress loops merely to seek a passing
   result. Preserve the `0xC0000005` evidence for host diagnostics.
3. Do not approve a host-gate record until the separately specified diagnostic,
   administrator-review, stable-window, evidence-hash, and explicit user-risk
   acceptance requirements are satisfied.
4. Reuse this external environment by absolute interpreter path. Do not delete
   or silently recreate it; inspect it first and record any drift.
5. Keep V3 frozen. A V4 campaign, if later authorized after a valid host-gate
   approval, must use new V4 campaign identities and a sibling V4 output root.

Until those gates are satisfied, the only defensible Task 1 conclusion is:
**environment constructed and importable; full offline baseline BLOCKED;
host gate BLOCKED; no model access authorized.**
