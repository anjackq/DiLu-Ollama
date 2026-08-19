# ICLR 2027 V3 Claim Interruption Recovery

## Status

The V3 claim campaign, `iclr2027-minimal-factorial-v3`, is retained as
immutable interrupted evidence under
`results/iclr2027_minimal_factorial_v3`. It is non-promotable and must not be
resumed, backfilled, overwritten, or used as a source of paper result rows.

The V3 root was reverified on 2026-08-19 and still contained 42 files totaling
111,834,047 bytes. Its inventory SHA-256 was unchanged:

`00eca60ab74f66594dca7aab2d7179931f72bab7626fc0e971d9882a66e70f3d`

The inventory serialization sorts Windows backslash-relative paths, serializes
each row as `relative_path|lowercase_sha256|length`, and joins the UTF-8 rows
with LF separators and no trailing newline.

## Failure boundary

Stage 1 completed 480/480 claim episodes. Stage 2 then completed 22 additional
episodes before Python exited natively with `0xC000001D`.

The retained claim state is:

- 502 completed episode summaries;
- one ambiguous started attempt;
- 337 unseen pending attempts;
- 8,476 attempt-ledger rows;
- 7,471 decision-trace rows;
- no baseline campaign.

The ambiguous attempt is:

- attempt: `episode-d008f123ba97f6f99f245d107a43012032be6806f93722888037566531709b7a`;
- request: `req-9a1f08e7da68f0a3d7391bda841166f9745c572dacd5b19a047e4018c14b7585`;
- model: `qwen3:0.6b`;
- condition: `c000`;
- case: `traffic_jam_escape_008`;
- simulator seed: `35008`;
- server response: HTTP 200;
- durable trace: line 7,471 with executed action `1` and disposition
  `ready_for_env_step`.

The attempt ledger contains `started` and `request_registered`, but no terminal
record. The episode summary was never published. The narrow evidence boundary
is after the successful response and durable decision-trace append, and before
the next durable episode lifecycle publication. Because the request may have
had effects in the interrupted process, exact-once recovery correctly forbids
resending it.

Core immutable artifact SHA-256 values at freeze time were:

- `llm_campaign/campaign_attempts.jsonl`:
  `b63bed1259b225719dfaaa70fb4861aa59d348d3573d42708f6d08a240bc28db`;
- `llm_campaign/episodes.jsonl`:
  `786bfa503baacfba316dfd00529515ad89164df66e921f7540e0a7176fd93768`;
- `llm_campaign/traces/decision_traces.jsonl`:
  `34375fda9a8e5887976a7d33008cc0eed2f3d6db4175228f27bb7d53cf8f62f0`.

## Host-stability boundary

The V3 Python exit cannot be treated as an isolated dependency failure. After
V3 was frozen, Windows recorded additional kernel failures between 2026-08-11
and 2026-08-17, including:

- three `SYSTEM_SERVICE_EXCEPTION (0x3B)` events with exception
  `0xC000001D`;
- `PAGE_FAULT_IN_NONPAGED_AREA (0x50)`;
- `KERNEL_SECURITY_CHECK_FAILURE (0x139)`;
- `KMODE_EXCEPTION_NOT_HANDLED (0x1E)` with exception `0xC0000096`;
- `HYPERVISOR_ERROR (0x20001)`.

This diversity is evidence of unresolved host-level instability. It does not
identify one CPU, RAM, driver, storage, hypervisor, or Python module as the
root cause. No claim-bearing model campaign may start until the dump audit and
host-stability gate in the implementation plan have passed.

## Versioned recovery contract

The next eligible recovery campaign will use:

- claim campaign `iclr2027-minimal-factorial-v4`;
- smoke campaign `iclr2027-minimal-factorial-smoke-v4`;
- output root `results/iclr2027_minimal_factorial_v4`.

V4 preserves the registered V1 selection prefixes, cases, simulator seeds,
generation-seed master, models, 2x2x2 conditions, prompts, parsers, shields,
scorers, and 840-row union schedule. Versioned campaign IDs create new pair,
request, trace, and attempt identities. Per-request derived generation seeds
are intentionally versioned because their derivation includes the versioned
pair identity; interrupted V2/V3 rows are not mixed with or compared as
replicates of V4.

A new process-level batch limit may select only the first N ledger-approved
pending rows for one invocation. It must not alter the frozen schedule,
denominator, ordering, identity derivation, completion gates, or analysis
inputs. Each subsequent invocation uses `--resume`; baselines remain forbidden
until the claim campaign independently validates as 840/840 completed with no
failed, blocked, resumable, pending, or ambiguous attempts.
