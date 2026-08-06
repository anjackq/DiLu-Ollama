# ICLR 2027 v2 Claim Interruption Recovery

## Status

The v2 claim campaign, `iclr2027-minimal-factorial-v2`, is retained as
immutable interrupted evidence under `results/iclr2027_minimal_factorial_v2`.
It is non-promotable and must not be resumed, backfilled, or overwritten.

## Failure boundary

The attempt ledger records all 840 scheduled attempts as completed, while the
episode summaries contain 839 rows. The missing summary belongs to attempt
`episode-09df51ed3213d68ba66e1701e73294614b79c0c75b32bb7272b5ca0aa9771935`:

- model: `qwen3:0.6b`;
- condition: `c111`;
- case: `cut_in_then_recover_006`;
- seed: `39006`;
- committed requests/traces: 26;
- terminal record SHA-256:
  `71b3d7e72df30dce9e942b96edb9e1a5bc3a8fbff36f67035b8d637c979ed01a`.

Python exited natively with `0xC000001D` after the terminal attempt commit and
before the episode summary was published. The host evidence reviewed for this
interruption contained no WER, WHEA, or Disk record that established a more
specific cause. The evidence therefore supports the observed process-exit
boundary, not a hardware or driver root-cause claim.

## Versioned recovery

The new recovery campaign uses:

- claim campaign `iclr2027-minimal-factorial-v3`;
- smoke campaign `iclr2027-minimal-factorial-smoke-v3`;
- output root `results/iclr2027_minimal_factorial_v3`.

The summary-first recovery commits `99d298e`, `50a890d`, and `2944785` protect
v3 only. They do not repair or promote v2. The frozen v1 selection prefixes
remain unchanged, so v3 retains the registered case and seed schedule while
the versioned campaign IDs create distinct pair and attempt identities.

## Retained v2 evidence fingerprint

At recovery registration, the v2 root contained 42 files totaling 185252177
bytes. Its inventory SHA-256 is
`d5947d9198c1be2283a9c81e040edb62b2e287ac38608420d9dd70126e0294b0`.

The inventory serialization follows the v1 recovery memo convention: sort
Windows backslash-relative paths, serialize each row as
`relative_path|lowercase_sha256|length`, and join the UTF-8 rows with LF
separators and no trailing newline.

Core immutable artifacts are:

- `campaign_manifest.json`: SHA-256
  `a5fb572b2c09523fa2737794fd55b684c7acf3e8d6e0a87bfd7aa0fc39e2304e`;
- `campaign_attempts.jsonl`: SHA-256
  `a29196fb9970a10ce549e6061c57e4636891b7122d5d976a64ddcfecfc71aca3`,
  14354 lines;
- episode summaries: SHA-256
  `17c9da200440541459ce70dc7e898fa0c4f98da39070a946dab3e62a4a29123b`,
  839 rows;
- decision traces: SHA-256
  `eedb6274f1296836febac4d261f3c47e7dcc3576ba5709613b5b84b5bb4d1c69`,
  12674 rows.
