# ICLR 2027 v1 Smoke Failure Recovery

## Status

The first live smoke campaign, `iclr2027-minimal-factorial-smoke-v1`, is retained
as failed infrastructure evidence under `results/iclr2027_minimal_factorial`.
It is not eligible for promotion and must not be resumed or overwritten.

## Failure boundary

All 16 scheduled attempts reached a terminal `trace_write_failure` before the
first committed decision trace. The simulator exposed the valid availability
order `[1, 0, 2, 3, 4]`, while the trace contract requires sorted unique action
IDs. Ollama requests returned successfully; the shared failure occurred while
composing the first scientific trace.

The runtime fix canonicalizes action availability at the driver ingress before
the immutable tuple is retained. It does not change the action domain, fixed
IDLE fallback, parser, shield, transport, model, benchmark, or factorial design.

## Versioned recovery

The repaired pre-claim campaign uses:

- claim campaign `iclr2027-minimal-factorial-v2`;
- smoke campaign `iclr2027-minimal-factorial-smoke-v2`;
- output root `results/iclr2027_minimal_factorial_v2`.

The v1 selection prefixes remain frozen so v2 selects the same smoke and claim
cases. Changing the campaign IDs creates distinct pair and attempt identities;
changing the output root prevents any overwrite of v1 evidence. S1 and every
runtime lock must be regenerated from the clean v2 code revision before the v2
smoke is run.

## Retained v1 evidence fingerprint

At recovery registration, the v1 root contained 37 files with inventory digest
`398c5350e6f56633462e60baca71a37da3d63f580245645c1d5e2080f8868392`.
The inventory serialization sorts Windows backslash-relative paths and joins
`relative_path|lowercase_sha256|length` rows as UTF-8 with LF separators and no
trailing newline. The v1 smoke manifest SHA-256 was
`705fdc6645850c888217cfd46fac9b9bb1bd452bea296c33401441ec1b55c98c`;
the v1 attempt ledger SHA-256 was
`79353b4ca8d38e876ca80b08d5a543eab31a2d4868221a775ce1e0243cdcd8e2`.
