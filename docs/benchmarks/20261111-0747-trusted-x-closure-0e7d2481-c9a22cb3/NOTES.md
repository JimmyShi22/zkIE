# Trusted activation closure

## Scheme and baseline

- Scheme: `trusted-x-closure`, the conditional output-head MatMul local
  closure with an independent trusted activation.
- Baseline: commit `0e7d2481` on `refactor/v2` (previous `c9a22cb3`).
  This directory is committed with the scheme change itself; the baseline is
  the commit the scheme was developed on.
- This round changes code and tests. It does not measure a benchmark: no
  proof bytes, wall time or VmHWM are reported, and no performance claim is
  made.

## Purpose

Close the local output-head relation on a formal path that consumes an
independent trusted activation `X` and keeps both `X` and the MatMul output
`Y` out of the verifier boundary, while binding the vocabulary mask, valid
range, ordering comparison, first-maximum tie rule and final output through
the trusted graph.

## Changes

- Trusted policy: a versioned, domain-separated activation commitment identity
  (root, positions, hidden width, salt length, configuration) bound to the
  model, version, case, anchor and model id. The operator writes it through a
  bind entry; the formal verifier compares the request session commitment
  against it and fails closed when the entry is absent.
- Hidden-input mode in the merged session: an empty public weights list makes
  the verifier take Eval targets from the message and bind each target to the
  committed reduced row through a BAT row. The reduced rows are the MatMul
  output and are committed under the hiding product-form encoding, so no
  plaintext `Y` is supplied. The activation-dependent Dot row is not part of
  this mode.
- Output relation: the mask step, valid range, argmax steps and expected
  output are trusted graph values; their canonical digest is absorbed into the
  first statement. The prover runs a full-row closure over all padded columns
  with a strict first-maximum scan and must reproduce the expected output.
- Diagnostic isolation: the plaintext-trace `X`/`Y` entries moved into a
  separate `diagnostic` module; the formal path does not call them.
- Tests: toy formal closure accept and negative matrix; real-fixture
  diagnostic test kept, with the formal real section gated by
  `ZKIE_SESSION_FORMAL`.

## Verification

Commands and results (logs in `logs/`):

- `cargo test -p zkie-ops --lib`: 149 passed, 0 failed.
- `cargo test -p zkie --release --lib`: 71 passed, 0 failed, 3 ignored.
- `cargo test --workspace`: exit 0, 48 suites ok, 652 passed, 0 failed.

The real-fixture test remains ignored because this environment has no
multi-GiB fixture. No real-fixture number is reported here.

## Negatives covered

Changed activation commitment against the policy pin, arbitrary activation,
changed output commitment, changed Eval target, changed MatMul step position,
missing output-relation declaration, forged mask constant, forged tie or
expected argmax, unsigned modular encoding of a negative value, masked
arithmetic overflow, spliced request material and activation-opening bytes
appended to the request encoding.

## Open items

- The Dot identity with hidden `X` is not checked: it needs a committed
  inner-product mechanism that the repository does not have. The formal path
  omits that row; the diagnostic path keeps it with plaintext weights.
- The hiding property of the activation commitment is not proven here.
- The previous-layer relation from the model to `X` is not proven.
- Plaintext `X` sources are diagnostic only.
- Production entry wiring is pending.
- Formal soundness is not done. No equivalence with any reference system and
  no equal security or performance claim follows.

## Artifact housekeeping

No binary artifacts were created or deleted in this round.
