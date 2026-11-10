# Output-head MatMul wired into the LIN request

## Scheme

- Scheme: `output-head-lin`.
- Current code: `e4021cc5`.
- Previous code: `29a99472` (the three-stage component).

## Purpose

Wire the real output-head MatMul into the three-stage LIN request: per
position `t`, `g_t[y] = Fp(X[t][y])`, `M[y][j] = Fp(W[y][j])`, `psi_t` over
the 16 column bits, and the target from the full 65536-column output. The
verifier takes X and Y from the verified trace and W from the trace's
authenticated output-head asset opening. The vocabulary mask, the masking Add
and the argmax steps stay separate trusted-plan items; no argmax is inferred
from sampled columns. All 65536 columns take part, including those above the
valid vocabulary.

This is a controlled diagnostic: the trace proof carries plaintext weight rows
and activations, and this request proof adds no hiding.

## Measurement basis

- Same machine, release build.
- `ZKIE_SESSION_FIXTURE=/tmp/lin-fixture cargo test -p zkie --release --test
  lin_session selected_gpt2_output_head_request_session -- --ignored
  --nocapture`
- The fixture links the production `gpt2-seq16` public commitment, trace proof
  and policy; the trace proof is 4,161,782,574 B.
- Unit tests: `cargo test -p zkie-ops --lib output_head` (12 tests).
- Workspace: `cargo test --workspace`.

## Results

- The selected output-head step is 4444.
- 16 positions: every Eval, Dot and Fold check holds; the full-value W and
  Omega controls hold.
- Request proof: 1,254,646,928 B.
- Full verification bytes: 4,161,782,574 (trace) + 1,254,646,928 (request) =
  5,416,429,502 B. The request artifact size is not the full proof size.
- Prove 611.637 s; verify 403.797 s; the whole test 1474.04 s.
- Unit tests 12 passed; workspace exit 0.

## Security boundary

- Field equality is equality modulo the base field order. The traced MatMul
  uses checked i128, requantize(shift = 0) and i64 checks; this request does
  not re-prove those integer bounds. No integer-equivalence claim is made
  beyond what the trace relation enforces.
- Signed values use the sign-extended field conversion, not an unsigned
  reinterpretation.
- The 16 claims are independent: no batching coefficients and no combined
  error bound.
- The trace and request carry plaintext weights and activations. This is a
  controlled diagnostic, not a hidden production proof; the hidden relation is
  a later round.
- The request is not yet connected to the production commit/prove/verify
  command or to the limited key artifact.

## Logs

- `real-trace.txt`: the full test output.
