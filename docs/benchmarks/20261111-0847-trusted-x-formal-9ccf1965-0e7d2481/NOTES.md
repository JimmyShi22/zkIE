# Formal fixture run with a fresh table source and a hidden-proof manifest

## Scheme and baseline

- Scheme: `trusted-x-formal`.
- Baseline: `9ccf1965` (conditioned output-head relation); test harness changes only.
- No benchmark claim: the numbers below are one local fixture run.

## Purpose

Run the formal hidden-input section on the production 4.16 GB fixture with a
fresh read-only table source (as a separate prover process would), and bind the
formal verification manifest to the hidden proof rather than the diagnostic
request.

## Results

- hidden proof: 8,984,536 B.
- hidden prove 34.482 s; hidden verify 7.899 s; whole test 200.78 s.
- formal manifest total: 4,170,809,603 B = trace 4,161,782,574 +
  hidden proof 8,984,536 + public commitment 42,493.
- diagnostic request: 8,984,312 B; diagnostic manifest total 4,170,809,379 B,
  reported separately and not used as the formal total.
- A one-byte change to the hidden request file is rejected by the formal
  manifest. The formal negatives in the test also reject changed X/Y, forged
  mask or tie, missing declarations, negative modular alternatives, spliced
  requests and opening leakage.

## Boundaries

- The formal path still omits the hidden-X Dot identity; this run is not the
  final Dot closure and must not be called a conditioned MatMul acceptance.
- The X commitment is operator-pinned; the operator's safe generation and
  acceptance of the root is a deployment assumption still to be documented.
- Output closure (masked valid range, ordering, tie rule and public output) is
  not verified by this run; it is a hard gate for the next round.
- The combined accounting is not rebuilt; the local numbers are not a security
  bound.
- The activation commitment's hiding property and the previous-layer relation
  are not proved. Plaintext X sources are diagnostic only.
- No equivalence or equal security/performance claim follows.

## Logs

- `formal-manifest.txt`: the full fixture run.
