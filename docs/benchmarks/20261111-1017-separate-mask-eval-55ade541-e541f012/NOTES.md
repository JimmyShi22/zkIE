# Separate-mask schema and masked Eval binding

## Scheme and baseline

- Scheme: `separate-mask-eval`.
- Baseline: `55ade541` (separate-mask schema and Eval binding); previous
  `e541f012` (commitment ledger).
- This is one local fixture run; no performance claim beyond the row below.

## Purpose

Record the real 4.16 GB formal run after the 73-entry separate-mask schema and
the masked Eval binding. The verifier receives `tau_t = W_t(psi_t) + e_t` and
`tilde_t = W_t(psi*) + g_t`; the Eval and Dot rows are bound to the committed
reduced rows and the separate mask entries. The commitment ledger charges the
actual mask/opening reservations.

## Results

- hidden proof: 8,997,192 B.
- hidden prove 36.011 s; hidden verify 8.304 s; whole test 205.84 s.
- formal manifest total: 4,170,822,259 B = trace 4,161,782,574 +
  hidden proof 8,997,192 + public commitment 42,493.
- ledger: 653 logical openings; table reuse 0.
- The separate-mask unit tests cover the positive Eval/Dot row binding and the
  negatives: changed W root/row, changed g_t/c_t/e_t slot, changed tau_t or
  psi_t, missing Eval row, reused e_t, wrong challenge order.

## Boundaries

- The hidden Dot identity `<X_t, Omega> = W_t(psi*)` itself is still not wired
  through the PSC sumcheck; this run is not the Dot closure and must not be
  called a conditioned MatMul acceptance.
- The 57-entry primary allocation is not used; this is the 73-entry fallback.
  The joint mask/Eval hiding argument remains unproven.
- Output closure (valid range/order/tie/public output) is not verified here.
- The combined accounting is not rebuilt; these numbers are not a security
  bound.
- The X pin is operator policy; the previous-layer relation and X hiding are
  not proved.
- No equivalence or equal security/performance claim follows.

## Logs

- `formal-eval-mask.txt`: the full fixture run.
