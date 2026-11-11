# Committed sumcheck row binding for the hidden Dot identity

## Scheme and baseline

- Scheme: `psc-row-binding`.
- Run basis: the working tree whose parent commit is `af7b5d4d` (the masked
  Eval binding trace). The previous scheme commit is `55ade541` (separate-mask
  schema and masked Eval binding). The scheme commit is the commit that
  carries this trace; its own hash cannot be embedded in its tree, so the
  directory names the run basis instead.
- One same-machine formal fixture run; no performance claim beyond the
  numbers below.

## Purpose

Wire the planned second-stage committed row binding for the hidden Dot
identity `<X_t, Omega> = W_t(psi*)`. The change commits a shown-map message of
the activation rows with its own encoding tail, salts and sample pad. After
the tilde values and the batching challenge it sends the masked
product-sumcheck transcript and rebuilds the `(U)` row and the two `(E_j)`
end-value rows; the public `c_t = g_t` tie rows join the claim masks to the
tilde masks. All new rows are registered in the own proximity group, so the
real BAT/FOLD/BASE test checks them. The one-time entries come from the
73-entry separate-mask schema, and the durable ledger records the new
commitment and its logical opening.

## Results

- hidden proof: 8,982,392 B.
- hidden prove 36.762 s; hidden verify 8.895 s; fixture test 206.69 s.
- whole test process: wall 3:26.91; maximum resident set size 18,337,768 KiB
  (measured with the system time tool on the test process; this is a
  whole-test figure, not the prove-stage peak recorded by the previous
  scheme).
- formal manifest total: 4,170,807,459 B = trace 4,161,782,574 + hidden proof
  8,982,392 + public commitment 42,493.
- ledger: 655 logical openings; table reuse 0. The shown-map commitment
  contributes one recorded commitment and two logical openings.
- Previous scheme, same machine and fixture: hidden proof 8,997,192 B, hidden
  prove 36.011 s, hidden verify 8.304 s, fixture test 205.84 s, ledger 653
  openings. The proof bytes are challenge-dependent and this run drew a
  different opening set; the figures are reported as measured, not as a gain.
  The hidden proof is 14,800 B smaller, the hidden prove is 2.1% longer
  (within the 3% noise band), the hidden verify is 7.1% longer (the new
  transcript check and the extra proximity member), and the whole test is 0.4%
  longer. No optimization is claimed.

## Tests

- `cargo test -p zkie-ops --lib`: 170 passed.
- `cargo test -p zkie`: all test binaries green.
- `cargo test --workspace`: all test binaries green (268 in the largest
  suite).
- Real fixture: this run, `selected_gpt2_output_head_request_session`,
  `/data/lin-fixture/gpt2-seq16`, formal hidden path.
- Negatives in the session tests: changed `(U)` round message, changed hiding
  sum, changed first and second end value, wrong challenge order (the last
  round sent first), changed shown-map commitment, changed `Omega`
  commitment, missing shown-map message in the first and second stage,
  missing message plan, changed or reused mask slots (claim, tilde, end,
  hiding, variable count), wrong shown entry count, colliding tree number and
  a non-single-answer shown message. The arithmetic test
  `every_sent_value_is_absorbed_before_its_challenge` checks that every sent
  value is absorbed before the challenge that tests it.

## Boundaries

- The verifier pins the shown-map commitment and the activation commitment
  separately; that the two commitments carry the same `X` is an
  operator-policy warranty, not a cross-commitment proof.
- The mask and Eval hiding argument, the committed table distances, the
  previous-layer relation and the full-row output closure are unchanged from
  the previous scheme. Formal soundness, single-shot decode semantics and an
  equivalence or equal-security claim are not made.
- The hidden proof bytes are a whole-request figure; no per-stage byte split
  is claimed.

## Logs

- `formal-psc-row-binding.txt`: the complete fixture run on the final tree.
