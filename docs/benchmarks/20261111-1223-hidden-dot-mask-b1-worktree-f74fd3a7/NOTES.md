# Independent claim and tilde masks with a public Dot residual (auxiliary-check round)

## Scheme and baseline

- Scheme: `hidden-dot-mask-b1`.
- Run basis: HEAD `f74fd3a7` (the committed sumcheck row binding of the hidden
  Dot identity). The scheme commit is the commit that carries this trace and
  cannot embed its own hash.
- This round runs the formal fixture with the optional plaintext witness
  agreement check enabled. The clean formal run without the optional check is
  recorded in `docs/benchmarks/20261111-1227-hidden-dot-mask-b1-worktree-f74fd3a7/`
  and is the measurement of record.

## Purpose

Exercise the independent-claim-mask construction on the real 4.16 GB fixture
and check the prover's private mask entries against the tilde row, the public
residual, the difference-row target and the product-sumcheck output rows. The
check uses the public outcome that was already computed; it does not replay
the verifier.

## Results

- hidden proof: 8,973,672 B.
- hidden prove 36,752 ms; hidden verify 8,859 ms; the optional witness
  agreement check 3,454 ms.
- own-group FOLD rounds: 6.
- fixture test: 210.63 s; whole test process wall 218.94 s; maximum resident
  set size 19,236,772 KiB.
- ledger: 655 logical openings; table reuse 0.
- formal manifest total: 4,170,798,739 B
  (trace 4,161,782,574 + hidden proof 8,973,672 + public 42,493).
- Baseline `f74fd3a7`: hidden proof 8,982,392 B; hidden prove 36,762 ms;
  hidden verify 8,895 ms; fixture wall 206.69 s; whole-test max RSS
  18,337,768 KiB; ledger 655 openings. The proof bytes are
  challenge-dependent and the message shape changed; the figures are
  reported as measured, not as a gain.

## Conclusion and open item

The optional check passes on the real fixture. It adds 3,454 ms to the test
and raises the whole-test peak memory; the hidden prove and verify stage
figures are unchanged within noise. The clean run in the `1227` directory
keeps the established formal measurement basis. No protocol code differs
between the rounds.

## Logs

- `formal-fixture.txt`: the complete formal fixture run with the optional
  witness agreement check.
