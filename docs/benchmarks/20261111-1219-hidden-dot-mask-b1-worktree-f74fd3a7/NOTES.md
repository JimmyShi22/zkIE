# Independent claim and tilde masks with a public Dot residual (preliminary round)

## Scheme and baseline

- Scheme: `hidden-dot-mask-b1`.
- Run basis: HEAD `f74fd3a7` (the committed sumcheck row binding of the hidden
  Dot identity). The previous scheme commit is the run basis; the scheme
  commit is the commit that carries this trace and cannot embed its own hash.
- This is the first fixture round of the scheme. The auxiliary-check round is
  recorded in `docs/benchmarks/20261111-1223-hidden-dot-mask-b1-worktree-f74fd3a7/`;
  the clean formal measurement of record is in
  `docs/benchmarks/20261111-1227-hidden-dot-mask-b1-worktree-f74fd3a7/`.

## Purpose

Exercise the independent-claim-mask construction on the real 4.16 GB fixture
and collect the hidden Dot proof bytes and stage times. This round also runs
an auxiliary plaintext witness check that re-verifies the public proof before
checking the mask entries.

## Results

- hidden proof: 8,973,672 B.
- hidden prove 36,734 ms; hidden verify 8,866 ms; auxiliary witness check
  12,843 ms (a second full verifier replay plus the plaintext mask checks).
- own-group FOLD rounds: 6.
- fixture test: 220.26 s; whole test process wall 228.59 s; maximum resident
  set size 19,232,416 KiB.
- ledger: 655 logical openings; table reuse 0.
- formal manifest total: 4,170,798,739 B
  (trace 4,161,782,574 + hidden proof 8,973,672 + public 42,493).
- Baseline `f74fd3a7`: hidden proof 8,982,392 B; hidden prove 36,762 ms;
  hidden verify 8,895 ms; fixture wall 206.69 s; whole-test max RSS
  18,337,768 KiB; ledger 655 openings. The proof bytes are
  challenge-dependent and the message shape changed; the figures are
  reported as measured, not as a gain.

## Conclusion and open item

The hidden stage numbers are within the recorded noise of the baseline. The
fixture wall and whole-test RSS rose because the auxiliary check replayed the
verifier while the first outcome was still alive. The later rounds remove the
replay and then gate the optional check, keeping the clean formal run on the
established measurement basis; no protocol code changed between the rounds.
The `1227` trace is the measurement of record; this directory keeps the
preliminary evidence.

## Logs

- `formal-fixture.txt`: the complete preliminary fixture run.
