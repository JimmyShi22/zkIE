# Independent claim and tilde masks with a public Dot residual

## Scheme and baseline

- Scheme: `hidden-dot-mask-b1`.
- Run basis: HEAD `f74fd3a7` (the committed sumcheck row binding of the hidden
  Dot identity). The scheme commit is the commit that carries this trace; its
  own hash cannot be embedded in its tree, so the directory names the run
  basis instead.
- One same-machine formal fixture run plus the unit, crate and workspace test
  suites. No performance claim beyond the numbers below.
- Two earlier rounds are recorded in
  `docs/benchmarks/20261111-1219-hidden-dot-mask-b1-worktree-f74fd3a7/` and
  `docs/benchmarks/20261111-1223-hidden-dot-mask-b1-worktree-f74fd3a7/`.
  They ran the optional plaintext witness agreement check; this round is the
  clean formal measurement without that optional check. The check is enabled
  only by `ZKIE_SESSION_WITNESS`; the protocol code is identical in all three
  rounds.

## Purpose

Remove the value-level correlation between the PSC claim mask `c_t` and the
tilde mask `g_t`. The two classes keep their 73-slot allocation but are drawn
independently in M1; the 16 per-position equality rows are replaced by one
public difference row on the chi-weighted residual
`R = sum_t chi^t (g_t - c_t)`. The prover sends `R` once after `chi`; the
verifier absorbs `R` and its relation description before the product-sumcheck
challenges, adjusts the PSC target to `sum_t chi^t tilde_t - R`, and rebuilds
the complete difference row against the same committed mask member. The
corrected joint-view and rank argument and the updated error terms are in
`docs/hidden-dot-mask.md`.

## Changes

- The mask generator no longer copies `g_t` into the claim slot; both classes
  keep independent draws from the stage-1 stream.
- `HiddenSecondMessages` carries one optional residual field; the canonical
  encoding places it with the Dot proof and rejects a residual without a Dot
  proof (and the reverse).
- The second-stage transcript absorbs the residual with a relation tag and
  version after `chi` and before `mu'`/`eps`.
- `DotRowBinding` exposes the batching challenge, the residual and the
  product-sumcheck output claims; `HiddenLinOutcome` carries the binding in
  the hidden mode.
- One difference row is registered in the own group: two units per position
  (tilde slot with `chi^t`, claim slot with `-chi^t`) and the residual as the
  target. The row builder checks the slot classes, the mask-member shape and
  the exact four-row count.
- The auxiliary witness agreement entry checks the plaintext tilde row, the
  residual, the adjusted target and the product-sumcheck output rows for the
  same mask entries. The formal fixture enables it only under
  `ZKIE_SESSION_WITNESS` so the default measurement basis is unchanged.
- The canonical schema version is bumped so old messages do not decode.

## Results

- hidden proof bytes: 8,973,672 B.
- hidden prove: 36,366 ms; hidden verify: 8,690 ms.
- own-group FOLD rounds recorded by the fixture: 6.
- fixture test: 205.16 s; whole test process wall 206.60 s; maximum resident
  set size 19,243,676 KiB.
- commitment ledger openings: 655; table reuse: 0.
- formal manifest total: 4,170,798,739 B
  (trace 4,161,782,574 + hidden proof 8,973,672 + public 42,493).
- Baseline `f74fd3a7`: hidden proof 8,982,392 B; hidden prove 36,762 ms;
  hidden verify 8,895 ms; fixture wall 206.69 s; whole-test max RSS
  18,337,768 KiB; ledger 655 openings; table reuse 0. The proof bytes are
  challenge-dependent and this scheme changes the message shape. The hidden
  proof is 8,720 B smaller, prove is 1.1% shorter, verify is 2.3% shorter and
  the fixture wall is 0.7% shorter; the whole-test maximum RSS is 4.9% higher
  in this run. The figures are reported as measured, not as a gain.

## Hiding and error terms

- The public `(tilde, R)` forms have rank `17` on the `32`-dimensional
  `(g, c)` space; the kernel is the `15` claim-mask directions orthogonal to
  `chi`. `R` is independent of `tilde` for a fixed witness.
- The 23 sent product-sumcheck scalars are a full-rank affine map of the 23
  hiding coefficients for every fixed claim; the claim direction is one
  hiding direction, so the correlation between `R` and `sigma` adds no rank.
- Total mask leakage rank: `16 (tilde) + 16 (tau) + 2 (end) + 23 (hiding map)
  + 1 (claim direction) = 58` of 73 entries, kernel 15.
- Updated bound: `eps_total <= (t + q_V + 1) 2^-128.00
  + (17/2)(t + q_V)^2 / 2^256 + eps_prox + 15/|E|`, with
  `15/|E| ~= 2^-188.09` for the 16-position chi-batching term and
  `eps_prox <= 10 * 2^-128.00` for the recorded own-group round count. The
  conservative union bound is `<= 2^-124.25`. The Dot plain rows drop from 19
  to 4, so the BAT row-count term does not increase.
- The argument is stated, not a formal proof. The hiding-coefficient rank is
  tested at one deterministic challenge point for the production shape.

## Tests

- `cargo test -p zkie-ops --lib`: 176 passed, 0 failed.
- `cargo test -p zkie`: all test binaries green; 84 passed and 4 ignored in
  the largest binary.
- `cargo test --workspace`: green on the rerun; 268 passed in the largest
  binary and 48 test-result lines without a failure. The first workspace run
  had one unrelated transient failure in
  `anchor::tests::selects_the_entry_matching_the_input` (an `InvalidConfig`
  from a temporary-directory selection test); the same test passed 5 of 5
  isolated reruns and the full rerun is green. Both workspace logs are kept.
- Real fixture: `selected_gpt2_output_head_request_session`,
  `/data/lin-fixture/gpt2-seq16`, `ZKIE_SESSION_FORMAL=1`.
- The optional plaintext witness agreement check passed on the same fixture
  in the `1219` and `1223` rounds; this round leaves it disabled so the
  formal wall/memory figures stay on the baseline basis.
- New negatives: changed residual, opposite sign, missing residual, residual
  without a Dot proof, a residual not absorbed before the product-sumcheck
  challenges, changed claim/tilde/hiding/end mask entries, wrong residual in
  the checked binding, non-canonical encodings with trailing bytes or
  appended per-position differences.
- New positives: independent claim/tilde draws, the single batched difference
  row with the correct indices and target, the `(tilde, R)` rank and the
  23-coefficient hiding-map rank.

## Boundaries

- Not a formal zero-knowledge or soundness proof; the PSC/proximity/hash
  composition remains unproven.
- The tail-union check (`|SEND ∪ BASE| <= r`) is still a plan/ledger gap in
  the surrounding session.
- Reusing the same mask values under a fresh root is not detectable by the
  verifier or a root-only ledger; root reuse is rejected by the durable
  ledger in the fixture.
- The Eval mask value equality `e_t = g_t` is not rejected; the honest draw is
  the basis.
- The durable ledger is exercised by the integration fixture, not the
  production command path; the output relation and the table/asset
  correspondence remain trusted-plan items.

## Logs

- `cargo-zkie-ops.txt`
- `cargo-zkie.txt`
- `cargo-workspace.txt` (first run, transient unrelated failure)
- `cargo-workspace-rerun.txt` (green rerun)
- `formal-fixture.txt`
