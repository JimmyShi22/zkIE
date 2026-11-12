# rework12 final: corrected Dot rows, online simulator and soundness reduction

## Scheme and baseline

- Scheme: `rework12-final`.
- Code: document commit `7dfc0754`; previous round head `8aae2c23`.
- This round changes only `docs/lin-bind-closure-security.md` (v9 -> v10).
  The production code, test path, plan, parameters, commitments, challenge
  order and proof format are unchanged.
- Fixture for the reused performance numbers: `/data/lin-fixture`, formal
  hidden session, GPT-2 output head, 16 positions.

## Purpose

Record the K1-K5 batch of the twelfth rework round: the corrected Dot row
equations and mask attrs, the online simulator construction with its
blockwise witness-independence and the explicit global-composition
obstruction, the soundness reduction with the kernel-coverage table and the
proximity sub-term, the base-position sampling record, and the wording
boundary.

## Validation basis

This is a document-only round. There is no new seq16x3 measurement: the test
path and the production protocol are unchanged, so the latest measured
seq16x3 remains the rework10 runs at `eeb0f163` (trace
`docs/benchmarks/20261112-0914-rework10-final-473e395d-10cf4df3/`). The
workspace run below is a new run on the v10 document commit.

The five-case set is not run because no seq512 logic or all-case structural
path changed.

## Workspace

`cargo test --workspace`: 48 test binaries, 751 passed, 0 failed, 17 ignored.
The complete log is `workspace-test.txt`.

## Reused seq16x3 numbers

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 | accepts | rejects |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 359,802 | 50,538 | 646.80 | 169,394,600 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 2 | 358,777 | 50,224 | 644.51 | 169,315,672 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 3 | 358,806 | 49,829 | 644.82 | 169,404,380 | f8f512c0…3215f5 | 276,343,251 | 0 |

These are the rework10 measurements, reused with provenance; they were not
re-measured in this round. The proof length is 9,939,144 B in all three runs.

## Conclusions

- K1: the Dot equations are corrected. `tilde_t = W_t(psi*) + g_t`,
  `R = sum_t chi^t(g_t - c_t)`, and the PSC target is
  `T = sum_t chi^t tilde_t - R = sum_t chi^t(W_t(psi*) + c_t)`; the difference
  row is the mask-entry relation `sum_t chi^t g_t - sum_t chi^t c_t = R`.
  `(E_0)` is the chi-batched activation/shown-map `X_chi(point)` (not W);
  `(E_1)` is the single `Omega(point)`. The closure virtual row uses
  `shown1` for `ends[1].value` and `shown2` for `b_point`; `(E_0)` uses
  `shown0`. The W direction is on the PSC-target side, masked by `g_t` and
  `c_t`; the joint per-block cover is rewritten accordingly.
- K2: the simulator is online in the real transcript order (no preselected
  PSC/FOLD/BASE values). The dummy-witness strategy is written blockwise with
  witness-independence from the pads, `q <= r` Vandermonde tails and constant
  determinant maps. The global composition is `[UNPROVEN]`: the full joint
  object across all shared columns, roots and PSC/BASE transcripts is not
  assembled; the opening-extension coupling is identified as the mechanism
  obstruction, with candidates (full joint matrix, hybrid/coupling lemma,
  different invariant). The construction is not a witness-free simulator;
  that remains open if required.
- K3: the soundness reduction enumerates the verifier acceptance conditions,
  states the false-statement detection union (SZ for algebraic checks,
  sampled-position miss for proximity, hash-model terms for bindings), gives
  the algebraic preconditions, and carries the direct FOLD-to-BASE proximity
  sub-term as `UNKNOWN` with its obstruction and candidates. Kernel coverage
  is treated as a proposition, not a premise: the error-class/coverage table
  marks the open cases and the candidates; the note does not assert that
  kernel differences are irrelevant to soundness.
- K4: the base-position record is `t = 2` draws, `s <= 2` unique positions and
  `s` variable; deduplication does not change the original-draw hit event, and
  the miss term is `(1 - delta)^t` for that event.
- K5: the target row list is the next-round basis; the full joint distribution
  of all shared columns with the roots/PSC/BASE cannot be derived from
  per-block pivots or same-stream lists. The workspace/hash/HEAD figures are
  TL-reported, not independent validation; document/test revisions are not a
  new protocol validation.

## Open items

- `UNPROVEN`: global online simulator composition and full joint distribution;
  kernel coverage in the soundness reduction; concrete-hash RO/CR advantage;
  deployment randomness of the asset tails and salts; single-equality
  degree/event derivation; local proximity distance/query numeric bound;
  exact final-verifier negative for the tree-reuse tamper.
- No total security bit count is claimed; `eps_proximity` is `UNKNOWN`.
- No push: origin remains `831e7134`.

## Ledger

- Tests passed: 751 workspace tests, 0 failed, 17 ignored.
- Proof bytes, wall time and VmHWM: not re-measured this round; the reused
  rework10 figures are listed above with their trace provenance.
