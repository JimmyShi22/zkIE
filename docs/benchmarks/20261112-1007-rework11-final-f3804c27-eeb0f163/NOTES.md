# rework11 final: BAT target inventory, ordered simulator and corrected proximity accounting

## Scheme and baseline

- Scheme: `rework11-final`.
- Code: document commit `f3804c27`; trace and final workspace head `46de871f`;
  previous round head `eeb0f163`.
- This round changes only `docs/lin-bind-closure-security.md` (v8 -> v9). The
  production code, test path, plan, parameters, commitments, challenge order
  and proof format are unchanged.
- Fixture for the reused performance numbers: `/data/lin-fixture`, formal
  hidden session, GPT-2 output head, 16 positions.

## Purpose

Record the J1-J5 batch of the eleventh rework round: the complete BAT target
inventory and joint conditional argument (closure, dot and own plain rows),
the ordered random-oracle simulator construction, the proximity miss-formula
and BASE-term correction, the strict-constant and kernel-case correction, and
the wording boundary for this round.

## Validation basis

This is a document-only round. There is no new seq16x3 measurement: the test
path and the production protocol are unchanged, so the latest measured
seq16x3 remains the rework10 runs at `eeb0f163` (trace
`docs/benchmarks/20261112-0914-rework10-final-473e395d-10cf4df3/`). The
workspace run below is a new run on the v9 document commit.

The five-case set is not run because no seq512 logic or all-case structural
path changed.

## Workspace

`cargo test --workspace`: 48 test binaries, 751 passed, 0 failed, 17 ignored.
The first run was on the v9 document commit `f3804c27` (`workspace-test.txt`);
the final rerun was on `46de871f` (`workspace-final.txt`). Both give the same
counts. The only commits after `f3804c27` are this trace note and its log, so
the tested production/test tree is unchanged.

## Reused seq16x3 numbers

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 | accepts | rejects |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 359,802 | 50,538 | 646.80 | 169,394,600 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 2 | 358,777 | 50,224 | 644.51 | 169,315,672 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 3 | 358,806 | 49,829 | 644.82 | 169,404,380 | f8f512c0…3215f5 | 276,343,251 | 0 |

These are the rework10 measurements, reused with provenance; they were not
re-measured in this round. The proof length is 9,939,144 B in all three runs.
The counter is process-level and includes `redraw` and repeated evaluations;
no best-run selection is used.

## Conclusions

- J1: every BAT target in the three plain-row families is a public constant,
  a verifier-reconstructed transcript value, or an already-published masked
  value. The five shown slots per closure position are the pivots of the five
  published values, the derived virtual/difference rows are their linear
  combinations, and the U/transcript rows use the invertible hiding maps. No
  target reveals `Y` outside winner/valid/tie plus the public statement; the
  stop condition did not trigger.
- J2: the simulator is written in the ordered construction (domain values,
  salt once, uniform commitment, root, challenges, opening-time message/tail
  choice, path consistency) without the witness or the opening set. Bad
  events are the pre-opening guess `q * 2^-320` and the
  collision/programming event `(q + 1)^2 / 2^256`.
- J3: the base-position miss probability is `(1 - delta)^t` for `t` draws
  (`s <= t` unique positions); the BASE lambda and the two codeword-index
  checks are withdrawn from `eps_components` and carried in
  `eps_proximity = UNKNOWN`.
- J4: the strict bounds are corrected (`50/p^3 < 2^-186.3`,
  `2/p^3 < 2^-190.9`, `30/p^3 < 2^-186.9`; `128/p^3` is slightly above
  `2^-185` and is not used). The `6/p^3` merge is not used; the
  single-equality term without a written degree derivation is excluded into
  `eps_proximity`. The kernel-difference case is marked `UNPROVEN`, not
  harmless.
- J5: the previously closed counter/sample/fixture wording is maintained; the
  751 tests, three hashes and HEAD are reported delivery figures, not an
  independent validation, and document/test revisions are not a new protocol
  validation.

## Open items

- `UNPROVEN`: mechanized adaptive composition; concrete-hash RO/CR
  advantage; deployment randomness of the asset tails and salts; the
  single-equality degree/event derivation; the local proximity
  distance/query numeric bound; an exact final-verifier negative for the
  tree-reuse tamper.
- No total security bit count is claimed; `eps_proximity` is `UNKNOWN`.
- No push: origin remains `831e7134`.

## Ledger

- Tests passed: 751 workspace tests, 0 failed, 17 ignored.
- Proof bytes, wall time and VmHWM: not re-measured this round; the reused
  rework10 figures are listed above with their trace provenance.
