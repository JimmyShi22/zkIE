# rework13 final: block-diagonal hiding closure and the actual proximity miss

## Scheme and baseline

- Scheme: `rework13-final`.
- Code: document commit `c297f484` (the v11 block-diagonal proof), corrected
  by `6aff57f9` (the code-length basis of the proximity miss); preceding v11
  commit `8b71bbf3`; previous round head `100ec87d`.
- This round changes only `docs/lin-bind-closure-security.md` (v10 -> v11).
  The production code, test path, plan, parameters, commitments, challenge
  order and proof format are unchanged.
- Fixture for the reused performance numbers: `/data/lin-fixture`, formal
  hidden session, GPT-2 output head, 16 positions.

## Purpose

Record the L1/L2 batch of the thirteenth rework round: the block-diagonal
witness-independence proof (or its failure), the actual soundness reduction
with the kernel-coverage table, and the direct FOLD-to-BASE proximity miss at
the actual code parameters.

## Validation basis

This is a document-only round. There is no new seq16x3 measurement: the test
path and the production protocol are unchanged, so the latest measured
seq16x3 remains the rework10 runs at `eeb0f163` (trace
`docs/benchmarks/20261112-0914-rework10-final-473e395d-10cf4df3/`). The
workspace run below is a new run on the v11 document commit `8b71bbf3`; the
only later commit (`c297f484`) adds the activation block B0 to the document,
so the tested production/test tree is unchanged.

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

- L1: the block-diagonal closure is written. The fresh random variables
  partition into B0 (activation commitment), B1 (W/SEND), B2 (index-mask and
  Dot), B3 (closure), B4 (shared/FOLD), B5 (own/FOLD) and B6 (BASE). The
  conditional block lemma shows that a fresh block vector, uniform and
  independent of the prior view and the secret, pushes forward to a uniform
  block public distribution whenever the block map is surjective; the block
  covers give the surjectivity from the Vandermonde tails, the constant
  determinants, the shown/end/star pivots and the fixed SEND pads. The online
  order and adaptive openings follow because the covers are challenge
  independent (`chi != 0` is the counted event) and the ledger keeps `q <= r`.
  The hash hybrid programs only the prover's commitment trees; the public
  activation/table roots are public inputs. The conclusion is
  witness-independence for every two valid witnesses of the same public
  statement; a witness-free simulator is not claimed (the dummy witness must
  satisfy the same statement).
- L2: the soundness reduction enumerates the verifier acceptance conditions,
  the statement-component binding map, the difference classes and their
  nonzero/degree conditions, and the kernel-capture table. The direct
  FOLD-to-BASE proximity is computed at the actual code parameters:
  `delta_main = (131072 - 73730)/131072 = 0.4374847412`,
  `delta_mask = (131072 - 767)/131072 = 0.9941482544`; with `t = 2`, the
  misses are `0.3164234164 = 2^-1.6601` (main) and
  `3.424292663e-5 = 2^-14.8338` (mask). The code length is
  `n = 2^17 = 131,072`, not `ell + r`; the product-form encoder returns `n`
  symbols and the RS minimum distance is `n - (ell + r) + 1`.
  A fold-consistent common-error pattern survives the fold with the same
  relative distance, so the folded check does not improve the main-code bound.
  The proximity sub-term is therefore **ineffective**, not a small unknown;
  the direct FOLD-to-BASE path does not provide a security bound at the
  current parameters. This is a real soundness gap (`[GAP]`), with candidates:
  about 155 independent query positions for `2^-128`, a complete IOPP
  proximity round set, or a lower-rate/larger-`r` code parameter change.
- The honest ledger has `eps_algebraic < 2^-179.6` (conditional on the
  nonzero-difference/order preconditions), `eps_proximity` ineffective,
  `eps_hash` model-level, and `eps_kernel` with open error classes. No total
  soundness bound is claimed.

## Correction

The first v11 draft used `n = ell + r = 73,731` as the codeword length and
reported a main miss of `0.7901314940`. The actual code length is
`n = 2^log_n = 131,072` (`CodeParams::n()`; the product-form encoder returns
`n` symbols), and the RS minimum distance is `n - (ell + r) + 1`. The corrected
values are `0.3164234164` (main) and `3.424292663e-5` (mask); the candidate
query count is 155, not 754. The qualitative conclusion (no security bound at
`t = 2`) is unchanged. The correction is in document commit `6aff57f9`; the
open-items gap entry uses the corrected distance from `b46dcd39`.

## Open items

- `GAP`: direct FOLD-to-BASE proximity ineffective at the current parameters;
  candidates as above.
- `UNPROVEN`: kernel coverage for the error classes that the ineffective
  proximity was supposed to capture; full kernel/coverage matrix.
- `UNPROVEN`: witness-free simulator (the proof gives witness-independence
  with a dummy witness satisfying the same statement).
- `UNPROVEN`: concrete-hash RO/CR advantage; deployment randomness of the
  asset tails and salts; exact final-verifier negative for the tree-reuse
  tamper.
- No push: origin remains `831e7134`.

## Ledger

- Tests passed: 751 workspace tests, 0 failed, 17 ignored.
- Proof bytes, wall time and VmHWM: not re-measured this round; the reused
  rework10 figures are listed above with their trace provenance.
