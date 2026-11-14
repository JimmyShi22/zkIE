# The evidence row and the pending claim

## Scheme

- Scheme: `eval-tie`.
- Current code: `32bc9fd4`, following `a07c7c69`.
- Previous code: `a07c7c69`.

## Purpose

The first chain's (Eval) row and the pending claim read the same object: the
combined matrix rows at the claim's column point. This block checks it on real
vectors: W = g^T M built from the rows, the pending claim's value and the
evaluation of W at the point coincide; another matrix cell moves the pending
value away from the fixed target. The check joins the two existing
computations and adds no mechanism.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_eval_row_matches_the_pending_claim`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused eval-tie test | 1 passed |
| workspace | 51 suites, 1073 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The (Eval) row's value is the pending claim's value on the same object | connection rule | `lin_eval` against `pending_value` on W = g^T M | focused test |
| Another matrix cell breaks the fixed target | source rule | changed cell | focused test |
| No new mechanism | reuse rule | the two existing computations joined | the test only compares them |

## Conclusions

- The evidence row and the pending claim agree on real vectors.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The first/shared/second wiring with the fold and dot rows, then the frozen
  layout and the batch report.
