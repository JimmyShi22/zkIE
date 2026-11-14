# Request-level weight claims and merge arithmetic

## Scheme

- Scheme: `request-weights`.
- Current code: `4ea86857`, following `569de4f9`.
- Previous code: `569de4f9`.

## Purpose

First block of the request-level claim structure: fix the weight matrix's
layout for a request-level claim and the arithmetic that turns a claim into
its row coefficients. The layout splits each index group between the matrix's
columns and its rows and stores weights as digits. A claim's column factor at
a point, its row factor and the per-row coefficients follow the rules that the
merge and the pending row use. The merge batches claims with one coefficient
each, folds each claim's column factor, accumulates the row coefficients into
one vector and leaves the pending row with the point consumed lowest column
bit first and the mask term.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_weights`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 7 passed |
| workspace | 51 suites, 1031 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The layout splits each index group between the matrix's columns and rows, with digit place values | weight-layout column-bit rules | `WeightGroup`, `WeightLayout::add`, `WeightUnit` | focused test checks the taken bits, the column offsets, the row counts and the digit rows |
| A claim's column factor is the product of its group factors at the point and the untaken column bits | claim column-factor rule | `eps_at`, `eq_table`, `eq1` | focused test computes the factor by hand for low, high and class-A groups |
| What remains after the columns are fixed is the row factor, and the row coefficients accumulate by digit and group order | row-part and coefficient rules | `row_part`, `RowClaim`, `gamma_add` | focused test checks the untaken high bit and the second digit's place value |
| The merge folds the batch coefficient into the claim's column factor and accumulates one row vector; the pending row reverses the point and carries the mask term | merge and pending-row rules | `merge_claims`, `pending_row` | focused test checks each claim's coefficient, the summed row vector and the reversed point |

## Conclusions

- The layout and merge arithmetic are implemented and tested.
- No session, proof message or existing check is touched; the toy proof is
  unchanged.

## Open items

- The sumcheck transcript that fixes the claim values, the batch challenge,
  the point and the masked end value is not wired; only its arithmetic is.
- The source-layer reads and the two-layer identity are the following block.
