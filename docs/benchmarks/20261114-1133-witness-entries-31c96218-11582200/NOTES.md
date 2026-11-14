# Reduced witness entries

## Scheme

- Scheme: `witness-entries`.
- Current code: `31c96218`, following `11582200`.
- Previous code: `11582200`.

## Purpose

Compute a reduced-witness table's entries: each entry is the scalar product of
one run of matrix cells with the run's point, in row-major then run order. This
is the prover side of the reduced witness; the read binding and the first-chain
wiring follow.

## Measurement basis

`cargo test -p zkie-ops --lib hidden_member` on the test machine.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 7 passed |
| workspace (previous group) | 51 suites, 1020 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Each reduced-witness entry is the inner product of a specified matrix fragment with the point, not a prover-chosen row | reduction-table entry rule | `entry_value`, `table_entries` | focused test checks the order, the inner product and the shape rejections |

## Open items

- The entries are not yet committed into a session or bound to the matrix
  reads; that is the first-chain wiring.
