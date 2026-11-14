# Weight-claim merge plan

## Scheme

- Scheme: `merge-plan`.
- Current code: `9dce7785`, following `91a949cc`.
- Previous code: `91a949cc`.

## Purpose

Plan the merge that batches a sub-proof's weight claims into the one pending
claim: the claims' end-value slots are registered before any challenge, one
challenge batches them, a product-sumcheck over the matrix column bits gives
the point, and the masked end value leaves the pending claim with the mask
factor as the index-mask coefficient. The end value's slot may not reuse a
claim's slot. The pending claim consumes the point lowest column bit first.
The runtime reduced-witness plan is now built from settled plan tables with
contiguous offsets, checked entry counts and every point set.

## Measurement basis

`cargo test -p zkie-ops --lib hidden_member` on the test machine.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 9 passed |
| workspace (previous group) | 51 suites, 1020 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The claims' end values are fixed before the merge challenge; one challenge batches them; the sumcheck gives the point and the masked end value | weight-merge rules | `MergePlan` phases and `pending_claim` | focused test: order violations, distinct slots, free slot, point reversal |
| The pending claim is the merged claim with the index-mask factor, not a second commitment | pending-claim rule | `MergePlan::pending_claim` builds one `PendingRow` | focused test checks psi, target, mask term and weights |
| Runtime witness tables come from settled plan tables, contiguous and complete | reduction-table order | `ReductionWitnessPlan::from_tables` | focused test rejects gaps, overlaps, wrong counts and unset points |

## Open items

- The transcript's challenge values are not wired into a session yet; the
  first-chain message and the read binding follow.
