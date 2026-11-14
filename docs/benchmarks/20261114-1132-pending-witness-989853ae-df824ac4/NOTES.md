# Pending claims and the reduced-witness plan

## Scheme

- Scheme: `pending-witness`.
- Current code: `989853ae` (this scheme), following `df824ac4`.
- Previous code: `df824ac4`.
- Scope: a new proof module only; it is not wired into any session yet.

## Purpose

Start the consumer wiring with its first logical group: the pending claim a
sub-proof leaves on the committed matrix, and the reduced-witness table the
consumer's other claims read. The pending claim is never a separate
commitment: it carries row weights, the column point, the mask slots with
their coefficients, and the masked target, all fixed by the operator
reduction. A chain of one pending row passes through with coefficient one; no
batching sumcheck is run. The reduced witness fixes, per table, the matrix,
the run length, the point and the entry offsets, and enforces that every
point is set before the witness root, that no entry is read before the root,
that the batching challenge follows the root, and that the reads cover every
table entry exactly once with a non-zero weight.

## Measurement basis

- Same machine, debug build.
- Focused: `cargo test -p zkie-ops --lib hidden_member`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and reverted.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 6 passed |
| workspace | 51 suites, 1020 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| A pending claim is row weights, a column point, mask slots with coefficients and a masked target; it is not a commitment | pending-row and member-claim rules | `PendingRow`, `MaskTerm`, `claim_of` | focused tests: pass-through, mask/point geometry, refused batch |
| A chain of one passes through; a batch is not this consumer's chain | batch-of-one rule | `claim_of` accepts one row with coefficient one only | focused test rejects a second row and another coefficient |
| Row weights are aligned blocks, at most two of each size | row-weight stretch rule | `stretch_weights` | focused test checks the block cut and the cube bound |
| The pending rows must name the index-mask message, its mask code and the matrix geometry | member rows-fit rule | `rows_fit` | focused tests reject every named mismatch |
| The reduced witness fixes matrix, run length, point and offsets; points before the root, root before the batching challenge | reduction-table order rules | `ReductionTable`, `ReductionWitnessPlan` phases | focused tests: read before root refused, root before points refused, challenge before root refused |
| Every table entry is read exactly once with a non-zero weight, and nothing outside | read-coverage rule | `check_reads` | focused tests reject a gap, a duplicate, a zero weight and an out-of-table read |

## Conclusions

- The pending claim and the reduced-witness plan are in the crate with their
  ordering and coverage rules and focused tests.
- No session, proof message or existing check changed; the toy proof is
  expected to stay at 45,080 bytes.

## Open items

- The module is not wired into the hidden session or into the first chain.
- The merge that batches a sub-proof's weight claims into this one pending
  claim, and the reduced-witness entry computation and read binding to the
  original matrix cells, are the next groups.
