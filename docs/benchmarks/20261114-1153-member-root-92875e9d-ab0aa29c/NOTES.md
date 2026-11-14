# Reduced-witness root in the first message

## Scheme

- Scheme: `member-root`.
- Current code: `92875e9d`, following `ab0aa29c`.
- Previous code: `ab0aa29c`.

## Purpose

Wire the consumer's pending-claim plan and its reduced witness into the hidden
send chain. The plan carries the pending claim, the reduced-witness tables and
the witness tree; the plan check pins the matrix geometry, the table geometry,
the tree shape and its number collisions. The prover computes the table
entries from the table rows, commits the witness tree and absorbs its root in
the first message after the claim and before any sample seed. The verifier
reruns the same absorption and rejects a changed root, a missing root or a
plan with another witness tree. The session encoding gains one layout: a
member session writes layout 15 with the root; a session without the reduced
witness keeps layout 14 and its exact bytes, so the toy output-head proof is
unchanged.

## Measurement basis

- Same machine, release build for the real run; debug build for the workspace.
- Focused: the hidden consumer test over the registered seq16 artifacts.
- Hard gate: the norm-free toy output-head proof, dumped and reverted.
- Workspace: `cargo test --workspace`.

## Key numbers

| figure | value |
|---|---:|
| hidden consumer run | 79.40 s, 1 passed (with the root negatives) |
| session package total | 2,191,281,203 B |
| session.bin bytes | 1,083,192 |
| session.bin sha256 | `168e59962a19d3021adf6ba7da6ab154a5462ad01a09de3d1395b9e79cd9b315` |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |
| workspace | 51 suites, 1023 passed, 0 failed, 19 ignored |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The pending claim and the reduced-witness tables are trusted plan data, not a second commitment | pending-row and reduction-table rules | `MemberPlan` in the hidden send plan and its validation | plan check rejects wrong geometry, table shape, tree shape and number collisions |
| The witness root is committed after the pending claim and before any sample seed | commit-point order | the prover absorbs the root after the claim, before the index-mask sample; the verifier reruns it | hidden run; changed, missing and different-tree roots are rejected |
| A session without the reduced witness keeps its exact encoding | canonical encoding rule | layout 15 only when the root is present; layout 14 otherwise | hard gate unchanged at 45,080 B |

## Open items

- The pending claim is not yet consumed by LIN's second chain; the merge
  transcript is planned but not wired, and the table reads are not yet bound
  to the matrix cells.
