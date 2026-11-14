# Reading the consumer's cells from the witness stream

## Scheme

- Scheme: `source-reads`.
- Current code: `e6851180`, following `ac51698f`.
- Previous code: `ac51698f`.

## Purpose

The chain's read values must come from the committed table rather than from
caller arrays. The session derives planned cells with the same stream and the
same row/cell layout as the reduced witness: one pass over the matrix rows,
each cell taken from the message before the run reductions. The rows must be in
nondecreasing order and every cell inside the plan's matrix and columns.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_source_reads_follow`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused source test | 1 passed |
| workspace | 51 suites, 1064 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Read values come from the committed table, on the witness's own stream and layout | source rule | `source_reads` | focused test matches the committed messages |
| Cells lie inside the plan's matrix and columns | range rule | `BoundViolation` checks | focused test |
| Rows are read in nondecreasing order on one pass | stream rule | order check before the pass | focused test refuses unsorted rows |
| No new commitment or mechanism | same-object rule | reuse of the witness stream | the derived values are the committed message cells |

## Conclusions

- The source-side derivation exists on the witness stream with its boundaries.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- Wiring the derived reads into the chain's stack segment, the changed-cell,
  swapped-row and swapped-object negatives, then the member chain follow.
