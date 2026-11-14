# Feeding the stack's sides from the witness stream

## Scheme

- Scheme: `stack-reads`.
- Current code: `8a13f34d`, following `e6851180`.
- Previous code: `e6851180`.

## Purpose

The stack's two sides are derivable from the committed table. The reduction's
panels fix how many cells each side takes, in panel and cell order; the cells
are fetched on the witness stream in row order and restored to panel order, so
the stack's arrays are the committed cells rather than caller data. The weight
panels' second side holds their weight cells, which the group turns into the
weight values.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_stack_reads_follow`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused stack-read test | 1 passed |
| workspace | 51 suites, 1065 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Each side's cells come from the committed table in the reduction's panel order | panel source rule | `stack_reads`, `fetch_cells` | focused test matches the committed messages |
| Unsorted cells are fetched in row order and restored to panel order | stream order rule | stable sort and restore | focused test sees panel order with unsorted rows |
| A short side or an out-of-range cell is refused | boundary rule | length and range checks | focused test |

## Conclusions

- The stack's sides now come from the witness stream with their layout rules.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- The session plan naming the read cells, and the changed-cell, swapped-row
  and swapped-object negatives, follow; then the member chain and the batch
  report.
