# The chain's read cells in the trusted plan

## Scheme

- Scheme: `read-plan`.
- Current code: `8cf677a4`, following `8a13f34d`.
- Previous code: `8a13f34d`.

## Purpose

The session plan names the request chain's table cells: one (row, column) pair
per cell of each side, in panel and cell order, carried only when the chain
reads the table on the witness stream. The stack's sides are then derived from
those names through the shared read path, so the trusted layout decides which
committed cells the chain ties.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib the_plan_names_the_chain_read_cells`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused read-plan test | 1 passed |
| workspace | 51 suites, 1066 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The trusted plan names the chain's read cells | plan layout rule | `RequestReadPlan`, the plan field | focused test reads the named cells |
| The sides come from those names on the witness stream | single path rule | `stack_reads` over the plan's lists | focused test matches the committed messages |
| Another cell list is another read set | negative rule | changed list yields another value | focused test |
| Existing paths are unchanged | stability rule | empty value in every plan literal | hard gate unchanged |

## Conclusions

- The read layout is trusted plan data and the sides derive from it.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- Chain-level changed-cell, swapped-row and swapped-object negatives, then the
  member chain and the batch report follow.
