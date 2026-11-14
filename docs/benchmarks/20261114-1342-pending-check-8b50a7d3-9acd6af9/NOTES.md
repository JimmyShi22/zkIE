# Pending row claim check

## Scheme

- Scheme: `pending-check`.
- Current code: `8b50a7d3`, following `9acd6af9`.
- Previous code: `9acd6af9`.

## Purpose

The pending claim a request's weight chain leaves is the row weights against
the matrix rows at the column point, plus the index-mask entries, against the
masked target. This block evaluates and checks that claim: the column point
pairs its bits lowest first, dense and scalar row blocks cover the row cube,
and the mask terms name their oracle and slot. The check refuses another
matrix value, another mask value, a block that leaves the row cube and a
malformed row.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib hidden_member`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused member tests | 12 passed |
| workspace | 51 suites, 1052 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The pending claim is the row weights against the matrix rows at the column point plus the mask entries, against the target | pending claim rule | `pending_value`, `check_pending` | focused test checks a hand-computed target |
| The column point pairs bit `j` with the coordinate of its lowest bit first | column point rule | `row_at` | focused test pins all four points of a two-bit point |
| Row weights may be scalar rows or aligned dense blocks inside the row cube | row weight rule | `WeightBlock` traversal | focused test covers both block kinds and refuses a row outside the cube |
| Another matrix value or mask value is refused | negatives | check refusal | focused test |

## Conclusions

- The pending claim is evaluable and checked on synthetic and structured
  weights; the claim shape matches the merge's output.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The weight fork's leftover claim enters the chain test, then the merge,
  pending row and stack claims enter the session transcript.
- The source-layer reads and the two-layer identity follow.
