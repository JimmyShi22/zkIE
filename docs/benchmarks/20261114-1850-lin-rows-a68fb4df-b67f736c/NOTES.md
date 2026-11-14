# The LIN connection rows

## Scheme

- Scheme: `lin-rows`.
- Current code: `a68fb4df`, following `b67f736c`.
- Previous code: `b67f736c`.

## Purpose

The LIN anchors leave three rows, and this block makes their relations
executable: the evaluation of a column vector at a point, the dot row tying the
row weights against Omega to the vector's value at the shared point, and the
fold row tying Omega to the folded row through the two equality tables. Every
form is cut at the matrix's rows, since a form that ran on would read entries
that mean nothing.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib lin_rows`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused LIN tests | 3 passed |
| workspace | 51 suites, 1071 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The evaluation pairs bit `j` with its coordinate | (Eval) rule | `lin_eval` | focused test pins all points of a two-bit cube |
| The dot row ties the row weights against Omega to the vector at the shared point, cut at the matrix's rows | (Dot) rule | `lin_dot` | focused test; a changed row is refused, and the cut keeps a later row out |
| The fold row ties Omega to the folded row through both equality tables, cut at the matrix's rows | (Fold) rule | `lin_fold` | focused test; a changed Omega row is refused |

## Conclusions

- The three LIN connection relations are executable and tested.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The W oracle's own SEND commitment, the first/shared/second chain order and
  their IOPP tests follow; the frozen layout waits for that wiring.
