# The stack group as the merge's source

## Scheme

- Scheme: `group-source`.
- Current code: `592d2d71`, following `82b97377`.
- Previous code: `82b97377`.

## Purpose

The stack's end group becomes a source for the rest of the chain. Its hidden
values are derived from the sent ones by subtracting the end factor times each
slot's mask entry; the two rest values are checked against the reads'
recombinations on each side; and the weight values are checked against the
weight coefficients as the sides of the terminal relation. The merge's inputs
come from the stack end constructor, so each value carries its sub-proof slot
and the stack's factor, while the merge's own end factor and the request's
index-mask entry stay separate.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_stack` and
  `cargo test -p zkie-ops --lib request_weights`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused stack tests | 9 passed |
| focused merge tests | 9 passed |
| workspace | 51 suites, 1060 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The hidden group values are the sent values minus the factor times their mask entries | source rule | `GroupValues::from_sent` | focused test derives the values and matches the reads |
| The rest values are checked against the reads and the weight values against their coefficients | group tie rule | `StackReduction::group_bases` | focused test; changed reads and a short group are refused |
| The merge's inputs come from the stack end values, not from constants | input source rule | `MergeValue::from_stack_end` in an end-to-end merge | focused test verifies the merge and its row claim |
| The merge's own end factor does not belong on the input values | two-layer mask rule | negative with a mixed factor | focused test refuses the mixed-factor merge |

## Conclusions

- The group's values flow into the merge with their source and factors kept
  apart.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The packet's claim list, the session entry and the source layer follow.
