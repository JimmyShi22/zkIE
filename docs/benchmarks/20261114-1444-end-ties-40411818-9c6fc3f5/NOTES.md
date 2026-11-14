# The stack's end-value ties

## Scheme

- Scheme: `end-ties`.
- Current code: `40411818`, following `9c6fc3f5`.
- Previous code: `9c6fc3f5`.

## Purpose

The stack leaves three linear claims. The sumcheck's own leftover row is the
output's row form; this block gives the other two their row shape. Each sent
end value is masked with the sumcheck's factor, so its hidden value is the
shown value minus the factor times its mask entry, and that hidden value must
equal the affine recombination of the panels' operand values — the first side
with the call and batch factors and the row weights, the second side with the
reads and the shown weights. A changed read, shown weight, end value or mask
entry breaks the tie.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_stack`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused stack tests | 8 passed |
| workspace | 51 suites, 1057 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Each sent end value's hidden value equals its affine recombination, with the shown weights on the second side | end-value tie rule | `EndValue`, `StackReduction::check_end_ties` | focused test binds both sides on a read panel and a weight panel |
| The hidden value is the shown value minus the factor times the mask entry | masked end-value rule | mask subtraction in the tie | focused test; a changed end value or mask entry is refused |
| The stack's third linear claim is the output's leftover row | leftover row rule | `RowForm::from_output` | row-claim block test |

## Conclusions

- All three of the stack's linear claims now have their row shapes.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The auxiliary end values for the weight claims wait for the ruling; the
  session entry and the source layer follow.
