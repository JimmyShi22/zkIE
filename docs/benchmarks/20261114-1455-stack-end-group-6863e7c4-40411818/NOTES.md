# The stack's whole end group

## Scheme

- Scheme: `stack-end-group`.
- Current code: `6863e7c4`, following `40411818`.
- Previous code: `40411818`.

## Purpose

The stack's batched sumcheck sends one masked end group instead of two end
values: the two sides' rest values and the never-sent and weight values with
their public coefficients, in the order the anchor fixes — A0, the u values,
B0, the v values — each with its own mask entry and the same factor. The last
round is the product of the recombined sides with their masks folded in, so the
weight values take part in the final round and in the terminal relation. The
inner sumcheck runs the aggregated maps' rounds and returns the whole group,
with the rest values subtracted as the anchor does.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib psc_general`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused general-sumcheck tests | 12 passed |
| workspace | 51 suites, 1058 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The whole end group is sent in the fixed order with one mask entry each and one factor | end group rule | `Summand::StackProduct`, shown count | focused test sees five end values and prove/verify agree |
| The last round is the product of the recombined sides with the group's masks | last round rule | `at` with the node-masked values | focused test; a changed last round leaves another output claim |
| The weight values take part in the final round and the terminal relation | not-an-appendix rule | terminal recombination | focused test; a changed weight map is refused |
| The group's size is fixed by the layout | layout rule | shown count check | focused test refuses a short group |

## Conclusions

- The end group and its last round are in place and tested.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- The package's As/Bs build from the group's slots, the merge's references to
  the same v slots, and the source layer follow.
