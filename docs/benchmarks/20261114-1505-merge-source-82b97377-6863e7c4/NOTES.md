# The merge value's source

## Scheme

- Scheme: `merge-source`.
- Current code: `82b97377`, following `6863e7c4`.
- Previous code: `6863e7c4`.

## Purpose

The weight values that enter the merge are the stack sub-proof's masked end
values, not independent caller constants. This block states the source in the
type: the shown scalar, its slot in the sub-proof's mask row and the stack
sumcheck's end factor, with a constructor that names it. The merge's own final
masked target uses the request's index-mask entry and the merge's own end
factor; the two factors and slots must not be mixed.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_weights`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused merge tests | 8 passed |
| workspace | 51 suites, 1058 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The merge's input values come from the stack sub-proof's masked end values | input source rule | `MergeValue` documentation, `from_stack_end` | focused tests stay green; the type names the source |
| The merge's final masked target uses the request index-mask entry and the merge's own end factor | two-layer mask rule | documentation of the two factors | statement in the type; used by the chain entry's pending construction |

## Conclusions

- The merge value's source is recorded in the type and constructor.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The group's As/Bs construction from the transcript's end slots, the v-slot
  claims and the negative matrix follow.
