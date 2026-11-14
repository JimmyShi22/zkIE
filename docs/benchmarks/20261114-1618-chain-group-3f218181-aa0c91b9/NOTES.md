# The whole end group in the request chain

## Scheme

- Scheme: `chain-group`.
- Current code: `3f218181`, following `aa0c91b9`.
- Previous code: `aa0c91b9`.

## Purpose

The chain's stack segment proves and verifies the whole end group. The start
form's target and claim masks feed the panel-product sumcheck; the received end
values derive the hidden group after checking that every end slot names the
mask entry the stack used; the tie check recomputes the terminal sides from the
group. The merge's inputs come from the same end values through the stack end
constructor, so no caller-supplied weight constants remain. Two boundaries are
fixed: a single-entry panel's empty equality range is the empty product, one,
and the last round's point may carry the further variable.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_chain`,
  `request_stack` and `psc_general`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused chain tests | 3 passed |
| focused stack tests | 10 passed |
| focused general-sumcheck tests | 13 passed |
| workspace | 51 suites, 1062 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The chain proves and verifies the whole end group | chain segment rule | `prove_stack_group`/`verify_stack_group` in the chain | focused chain test |
| The hidden group comes from the received end values and the slots' mask entries | source rule | `group_from_ends` with slot/mask equality | focused test; a mismatched slot or mask is refused |
| The merge's inputs are the stack's end values, not caller constants | input source rule | `MergeValue::from_stack_end` from the group | focused chain test |
| Empty equality ranges are one, and the last round's point may carry the further variable | boundary rules | `eq_range`; panel-product point tolerance | focused and workspace tests pass |

## Conclusions

- The whole end group is wired into the chain on one convention, from the
  stack's sent values to the merge's inputs.
- No session message or proof path is touched; the toy proof is byte-identical.

## Open items

- The session entry and the negative matrix follow, then the source layer and
  the member chain.
