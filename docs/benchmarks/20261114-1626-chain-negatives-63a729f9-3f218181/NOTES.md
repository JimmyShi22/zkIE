# Chain negatives: mask source and end slots

## Scheme

- Scheme: `chain-negatives`.
- Current code: `63a729f9`, following `3f218181`.
- Previous code: `3f218181`.

## Purpose

The chain's group derivation gains two refusals. Every end slot must be
distinct: a repeated slot would let one sent value stand in for another from
the same mask row. And every slot's mask entry must equal the entry the stack's
sumcheck used: a different closure over the same slot numbers is another
source and must not pass. Both join the chain's existing negatives (changed
token proof, changed merge proof, changed matrix or mask value).

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_chain`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused chain tests | 3 passed |
| workspace | 51 suites, 1062 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Every end value has its own mask slot | slot ownership rule | distinct-slot check in `group_from_ends` | focused test; a repeated slot is refused |
| The slot's entry must be the mask the stack used | source rule | slot/mask equality | focused test; another mask source is refused |
| The remaining chain negatives stay green | regression rule | existing tests unchanged | workspace passes |

## Conclusions

- The two source-side negatives are in place; the chain refuses another mask
  source and a repeated end slot.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The session entry and the source layer follow, then the member chain.
