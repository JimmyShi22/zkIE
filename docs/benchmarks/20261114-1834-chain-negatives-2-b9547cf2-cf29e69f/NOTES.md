# Chain negatives: batch and weight slots

## Scheme

- Scheme: `chain-negatives-2`.
- Current code: `b9547cf2`, following `cf29e69f`.
- Previous code: `cf29e69f`.

## Purpose

Two more chain negatives join the set. Another public batch coefficient breaks
the start form and the stack's target against the proof. A swapped weight slot
names another mask entry than the one the stack used, so the group's source
check refuses it instead of reading one value for another.

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
| workspace | 51 suites, 1067 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| Another public batch coefficient is refused | plan data rule | changed batch in the chain inputs | focused chain test |
| A swapped weight slot does not read another entry | slot ownership rule | changed slot order | focused chain test |
| The rest of the chain stays green | regression rule | existing tests | workspace passes |

## Conclusions

- The chain refuses another batch and another weight slot.
- No session or proof message is touched; the toy proof is byte-identical.

## Open items

- The layout registry skeleton, then the W opening's ruling, follow.
