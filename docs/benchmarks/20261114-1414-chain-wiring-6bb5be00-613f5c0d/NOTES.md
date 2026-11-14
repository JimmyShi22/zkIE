# The request chain in the session plan and message

## Scheme

- Scheme: `chain-wiring`.
- Current code: `6bb5be00`, following `613f5c0d`.
- Previous code: `613f5c0d`.

## Purpose

The session carries the request chain's transcript section. The plan gains the
three proofs' shapes, the message gains the three proofs, and the section is
written after the integrated norm and before the chain value only when the
message carries it. The decoder reads the section only when the plan carries
the shapes, so a message and a plan that disagree are refused by the stored
counts instead of being reinterpreted. Plans without the chain encode exactly
as before.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib request_chain`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused section tests | 2 passed |
| workspace | 51 suites, 1055 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The plan carries the chain's shapes | plan field rule | `HiddenSendPlan.request_chain` | all plan literals updated; workspace builds and passes |
| The message carries the chain's proofs and is written only when present | section placement rule | `HiddenSendMessages.request`, encode after the norm | hard gate byte-identical; the section's own codec tests |
| Old plans encode and decode exactly as before | byte stability rule | no presence byte; the plan decides | toy proof 45,080 B with the same hash |
| A plan and a message that disagree are refused | mismatch rule | decode reads by the plan; stored counts checked | section decoder tests |

## Conclusions

- The section's carrier is in the session plan and message with byte-stable
  old paths.
- No proof path produces a chain yet; the next block wires its prove and
  verify entries.

## Open items

- The chain's prove and verify entries, the claim checks (`wsForkLin`, the
  stack's three linear claims) and the source-layer reads follow.
