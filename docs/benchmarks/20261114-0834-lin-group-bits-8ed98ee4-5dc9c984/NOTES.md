# Group row bits from the actual member layout

## Scheme and baseline

- Scheme: `lin-group-bits` (first piece of the missing sizing derivation).
- Code: `8ed98ee4`; previous commit `5dc9c984`; branch `refactor/v2`.
- Scope: the trusted planner's group row-bit derivation from the actual
  oracle members. No session parameter is wired yet; the production session
  plan still takes its parameters from the diagnostic fixture.

## What changed

- `crates/zkie-ops/src/proof/proximity_planner.rs`:
  `lin_group_bits(main_rows, mask_rows)` sums the aligned power-of-two main
  and mask row blocks of the actual group members, takes the mask bits as the
  ceiling of the mask sum, and the main bits as the maximum of the main row
  bits, the mask bits and one. Empty or unequal member lists are rejected.
- Unit tests: a request-sized group, unaligned member blocks, the
  table-sized shape, a mask-only group and the invalid lists.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Group bits from the actual members | committed-group block layout: each member occupies an aligned power-of-two row block | `lin_group_bits` sums the blocks | unit test |
| Mask bits ceiling and main bits floor of one | group shape rule | `h = ceil(log2(mask sum))`, `k = max(main bits, h, 1)` | unit test |
| Invalid member lists rejected | plan validation | empty or unequal lists rejected | unit test |
| Codes, random tails, stage lists, full round budget | next steps | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib lin_group_bits`: 1 passed
  (`group-bits-tests.txt`).
- `cargo test --workspace`: 51 suites, 992 passed, 0 failed, 19 ignored
  (`workspace.txt`).
- Hard gate: norm-free toy output-head proof 45,080 bytes, sha256
  `7a5da6eb…de4cfb`, unchanged.
- No push.

## Boundaries

- This is the first piece of the derivation only. The code capacities, the
  random tails, the two-stage and full stage derivation, the shared/own joint
  solving and the whole-chain round budget are the next steps.
- The production session plan still uses the diagnostic fixture parameters
  with the planner check disabled; nothing claims the production parameter
  provenance yet. Fixture constants remain test-only.
- The layer-0 key projection consumer chain is unchanged; all existing
  full-weight checks remain; no plaintext is removed.
- No protocol object, message or verification condition changed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `group-bits-tests.txt`, `NOTES.md`.
