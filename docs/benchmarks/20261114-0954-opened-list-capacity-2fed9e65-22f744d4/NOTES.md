# Opened-list capacity mapping per encoded object

## Scheme and baseline

- Scheme: `opened-list-capacity` (position-capacity mapping).
- Code: `2fed9e65`; previous commit `22f744d4`; branch `refactor/v2`.

## What changed

- `proximity_planner.rs`: `StagePositions` and `opened_lists` derive each
  encoded object's accumulated opened lists over the stage list (folded and a
  split Omega by the shared test, Omega additionally by every request, the
  request's own objects by its tests; main and mask separate).
  `opened_capacity` checks every object's own list against its tail account
  and names the first object that does not fit, with its gap. The pad rule
  from the previous piece stays separate.
- Unit tests: the split and joined list derivation, a named Omega failure
  with the exact list, the request object checked on its own list.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Opened sets from the stage list and configuration | stage positions per group | `opened_lists` | unit test |
| Shared tail, Omega, request objects bound separately | per-object accounting | object lists derived separately | unit test |
| Spent = opened list, remaining = capacity - spent | capacity rule | `opened_capacity` | unit test |
| Negative remaining names object and gap | capacity failure | named error | unit test |
| main/mask separate, pads separate | prior account and pad rule | unchanged and reused | unit test |
| Table/range/link hiding conditions, whole-chain round budget | next steps | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib opened_list`: 2 passed.
- `cargo test --workspace`: 51 suites, 1003 passed, 0 failed, 19 ignored.
- Hard gate: 45,080 bytes, sha256 `7a5da6eb…de4cfb`, unchanged.
- No push.

## Boundaries

- Position capacity only: the table, range and link hiding conditions of the
  full checker and the whole-chain round budget are not implemented.
- The production session still uses the diagnostic fixture parameters; the
  planner check stays disabled; fixture constants remain test-only.
- No protocol object, message or verification condition changed; all existing
  full-weight checks remain; no plaintext is removed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `opened-tests.txt`, `NOTES.md`.
