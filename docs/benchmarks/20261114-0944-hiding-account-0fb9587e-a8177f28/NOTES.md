# Position hiding capacity per encoded object

## Scheme and baseline

- Scheme: `hiding-account` (first piece of the capacity mapping).
- Code: `0fb9587e`; previous commit `a8177f28`; branch `refactor/v2`.

## What changed

- `proximity_planner.rs`: `HidingAccount{tail, spent}` keeps one account per
  encoded object with `remaining()` and `covers(list_len)`;
  `hiding_capacity(names, accounts, list_len)` requires the unified stage
  list to fit every affected object and names the first object with its gap;
  `ood_pads_fit(entries, rows, ell, answers)` counts the sampled answers'
  pads on the message's own rows, separate from position capacity.
- Unit tests: per-object spent/remaining, no pooling between main and mask,
  a named capacity failure, and pads counted separately from hiding capacity.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| One account per encoded object | tail hides opened positions | `HidingAccount` | unit test |
| Spent and remaining per main/mask, no pooling | per-object accounting | `hiding_capacity` | unit test |
| Unified stage list within every object's remaining capacity | capacity rule | named failure with gap | unit test |
| OOD answers and pads on separate books | pad rule | `ood_pads_fit` | unit test |
| Table/range/link hiding conditions, whole-chain budget | next steps | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib hiding_account`: 2 passed.
- `cargo test --workspace`: 51 suites, 1001 passed, 0 failed, 19 ignored.
- Hard gate: 45,080 bytes, sha256 `7a5da6eb…de4cfb`, unchanged.
- No push.

## Boundaries

- This is the per-object position account and the pad rule only. The table,
  range and link hiding conditions of the full checker, the capacity mapping
  over the actual opened lists and the whole-chain round budget are not
  implemented.
- The production session still uses the diagnostic fixture parameters; the
  planner check stays disabled; fixture constants remain test-only.
- No protocol object, message or verification condition changed; all existing
  full-weight checks remain; no plaintext is removed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `hiding-tests.txt`, `NOTES.md`.
