# Chain round budget and capacity recheck

## Scheme and baseline

- Scheme: `round-budget` (round counts and the composed capacity recheck).
- Code: `21cc0e96`; previous commit `3129d9f3`; branch `refactor/v2`.

## What changed

- `proximity_planner.rs`: `chain_round_budget(stages)` counts one batch
  round, one epsilon round plus the stage's fold rounds and one code-switch
  round for every stage but the last, and the last stage's masked-target and
  base rounds; `budget_within(budget, reserved)` names the chain's overage
  over the reserved rounds; `capacity_recheck` composes the opened-list
  derivation with the per-object account for the whole chain.
- Unit tests: a three-stage chain's counts and total, the named budget
  overage, and the composed capacity recheck naming the request object.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Round counts from the stage list | round formulas (batch, epsilon + fold, switch, base) | `chain_round_budget` | unit test |
| A chain over its reserved rounds is named | budget check | `budget_within` | unit test |
| Position capacity rechecked over the chain | opened lists + per-object account | `capacity_recheck` | unit test |
| Production session assembly with the derived parameters | next step | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib round_budget`: 2 passed.
- `cargo test --workspace`: 51 suites, 1007 passed, 0 failed, 19 ignored.
- Hard gate: 45,080 bytes, sha256 `7a5da6eb…de4cfb`, unchanged.
- No push.

## Boundaries

- The round count is the stage-structure count (batch, folds, switches, base);
  the per-round error terms and message bytes are the session's own planning,
  not re-derived here. The reserved budget is a caller configuration value.
- The hiding-conditions recheck covers the position capacity only; the table,
  range and link conditions stay outside per the agreed scope.
- The production session still uses the diagnostic fixture parameters; the
  planner check stays disabled; fixture constants remain test-only.
- No protocol object, message or verification condition changed; all existing
  full-weight checks remain; no plaintext is removed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `round-budget-tests.txt`, `NOTES.md`.
