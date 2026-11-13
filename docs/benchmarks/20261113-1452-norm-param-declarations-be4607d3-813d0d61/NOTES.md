# Parameter claim declaration rows on both sides

- Scheme: `norm-param-declarations`
- Change: code commit `be4607d3` (claim message code aligned with the shared
  group code; record/column declaration row helper; one test).
- Baseline: previous commit `813d0d61` (G_u commitment; trace `dfacb3f0`).
- Purpose: build the record and column declarations that tie the folded
  parameter message to the committed table, on both the prover and the verifier
  side, before they join the proximity group.

## What changed

- The fixture claim's G_u message code aligns with the shared proximity group
  code; the committed table index keeps its single-row shape, and the folded
  row still fits the message beside its pads. The claim plan tests use the same
  group code.
- `norm_param_claim_rows` registers the folded message root against the claim
  shape, rebuilds the record seeds from the table name and shape, and turns
  each record answer and each opened column into a relation row. Both sides
  call the same helper with the same message and plan reader.
- A test proves the chain, builds the rows twice and checks they agree, and
  rejects swapped positions and a tampered column symbol.

## Evidence

- Focused: norm 38, rstd 16, lookup 24, norm_param_table 4 passed
  (`focused.txt`).
- Workspace: 48 suites ok, 903 passed, 0 failed, exit 0
  (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 500) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| compatible group code reuse | claim message code equals the shared group code; table index unchanged | positive chain with the aligned shape | member join is the next step |
| record declarations | record seeds from the table name and shape, one row per answer | rows test | rows not yet in BAT/FOLD |
| sampled column declarations | one relation row per drawn column via the verified opening | rows test; swapped/tampered negatives | group join follows |
| no prover-supplied values | both sides build the rows from the same message and reader | equal-rows test | claim binding in the group is next |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 82 tests across the four filters.
- Workspace: 903 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- Join the claim members and declaration rows into the shared proximity group
  with the member-derived fold/base/mask budgets; if infeasible, form a
  separate group with the concrete obstacle.
- Evaluation consumption into the norm rows; the output relations; the hd
  group; ledger categories at the zkie path.
- Production source closure remains a dependency.
- No push in this round.
