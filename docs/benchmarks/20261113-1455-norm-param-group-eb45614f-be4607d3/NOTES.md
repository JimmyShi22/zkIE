# Parameter claims joined into the shared proximity group

- Scheme: `norm-param-group`
- Change: code commit `eb45614f` (parameter claim members and declaration
  rows joined into the shared proximity group; prover claim material through
  the norm witness; verifier rows from the first message; join assertion).
- Baseline: previous commit `be4607d3` (declaration rows; trace `e0a60241`).
- Purpose: carry the parameter claim members and their record/column
  declarations into the existing BAT/FOLD/CS/BASE closure, reusing the
  compatible shared group.

## What changed

- The shared stage adds one member per parameter claim: the folded message
  committed as G_u with the shared group code, no mask rows, its own oracle and
  runtime tree.
- The prover rebuilds the G_u tree from the committed material stored in the
  norm witness and checks the root; the verifier constructs the member from the
  first message root and the plan tree.
- Both sides build the sample row, the record rows and the column rows from the
  same claim message and plan reader; the rows enter the shared group's plain
  rows and the claims returned by the verifier.
- The member and row counts, the fold shape and the base and mask budgets stay
  member derived; the fixture's existing shared plan values accommodated the
  added members and rows, so no budget change was needed and the reuse
  condition holds.
- A test checks that the shared claims carry the parameter sample, record and
  column rows.

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
| compatible existing group reused | claim member with the shared group code joins the shared proximity group | positive chain; join assertion | fixture budgets only; no sampling change |
| member and row counts member derived | fold shape from the members; existing base and mask budgets | positive chain | planner-level budget derivation at the production path |
| record and sampled column declarations registered | rows enter the shared plain rows and claims | join assertion | evaluation consumption next |
| no per-position weight copies | one member per parameter table, shared by positions | member count | production source closure remains a dependency |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 82 tests across the four filters.
- Workspace: 903 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- Evaluation consumption into the norm relation rows.
- The output relations (image, shift, bias, clamp, padding) and the hd group.
- Ledger categories at the zkie path; production source closure.
- No push in this round.
