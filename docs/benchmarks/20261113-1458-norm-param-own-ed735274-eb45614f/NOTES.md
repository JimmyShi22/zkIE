# Folded parameter members registered in the own group

- Scheme: `norm-param-own`
- Change: code commit `ed735274` (the same physical parameter claim members
  and their sample rows registered in the own proximity group; the tree source
  rebuilds the folded message tree).
- Baseline: previous commit `eb45614f` (shared group join; trace `d45a0fa4`).
- Purpose: make the committed parameter message available to the own-group
  relation rows by physical identity, without committing a second copy.

## What changed

- The own group's member table gains the same parameter claim members as the
  shared group: the same roots and runtime trees, the group main code, no mask
  rows, one member per parameter table.
- The prover keeps the committed folded message material in the norm witness;
  the own tree source rebuilds the folded message tree from it and checks the
  plan and the root. The verifier builds the same members from the first
  message.
- The sample rows of the parameter members are built from the same claim
  message seeds and answers in both groups.
- The member and row counts stay member derived; the fixture budgets
  accommodated the reuse, so no second commitment was needed.

## Evidence

- Focused: norm 38, rstd 16, lookup 24, norm_param_table 4 passed
  (`focused.txt`).
- Workspace: 48 suites ok, 903 passed, 0 failed, exit 0
  (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 500.4) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| same physical root/tree registered by physical identity | the own group reuses the shared member roots and trees | positive chain | ledger counting is at the zkie path |
| no prover-supplied copy | the tree is rebuilt from the committed material and checked against the plan and root | positive chain; tamper negatives | the sample rows bind the answers |
| claim object available to the own rows | the member and sample rows enter the own group | positive chain | the consumption rows are the next step |
| declaration-driven evaluation | not yet wired | — | next step |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 82 tests across the four filters.
- Workspace: 903 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- NormParamEval consumption rows from the shared declarations into the own
  relation rows; the output relations; the hd group.
- Ledger categories at the zkie path; production source closure.
- No push in this round.
