# Folded-message claim plans for the norm parameter tables

- Scheme: `norm-param-claim`
- Change: code commit `f8737593` (per-parameter folded-message claim plan;
  plan validation and digest; one plan test).
- Baseline: previous commit `a0981bbf` (opening verification; trace `fbdf01dc`).
- Purpose: fix the G_u message shape and the claim parameters in the trusted
  plan before the first stage commits and samples the folded row.

## What changed

- `NormParamClaimPlan`: the parameter kind, the G_u shape of the single-row
  fold (the folded row message with its own code, sample answers and runtime
  tree over the committed table index) and the column draw count. The table has
  one row, so the fold weights are the single entry one and there is no gamma.
- The tables plan can carry the gain and bias claims; validation requires the
  table index shape, the parameter width fit in the G_u message beside the
  sample pads, nonzero oracle and group ids, the column budget, and both
  claims present or both absent. The plan digest absorbs the claims.
- Unit test: a legal single-row fold shape (two main message rows, one fold
  weight, no gamma) binds to the plan; wrong kind, wrong index shape, zero
  draws, an unfolded message (one main row), a missing bias claim and a zero
  oracle are rejected.

## Evidence

- Focused: norm 37, rstd 16, lookup 24, norm_param_table 4 passed
  (`focused.txt`).
- Workspace: 48 suites ok, 902 passed, 0 failed, exit 0
  (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 500) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| form (a): G_u is the committed single-row fold, sent as usual | per-parameter GuShape (code/sample/tree/index) in the plan | plan test with a legal fold shape | the commitment itself is the next step |
| v=0 fold weights [1], no gamma | no gamma field; the fold shape is fixed by the plan | plan test | the fold computation is the next step |
| keep record and sampled column declarations | claim carries the column draw count and the table index | validation test | the declarations are wired with the G_u step |
| parameter width fits beside the pads | width bound against the G_u message room | one-main-row rejection | no sampling reduction |
| grouping by actual layout (next) | claim group id in the plan | plan validation | the group join is the next step |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the four filters.
- Workspace: 902 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- Commit and sample G_u (the folded row message) before drawing the columns;
  reorder the current parameter column draw behind the G_u commitment; keep the
  record and column declarations; group join per the actual layout.
- Evaluation consumption into the norm rows; the output relations; the hd
  group; ledger categories at the zkie path.
- Production source closure remains a dependency; the i32 mapping flag is an
  assumption.
- No push in this round.
