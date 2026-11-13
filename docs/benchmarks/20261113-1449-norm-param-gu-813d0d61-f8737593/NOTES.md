# Folded parameter row committed and sampled before its columns

- Scheme: `norm-param-gu`
- Change: code commit `813d0d61` (G_u commitment, sample and answers for
  each parameter claim; column draws moved behind the G_u steps; claim message
  wire fields; fixture claims; single-row claim rule).
- Baseline: previous commit `f8737593` (claim plan; trace `dd5d1386`).
- Purpose: follow the fixed order of the ruling: fix the folded-row commitment
  and its sample before any column challenge.

## What changed

- One claim helper per parameter: fold the single table row (weights [1], no
  gamma), build the folded message as G_u with its own code, pads, tails, salts
  and runtime tree, absorb the root, draw and absorb the sample answers, then
  draw the table columns and open them.
- The first-message claim carries the G_u root, the sample seeds and answers,
  and the column openings; the codec binds the seed count and cube bits, the
  answer count, the opening count, the salt length, the row count and the
  strictly increasing positions.
- The verifier replays the root, seeds, answers and columns in the same order,
  compares the seeds and positions with the message and verifies the openings
  against the plan reader.
- Claims are single-row only in this stage; a packed multi-row table is
  rejected at plan validation (the gamma fold is not part of this step).
- The fixture attaches a legal fold shape to both parameter tables; the
  positive chain and the tampering negatives pass through the real verifier.

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
| fix G_u commitment and sample before columns | root absorbed, seeds drawn and answers absorbed before the column draw | order enforced in prover and verifier; seed tamper rejected | the claim declarations are the next step |
| G_u is the committed folded row, sent with its shape | own code/pads/tails/salts/tree; folded message committed | positive chain | the record/column declarations and group join follow |
| v=0 fold weights [1], no gamma | empty gamma fold; single-row rule | plan rejection of a multi-row claim | packed layouts wait for the gamma fold |
| record and sampled column declarations kept | column draws and openings carried after G_u | position/symbol negatives | declarations enter BAT/FOLD in the next step |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the four filters.
- Workspace: 902 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- Register the record and column declarations into the claim group and join it
  through the existing BAT/FOLD/CS/BASE; ledger categories at the zkie path.
- Evaluation consumption into the norm rows; the output relations; the hd
  group.
- Production source closure remains a dependency; the i32 mapping flag is an
  assumption.
- No push in this round.
