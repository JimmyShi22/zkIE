# Norm parameter table openings carried in the first norm message

- Scheme: `norm-param-message`
- Change: code commit `d34fcac8` (parameter opening/message types with the
  first-message section and plan gating; witness parameter vectors; width bound
  on the usable plain message region; one round-trip/gating test).
- Baseline: previous commit `3d9ef0f8` (plan binding; trace `47f04901`).
- Purpose: carry the authenticated parameter table openings in the first norm
  message so the verifier can check them against the plan's committed readers
  before any dependent challenge.

## What changed

- `NormParamOpenings` and `NormParamMessage`: the gain openings plus optional
  bias openings; the section is encoded only when the plan carries the tables.
  The codec bounds the opening count, row count, code depth, salt length and
  column range, and requires strictly increasing positions.
- `NormFirstMessages` gains the optional parameter section; the decode path is
  plan gated and the trusted check requires the message and plan to agree.
- `NormWitness` gains the private gain/bias parameter vectors for the later
  relation rows.
- The parameter width bound is the usable plain message region
  (`plain = ell - pads`), not the codeword length; the table pads are reserved
  for sampling.
- A unit test round-trips the first message with the parameter section, rejects
  the bytes under a plan without the tables, rejects decreasing positions and
  requires the bias openings when the bias table is separate.

## Evidence

- Focused: norm 34, rstd 16, lookup 24, norm_param_table 3 passed
  (`focused.txt`).
- Workspace: 48 suites ok, 899 passed, 0 failed, exit 0
  (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 498) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| openings carried and checked against the committed reader | opening/message types; plan-gated wire section | round-trip and plan-gating test | the verifier opening check is the next step |
| one commitment per parameter, shared across positions | one identity/reader per kind or one packed table | existing tables-plan tests | the fixture table build is the next step |
| witness material for the relations | private gain/bias vectors in the witness | witness field compiled through the fixtures | not consumed by a relation yet |
| width/mapping bound | width fits the plain message region | skeleton validation tests updated | evaluation reduction is the next step |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 77 tests across the four filters.
- Workspace: 899 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- Fixture parameter tables built from read-only rows; the position draw and the
  opening verification block in the first stage; ledger categories.
- Challenge order: identity/root/config/shape/mapping absorbed before sampling;
  LIN/FOLDEQ evaluation consumption and the output relations; the hd group.
- Production source closure remains a dependency; the i32 mapping flag is an
  assumption.
- No push in this round.
