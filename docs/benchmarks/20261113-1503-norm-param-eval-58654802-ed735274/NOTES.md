# Parameter message evaluation coefficients for the own relation rows

- Scheme: `norm-param-eval`
- Change: code commit `58654802` (equality-weight scattering of a parameter
  vector point into the Gu message BAT coefficients; one test).
- Baseline: previous commit `ed735274` (own group registration; trace `2d25778a`).
- Purpose: let the own relation rows bind gain and bias evaluations to the
  committed folded message, declaration driven and never from a prover-supplied
  value.

## What changed

- `norm_param_eval_coefficients` takes the column point and the parameter
  width, computes the equality weights, scatters them through the Gu message
  layout into the member's BAT table coefficients, and checks the width, the
  point length and the usable message room.
- The helper never reads a proof value: the coefficients reference the
  committed folded message registered in the own group by physical identity.
- A test checks the coefficient mapping for both parameter claims and rejects a
  wrong point length and a vector wider than the message room.

## Evidence

- Focused: norm 39, rstd 16, lookup 24, norm_param_table 4 passed
  (`focused.txt`).
- Workspace: 48 suites ok, 904 passed, 0 failed, exit 0
  (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 500) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| evaluation bound by declaration, not prover value | eq weights scatter into the committed message coefficients | coefficient test | the row term is added by the output relations |
| same physical member referenced | coefficients use the own-group member registered by physical identity | positive chain | ledger counting at the zkie path |
| width and layout fixed | point length, width and message room checks | wrong-length and too-wide negatives | packing layouts still single-row |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 83 tests across the four filters.
- Workspace: 904 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The output relations: the pointwise products (rstd times gain, centered times
  it) need the existing product summand machinery wired into the norm rows,
  then the shift, bias, clamp and padding relations.
- The hd group and the MatMul consumer object.
- Ledger categories at the zkie path; production source closure.
- No push in this round.
