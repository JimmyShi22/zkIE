# Output tail groups fixed in the norm plan

- Scheme: `norm-output-plan`
- Change: code commit `b7e5114b` (output tail plan with four committed
  column groups, the output shift and the optional clamp; plan binding and
  digest; bind version 8; one plan test).
- Baseline: previous commit `58654802` (evaluation coefficient binding; trace
  `f05524a4`).
- Purpose: fix the output tail groups and the runtime parameters in the trusted
  plan before the products and the rounding, bias, clamp and padding relations
  are wired.

## What changed

- `NormOutputPlan`: the broadcast rstd row, the pointwise product row, the
  centered image row and the output hd row, plus the output shift and the
  optional clamp interval.
- Validation requires the parameter tables, binds every output group to the
  padded column domain, requires distinct oracle bases inside the section and
  against the five norm groups, and checks the shift and clamp bounds.
- The plan digest absorbs the section; the bind version moves to 8.
- A plan test checks the digest change, the column-domain binding, the shift
  and clamp rejections and the oracle-base collisions.

## Evidence

- Focused: norm 40, rstd 16, lookup 24, norm_param_table 4 passed
  (`focused.txt`).
- Workspace: 48 suites ok, 905 passed, 0 failed, exit 0
  (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 500 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| output tail groups fixed before challenges | groups and runtime parameters in the plan digest | plan test | the product instances are next |
| rstd broadcast, product, image and hd rows | one committed group per row | plan test | no proof wiring yet |
| shift and clamp trusted parameters | shift/clamp in the plan; bounds checked | rejection tests | the rounding and clamp relations are next |
| hd is the out group | the out group is the hd object | plan test | the MatMul consumer converges in 2c |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 84 tests across the four filters.
- Workspace: 905 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The two pointwise product instances through the existing product summand and
  the claim-to-message rows; the rounding, bias, clamp and padding relations;
  the witness rows and the no-wrap derivation.
- The hd binding to the same-position parameters and the MatMul consumer.
- Ledger categories at the zkie path; production source closure.
- No push in this round.
