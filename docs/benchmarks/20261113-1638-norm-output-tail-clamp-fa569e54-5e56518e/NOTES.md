# Clamp value and selector exclusivity relations

- Scheme: `norm-output-tail-clamp`
- Change: code commit `fa569e54` (tail packing widened to sixteen products per
  column; committed out value as a factor; clamp value relation five;
  selector exclusivity relation six; tail instance count follows the relation
  constant).
- Baseline: previous commit `5e56518e` (trace of the clamp slot carrier
  `5794f907`).
- Purpose: close the clamp value equation over the committed slots and bind
  the out group (the hd object) to the pre-clamp chain, before the selector
  consistency slacks and the padding rule.

## What changed

- `NORM_OUTPUT_TAIL_PAIRS = 16` and `NORM_OUTPUT_TAIL_RELATIONS = 7`; the tail
  relation variables follow (6 on the four-column fixture); the committed mask
  allocation and the tail transcript section follow the same constant.
- `TailFactor::Out`: the committed out group value of the term's column; the
  row builder carries out coefficients per side.
- Relation 5: `out = pre - s_lo * (pre - low) - s_hi * (pre - high)` with the
  committed offset distances and selectors and the public half-range
  constants; when the plan has no clamp interval it degenerates to the
  identity `out = pre`.
- Relation 6: selector exclusivity `s_lo * s_hi = 0`; without a clamp interval
  it contributes no terms.
- The tail group is now 128 columns (26 slots per column); the transcript and
  mask counts follow the relation constant.

## Evidence

- The relation test passes all seven inner products on the exact rows
  (clamp-free fixture rows for the identity form) and rejects changed image,
  selector and remainder values.
- The full output chain passes prove and verify with seven tail instances per
  position and the out coefficients in the rows.
- Focused: hidden_norm 36, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 912 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 496.5 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| clamp value relation | relation 5 over committed out/pre/distance slots | chain + relation test | selector consistency slacks are not wired |
| selector exclusivity | relation 6 `s_lo s_hi = 0` | relation test | selector booleanity/consistency are separate |
| out is the hd object | out group term in the rows; clamp row binds out to pre | chain pass | MatMul same-object convergence is 2c |
| no clamp interval | identity `out = pre` | relation test (clamp-free rows) | plan-gated |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 912 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- Selector consistency (`s_lo = [pre < low]`, `s_hi = [pre > high]`) needs
  committed slack slots; not wired.
- Padding (`out = clamp(bias)` on non-valid columns) needs a per-column bias
  object; the parameter member only gives random-point evaluations. Stopped
  here and reporting per the task rule; no self-invented mechanism.
- The hd same-position binding to x/mean/rstd/gain/bias is the conjunction of
  the existing rows; the MatMul consumer convergence is 2c.
- The ledger categories at the zkie path; production source closure.
- No push in this round.
