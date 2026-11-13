# Bias addition bound to the committed shifted and pre-clamp slots

- Scheme: `norm-output-tail-bias`
- Change: code commit `bae4fc2e` (one bias point per run and one linear row per
  position binding `pre_clamp = shifted + bias`).
- Baseline: previous commit `dc21d684` (trace of the tail row binding
  `93c19b4c`).
- Purpose: bind the bias addition to the committed tail slots and the
  parameter bias evaluation.

## What changed

- A dedicated bias point is drawn from the chain after the tail instances with
  its own prefix tag; prover and verifier draw it identically and carry it in
  the row binding.
- One row per position: the pre-clamp digit reconstruction minus the shifted
  digit reconstruction on the committed tail slots, minus the bias evaluation
  coefficients of the parameter member at the bias point; the public
  half-range offsets are folded into the row target.
- The row is built from the plan and the drawn point only; no message field is
  added.

## Evidence

- The full output chain passes prove and verify with the bias rows in the BAT
  closure; the honest fixture bias values (1/2/-1/2) match the committed
  parameter table and the pre-clamp slots.
- The earlier tail commitment, transcript and row negatives still hold.
- Focused: hidden_norm 36, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 912 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| pre_clamp = shifted + bias | linear row per position | chain pass | one fresh point per run |
| bias from the parameter member | evaluation coefficients on the committed member | chain pass | the packed (single-reader) bias case is not carried by this row |
| public offsets | half-range offsets in the target | chain pass | no new commitments |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 912 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The clamp interval (selectors and slacks) and the padding rule; the hd
  binding.
- A wrong-bias negative needs a fixture witness whose pre-clamp slots disagree
  with the parameter table; the wire-visible path cannot inject it (recorded).
- The ledger categories at the zkie path; production source closure.
- No push in this round.
