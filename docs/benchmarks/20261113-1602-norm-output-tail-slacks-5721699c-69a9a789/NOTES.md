# Halfway selector slack slots in the tail layout

- Scheme: `norm-output-tail-slacks`
- Change: code commit `5721699c` (doubled remainder and tie slack widths and
  digit rows in the tail layout).
- Baseline: previous commit `69a9a789` (trace of the tail no-wrap bounds
  `e006dfaf`).
- Purpose: carry the slack values the half-up selector relation needs as
  committed digit rows, with the widths derived from the shift divisor.

## What changed

- `NormOutputTailWidths` gains the doubled remainder (`2 r`, below twice the
  divisor) and the tie slack (`2 r - 2^shift * selector`, below the divisor);
  both widths are bounded and capacity-checked against the divisor.
- The slot order of one column is now: offset quotient digits, remainder
  digits, doubled remainder digits, tie slack digits, the half selector, the
  offset shifted digits and the offset pre-clamp digits.
- The slot values are recomputed from the exact output rows; the tie slack is
  derived as `2 r - divisor * selector`, so a witness whose selector
  disagrees with the remainder bound is a bound violation.
- The plan encoding absorbs the new widths; the integrated fixture uses
  widths 3/2/2/2/3/3 (16 slots per column, still 64 columns for the four
  column domain).

## Evidence

- The plan test accepts the extended widths, keeps the 64-column size and
  rejects the previous rejection cases.
- The full output chain still passes prove and verify with the extended tail
  slots committed and opened.
- Focused: hidden_norm 35, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 911 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| half-up selector with nonnegative digits | doubled remainder and tie slack rows | chain + plan test | the selector relation is not wired yet |
| widths from the trusted plan | capacity checks against the divisor | rejection cases | shift stays the plan value |
| no degraded layout | the existing slots keep their order and meaning | chain pass | the relations reuse the rows |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 80 tests across the five filters.
- Workspace: 911 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The rounding, bias, clamp and padding relations over the committed slots,
  reusing the no-wrap bounds.
- The hd binding; the ledger categories at the zkie path; production source
  closure.
- No push in this round.
