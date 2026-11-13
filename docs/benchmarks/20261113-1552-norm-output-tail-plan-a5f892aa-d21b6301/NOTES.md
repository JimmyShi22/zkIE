# Output tail digit rows carried in the plan

- Scheme: `norm-output-tail-plan`
- Change: code commit `a5f892aa` (trusted plan of the output tail witness rows:
  committed group, digit widths, capacity and oracle-base checks; bind version
  11).
- Baseline: previous commit `d21b6301` (trace of the pointwise row binding
  `219cf5f8`).
- Purpose: fix the committed group and the digit widths of the rounding,
  bias and clamp rows before the relations over them are wired.

## What changed

- `NormOutputTailWidths`: byte widths of the quotient, remainder, shifted and
  pre-clamp digit rows; every width is bounded and must stay below the
  modulus, and the remainder width must cover the shift divisor.
- `NormOutputTailPlan`: the committed column group of one position (quotient,
  remainder, half selector, shifted and pre-clamp slots per column) with the
  widths; the shift and clamp stay the output plan's trusted values.
- Validation: the group width equals the slot count rounded up to a power of
  two, the group is a valid message shape with one answer, and its oracle base
  is distinct from every other norm group (the five core groups, the round
  group when present, and the four output groups).
- The output plan carries the tail section optionally; the plan digest
  absorbs it and the bind version moves to 11.

## Evidence

- The plan test builds a tail section with widths (3, 2, 3, 3) and 12 slots
  rounded to 16 columns, accepts it, and rejects: a group whose column count
  differs from the slot count, a shift too large for the remainder width, an
  oracle base equal to the out group's, and a quotient width outside the byte
  bound.
- Focused: hidden_norm 34, norm_output 1, norm_pointwise 1, rstd 16,
  lookup 24, norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 910 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| committed rows for the rounding/bias/clamp tail | tail group in the output plan | plan test | the row values are not committed yet |
| digit widths with a single derivation | widths struct with bounds and capacity checks | rejection cases | the no-wrap real bounds are next |
| group identity fixed before challenges | group declaration and digest | plan test | the relations are next |
| shift/clamp stay trusted plan values | tail validates against the output plan | shift rejection | no scheme change |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 80 tests across the six filters.
- Workspace: 910 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The tail witness values and their commitment in the first-stage message.
- The rounding, bias and clamp relations over the committed rows (digit,
  selector and slack rows); the padding rule (non-valid columns output the
  clamped bias).
- The no-wrap real-bound re-derivation for every tail equation (< p/2), with
  stop-and-report if a bound fails.
- The hd binding to the same-position parameters; the ledger categories at
  the zkie path; production source closure.
- No push in this round.
