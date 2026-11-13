# Output tail group sized by the column domain

- Scheme: `norm-output-tail-width`
- Change: code commit `122ef0c2` (the tail row carries every slot of every
  column, so the group width is the column count times the slot count rounded
  up to a power of two).
- Baseline: previous commit `95447e03` (trace of the tail plan carrier
  `a5f892aa`).
- Purpose: correct the tail group width before the tail witness values are
  committed; the previous width covered one column's slots only.

## What changed

- `NormOutputTailWidths::columns(columns)` multiplies the column domain by the
  slot count and rounds up to a power of two.
- The tail plan validation compares the committed group width against this
  column-domain width.
- The plan test expects 64 columns for the integrated fixture (4 columns times
  12 slots) and rejects a group whose width is only the per-column slot
  round-up.

## Evidence

- Plan test passes with the corrected width and the four rejection cases.
- Focused: hidden_norm 34, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 910 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| tail rows cover the column domain | width = columns × slots rounded up | plan test | values are not committed yet |
| no degraded layout | the slot count per column is unchanged | slot assertion | the witness layout is next |
| digest and declaration checks | validation follows the corrected width | rejection cases | no scheme change |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 79 tests across the five filters.
- Workspace: 910 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The tail witness values and their commitment in the first-stage message.
- The rounding, bias and clamp relations; the padding rule; the no-wrap
  real-bound re-derivation; the hd binding.
- The ledger categories at the zkie path; production source closure.
- No push in this round.
