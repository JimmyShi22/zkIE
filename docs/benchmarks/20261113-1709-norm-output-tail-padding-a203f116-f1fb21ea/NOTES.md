# Padding columns bound to the committed bias rows

- Scheme: `norm-output-tail-padding`
- Change: code commit `a203f116` (one linear row per non-valid column binding
  the pre-clamp digits to the bias entry read from the committed G_u message
  at the column unit point).
- Baseline: previous commit `f1fb21ea` (trace of the clamp selector sign
  consistency `51024736`).
- Purpose: implement the TURN 504 二 (a) padding direction: `out = clamp(bias)`
  on non-valid columns with the same physical bias object as the main
  relations.

## What changed

- `output_tail_padding_rows`: for every position and every non-valid column
  of the trusted valid count, a linear row over the committed tail pre-clamp
  digit slots (offset reconstruction) minus the bias evaluation coefficients
  of the parameter member at the column unit point; the public half-range
  offset is the row target.
- The valid split is the trusted plan's valid count; the rows enumerate
  exactly the non-valid columns.
- The clamp value relation over the same pre-clamp slots then yields
  `out = clamp(bias)` on those columns.
- The bias read uses the same parameter G_u message, the same column-to-slot
  mapping and the same encoding as the bias addition row.

## Evidence

- Full output chain prove/verify passes with the padding rows in the BAT
  closure; the fixture bias entries for the non-valid columns (-1, 2) match
  the committed pre-clamp slots.
- Focused: hidden_norm 36, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 912 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 504 二) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| `bias_i` read from the G_u main column i | unit-point evaluation coefficients on the same member | chain pass | no new tree or message |
| same object as the main relations | same parameter message, mapping and encoding | chain pass | - |
| exhaustive valid split | rows enumerate exactly the non-valid columns | row construction | valid is the trusted plan value |
| `out = clamp(bias)` | clamp value relation on the same pre slots | chain pass | the selector sign consistency is the previous round |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 912 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The public byte-table range lookup for the committed slack digits
  (TURN 504 一 ①) is not wired yet; the slack widths bound the values but do
  not prove byte-ness through the lookup. This is the remaining mandatory
  item of the clamp mechanism.
- A direct wrong-bias/wrong-column padding negative needs a witness or
  message mutation that the current wire path does not expose (recorded).
- The hd same-position binding; the MatMul object convergence is 2c.
- No push in this round.
