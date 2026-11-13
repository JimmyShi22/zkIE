# Clamp selector sign consistency

- Scheme: `norm-output-tail-clamp-consistency`
- Change: code commit `51024736` (four sign-consistency relations for the
  clamp selectors over the committed slacks and distances; wider tail
  packing; fixture mask rows raised).
- Baseline: previous commit `86e28e41` (trace of the clamp slack slots
  carrier; trace path below).
- Purpose: enforce both directions of `s_lo = [pre < low]` and
  `s_hi = [pre > high]`, so the selector cannot be chosen freely by the
  prover.

## What changed

- `NORM_OUTPUT_TAIL_PAIRS = 32` (worst relation uses seventeen products per
  column) and `NORM_OUTPUT_TAIL_RELATIONS = 11`.
- Relations 7..10:
  - 7: `s_lo (u + 1 + alpha_lo) = 0` with `u = pre - low` (strict side);
  - 8: `(1 - s_lo) (u - beta_lo) = 0` (non-strict side);
  - 9: `s_hi (v - 1 - alpha_hi) = 0` with `v = pre - high`;
  - 10: `(1 - s_hi) (v + beta_hi) = 0`.
  The selector is multiplied through the committed distance and slack digit
  rows, folding the public half-range offsets into the products.
- The relation constant drives the tail instance counts everywhere (mask
  plan limit, transcripts, row bindings); the fixture mask message grows to
  64 rows to cover the wider mask allocation.

## Evidence

- Relation test: all eleven inner products are zero on the exact rows
  (clamp-free rows make relations 5..10 degenerate correctly).
- Full output chain prove/verify passes with eleven tail instances per
  position and the new relations in the BAT closure.
- Focused: hidden_norm 36, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 912 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 504 一 / 492.A) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| committed slacks | slack digit slots in the clamp section | chain pass | byte range lookup for the slack digits is not wired |
| strict/non-strict identities | relations 7..10 with the selector multiplied through | relation test | the identities hold over the committed digits |
| both directions | s=1 branch and s=0 branch both enforced | relation test | - |
| no-wrap evidence | distance/slack products use the no-wrap bounds | bounds from the earlier round | the slack digit non-negativity rests on the committed width and byte range lookup (next) |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 912 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The byte range lookup for the slack digits (public byte table), binding the
  non-negativity of the committed slack digit rows.
- Padding: `out = clamp(bias)` on non-valid columns with `bias_i` read from
  the G_u main column slots (TURN 504 二 (a)).
- The hd same-position binding; the MatMul object convergence is 2c.
- No push in this round.
