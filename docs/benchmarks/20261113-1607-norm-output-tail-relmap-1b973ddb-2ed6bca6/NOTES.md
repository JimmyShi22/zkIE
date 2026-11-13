# Value-level rounding relation maps of the output tail

- Scheme: `norm-output-tail-relmap`
- Change: code commit `1b973ddb` (five tail relations as bilinear maps over the
  committed slots, with a per-column product packing of eight slots).
- Baseline: previous commit `2ed6bca6` (trace of the tail slack slots
  `5721699c`).
- Purpose: fix the algebra of the rounding relations before the product
  instances and their rows are wired.

## What changed

- `output_tail_variables`: the product-sumcheck variable count of the tail
  relations (columns × eight packed products, rounded up).
- `output_rel_map`: relation 0 is the rounding reconstruction
  `image = q * 2^shift + r` with the offset quotient and the half-range offset
  constant; relation 1 is `shifted = q + selector`; relation 2 is the
  selector booleanity `s - s^2`; relation 3 is the doubled remainder
  `u = 2 r`; relation 4 is the tie slack `w = u - 2^shift * s`. Every
  coefficient is a field power of the digit base or a public constant; the
  maps are the witness-side algebra over the committed slot values.
- The packing uses eight product slots per column, which covers the largest
  relation (seven products) and keeps a power-of-two domain.

## Evidence

- On the exact fixture rows (positive and negative images, a tie, an exact
  division, non-valid columns) every relation inner product is zero.
- A changed image makes relation 0 nonzero; a flipped selector makes
  relation 1 nonzero; a changed remainder makes relation 0 nonzero; relations
  3 and 4 are derived from the row by construction and stay zero.
- Focused: hidden_norm 36, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 912 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| rounding reconstruction | relation 0 with digit powers and the half offset | exact-row and tamper checks | not yet a product instance |
| shifted = q + selector | relation 1 | selector flip check | the rows are next |
| half-up with nonnegative digits | relations 2..4 over selector, doubled and tie slots | exact-row checks | the range half of the proof is the slot widths |
| no-wrap reuse | field coefficients, integer identity below p/2 | existing bounds | the maps are field algebra |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 912 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The product instances over the tail relation maps and their rows binding
  the maps to the committed tail and image groups.
- The bias addition, the clamp interval and the padding rule; the hd binding.
- The ledger categories at the zkie path; production source closure.
- No push in this round.
