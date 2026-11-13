# Exact output tail witness rows of the norm

- Scheme: `norm-output-tail`
- Change: code commit `0a09f845` (the per-position output tail witness: the
  rstd broadcast, the rstd-weighted product, the image, the half-up shift
  rounding decomposition and the bias/clamp output; checked against the
  runtime norm operator).
- Baseline: previous commit `8580b55f` (trace of the output tail plan
  `b7e5114b`).
- Purpose: compute the exact integer rows that the output relations
  (`image = centered * rstd * gain`, shift rounding, bias, clamp, padding)
  will bind, and fix their decomposition against the runtime semantics
  before the proof wiring.

## What changed

- `shift_round`: the half-up rounding of an image by `2^shift` decomposed into
  the Euclidean quotient and remainder, the selector that adds one when twice
  the remainder reaches the divisor, and the rounded quotient.
- `NormOutputRow`: the column-domain broadcast row (rstd repeated), the
  product row (`rstd * weight`), the image row (`centered * product` on valid
  columns, zero on non-valid columns), the quotient/remainder/selector rows,
  the shifted row, the pre-clamp row and the output row.
- The non-valid columns keep `shifted = 0` and output the clamped bias, not
  zero, which is the runtime convention; the bias is added before the clamp.
- The witness is prover/fixture-side; no verifier relation is added in this
  round.

## Evidence

- `output_tail_matches_the_runtime_operator`: the committed tail rows are
  compared against `ProfileLayerNormOp::forward_values` on the same values
  (valid and non-valid columns, positive and negative images, a tie, the
  clamp); the images and outputs match element by element.
- `shift_rounding_follows_the_half_up_discipline`: positive, negative, tie,
  exact and zero-shift cases plus the shift bound.
- `output_tail_tampering_is_rejected`: image, quotient, selector, output,
  broadcast and shape tampering are rejected by the row check.
- Focused: norm 34, rstd 16, lookup 24, norm_param_table 4 passed
  (`focused.txt`).
- Workspace: 48 suites ok, 908 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 500 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| image = centered * rstd * gain | product and image rows computed exactly | runtime operator comparison | rows are not yet bound by a proof relation |
| round_half_up on 2^shift | quotient/remainder/selector decomposition | tie/negative/exact cases | digit/selector relations are next |
| bias then optional clamp | pre-clamp row and clamped output | clamp boundary cases | clamp selector/slack rows are next |
| padding/non-valid columns | image zero, output clamp(bias) | runtime operator comparison | no relation yet |
| exactness | values match the runtime forward path | element-wise comparison | no claim of complete norm or proved rounding |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 78 tests across the four filters.
- Workspace: 908 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- Bind the two pointwise products through the product summand with equality
  weights and the claim-to-message rows; then the rounding, bias, clamp and
  padding relations; then the hd binding.
- The no-wrap real-bound re-derivation for the product, rounding and bias
  equations.
- Ledger categories at the zkie path; production source closure.
- No push in this round.
