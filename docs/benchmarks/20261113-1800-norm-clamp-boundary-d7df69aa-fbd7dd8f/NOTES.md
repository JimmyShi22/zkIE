# Clamp boundary matrix on the hidden chain

## Scheme and baseline

- Scheme: `norm-clamp-boundary`.
- Code: `d7df69aa`; previous commit `fbd7dd8f`.
- This is partial progress on the norm output tail: the clamp branch matrix
  now runs through the real hidden chain with the slack range lookup attached.
  The boundary cases found and fixed a mistake in the non-strict distance
  relations.

## What changed

- Relations eight and ten (`(1 - s_lo)(u - beta_lo) = 0` and
  `(1 - s_hi)(v + beta_hi) = 0`) subtracted the half-range constant
  unconditionally. The constant is now scaled with the selector complement, so
  the strict branches close and the equality branches keep the identity value.
  Before the fix every narrow clamp interval failed the honest chain; the wide
  fixture interval had kept the strict branches unexercised.
- The output fixture takes a clamp interval and attaches the tail slack lookup,
  so the boundary tests exercise the clamp relations and the digit range lookup
  together.
- Boundary placements around a real pre-clamp value: one below the low bound,
  the low bound, one above it, one below the high bound, the high bound, one
  above it, and the collapsed interval where both bounds coincide.
- Negative tests: a changed clamped output, a flipped low selector in the
  committed message, a changed low distance byte, and a clamp distance width
  that cannot hold the span.

## Results

- `cargo test --workspace`: 48 suites, 924 passed, 0 failed, 17 ignored, exit 0.
- Focused: hidden_norm 37, norm_output 1, rstd 16, lookup 31, norm_param_table 4,
  norm_tail_slack 6, norm_tail_clamp 5 passed (`focused.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical (`proof-bytes.txt`).
- Verification set: the change is a module and fixture block on the hidden
  verifier path and does not touch seq512; the norm-free byte gate is the
  required byte gate for this step. No end-to-end seq512 case was run because no
  model-level path changed.

## Positive and negative coverage

- All seven boundary placements prove and verify through the real hidden chain
  with the slack lookup attached: below, on and above each bound, the half
  branch, and the collapsed interval.
- The flipped selector, the changed clamped output, the changed distance byte
  and the insufficient distance width are rejected in the real hidden path.
- The out-of-range slack digit and the tampered committed slack cell are covered
  by the tail-keyed lookup tests recorded in the previous trace.
- These are local tail relations; no soundness or exact-rounding claim for the
  complete norm is made.

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| Selector and sign equivalence in both directions | the non-strict half constant is scaled with the selector complement | all boundary chains; flipped-selector negative | local clamp relations |
| Exhaustive partition with the equality branch | the equality placements land on the identity branch | low, high and collapsed placements | local clamp relations |
| Differences stay below the modulus | the distance width gate rejects a width that cannot hold the span | width-gate negative | trusted plan check |
| Trusted bounds fixed before the challenges | the clamp interval stays a plan value; the version was raised with the tail layout | plan digest tests | trusted plan |
| All relations share the same pre and position | the tail rows carry every selector, slack and value term | positive chain; changed-output negative | local tail section |

## Scope and next

- No claim that the complete norm or exact rounding is proven; no production
  source closure for the gain and bias tables.
- Open: the padding negatives, the hidden-output binding, and the model-level
  assembly.
- No push in this round.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
