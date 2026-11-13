# Masked output product claim

## Scheme and baseline

- Scheme: `norm-output-claim-mask`.
- Code: `72b6041d`; previous commit `e5bb6b59`.
- This is the rework of the output product claim hiding: the sumcheck target
  sent to the verifier is now masked by a committed claim-mask entry, and the
  verifier's view carries no unmasked private product evaluation.

## What changed

- The pointwise product instances use one reserved claim mask entry per
  instance from the first-stage norm mask message. The prover sends
  `masked = value + mask`, runs the product sumcheck with the claim mask term,
  and the verifier checks the same term from the plan-derived slot.
- The claim row carries the mask unit, so the masked target is bound to the
  product or image map at the original column point drawn for the instance.
  The unit row carries the scaled claim-mask terms, so the mask cancels exactly
  where the sumcheck's own row consumes it.
- The mask plan requires at least one reserved claim entry per pointwise
  instance; the fixtures set the count. The entries are committed and sampled
  in the first stage, before the pointwise product-sumcheck challenges are
  drawn.
- The verifier outcome exposes the checked output bindings (original point and
  masked claim) for positive evidence.
- The private product evaluation no longer appears in the message: the prover
  target is masked and the sumcheck rounds, hiding sums, end values and mask
  forms stay as before.

## Results

- `cargo test --workspace`: 48 suites, 943 passed, 0 failed, 17 ignored, exit 0.
- Focused: hidden_norm 37, norm_output 9, norm_output_claim 4, rstd 16,
  lookup 31, norm_param_table 4, norm_tail_slack 6, norm_tail_clamp 5,
  norm_tail_padding 4, norm_output_hd 4, norm_hd 7 passed (`focused.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical (`proof-bytes.txt`).
- Verification set: the change is at fixture scale on the hidden verifier
  chain and does not touch seq512; the norm-free byte gate is the required byte
  gate for this step. No end-to-end seq512 case was run because no model-level
  path changed.

## Positive and negative coverage

- For every pointwise instance the verifier-accepted claim differs from the
  private evaluation computed from the witness rows at the same original point,
  and the private evaluation's field encoding does not appear anywhere in the
  encoded proof bytes.
- Rejected in the real hidden path: a changed sent claim, a changed norm mask
  root, a changed mask sample answer, and a changed committed claim mask entry.
  The original-point binding and the product/image map binding are untouched
  from the earlier fix; a wrong point would require a prover-side harness
  because the point is drawn from the transcript on both sides.
- These are fixture-scale hiding and binding negatives; no formal hiding bound
  is claimed for the complete norm.

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| The claim uses a committed mask before the challenges | reserved claim mask entries, committed in the first-stage mask message | chain order; mask root/sample negatives | fixture-scale chain |
| The sent claim is not the private evaluation | prover sends `value + mask`; verifier checks the mask term | claim/value inequality; plaintext scan | no formal hiding bound |
| The mask is bound to the committed declaration | claim row mask unit; unit row scaled claim-mask terms | changed claim, changed mask entry negatives | same committed mask message |
| The original point and maps stay | the claim row still uses the original column point and product/image map | existing point/map negatives | unchanged |
| No zero-mask workaround | the claim term is non-empty and plan-derived | code path and tests | - |

## Scope and next

- No claim that the full chain or the topic is complete; the production source
  closure for the gain and bias tables remains a dependency.
- Open: the expert's remaining batch review items, the model-level entry
  convergence and the real-path ledger budget.
- No push in this round.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
