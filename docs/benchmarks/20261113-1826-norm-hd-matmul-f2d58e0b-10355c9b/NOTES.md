# Norm hidden-output object consumed by the MatMul dot

## Scheme and baseline

- Scheme: `norm-hd-matmul`.
- Code: `f2d58e0b`; previous commit `10355c9b`.
- This is partial progress on the consumer connection: the norm hidden-output
  object is the input of the existing hidden MatMul/Dot path, and the logits
  close to the argmax in the same session.

## What changed

- `HiddenSendPlan` carries an activation source. The new norm mode makes the
  Dot read its activation rows from the committed norm output messages; no
  separate activation matrix and no separate shown-map message is created.
- The statement carries the derived identity of the ordered norm output roots
  (a versioned hash over the plan numbers and the committed roots). The
  verifier recomputes it from the first-stage norm message and compares it with
  the prover's statement, so the consumer is bound to the object the norm
  relation proves, not to a second root and not to a plan field equality.
- The reduced rows and logits derive from those same committed messages. The
  first end row of the Dot identity addresses the norm output members per
  position with the position challenge weight and the source indexing of the
  output message.
- The output-closure plan runs in the same session: the logits are closed to
  the first-maximum argmax over the full masked vocabulary, and the closure
  messages, relations and masks flow through the existing BAT/FOLD/CS/BASE.
- The independently committed activation path keeps its statement bytes; the
  new mode appends its own mode tag, and its identity has its own version. The
  base send schema version stays unchanged so the norm-free gate is preserved.

## Results

- `cargo test --workspace`: 48 suites, 939 passed, 0 failed, 17 ignored, exit 0.
- Focused: hidden_norm 37, norm_output 5, rstd 16, lookup 31, norm_param_table 4,
  norm_tail_slack 6, norm_tail_clamp 5, norm_tail_padding 4, norm_output_hd 4,
  norm_hd 7 passed (`focused.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical (`proof-bytes.txt`).
- Verification set: this is the hidden verifier chain at fixture scale; it does
  not touch seq512. The norm-free byte gate is the required byte gate for this
  step; no end-to-end seq512 case was run because no model-level path changed.

## Positive and negative coverage

- The full chain norm object -> Dot MatMul -> logits -> closure -> argmax proves
  and verifies through the real hidden verifier. The committed output cells are
  the activation rows; the logits are the reduced rows; the closure winner is
  the first maximum of the masked logits.
- Rejected in the real hidden path: another committed output object in the same
  position, a changed committed output row under a consistent tree, a plan that
  names another output object, an independent-activation plan against this
  proof, a missing output section, a layout whose activation width differs from
  the norm valid width, and a closure winner that differs from the logits.
- These are fixture-scale structural and relation negatives; no soundness or
  complete-norm claim is made.

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| The hd group is the MatMul activation object | the Dot reads the committed norm output messages; no second commitment or shown map | positive chain; object-swap negative | fixture-scale hidden chain |
| The consumer and producer use the same object | derived statement identity over the ordered output roots; verifier recomputes it from the norm message | identity check; foreign-object negatives | the model-level entry is not switched |
| lm_head side unchanged | the reduced rows still derive from the same committed table and the same W group | positive chain | existing table mechanism |
| logits to closure to argmax unchanged | the existing closure plan runs in the same session over the logits | positive chain; wrong-winner negative | first-maximum over the masked vocabulary |
| Versioning | the mode tag is appended only on the norm path; the identity has its own version | old/independent plan negative | base schema version preserved for the gate |

## Scope and next

- No claim that the full local chain or topic is complete; the production source closure for the gain
  and bias tables remains a dependency.
- Open: switching the model-level output-head entry to build the norm mode
  (the hidden session path itself is wired), the real-path ledger and opening
  budget, and the model-level assembly.
- No push in this round.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
