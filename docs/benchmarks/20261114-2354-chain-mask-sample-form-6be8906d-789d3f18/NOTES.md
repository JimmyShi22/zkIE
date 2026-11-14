# chain-mask-sample-form

- Scheme: chain-mask-sample-form (B2 request chain, TURN 540-C1 V1: the
  sub-proof mask table sample form width).
- Purpose: fix the seed-coordinate count of the table mask sample form before
  the answer derivation and root verification.
- Change: ChainMaskPlan::sample_bits = mask table row bits + mask code message
  bits, refused when the table plan faults; fixture checks two bits for four
  rows plus four message bits for a sixteen-place code.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace numbers in workspace.log.
- Conclusions: the table sample form width follows the trusted layout; a broken
  plan fails before any sample is drawn.
- Open items: answer derivation, root verification, four row bindings, the
  first-chain segment order, the transcript/version bump, member_rows_fit and
  the negatives.
