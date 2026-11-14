# merge-remaining-row

- Scheme: merge-remaining-row (B2 request chain, TURN 540-C1 V2: the merge
  sumcheck remaining relation lands on its owning member).
- Purpose: give the wsForkLin remaining row the same trusted landing as the
  token kept row.
- Change: request_chain::merge_remaining_row takes the merge output row form
  through chain_mask_row_form with the trusted slot list and kept owners; a
  slot without a trusted entry is refused.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace in workspace.log.
- Conclusions: the merge remaining row is expressed on the mask table that owns
  its slots, with the target and kept owners explicit.
- Open items: the stack terminal recombination (Y-round reconstruction already
  covers the sent masked end group), first-chain segment order,
  transcript/version, member_rows_fit, negatives, W layout freeze.
