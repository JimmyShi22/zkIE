# token-kept-row

- Scheme: token-kept-row (B2 request chain, TURN 540-C1 V2: the first of the
  four remaining relations lands on its owning member).
- Purpose: bind the token kept row form to the sub-proof mask table with a
  trusted slot-to-entry list and explicit kept-value owners.
- Change: request_chain::chain_mask_row_form builds the RowBinding (hiding
  entries at the first table entries in order, counted slots through the plan
  slot list, kept values on their owners) and delegates to the row form landing;
  a slot without a trusted entry is refused.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace in workspace.log.
- Conclusions: the token kept relation is expressed on the table that owns its
  slots; no table position is chosen by a proof byte.
- Open items: stack terminal recombination and A0-Alin/B0-Blin source rows,
  merge pscRow kept row, first-chain segment order, transcript/version,
  member_rows_fit, negatives, W layout freeze.
