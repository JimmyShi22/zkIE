# stack-source-rows

- Scheme: stack-source-rows (B2 request chain, TURN 540-C1 V2: the A0-Alin and
  B0-Blin source relations land on their owning members).
- Purpose: express the two stack sources as rows on the reduction-table owner
  with the equation weights and the sub-proof mask slots.
- Change: request_chain::stack_source_rows builds the two rows (table side
  units with the equation weights plus the value own mask slot with beta_stack,
  target = shown value); mismatched weight lengths and a zero factor are
  refused.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace in workspace.log.
- Conclusions: A0 and B0 sources reference the table owner and the mask table
  by (member, entry) identity; the sent shown values are the targets.
- Open items: stack terminal recombination row, merge pscRow kept row,
  first-chain segment order, transcript/version, member_rows_fit, negatives,
  W layout freeze.
