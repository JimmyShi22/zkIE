# chain-mask-openings

- Scheme: chain-mask-openings (B2 request chain, TURN 540-C1 V1: the sub-proof
  mask table opening list and budget per object).
- Purpose: wire the table opening list into the plan checks and the table own
  read ledger before the real root verification.
- Change: ChainMaskPlan::check_positions: the list size is the plan openings,
  every position lies inside the tree columns, no position repeats, and the
  accumulated reads pass ChainMaskReads; the root is verified elsewhere.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  tests green; workspace numbers in workspace.log.
- Conclusions: the table opening list and budget are checked and counted on the
  owning object; wrong counts, repeated or out-of-tree positions are refused.
- Open items: root verification of the opening, the four remaining row
  bindings, the first-chain segment order, the transcript/version bump,
  member_rows_fit and the negatives.
