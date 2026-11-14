# chain-mask-root

- Scheme: chain-mask-root (B2 request chain, TURN 540-C1 V1: opening root
  verification for the sub-proof mask table).
- Purpose: bind the table opening to its committed root, with the list supplied
  by the trusted policy rather than the proof.
- Change: verify_chain_mask_opening(plan, root, list, opening) rejects a broken
  plan and delegates to the runtime tree opening verification with the explicit
  list; focused test covers the broken-plan refusal.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace in workspace.log.
- Conclusions: the table opening root path exists with the list policy outside
  the proof; the plan fault is checked first.
- Open items: a committed mask-only tree opening fixture (positive/negative
  path), the four remaining row bindings, first-chain segment order,
  transcript/version, member_rows_fit, negatives.
