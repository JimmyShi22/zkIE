# chain-mask-answer

- Scheme: chain-mask-answer (B2 request chain, TURN 540-C1 V1: the sub-proof
  mask table answer derivation).
- Purpose: the table has its own sample form and answer, distinct from the
  request index-mask table.
- Change: ChainMaskPlan::answer reads the mask rows through the sample cube
  (row bits + mask code message bits), the last entry takes the seed factor
  instead of its cube coefficient; zero factor, short seed and wrongly shaped
  rows are refused.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green (independent cube recomputation); workspace in workspace.log.
- Conclusions: the table answer follows the trusted layout with no shared
  state with the index-mask; malformed inputs are refused before any challenge.
- Open items: root verification of the opening, four row bindings, first-chain
  segment order, transcript/version, member_rows_fit, negatives.
