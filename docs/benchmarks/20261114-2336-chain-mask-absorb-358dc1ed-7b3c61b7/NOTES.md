# chain-mask-absorb

- Scheme: chain-mask-absorb (B2 request chain, TURN 540-C1 V1 first step: the
  sub-proof mask table is fixed in the chain before the segment challenges).
- Purpose: give the sub-proof mask table its transcript entry points so the
  segment's slots are bound to a committed object (distinct from the request
  index-mask) before any relation row references them.
- Change: hidden_send_session::absorb_chain_mask_root (tag 18: schema, oracle,
  group, mask rows, entries, mask code, tree plan, root) and
  absorb_chain_mask_answer (tag 19: schema, answer); fixture test shows the
  same root and answer reproduce the chain value and a changed root changes it.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace numbers in workspace.log.
- Conclusions: the sub-proof mask table enters the chain under its own identity
  and layout; its root and answer are order-bound; the table plan (ChainMaskPlan)
  passes its structural checks in the fixture.
- Open items: the table's opening and capacity/read ledger, the four remaining
  row bindings, the first-chain segment order, the transcript/version bump.
