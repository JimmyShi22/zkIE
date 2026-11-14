# request-w-session

- Scheme: request-w-session (B2 production projection consumer, TURN 538
  first part: the request W and the consumer member ride the second session).
- Purpose: run one whole hidden session with the plan's consumer member, the
  request-level W oracle and the chain's read cells, and fix the three defects
  the fixture exposed.
- Change: derive_request_w now uses the protocol's matrix map (the table cell
  at column >> k_lo, place low + y*2^k_lo), the same map the per-position W
  rows use; WCommitRandomness::from_coins draws one request W salt per tree
  column; witness_tree_for returns the retained request W tree; the member
  fixture precomputes the claim target from the same seed-derived randomness.
- Key numbers: hard gate 45,080 B, sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb;
  merged tests 103 passed; workspace numbers in workspace.log.
- Conclusions: with the corrected mapping the Eval row and the Dot row hold on
  the committed members and both proximity tests (shared and own) pass in prove
  and verify; the wrong mapping failed the BASE target linear check, which is
  the actual chain rejecting it.
- Open items: chain section (token/stack/merge) wiring into the session
  transcript; at_start slot ledger; request-level parameter re-derivation per
  TURN 530/532; the TURN 524 consumer negatives on the whole chain; the W
  layout (oracle count, encoded rows, slots, opening capacity) stays unfrozen
  per TURN 538.
