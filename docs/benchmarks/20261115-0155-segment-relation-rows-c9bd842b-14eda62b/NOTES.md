# segment-relation-rows

- Scheme: segment-relation-rows (the request chain segment's four remaining
  relations land in the own proximity group on the segment's committed mask
  table).
- Based on: the segment pending binding (commit 2bde0889, trace 14eda62b).
- Purpose: commit the sub-proof mask table as its own object in the first chain
  and check the segment's four remaining relations as plain BAT rows on that
  member, with the request index-mask kept as a separate object.
- Change: HiddenSendPlan carries the segment mask table plan; the first-chain
  segment commits it after the segment runs and before the section is fixed,
  and the message carries its root, sample, drawn positions and opening (the
  verifier checks all four). The own group includes the table as a mask-only
  member; its sample form and its fresh mask rows are part of the own
  proximity test. The four rows (token kept, A0 against the read cells, B0
  against the second side, merge remaining) are built from the segment outputs
  and the trusted inputs, rebound to the member's BAT registration and appended
  to the own group's plain rows. The fresh mask tree carries the sum of the
  padded mask-row blocks (request index-mask plus segment table).
- Key numbers: workspace 1093 passed / 0 failed (workspace.log); merged tests
  111 passed; hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; release
  gate green.
- Conclusions: the four relations are checked by the own proximity IOPP
  against the committed segment table. Session negatives refused: a changed
  root, sample answer or message plan, a missing mask message, a mixed oracle,
  a changed stack end value, a changed source cell, and missing, repeated or
  swapped weight slots. A changed section still changes the first chain value
  at its position.
- Boundaries: the table also carries the fixture's source cells and kept owner
  entry in its free entries; the production source-layer owners (committed
  table reads and reduction witness) and the frozen mask slot/opening layout
  remain open. The chain section's verifier still takes the caller's mask
  closure for the direct checks; the committed rows are the production-shaped
  checks.
- Not pushed.
