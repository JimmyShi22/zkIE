# chain-mask-reads

- Scheme: chain-mask-reads (B2 request chain, TURN 540-C1 V1: per-object
  capacity/read accounting for the sub-proof mask table).
- Purpose: the sub-proof mask table has its own opening budget and ledger, so
  the segment slots and openings are accounted per committed object and never
  merged with the request index-mask.
- Change: ChainMaskPlan gains the openings count (1..=code tail, checked in
  fault) and ChainMaskReads, a per-object ledger whose accumulated openings may
  not pass the table's code tail; zero or overflowing reads are refused.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  tests green; workspace numbers in workspace.log.
- Conclusions: the table's slots and openings are bounded and counted on the
  object that owns them; the fixture plan passes and the ledger rejects an
  over-budget read.
- Open items: table opening verification, the four remaining row bindings, the
  first-chain segment order, the transcript/version bump, member_rows_fit and
  the negatives.
