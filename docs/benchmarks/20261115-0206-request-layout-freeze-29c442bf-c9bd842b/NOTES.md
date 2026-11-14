# request-layout-freeze

- Scheme: request-layout-freeze (the request-level parameters, the source-owner
  cells and the mask slot/opening layout are re-derived from the trusted
  shapes and the named layout is frozen for both sides).
- Based on: the segment relation rows (commit c9bd842b, trace 1e086f21).
- Purpose: derive every request-level quantity from the plan shapes instead of
  fixed numbers, check each object's opening list against its own code tail and
  its own ledger, and freeze the named request layout before any challenge.
- Change: RequestParameters gains the request W's encoded rows, oracle count
  and opening capacity, the segment mask table's shape, and the source-read
  cell counts. check_chain validates the W columns against its encoded rows,
  the pending allocation plus pending slots against the index-mask free
  entries, the segment entries and openings against its own table and tail, and
  the source cells against the committed table bounds. freeze_layout fills the
  named request layout and plan validation refuses a skeleton or a shape
  mismatch on both sides. The segment table's slot layout is derived from the
  chain inputs and the table capacity (token hiding, sent end values, merge
  hiding, source cells and counted claim slots in consecutive free entries)
  instead of fixed slot numbers.
- Key numbers: workspace 1093 passed / 0 failed (workspace.log); merged tests
  112 passed; hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; release
  gate green. Frozen values pinned by the test: row bits 1, column bits 2, W
  oracle 1, encoded rows 1, sample answers 1, opening capacity 4096, mask rows
  8, mask slots 5, end values 3, stack rounds 1, merge rounds 2, opening budget
  9, sample plan 3, tree layout 9, stage plan 0.
- Conclusions: every request-level number follows a trusted shape and is
  checked on both sides before challenges; a zero-row W message and a source
  cell outside the committed table are refused.
- Boundaries: openings_per_oracle = 2 stays the transitional W read of the
  approved first consumer (not the final proximity budget); the plain fixture's
  stage plan is zero while the staged fixture exercises the solved stages
  separately; the source-owner objects remain the committed table and read
  cells (bounds-checked registration), not yet the full source relations.
- Not pushed.
