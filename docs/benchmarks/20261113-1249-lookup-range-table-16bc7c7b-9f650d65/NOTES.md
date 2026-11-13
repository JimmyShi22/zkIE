# Range-read table view for frozen public tables

- Scheme: `lookup-range-table`
- Change: code commit `16bc7c7b` (range-read table view, content check,
  shared adjacent fold).
- Baseline: previous commit `9f650d65` (lookup table generalization).
- Purpose: let a verifier evaluate a frozen public table at a random point
  through bounded range reads, with the trusted identity, length, encoding and
  content digest bound before evaluation, and with the padded rows fixed to a
  copy of the first real row.

## What changed

- `TableConfig` fixes the metadata identity, the canonical content root, the
  real length, the row encoding (`i32-le` / `i64-le`) and the encoding
  version.
- `TableRowSource` is the read-only row source: identity, content root, real
  length, encoding and range row reads. The adapter maps logical rows to the
  base and extension assets; the base/extension concatenation and the
  first-row padding are exercised by a split source.
- `RangeTable` opens the source against the configuration, exposes the padded
  power-of-two domain and implements `LookupTable` with two columns: the row
  index column (padding rows copy index zero) and the value column (padding
  rows copy the first value).
- `column_at` splits the point into two halves, reads each intersecting row
  chunk once, folds the chunk over the low half and accumulates with the high
  equality weight. Peak memory is `O(2^(log/2))`; the index column needs no
  IO and the value column reads each real row at most once.
- `TableIoStats` records reads, rows, bytes and the peak range buffer.
- `verify_content` streams the real rows in 4096-row chunks and compares the
  canonical SHA-256 content digest with the trusted root.
- The byte table and the leaf fold share one adjacent-fold helper, so the
  coordinate order has a single implementation.

## Evidence

- Focused: `ZKIE_RSQRT_TABLE=<external gpt2 rsqrt table>`
  `cargo test -p zkie-ops --release --lib proof::lookup_table::tests`:
  6 passed, including the real 7,340,032-row table. Streaming evaluation at a
  23-coordinate point equals the direct full fold; IO rows = 7,340,033
  (one open row plus every real row once), bytes = 29,360,132, peak range
  buffer = 4,096 rows. The content check passes. See `focused.txt`.
- Unit cases: base/extension concatenation and first-row padding; content
  digest rejects tampered rows; mismatched identity, content root, real length
  and version are rejected at open; a 16,000-row table at log 14 streams the
  same value as the direct fold.
- Workspace: 48 suites ok, exit 0 (the real-table test is env-gated and
  returns without the external asset; it ran explicitly in the focused
  release command above).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (see `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 492) | Change | Evidence | Boundary / open |
| --- | --- | --- | --- |
| B: real length, ordering, i32 decoding, base/extension concatenation, identity and encoding version in trusted public configuration | `TableConfig` and `TableRowSource`; identity/content/length/encoding/version checks at open; split base+extension source | open negatives; concatenation and padding test; real asset length 7,340,032 and encoding i32-le | the model-level config and asset adapter are wired at integration; this round proves the table layer |
| B: padded rows are a public fixed valid row copy | `RangeTable` exposes padded index/value columns with first-row copies; the key-side padding is the same public first row | padding test and two-column lookup padding from the previous round | the index range `index < N` is an A-side relation, not yet landed |
| C: verifier evaluates `TableAt(point)` by range reads, no whole-table materialization; trusted identity/length/decoding/content digest bound; actual reads match; record IO/time/peak | split-coordinate chunked evaluation; `TableIoStats`; streaming content check; base/extension row adapter | real 7.34M-row table streams the direct value; rows/bytes/peak recorded; tamper rejected by the content digest | no timing claim is made; the production asset reader adapter and the hidden verifier wiring come at integration |
| A/D table identity is part of the lookup transcript | the table view exposes the trusted identity; the lookup absorbs it before theta and alpha | changed identity is rejected before any proof; byte and two-column tests | the rstd table is not yet in the hidden second BAT/FOLD |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused test wall time: 0.37 s (release, six table tests, includes the real
  table read); this is a test measurement, not a benchmark ledger entry.
- IO on the real table: 7,340,033 rows read, 29,360,132 bytes, peak buffer
  4,096 rows.
- Workspace: 48 suites ok, 0 failed.

## Open items

- A relations: variance nonnegativity and real range, raw rounding through
  `2^18`, committed segment selectors, segment slacks and per-segment index
  formulas, then `index < N`.
- Wire the rstd table view and the index/rstd key columns into the hidden
  second message and BAT/FOLD; run the negative matrix through the real hidden
  verifier.
- Production asset adapter for the frozen rstd table (mmap/range reads under
  the asset reader and governor), with the trusted configuration supplied by
  the model plan.
- No push in this round.
