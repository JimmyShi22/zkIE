# Lookup table generalization and multi-column key binding

- Scheme: `lookup-multicolumn-table`
- Change: code commit `9f650d65` (lookup plan columns, key source columns,
  combination challenge, combined table view, leaf table values).
- Baseline: previous commit `8742e46a` (2b.2c closure).
- Purpose: prepare the second public table (rsqrt index to rstd value) by
  making the existing product-tree lookup table-generic and by binding an
  index column to a value column under one random combination challenge.

## What changed

- `LookupPlan` carries a column count (1..=4) and accepts any `table_log`
  in 1..=30; the byte table stays a one-column table with eight rows.
- `LookupKeyPlan` records one source per key and column, ordered key-major,
  with one public padding value per column. Validation rejects duplicate
  physical sources within a column and mismatched column order.
- `LookupTable` is the trusted table view: column count, padded row count,
  trusted identity, per-column point evaluation and per-column row reads.
  `CombinedTable` forms the theta-weighted linear combination of the table
  columns and evaluates the same combination on the key side.
- Challenge order: plan digest, trusted table identity, message roots, then
  theta (only when more than one column), then alpha, then the product
  sumchecks. `LookupOutcome` carries theta so the key claim expansion can
  weight each column by `eq * theta^column`.
- The leaf inner sumcheck previously used the row number as the table value
  (correct only for the byte table, whose value equals its row index). It now
  reads the table row values supplied by the combined view.
- `NormLookupPlan` binds the source column and the per-column padding values
  in its digest. The byte lookup keeps identity `0x63..63` and one column.

## Evidence

- Focused tests: lookup 14 passed, hidden norm 19 passed, hidden session 15
  passed (see `focused.txt`).
- Workspace: 48 suites reported ok, exit 0 (see `workspace.txt`).
- Two-column test `two_column_lookup_binds_the_index_value_correspondence`:
  - honest index/value rows over a four-row table (padded rows copy the first
    row) prove and verify;
  - the key claim expansion reconstructs the combined key vector from the
    committed index and value cells;
  - moving a value to another index (same value set, wrong correspondence)
    cannot be proved;
  - a changed trusted table identity is rejected before any proof.
- Hard gate: norm-free toy proof 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  byte-identical to the baseline (see `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 492) | Change | Evidence | Boundary / open |
| --- | --- | --- | --- |
| B: lookup binds the same table's index and value correspondence, values and table fixed first, random combination drawn after | column count in the plan; key sources per column; combined table view; theta drawn after plan/table/root absorption; theta in the outcome | two-column honest proof; wrong-correspondence proof fails; expansion reconstructs the combined keys | the rstd table view, real length and padding rows are not wired yet |
| B: padding rows are a public fixed valid row copy, not prover-chosen | per-column padding values in the key plan; table view rows expose the padded column values | two-column test uses the first-row copy for the padded keys and the table's padded rows; key claim expansion includes the padding constant | the rstd source and its real-length bound are the next step |
| D: key=index and the value column is bound by the multi-column form | key column 0 is the index source, column 1 the value source; the table combines index and value columns | the wrong-correspondence negative only fails because the pair is combined, not because a value alone is absent from the table | the range proof index<N and the A segment relations are not landed yet |
| A/C table identity and trusted view | trusted identity absorbed into the lookup transcript; table view supplies identity, row count, columns, point evaluation and rows | changed table identity rejected; byte and two-column paths verify | streaming range reads, content digest and IO/peak records are not landed yet |

## Key numbers

- Proof bytes: 45,080 (hard gate, unchanged).
- Wall time / VmHWM: this is a mechanism test round, not a benchmark matrix
  run; no performance number is claimed or recorded in the ledger.
- Tests: focused 14 + 19 + 15 passed; workspace 48 suites ok, 0 failed.

## Open items

- Streaming table view over a range-read source, real length, i32 decoding,
  base/extension concatenation, content digest and IO/peak records.
- A relations: nonnegative variance range, raw rounding quotient/remainder and
  no-wrap, committed segment selectors and per-segment index formulas.
- Wiring the rstd table and its index range into the hidden second BAT/FOLD,
  then the negative matrix through the real hidden verifier.
- No push in this round.
