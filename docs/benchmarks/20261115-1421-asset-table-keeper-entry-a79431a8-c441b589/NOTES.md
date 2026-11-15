# asset-table-keeper-entry

- Scheme: asset-table-keeper-entry (TURN 568 F2: the persistent column budget
  reaches the actual asset table consumption entry).
- Based on: the column count initialization (174bbd28) and its trace
  (c441b589); previous baseline commit c441b589.
- Purpose: connect the chosen asset table shared verification entry to the
  existing keeper and one persistent budget owner; production reuse cannot
  silently bypass the budget.
- Change:
  - The production hidden-input entry takes the concrete asset table keeper,
    so the table's durable column budget is part of the entry type.
  - The authenticated table builder accepts the caller's shape; the hidden
    fixture's table shape is built from a committed asset opening through that
    builder instead of hand-built rows.
  - The authenticated table test proves and verifies the full hidden chain
    over the keeper: one batch reserved before any use, a restarted keeper
    continues the count, the exhausted ledger refuses the next proof, and
    every proof verifies against the keeper's reader.
  - The diagnostic table always carries a durable budget, so the real entry
    no longer runs with no persistent count.
- Numbers:
  - focused: authenticated keeper 1
  - workspace: 1135 passed / 0 failed / 20 ignored over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries: the fixture-oriented trait entry remains for isolated component
  runs; the CLI does not yet run the hidden session; the diagnostic's full
  entry still fails on the placeholder claim and zero maps (not R5); the
  diagnostic segment evidence path does not count a batch. LATER stays
  ONE_TIME-only. Not pushed.
