# column-count-initialization

- Scheme: column-count-initialization (TURN 568 F1: the durable column count
  is initialized with the table, opened only when it exists, and never reset).
- Based on: the B3 four-item closure (f8f513ff); previous baseline commit
  f8f513ff.
- Purpose: distinguish a new table's legal initialization from an existing
  table's recovery; concurrent first opens must not truncate; an existing
  table with a missing ledger must not be zeroed; keep reserve-before-use and
  no refund.
- Change:
  - The engine count store has one initializer: under the count lock it
    creates an empty count with a create-new file and writes an
    initialization marker through a synced temporary file and rename. An
    existing count is adopted unchanged; a non-empty count without a marker
    is refused; an initialized table whose count is missing is refused.
  - The durable budget's constructor is open-only: it requires the marker and
    the count and never creates or truncates them; a separate initializer is
    the creation path.
  - The asset table commit initializes each committed table's ledger under the
    real asset directory and the table root; a repeated commit leaves a
    consumed count untouched. The prove-side keeper opens the ledger and never
    creates it.
  - Tests: concurrent initialization, repeated initialization after a
    reservation, a missing count on an initialized table, an unmarked
    non-empty count, commit-time creation and a repeated commit, a keeper that
    never creates its ledger, and a full first part across a keeper restart.
- Numbers:
  - focused: engine initialization 4; keeper 2; commit ledger 1
  - workspace: 1134 passed / 0 failed / 20 ignored over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries: the single-machine trusted directory and lock limits are
  unchanged; one count per ledger root; a deleted whole ledger directory is
  indistinguishable from a new table, which stays the store's documented
  copy/restore limit. LATER stays ONE_TIME-only. Not pushed.
