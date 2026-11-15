# asset-keeper-wiring

- Scheme: asset-keeper-wiring (TURN 566 item 1: the production keeper is
  opened from the real asset directory and the built table identity).
- Based on: the durable asset budget count (2978c7ad) and its trace
  (872302b9); previous baseline commit 872302b9.
- Purpose: connect the real asset table directory held by the model/CLI
  provider path to the persistent column reservation, with the ledger in a
  new directory and no write to any asset body.
- Change:
  - `AssetTableKeeper` wraps the authenticated encoding table with the
    persistent hiding state and the durable table-life column budget. It is
    the row source and the column opener; the first part reserves one batch,
    written and synced before any use; a new keeper over the same ledger
    continues the count; an exhausted ledger refuses the next reservation;
    there is no refund.
  - `column_ledger_dir(asset_dir, table_root)` derives
    `<asset_dir>/.zkie-column-ledger/<table-root-hex>`; the production
    constructor `open_asset_table_keeper` builds the authenticated table
    with the session configuration and opens the keeper over that ledger.
  - `ProviderConfig::column_ledger_dir` exposes the same derivation for the
    provider's real asset root, so the model/CLI path has one derivation.
  - Tests: derived path; keeper over the fixture's asset directory (two
    owners on one ledger, exhaustion, untouched asset body, row source and
    opener); provider derivation; a full first part whose failed proof keeps
    the durable count across a keeper restart.
- Numbers:
  - focused: keeper 2 (column session and first part) and provider 1, green
  - workspace: 1125 passed / 0 failed / 20 ignored over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries: the durable store keeps its recorded single-machine limits
  (trusted same-machine lock and directory; not rollback-proof or
  cross-machine; a copied directory has an independent count). The CLI does
  not yet run the hidden session; the keeper constructor and the provider
  derivation are the production edge, and the hidden-session caller passes
  the provider's asset root. LATER qualification stays in-session only:
  current support is ONE_TIME; LATER is not enabled. The Omega per-tree
  runtime opening count is not part of this scheme. Not pushed.
