# omega-opening-count

- Scheme: omega-opening-count (TURN 566 item 2: the Omega tree's runtime
  opening count is owned by the Omega stage and shared by every consuming
  chain).
- Based on: the asset keeper wiring (5f896a23) and its trace (f4ea26a8);
  previous baseline commit f4ea26a8.
- Purpose: accumulate the Omega tree's openings against its tail hiding
  capacity before each actual opening, share one count across the chains that
  open the tree, refuse exhaustion and keep the count when a run is
  abandoned.
- Change:
  - `TreeOpeningCount` in the tree bookkeeping module: one count shared
    through its handle, raised before the opening, checked against the tail
    budget and never refunded; equality is handle identity.
  - The shared stage creates the count when it commits the Omega tree and
    carries it into the shared proximity test; the second chain receives the
    same count through the shared witness.
  - The single-stage BASE opening and the first switch of the multi-stage
    loop both raise the count for the member whose tree number and root
    match, before opening that member; a count that matches no opened tree
    is refused. One-time mask trees are not represented and are never
    adopted across chains.
  - Tests: shared and fail-closed count; both-chain count in the
    single-stage path; both-chain count assertions in the staged path.
- Numbers:
  - focused: tree count 1; Omega count 1; staged path 1; multi-stage loop 28
  - workspace: 1127 passed / 0 failed / 20 ignored over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries: the count is run-scoped and in-memory; it is not a durable
  store (the asset table column budget remains the durable one). The count
  covers the Omega tree only. LATER qualification stays in-session only:
  current support is ONE_TIME; LATER is not enabled. Not pushed.
