# Projection reduction claim core

## Scheme and baseline

- Scheme: `projection-claim` (first part of the accepted B2b first delivery).
- Code: `2959e7f2`; previous commit `99222745`; branch `refactor/v2`.
- Scope: the projection geometry and the reduction claims over verified
  activation/output rows. The generic pending/member glue into the production
  hidden LIN session, the proximity closure and the production consumer
  wiring are the next parts and are not claimed here.

## What changed

- New module `crates/zkie-ops/src/proof/projection_claim.rs`:
  - `ProjectionShape`: `matrix_rows`, `k_lo`, `columns = matrix_rows << k_lo`,
    positions; validated as a whole.
  - `projection_claims`: builds the row weights and the output rows from the
    verified activation rows and the operator's actual output rows. Both
    sides derive the row weights from the activation values; no
    prover-supplied equation enters the claim.
  - `check_projection_reduction`: recomputes every output entry from the row
    weights and the trusted encoded table rows in the session's layout
    (`table_row = column >> k_lo`, `entry = low + row << k_lo`) and rejects a
    mismatch.
- Real-size test: 1024 x 1024 matrix, 2 positions. Round trip passes;
  negatives cover a changed output entry, an invalid column geometry, and
  short activation or output rows.
- No existing path changed; no proof or protocol change; no plaintext
  removal.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Projection geometry and table layout | hidden session reduction layout (`column >> k_lo`, `low + row << k_lo`) | `ProjectionShape`, `entry` | real-size round trip |
| Claims from verified values | verified trace outputs / activation (adapter template in the output-head test) | `projection_claims` | real-size test |
| Verifier-side reduction check | same layout, trusted table rows | `check_projection_reduction` | test and changed-output negative |
| Pending/member/BATT glue and hidden LIN closure | `hidden_send_session` plans, `fold_member_firsts` | **not yet** | gap |
| Production Projection consumer | — | **not yet** | gap |

## Tests and evidence

- `cargo test -p zkie-ops --lib projection_claim`: 2 passed.
- `cargo test --workspace`: 49 suites, 980 passed, 0 failed, 17 ignored.
- Hard gate: 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  `cmp` identical to the earlier acceptance reference; the temporary dump was reverted.

## Boundaries

- This is the reduction and claim core. The general pending/member glue
  (member dimensions and mask ownership, BATT reduction, first/shared/second
  requests, proximity and output checks) and the real consumer wiring are the
  remaining parts of B2b.
- The 1024 x 1024 fixture uses deterministic test values, not production
  activations; the next part uses the real consumer values from the verified
  trace and the B1 table.
- The expert's acceptance (one real Projection consumer fully closed) is not
  claimed. No push.

## Artifacts

`identity.txt`, `change.diff`, `projection-claim-tests.txt`, `workspace.txt`,
`proof-bytes.txt`, `NOTES.md`.
