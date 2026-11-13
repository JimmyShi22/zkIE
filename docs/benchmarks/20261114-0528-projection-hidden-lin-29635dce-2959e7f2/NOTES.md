# Real-size projection consumer closure on the hidden LIN session

## Scheme and baseline

- Scheme: `projection-hidden-lin` (closure part of the accepted B2b first
  delivery).
- Code: `29635dce`; previous commit `2959e7f2` (claim core); branch
  `refactor/v2`.
- Scope: the projection claim core wired into the production hidden LIN
  session's diagnostic path, closed at the real 1024 x 1024 projection
  geometry, with negatives. The production consumer (real asset, activation
  and output values; hidden-input mode; index-mask slot; bias) is the second
  commit.

## What changed

- New integration test `crates/zkie-ops/tests/projection_hidden_lin.rs`:
  - builds a 1024 x 1024 projection table in the session layout (table rows
    1024, table code log_n 12 / ell 2048 / r 8, columns 1024, matrix_rows
    1024, k_lo 0) with deterministic signed weights;
  - commits a real 2 x 1024 activation and computes the actual output rows
    from the same table layout;
  - derives the row weights and output claims from those values and builds
    the hidden `HiddenLinPlan` with the session's G_u, Omega, tree and mask
    shapes (proximity code log_n 14 / ell 8192 / r 4);
  - runs the production diagnostic entries `prove_hidden_lin` and
    `verify_hidden_lin` end to end (78 s per full run);
  - re-checks the same values with the claim core's verifier-side reduction.
- Parameters were fitted to the production relationships: the folded table
  row fits beside the G_u pad (2056 <= 8191), the table column budget does not
  exceed the table code tail (4 <= 8), and the proximity code tail covers the
  SEND and BASE openings (2 + 2 <= 4).
- Negatives: a changed output row, changed coefficients in the verifier plan,
  and a changed table identity are all rejected by the session.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Operator reduction to claims | projection claim core (`2959e7f2`) | claims from the verified activation and actual output rows | positive test |
| Pending/member/BATT and LIN first/shared/second plus proximity | production `hidden_send_session` plans, fold and prove/verify entries | the plan is built from the claims and the session runs its own pending/member/BATT, first/shared/second and proximity tests | prove and verify pass at 1024 x 1024 |
| Verifier reconstructs the sources | activation rows -> row weights; output rows -> targets | claims derived on both sides, not prover-shaped | changed-coefficient negative |
| Table identity binding | reader table identity versus `send.table_identity` | plan identity is the reader identity | changed-identity negative |
| Mask ownership and bias | index-mask ledger slot; bias is not the protocol mask | not exercised in the diagnostic path | gap |
| Production consumer | real asset, activation and output | not yet | gap |

## Evidence

- `cargo test -p zkie-ops --test projection_hidden_lin`: 4 passed (positive
  closure and three negatives), 78 s.
- `cargo test --workspace`: 50 suites, 984 passed, 0 failed, 17 ignored.
- Hard gate: 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  `cmp` identical to the earlier acceptance reference on the final tree.
- No push.

## Boundaries

- The closure uses the diagnostic path: the row weights and output rows are
  plan values. The hidden-input mode (index mask, dot map and closure, the
  same-ledger mask slot, bias not absorbed into the target) is the next part.
  The mask-slot and bias negatives are therefore not applicable here.
- The fixture is real-size but uses deterministic weight, activation and
  output values, not the production asset or trace values; the production
  consumer wiring is the second commit.
- The expert acceptance (one real production Projection consumer fully
  closed) is not claimed.

## Artifacts

`identity.txt`, `change.diff`, `projection-hidden-lin-tests.txt`,
`workspace.txt`, `proof-bytes.txt`, `NOTES.md`.
