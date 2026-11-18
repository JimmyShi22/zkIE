# Gemma 3: the first consumer-authenticated weight

- Scheme: the layer-0 query projection of the Gemma 3 trace becomes the first
  registered consumer-authenticated asset. The request consumer chain proves
  the projection over the committed encoded table; the trace carries no
  opening for that weight.
- Code: `103c4560`; effective baseline `31825ece`.
- Selected asset: `models/gemma3/weights/L0_q_w_i32.bin`, a square 1024 by
  1024 registered weight consumed by the four per-head query projection steps.
  The layer-0 key and value projections are rectangular (1024 rows by 256
  columns); the existing key/range chain derives a square matrix and refuses
  that layout, so the first compatible registered weight is the query
  projection.
- Table geometry: log_n 12, ell 2048, r 128, v 10, matrix_rows_log 10,
  k_lo 0, later_batches 1; row width 1024, 1024 rows, 1,048,576 entries. The
  declared entry range is the existing public [-262144, 262143]; the actual
  query weights lie within +/-56,320. The full signed 32-bit range would make
  the consumer integer no-wrap gate fail, so the declared range is the fixed
  public range that covers the table.

## Mechanism mapping

| Requirement | Implementation | Test / negative | Evidence |
|---|---|---|---|
| One-time key qualification at commit | the commit key proof over the committed table, published in the commit public manifest | key bundle missing / changed refused | consumer-evidence.json key fields |
| Per-request key re-verification | the public key entry before the consumer chain; the accepted key is compared with this commit | key bundle missing / changed refused | key_request_verify_ms |
| Key and table manifest binding | commit and proof manifests bind the key and sidecar by length and digest | missing sidecar, container byte changed refused | consumer-evidence.log |
| Eval closes the PSC pending | request consumer first stage | member, member target, plan target changed refused | negatives |
| Raw weighted rows are bound | committed raw messages under the request coefficients | raw root changed refused | negatives |
| Target fixed before the first challenge | fixed PSC prefix | plan target changed refused | negatives |
| Residual consumer opening refused | public plan promotes the asset; trace opening check | residual opening and tampered row refused | negatives |
| Negative matrix | twelve executions over the eleven accepted classes plus one plan-member-target variant | all refused by the production verify entry | consumer-evidence.json `negatives: 12` |
| Byte accounting | opening removal, consumer sidecar and key bundle accounted separately | artifact delta -6,567,956 B; opening removal -8,413,688 B | table below |
| Real three-stage run | production commit/prove/verify | gemma3-seq16 PASS, GT digest `8683f5e22e51d1405fe3efcc7487a9a4df2e694c5d9f0bed64e921810d77c562` | acceptance-summary.json |

## Byte accounting

| item | value |
|---|---|
| proof artifact (three-stage) | 5,718,769,568 B |
| previous gemma3-seq16 artifact | 5,725,337,524 B |
| artifact delta | -6,567,956 B |
| proof.bin (three-stage) | 5,716,919,862 B |
| previous proof.bin | 5,725,333,550 B |
| proof.bin delta | -8,413,688 B |
| removed consumer opening | 8,413,688 B |
| trace with that opening | 5,725,333,550 B |
| consumer sidecar (evidence run) | 1,857,544 B |
| consumer sidecar (three-stage proof) | 1,845,216 B |
| statement / public inputs / manifest / COMPLETE | 218 / 72 / 4,135 / 65 B |
| key bundle | 4,551,344 B |
| commit | 246,215 ms |
| prove | 109,121 ms |
| verify | 95,499 ms |
| VmHWM commit/prove/verify | 27,314,244 / 12,992,936 / 11,226,792 KiB |
| consumer evidence commit/prove/verify | 261,015 / 106,075 / 92,624 ms |
| key prove / commit verify / request verify | 191,715 / 59,096 / 59,550 ms |
| negatives refused | 12 executions covering 11 classes |

The opening deletion is a deletion increment, not the end-to-end net gain of
the asset-substitution work. The consumer sidecar and the key bundle are
accounted separately; the artifact delta is reported beside, not instead of,
the removal increment. The stage times include key and consumer costs and are
not a performance claim.

## Negatives

The production verify entry refuses these twelve executions, covering the
eleven classes of the accepted chain plus one plan-member-target variant:

1. missing request proof file;
2. changed plan member (psi);
3. changed plan target;
4. changed mask root;
5. changed raw root;
6. changed container byte;
7. residual consumer opening (valid);
8. residual consumer opening with a changed row;
9. changed deferred output;
10. missing key bundle;
11. changed key bundle;
12. changed plan member target (the added variant of the plan-member class).

## Policy and harness notes

- The Gemma 3 policy now requires the committed limited table identity; the
  first registered asset id 0 is admitted as a real id. Old Gemma 3 policies
  without the identity are not accepted by this version. Whether the policy
  or model version should be bumped is reserved for review.
- The consumer evidence harness copies the whole proof directory for every
  negative. With the larger Gemma 3 proof (about 5.7 GB) that is expensive
  disk I/O; it does not affect the result, but the harness should reuse the
  unchanged proof file in a later round.

## Boundaries

- One registered weight of one model; not the full asset set.
- The layer-0 key and value projections remain on the raw row commitment.
- No claim of full-model hiding, all-asset substitution, or overall
  performance acceptance.
- The run uses a single seq16 round; seq512 is later evidence.
