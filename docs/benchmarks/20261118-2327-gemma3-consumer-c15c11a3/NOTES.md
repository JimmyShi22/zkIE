# Gemma 3: the first consumer-authenticated weight

- Scheme: the layer-0 query projection of the Gemma 3 trace becomes the first
  registered consumer-authenticated asset. The request consumer chain proves
  the projection over the committed encoded table; the trace carries no
  opening for that weight.
- Code: `c15c11a3`; effective baseline `31825ece`; Gemma 3 model version 2.
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
| Policy version discipline | Gemma 3 model version 2; the policy must carry the committed table identity | a policy without the identity is refused | version negative |
| Negative matrix | twelve executions over the eleven accepted classes plus one plan-member-target variant | all refused by the production verify entry | consumer-evidence.json |
| Byte accounting | opening removal, consumer sidecar and key bundle accounted separately | artifact delta -6,557,951 B; opening removal +8,413,688 B | table below |
| Real three-stage run | production commit/prove/verify | gemma3-seq16 PASS, GT digest `8683f5e22e51d1405fe3efcc7487a9a4df2e694c5d9f0bed64e921810d77c562` | acceptance-summary.json |

## Byte accounting

The sidecar account uses the three-stage proof artifact only. The
`consumer_proof_bytes` field of the consumer-evidence run is telemetry from a
different randomized instance; it is not used in this byte account.

| item | value |
|---|---|
| proof artifact (three-stage) | 5,718,779,573 B |
| previous gemma3-seq16 artifact | 5,725,337,524 B |
| artifact delta | -6,557,951 B |
| proof.bin (three-stage) | 5,716,919,862 B |
| previous proof.bin | 5,725,333,550 B |
| proof.bin delta | -8,413,688 B |
| removed consumer opening | 8,413,688 B |
| trace with that opening | 5,725,333,550 B |
| consumer sidecar (three-stage) | 1,855,208 B |
| key bundle | 4,543,544 B |
| commit | 248,988 ms |
| prove | 109,556 ms |
| verify | 95,750 ms |
| VmHWM commit/prove/verify | 27,313,884 / 12,993,296 / 11,227,624 KiB |
| consumer evidence commit/prove/verify | 258,660 / 105,974 / 92,516 ms |
| key prove / commit verify / request verify | 189,456 / 58,745 / 59,379 ms |
| negatives refused | 13 executions (12 consumer-matrix + 1 version) |

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

A thirteenth execution writes a policy without the committed table identity
and requires the production entry to refuse it. This is the version-discipline
negative for the model version 2 policy.

## Policy and harness notes

- Gemma 3 model version 2 makes the committed limited table identity
  mandatory. The first registered asset id 0 is admitted as a real id. Older
  Gemma 3 policies without the identity are refused by this version.
- The version-1 evidence directory was superseded by this version-2 run and is
  removed from the tree; its run remains historical.
- The consumer evidence harness copies the whole proof directory for every
  matrix negative. With the larger Gemma 3 proof (about 5.7 GB) that is
  expensive disk I/O; it does not affect the result, but the harness should
  reuse the unchanged proof file in a later round.

## Boundaries

- One registered weight of one model; not the full asset set.
- The layer-0 key and value projections remain on the raw row commitment.
- No claim of full-model hiding, all-asset substitution, or overall
  performance acceptance.
- The run uses a single seq16 round; seq512 is later evidence.
