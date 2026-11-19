# Norm segment whole chain acceptance (gpt2-seq16)

## Scheme and baseline

- Scheme: `norm-segment-whole-chain`.
- Code: `cf182085`; previous commit `3cc93db2` (that commit carries the byte
  width fix and the derived stage parameters; this one binds the segment table
  identity in the statement and derives the session from it).
- Purpose: real three-stage acceptance (commit / prove / verify as separate
  processes) of the final-norm (ln_f) -> lm_head segment on gpt2 at 16
  positions: derived shared/own stage parameters, byte-width relation weights
  (256^j), frozen rstd table read on both sides, and the segment table
  identity authored by the statement and absorbed into the session seed.
- 1 cold + 3 warm rounds on the same machine. The policy template block names
  the consumer table (L0_k_w); the segment no longer reads it.

## Results

| round | commit ms | prove ms | prove VmHWM KiB | verify ms | verify VmHWM KiB | proof bytes |
|---|---|---|---|---|---|---|
| 1 cold | 239,328 | 796,311 | 338,430,908 | 232,245 | 251,567,972 | 3,589,348,462 |
| 2 warm | 237,422 | 795,466 | 338,388,112 | 231,447 | 251,560,664 | 3,589,338,499 |
| 3 warm | 239,706 | 799,199 | 338,403,948 | 231,730 | 251,563,532 | 3,589,355,348 |
| 4 warm | 236,004 | 797,684 | 338,376,548 | 233,138 | 251,557,176 | 3,589,347,637 |

- Warm medians (rounds 2-4): commit 237,422 ms; prove 797,684 ms with VmHWM
  338,388,112 KiB; verify 231,730 ms with VmHWM 251,560,664 KiB; proof
  3,589,347,637 B. Cold numbers are not used in the ledger.
- Every one of the 12 stage runs passed (commit, prove, verify x 4).

## Byte account (round-1 artifact)

| file | bytes |
|---|---|
| proof.bin (trace) | 1,879,423,254 |
| norm-segment-carrier.bin | 1,707,601,596 |
| consumer_proof.bin | 2,318,368 |
| statement.bin | 431 |
| public-inputs.bin | 72 |
| manifest.json + COMPLETE | 4,741 |
| total | 3,589,348,462 |

- The carrier is the segment's cost and is counted in full; the trace payload
  is retained unchanged (no norm-step opening is dropped). No byte reduction
  is claimed for this round.

## Reuse mapping

| reused mechanism | source | landing in this segment | evidence |
|---|---|---|---|
| final-norm x rows | authenticated trace rows | `final_norm_rows` materializes the ln_f input rows; the norm plan binds them | 4 rounds verify PASS; activation root check |
| frozen rstd table | committed rsqrt table spec | `read_rstd_values` + `rstd_table_view`; the prover reads the same frozen view the verifier checks | bounds checks (B1-B3); 4 rounds PASS |
| gain / bias identity | committed parameter tables (asset metas 194/195) | `read_anchored_params` + the parameter claim tables; leaf-root checks | B1-B3 checks; 4 rounds PASS |
| hidden norm chain | accepted batch-2 norm plan/witness | `NormPlan`/`NormWitness` through the hidden session's rstd-table entry | 4 rounds PASS |
| batched switch / fold / CS / BASE | accepted batch-2/B5 layout | multi-stage IOPP with shared fold width and own stage parameters derived from the real member rows | pre-fold delta `[0,0,0]`; per-row scan 39,256 rows with 0 bad rows; 4 rounds PASS |
| byte-width relation weights | relation 8..=15 map definitions | byte digits reconstructed at 256^j in the BAT rows, matching the witness map | pre-fold delta zero after the fix; 4 rounds PASS |
| segment table identity | statement/plan authored | `SegmentTableIdentity` in the statement, cross-checked against the request plan and the public commitment, absorbed into the session seed (v2 domain) | `statement_identity` negative rejected; 4 rounds PASS |
| carrier transport | `norm-segment-carrier.bin` v1 | reader + canonical messages, manifest-bound by length and digest | carrier tamper negatives rejected; 4 rounds PASS |

## Negative matrix (real verifier, round-1 artifact)

| case | accepted? |
|---|---|
| control | PASS (control must pass) |
| carrier version bumped | rejected |
| carrier reader root changed | rejected |
| carrier record digest changed | rejected |
| carrier messages truncated | rejected |
| statement segment identity root changed | rejected |
| materialized source row (final norm output) changed | rejected |

All six tampered artifacts were rebuilt with a self-consistent manifest and
complete marker, so the artifact-integrity layer accepted each one and the
segment checks rejected it. Raw outcomes in `negatives/negatives.json`.

## Gates

- Workspace: 56 suites, 1,313 passed, 0 failed, 32 ignored (release).
- Hard gate: 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (byte-identical).

## Boundaries and open items

- The segment participates only at 16 positions; seq512 is outside this round.
- The verifier checks every identity field it can derive (code, length, salts,
  v, answers, distances, matrix rows, k-lo/k-col, asset id and asset table
  identity, configuration, shape) and pins the carrier to the statement
  identity; the table root and record digest are statement-authored and bound
  by the session derivation. The verifier does not recompute the external
  asset table.
- No claim is made beyond this segment; the byte account is additive.
- Deletion note: the previous checkpoint round under
  `20261119-0933-norm-segment-whole-chain-3cc93db2-d1812052/` kept its text
  traces; its gitignored proof binaries were deleted when this scheme started
  benching (recorded in that round's NOTES).

## Acceptance record

- 2026-11-19: the acceptance review approved this batch as delivered; the
  stage is closed.
- Follow-up evidence items (they do not block this close): coverage at 512
  positions, and the retirement of the old step messages (the byte reduction
  sub-step).
