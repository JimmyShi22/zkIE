# TimesFM: the first consumer-authenticated weight

- Case: `timesfm-16patch` (seq16; seq512 not run in this round)
- Code commit: `ab687693`; run date: 2026-11-19 00:25
- This is one release run; it is not a warm-median or a performance claim.
- Three-stage GT: `cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3` (32,768 elements, output binding `expected`).
- Result: accepted; 12 consumer-matrix negatives and 1 version negative refused (see matrix below).

## Four-column mapping

| Requirement | Code landing | Evidence | Status |
|---|---|---|---|
| Authenticated-table key chain: one-time key qualification, per-request verify_key, manifest binding | TimesFM first-projection table plan/identity and range plan; commit-side table build and key bundle; provider commit/prove/verify key checks; shared asset-key range parameter | Key bundle 3,206,184 B; key prove 115,879 ms; commit-side verify 35,527 ms; this run re-verifies the key per request (35,615 ms) | PASS |
| Consumer chain: Eval closes the PSC pending; raw weighted rows; target absorbed before the first challenge | TimesFM consumer witness over the registered first projection; shared consumer session/proof; provider request consumer verification | Accepted positive run; changed-member/member-target/target/mask/raw/container negatives refused | PASS |
| Residual consumer-asset opening rejection | Production plan marks the first projection consumer-authenticated; the trace omits its opening; trace verification rejects a residual opening | Trace carries no opening; residual-opening and tampered-residual-row negatives refused | PASS |
| Policy identity discipline | TimesFM model version 2 makes the committed limited table identity mandatory; the policy must carry the identity derived from the commit public bundle | A policy without the identity is refused (version negative) | PASS |
| Closed negative matrix (same classes as the accepted first-asset round) | Model-specific evidence test | 13 negative cases in the matrix below (12 consumer matrix + 1 version gate) | PASS |
| Real three-stage seq16 run | Production provider commit/prove/verify through the CLI | This run: commit/prove/verify accepted; GT digest `cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3` | PASS |
| Byte account: removed opening and same-basis before/after | Executor trace without the consumer opening; rebuilt opening from the same commit | Byte table below | PASS |
| NOTES four-column archive | This file | Four-column table plus byte/timing/memory tables | PASS |

## Byte account

The consumer-evidence run and the three-stage acceptance run are separate randomized instances. The opening-removal account uses the consumer-evidence trace; the proof artifact account uses the three-stage run. The two are reported separately and are not subtracted from each other.

| Item | Bytes | Note |
|---|---:|---|
| Consumer-run trace (proof.bin) | 3,613,502,964 | 237 asset openings; the consumer opening is absent |
| Removed consumer opening (same basis) | 33,604,088 | Increment only; not a net end-to-end gain |
| Trace with that opening (same basis) | 3,647,107,052 | Rebuilt from the same commit |
| Consumer proof sidecar (consumer run) | 2,028,208 | Telemetry only; not used in the artifact account |
| Three-stage proof artifact total | 3,615,796,947 | Whole proof artifact as measured; proof.bin is deleted afterwards |
| Three-stage proof.bin (measured, deleted) | 3,613,502,964 | Deleted after measurement |
| Proof directory on disk after deletion | 2,293,983 | Excludes proof.bin |
| Commit bundle total | 6,035,298 | Acceptance commit directory |
| Key bundle (subset of commit bundle) | 3,206,184 | Do not add to the commit bundle total |

## Timings and memory (single run)

| Stage | Wall time (ms) | Process VmHWM after stage (KiB) |
|---|---:|---:|
| commit | 153,400 | 13,821,444 |
| prove | 81,701 | 9,045,896 |
| verify | 64,746 | 6,442,368 |

| Consumer-evidence stage | Wall time (ms) | Process VmHWM after stage (KiB) |
|---|---:|---:|
| commit | 160,857 | 18,090,264 |
| prove | 79,990 | 18,090,264 |
| verify | 62,578 | 18,090,264 |

Key stage: prove 115,879 ms, commit-side verify 35,527 ms, request-side verify 35,615 ms.

## Negative matrix

| # | Negative | Expected | Result |
|---:|---|---|---|
| 1 | Missing consumer proof file | refused | refused |
| 2 | Changed plan member (psi) | refused | refused |
| 3 | Changed member target (not the PSC pending) | refused | refused |
| 4 | Changed plan target | refused | refused |
| 5 | Changed mask root | refused | refused |
| 6 | Changed raw root | refused | refused |
| 7 | Changed container byte | refused | refused |
| 8 | Residual consumer-asset opening | refused | refused |
| 9 | Tampered residual-opening row | refused | refused |
| 10 | Changed deferred output | refused | refused |
| 11 | Missing key bundle | refused | refused |
| 12 | Changed key bundle | refused | refused |
| 13 | Old policy without the committed table identity | refused | refused |

The 12 consumer-matrix cases cover the closed negative classes; the residual-opening class has two cases (valid residue and tampered residue). The 13th case is the version gate: a policy written without the committed table identity is refused by the production entry.

## Weight range fitting

- `pro_out_w` real extrema: [-5,320, 5,304].
- Declared signed column range: [-131,072, 131,071]; the real extrema fit inside it, so the range plan is fitted to the committed weight rather than assumed.

## Boundary

- One timesfm-16patch seq16 release run; seq512 is not covered and no stability claim is made.
- The first projection is the first non-alias square projection compatible with the registered square key witness; the input projection is 64x2048 and is not used as the consumer asset.
- The proof.bin of the three-stage run is deleted after measurement; the proof directory on disk therefore excludes it, while the artifact total row reports the measured value.
- This is the first registered asset only, not full-asset replacement.
- The numbers are not comparable across models; no performance target is signed.
