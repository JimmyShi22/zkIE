# TimesFM first-asset consumer evidence

- Case: `timesfm-16patch` (seq16; seq512 not run in this round)
- Code commit: `63db86eb`; run date: 2026-11-18 22:12
- This is one release run; it is not a warm-median or a performance claim.
- Result: accepted; 12 negatives refused (see matrix below).

## Four-column mapping

| Requirement | Code landing | Evidence | Status |
|---|---|---|---|
| Authenticated-table key chain: one-time key qualification, per-request verify_key, manifest binding | TimesFM first-projection table plan/identity and range plan; commit-side table build and key bundle; provider commit/prove/verify key checks; shared asset-key range parameter | Key bundle 3,214,840 B; key prove 113,741 ms; commit-side verify 35,088 ms; this run re-verifies the key per request (35,086 ms) | PASS |
| Consumer chain: Eval closes the PSC pending; raw weighted rows; target absorbed before the first challenge | TimesFM consumer witness over the registered first projection; shared consumer session/proof; provider request consumer verification | Accepted positive run; changed-member/member-target/target/mask/raw/container negatives refused | PASS |
| Residual consumer-asset opening rejection | Production plan marks the first projection consumer-authenticated; the trace omits its opening; trace verification rejects a residual opening | Trace carries no opening; residual-opening and tampered-residual-row negatives refused | PASS |
| Closed negative matrix (same classes as the accepted first-asset round) | Model-specific evidence test | 12 negative cases in the matrix below | PASS |
| Real three-stage seq16 run | Production provider commit/prove/verify | This run: commit/prove/verify accepted | PASS |
| Byte account: removed opening and same-basis before/after | Executor trace without the consumer opening; rebuilt opening from the same commit | Byte table below | PASS |
| NOTES four-column archive | This file | Four-column table plus byte/timing/memory tables | PASS |

## Byte account

| Item | Bytes | Note |
|---|---:|---|
| Trace (proof.bin) | 3,613,502,964 | 237 asset openings; the consumer opening is absent |
| Removed consumer opening (same basis) | 33,604,088 | Increment only; not a net end-to-end gain |
| Trace with that opening (same basis) | 3,647,107,052 | Rebuilt from the same commit |
| Consumer proof sidecar | 2,025,600 | Request proof file beside the trace |
| Statement | 133,214 | Request statement |
| Proof artifact total | 3,615,797,071 | Includes proof.bin, sidecar, statement, public inputs, manifest |
| Commit bundle total | 6,019,696 | Includes the key bundle and table randomness |
| Key bundle (subset of commit bundle) | 3,214,840 | Do not add to the commit bundle total |

## Timings and memory (single run)

| Stage | Wall time (ms) | Process VmHWM after stage (KiB) |
|---|---:|---:|
| commit | 158,154 | 18,220,096 |
| prove | 79,919 | 18,220,096 |
| verify | 62,130 | 18,220,096 |

Key stage: prove 113,741 ms, commit-side verify 35,088 ms, request-side verify 35,086 ms.

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

The 12 cases cover the closed negative classes; the residual-opening class has two cases (valid residue and tampered residue).

## Boundary

- One timesfm-16patch seq16 release run; seq512 is not covered and no stability claim is made.
- The first projection is the first non-alias square projection compatible with the registered square key witness; the input projection is 64x2048 and is not used as the consumer asset.
- This is the first registered asset only, not full-asset replacement.
- The numbers are not comparable across models; no performance target is signed.
- The proof artifact stays in the evidence directory; binary files are ignored by git.
