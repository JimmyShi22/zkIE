# Segment bounded liveness (gpt2 A04 prove side)

## Scheme and baseline

- Scheme: `segment-bounded-liveness`.
- Code: `1bf7d00a`; previous commit `6fa06244` (the accepted segment code).
- Purpose: turn the proven semantic-zero-change reductions into the shared
  row/session discipline: (1) the tail clamp-distance and padding plain rows
  emit sparse BAT units instead of dense member-table coefficient vectors;
  (2) the fold releases its truncation-retained capacity at the last round, so
  the stage-0 group messages and the folded tables do not stay resident past
  the fold.
- Method: the durable one-request column ledger is keyed by the asset
  directory and the table root, and there is no refund
  (`crates/zkie-ops/src/proof/column_session.rs`; the asset directory holds
  one ledger directory per consumed root). Two proves on one commit would
  therefore reuse a spent table life, so the old and the new binary run on
  separate fresh commits with the same input on the same machine, and the
  segment proof bytes are checked through the commit-invariant carrier hash.

## Results (old vs new, same machine, same input)

| stage | old (6fa06244, fresh commit) | new (1bf7d00a, fresh commit) | delta |
|---|---|---|---|
| commit wall | 245,280 ms | 235,045 ms | -10,235 ms |
| prove wall | 832,553 ms | 765,577 ms | -66,976 ms (-8.0%) |
| prove VmHWM | 338,362,564 KiB | 212,605,940 KiB | -125,756,624 KiB (-119.9 GiB, -37.2%) |
| verify wall | 240,252 ms | 195,936 ms | -44,316 ms (-18.4%) |
| verify VmHWM | 251,556,396 KiB | 125,866,204 KiB | -125,690,192 KiB (-119.9 GiB, -50.0%) |
| proof bytes | 3,589,346,220 B | 3,589,359,289 B | +13,069 B |

- All six stage runs PASS.

## Hard checks

- Carrier (the segment proof messages) sha256 on both runs:
  `3815f1af7ac5913195930449a8ecbf61175aa6df2e20595df549b21367d16819` -
  byte-identical, and the same invariant already measured across the four
  accepted rounds' four different commits. The segment proof bytes are
  unchanged.
- Artifact sizes: carrier 1,707,601,596 B on both runs; trace `proof.bin`
  1,879,423,254 B on both runs (same serialized length); `statement.bin` 431 B;
  `public-inputs.bin` 72 B; consumer sidecar 2,316,128 -> 2,329,184 B
  (+13,056 B). The trace and sidecar differences come from the per-commit
  table randomness, which also made the four accepted rounds differ (their
  traces were `7b2fcf6e...` and `35e111c2...`). The trace is produced by the
  model execution before the segment path, so the segment changes cannot alter
  it for a fixed commit.
- Workspace: 56 suites, 1,313 passed, 0 failed, 32 ignored (release). Hard gate:
  45,080 B sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (byte-identical).

## Bounded liveness accounting (prove side)

| phase | large live items | holder | release |
|---|---|---|---|
| commit load | provider config, model commitment (anchors, asset metas, table randomness) | prove entry | held as the run's trusted instance |
| model execution | trace (the artifact payload) and consumer sidecar | prove entry | written into the artifact |
| segment materialization | final-norm x/hd rows, gain/bias, rstd values, norm plan/witness (five committed group messages) | segment prove | witness consumed by the hidden session; rows feed the plan |
| table build | streamed row store on disk (bounded windows: 1 MiB read window, 4096-column panels, 1.7 GB cap, 1 GiB total, 4 GiB store) | segment table build | work directory removed at the end of the prove |
| BAT | statement sample rows (one per member, sized by the member table), plain rows (relation/output/tail rows; clamp-distance and padding rows now sparse units), combined group row | proximity prove | `begin_run` moves the rows into the run; the session (statement and rows) is dropped at `into_chain`; the combined row moves into the fold |
| fold | main coefficients and messages (combined row and group table) plus the mask vectors | fold session | folded in place each round; the last bind shrinks the fold-time capacity (this change); the folded values are copied into the stage-zero rows and the session is dropped before the IOPP loop |
| IOPP | per-stage switched rows (one main row and one mask row at the target codes), stage folds, proof records | IOPP loop | per-stage provers dropped after their fold; records retained as the proof messages |
| artifact | carrier bytes and trace serialization | prove return | written into the artifact manifest |

The peak sits in the BAT plus fold window: the combined group row together
with the two member tables it pairs with, while the trace, the witness and
the statement sample rows are resident. The sparse-unit change removes the
dense clamp-distance and padding coefficient vectors; the fold-capacity
release frees the two full-size table allocations as soon as the last round
binds. The measured prove peak drops from 338,362,564 KiB to 212,605,940 KiB
on the same workload.

## Boundaries

- Single machine, same protocol, old vs new binary with separate fresh
  commits; the commit-invariant carrier hash is the byte-identity evidence for
  the segment. No whole-model claim is made here.
- The expectation of at most 338 GiB was met with margin (the new prove peak is
  212,605,940 KiB).
