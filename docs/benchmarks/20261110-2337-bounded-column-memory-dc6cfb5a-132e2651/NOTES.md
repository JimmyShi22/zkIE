# Bounded column claim memory

## Scheme

- Scheme: `bounded-column-memory`.
- Current code: `dc6cfb5a` (`9204aa1c` engine claim construction,
  `381ec4a6` holder source release, `daea75b8` adapter targets,
  `dc6cfb5a` bounded driver).
- Previous code: `132e2651`, the state before this round.
- Scope: the real GPT-2 LM-head local chain only. The production three-stage
  command still does not carry this chain; that is an open integration item.

## Purpose

The effective prove memory budget is 1.7 GB. The previous chain peaked at
5,603,208 KiB (5.35 GiB), which does not pass. This round keeps the same claim
declarations, claim order, challenge order, roots, transcript and column
budget while making the witness computation bounded.

## Measurement basis

- Same machine for every row, release build:
  `cargo test -p zkie --release real_gpt2_lm_head_chain -- --ignored --nocapture`.
- Stage points read `/proc/self/status` (`VmHWM`, `VmRSS`, `RssAnon`,
  `RssFile`, `RssShmem`) at fold, record, queries, columns, Omega, BAT
  construction/freeze/combine/column carry, FOLD, every switch and BASE.
- The baseline row was rebuilt at `132e2651` in an isolated worktree with only
  the test-only memory trace added; `baseline-a.txt` is that run. The statement
  and public commit fixture files are external read-only inputs; the 4 GiB
  proof file is not read by this chain.
- Commit and prove run in one process for every row in this trace. The commit
  and prove columns are not independent-process stage measurements; that
  separation is still open.
- The temp store is written and read only during commit: 1,073,741,824 bytes
  written, 2,147,483,648 bytes read by the completion pass, then removed.
  It is not resident during prove.
- Values are KiB unless noted.

## Changes

1. Freeze the BAT claim set by moving the registered rows out of the group
   instead of cloning them. The frozen order (plain rows, then sample rows,
   then generator rows) is unchanged.
2. Build column relation rows on demand. A column join checks the shape once,
   validates each provided column and stores only its target; the verified
   read columns remain the source. One dense relation row is built and dropped
   at a time. The BAT side adds that row through a compact row-major carry
   that maps entries into the group row without materializing the virtual
   table.
3. Register one target-only claim slot per column draw before the challenge,
   draw x, then build each column row, scale it by its own x power, add it to
   the combined row and drop it. Slot count, slot order, targets and powers
   are the previous ones.
4. Release last-use buffers: the streaming holder drops its row source after
   the last column opening, and the real chain test drops the committed key
   material after the temp store is removed. The holder keeps the reader,
   name, tree and salts.

## Results

| run | commit wall | commit HWM | columns anon | fold-stage anon | prove wall | prove HWM | base bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: before | 29.416 s | 525,968 | 1,608,392 | 5,338,052 | 10.450 s | 5,603,208 | 62,968 |
| B-1: freeze by move | 29.535 s | 526,552 | 1,608,412 | 3,528,580 | 9.836 s | 3,794,376 | 63,144 |
| B-2: written-out units (rejected) | 29.916 s | 526,560 | 1,608,352 | 4,358,816 | 10.523 s | 4,624,428 | 63,912 |
| B-3: on-demand rows | 29.678 s | 526,516 | 394,612 | 461,840 | 11.828 s | 727,648 | 61,160 |
| B-3b: target-only join | 29.639 s | 526,508 | 392,020 | 461,832 | 8.442 s | 727,636 | 62,440 |
| B-4: last-use release | 29.584 s | 526,580 | 338,000 | 338,000 | 8.470 s | 603,680 | 62,696 |

The B-4 `columns` and `fold-stage` anon values are measured after the holder
source release; the source no longer contributes file-backed pages from that
point. B-4 `prove HWM` is 589.5 MiB, 2.9x below the 1.7 GB budget and 9.3x
below the baseline. Prove wall time is 8.470 s, 19% below the 10.450 s
baseline.

The base artifact size varies with the sampled positions and star material; it
is not a committed size constant in this local chain. All runs sealed, and the
claim order, target slots, powers, transcript steps, FOLD relation and BASE
seal path are unchanged.

## Rejected measurement

B-2 wrote one BAT unit per nonzero coefficient. Each column relation has
65,539 generator coefficients, so the per-entry metadata was larger than the
dense coefficient vector: prove HWM rose to 4,624,428 KiB and wall time to
10.523 s. The form was reverted; `b2-written-units-rejected.txt` is kept as the
negative result.

## Correctness evidence

- `workspace-tests.log`: `cargo test --workspace --release`, all suites green.
- Real chain: sealed nonzero digest, non-empty BASE artifact, commit and prove
  both complete on the real asset.
- Engine tests cover the frozen BAT order, compact carry shape checks and
  on-demand column evaluation; the adapter tests compare column targets.

## Open items

- Separate-process commit, prove and verify measurements for this local chain
  need an approved stage handoff for the committed rows and private material.
  The production three-stage command does not carry this chain yet.
- A single whole-chain artifact, the nine-class real-chain negative matrix and
  an independent verifier run remain open.
- This trace has one run per row. The rebuilt baseline reproduces the earlier
  attribution stage value within run-to-run variation, and the final difference
  is far outside that variation; `baseline-a.txt` is the archived measurement.
  Warm medians still belong in the final ledger when the separate-process
  measurement exists.
