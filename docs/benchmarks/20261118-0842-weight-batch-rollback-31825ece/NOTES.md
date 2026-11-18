# Post-rollback baseline verification on the asset-consume closure

- Scheme: `weight-batch-rollback` — the weight-batch common-table theme was
  terminated and the branch was returned to the asset-consume closure; this
  round verifies that the production baseline is back to that closure.
- Code: `31825ece` (asset-consume round closure); branch `refactor/v2`. The
  tracked tree is clean; the weight-batch heads are contained by no branch or
  remote ref (the heads sampled from the reflog are dangling). The code tree
  is identical to the previously accepted protocol version: `git diff
  460dd79a 31825ece` touches `docs/` only (279 evidence documents); no source,
  model, or configuration file changes.
- Purpose: confirm the three seq16 cases pass end to end on the rolled-back
  tree, and that the gpt2-seq16 prove wall time and proof artifact bytes are
  back at the accepted baseline: prove warm median 92,652 ms and proof
  artifact 1,881,722,126 B; 3% is the regression tolerance, applied to both.
- Reference round: `docs/benchmarks/20261117-0251-asset-consume-chain-close-accept-9bf0f30f-35c76e4e`
  (same machine, same driver, same case basis).

## Method

- Driver: `zkie/tests/acceptance_rounds.rs` (ignored test). One invocation runs
  four rounds; each round provisions a fresh commit (one table instance serves
  one request), derives the verifier policy from that round's commit public
  bundle, and runs `zkie commit`, `zkie prove` and `zkie verify` as separate
  CLI processes with `ZKIE_CASE_ID` propagated. Stage wall time and VmHWM are
  per-stage values. Round 1 is cold; rounds 2-4 are warm. Warm medians are the
  ledger values; the cold round is recorded and excluded.
- Per-case command (model and case identifiers substituted):
  `ZKIE_ACCEPT_CASE_FILE=/data/zkIE/models/<model>/case.toml
  ZKIE_ACCEPT_CASE_ID=<case> ZKIE_ACCEPT_MODEL=<model>
  ZKIE_ACCEPT_POLICY_TEMPLATE=/data/zkIE/models/<model>/case.policy
  ZKIE_ACCEPT_ROUNDS=4 ZKIE_ACCEPT_OUT=<this dir>/cases/<case>
  ZKIE_ACCEPT_BIN=/data/zkIE/target/release/zkie
  cargo test --release -p zkie --test acceptance_rounds -- --ignored --nocapture`
- Binary: `target/release/zkie` built from `31825ece` with
  `cargo build --release -p zkie` (EXIT 0); sha256
  `d2909dca75e306a6879eddcc3e7325c5e43622e646e5af3d82400165e2959fe1`.
- Machine: Intel Xeon 6982P-C, 64 CPUs, 495 GiB RAM, Linux 6.6.102; same
  machine as the reference round. The three cases ran sequentially from
  08:42 to 09:18.
- A manual probe commit of gpt2-seq16 (243,224 ms) preceded the batch while an
  invocation issue was diagnosed; it warmed the gpt2 commit path before
  round 1.
- The first invocation at 08:36 failed before any measurement: it passed
  relative `--case`/`--assets-manifest` paths, and under `cargo test` the
  stage processes inherit the package root as their working directory, so
  every commit stage returned `asset_mismatch` at 0 ms. Its textual artifacts
  are preserved under `attempt1-failed-invocation/`; no numbers from it are
  used. The batch was re-launched at 08:42 with absolute paths.
- The original incremental build log was lost while consolidating the failed
  first-attempt directory; `build.log` records a re-run of the same command on
  the unchanged tree (cargo reports the target up to date; the binary was not
  rebuilt and its sha256 is unchanged).

## Results

All three cases pass: 12 end-to-end rounds (3 x 4), 36 stage executions, exit
code 0. One cold plus three warm rounds per case. The table gives cold and
warm median (ms) with warm-stage VmHWM medians (KiB):

| case | proof artifact cold / warm median (B) | commit cold | commit warm med | prove cold | prove warm med | prove HWM warm med | verify cold | verify warm med | verify HWM warm med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-seq16 | 1,881,723,822 / 1,881,759,600 | 246,143 | 243,577 | 93,412 | 93,136 | 5,824,620 | 78,790 | 78,556 | 7,439,308 |
| gemma3-seq16 | 5,725,337,524 / 5,725,337,524 | 9,309 | 9,296 | 34,031 | 33,840 | 12,166,540 | 29,742 | 29,682 | 10,383,084 |
| timesfm-16patch | 3,647,375,061 / 3,647,375,061 | 5,506 | 5,397 | 23,275 | 23,395 | 7,859,444 | 23,314 | 23,254 | 4,037,228 |

Per-round detail (ms and KiB; proof artifact bytes):

| case | round | commit | commit HWM | prove | prove HWM | proof artifact B | verify | verify HWM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-seq16 | 1 | 246,143 | 27,295,172 | 93,412 | 5,824,820 | 1,881,723,822 | 78,790 | 7,439,004 |
| gpt2-seq16 | 2 | 243,577 | 27,295,320 | 93,136 | 5,826,744 | 1,881,741,581 | 78,253 | 7,439,360 |
| gpt2-seq16 | 3 | 246,616 | 27,295,580 | 93,356 | 5,822,952 | 1,881,762,067 | 78,823 | 7,439,308 |
| gpt2-seq16 | 4 | 243,225 | 27,295,368 | 92,856 | 5,824,620 | 1,881,759,600 | 78,556 | 7,439,288 |
| gemma3-seq16 | 1 | 9,309 | 1,086,108 | 34,031 | 12,166,596 | 5,725,337,524 | 29,742 | 10,382,932 |
| gemma3-seq16 | 2 | 9,296 | 1,086,100 | 33,840 | 12,166,876 | 5,725,337,524 | 29,507 | 10,383,084 |
| gemma3-seq16 | 3 | 9,272 | 1,086,108 | 33,786 | 12,166,456 | 5,725,337,524 | 30,071 | 10,381,216 |
| gemma3-seq16 | 4 | 9,503 | 1,086,096 | 34,386 | 12,166,540 | 5,725,337,524 | 29,682 | 10,385,628 |
| timesfm-16patch | 1 | 5,506 | 175,928 | 23,275 | 7,860,524 | 3,647,375,061 | 23,314 | 4,039,996 |
| timesfm-16patch | 2 | 5,385 | 176,348 | 23,342 | 7,859,444 | 3,647,375,061 | 23,254 | 4,040,912 |
| timesfm-16patch | 3 | 5,406 | 175,600 | 23,395 | 7,859,432 | 3,647,375,061 | 23,340 | 4,037,228 |
| timesfm-16patch | 4 | 5,397 | 177,064 | 23,783 | 7,860,872 | 3,647,375,061 | 23,092 | 4,026,852 |

The proof artifact figure is the whole proof output directory (trace, request
consumer sidecar, statement, public inputs and manifest), not `proof.bin`.
`proof.bin` is deleted after its measured bytes are recorded
(`summary-table.md` keeps that column); every textual artifact stays.

## Baseline comparison (gpt2-seq16, +/-3%)

- prove: baseline 92,652 ms; this run warm median 93,136 ms (+0.52%); band
  [89,872, 95,432] ms -> WITHIN.
- proof artifact: baseline 1,881,722,126 B; this run cold 1,881,723,822 B
  (+0.0001%) and warm median 1,881,759,600 B (+0.0020%); band
  [1,825,270,462, 1,938,173,789] B -> WITHIN for both.
- The other two seq16 cases are consistent with the reference round: gemma3
  warm medians commit 9,296 / prove 33,840 / verify 29,682 ms and
  5,725,337,524 B; timesfm 5,397 / 23,395 / 23,254 ms and 3,647,375,061 B
  (reference: 9,309 / 33,511 / 29,504 and 5,376 / 23,082 / 23,176). These are
  reference comparisons, not a performance sign-off.

## Tests

`cargo test --workspace`: 55 suites, 1,292 passed, 0 failed, 31 ignored,
EXIT 0 (`workspace-tests.log`) — identical to the closure baseline. The
45,080 B hard gate is not re-measured in this round: the code tree is
identical to the closed round that recorded it, and this round's scope is the
three-case baseline check.

## File sizes

`file-sizes.txt` lists the on-disk sizes. The directory is 37 MiB on disk and
203 text files carry 930,199 bytes; the `.bin` artifacts stay on
disk and are gitignored. The 42 GiB of gitignored binaries left by the terminated theme
(`20261118-0220-weight-batch-r2-recheck-...`) remain on disk; no deletion was
performed in this round.

## Open items

- No push and no flag in this round.
- This is a rollback regression check against the closure baseline, not a
  performance sign-off. The seq512 cases are outside this round's scope and
  were not run.

## Abandoned-theme binary disposal

- The gitignored binaries of the terminated weight-batch theme were disposed
  on 2026-11-18 after the rollback verification closed: the scratch run
  directories under /data (wb-b2..b21, wb-cost-a/b/c, wb-e2e, wb-smoke,
  wb-k2..k4, zkie-consumer-ev) and the gitignored binaries inside the
  weight-batch benchmark directories (20261117-1710, 20261117-1716,
  20261118-0148, 20261118-0220, 20261118-0710). Their committed textual
  evidence remains reachable in the dangling weight-batch git objects.
- Disk before the disposal: 215 GiB free (89 percent used). After: 290 GiB
  free (86 percent used). The rollback evidence of this directory was not
  touched; nothing disposed is referenced by this verification.
