# Final long-sequence (seq512) single-round validation

- Scheme: final long-sequence validation on the current production baseline,
  one single round per case. The directory name follows the task-mandated
  identifier; this is not a weight-batch result and does not revive the
  terminated theme.
- Code: HEAD `d7a7a863` (evidence-only commits after the effective baseline;
  `git diff 31825ece..HEAD` touches `docs/` only); effective code baseline
  `31825ece`; branch `refactor/v2`, tracked tree clean at run start.
- Binary: `/data/zkIE/target/release/zkie`, sha256
  `d2909dca75e306a6879eddcc3e7325c5e43622e646e5af3d82400165e2959fe1`
  (5,570,512 B; the release build from `31825ece`; unchanged since the
  20261118-0842 build; build provenance recorded there and re-checked here).
- Basis: the W05-0251 case blocks, inputs, integer ground truth and protocol
  parameters for `gpt2-seq512` and `gemma3-seq512`; the verifier policy is
  derived per round from the new commit public bundle (not copied from the
  0251 instance). seq512 is a sequence length, not 512 decode steps; no
  decode-step coverage is claimed here.
- Method: the acceptance driver (`zkie/tests/acceptance_rounds.rs`,
  `ZKIE_ACCEPT_ROUNDS=1`) drives `commit`, `prove` and `verify` as separate
  CLI processes per case; each stage is wrapped by
  `/tmp/zkie-stage-timeout.sh` (`timeout --signal=KILL 1800`), so a stage
  exceeding 30 minutes is a failure. No timeout occurred. Stage results,
  progress and metrics are recorded per round; `proof.bin` is deleted after
  its measured bytes are recorded. Single round: no cold/warm series, no
  warm median, no stability claim.
- Environment: Intel Xeon 6982P-C, 64 CPUs, 495 GiB RAM (331 GiB available
  at start), Linux 6.6.102; the CLI runs without a thread parameter. The
  live monitor observed up to 66 OS threads in the stage process; this is an
  observed OS thread count, not a measured compute-thread setting, and the
  machine core count is not written as a thread setting. Disk: 290 GiB free
  before and after, minimum 280 GiB during the run (proof.bin peak); no
  existing evidence was removed or squeezed. The two cases ran sequentially
  from 13:05 to 13:16.

## Identity and parameters

| case | case block / input / GT | policy identity (this round) | expected output |
|---|---|---|---|
| gpt2-seq512 | `models/gpt2/case.toml` block `gpt2-seq512`; input `cases/gpt2/tokens_512_i32.bin` sha256 `51dcd55b…f6a8`; GT `cases/gpt2/gt_int_argmax_512_i32.bin` sha256 `eac0949d…0fdfdd` | anchor `898ad20b…ac239`; model `6d35b057…438d271`; limited-table identity from this round's commit; key bundle 4,547,680 B | `eac0949d6463927eb9a961d61d618fb703e4b02169e89db10dbada94a5f0dfdd`, 512 elements |
| gemma3-seq512 | `models/gemma3/case.toml` block `gemma3-seq512`; input `cases/gemma3/tokens_512_i32.bin` sha256 `54d7bd85…2054`; GT `cases/gemma3/gt_int_argmax_512_i32.bin` sha256 `6d1df6b5…9344c` | anchor `fef99a16…23216`; model `d5114631…e332ec` (deterministic commit; same identity as the 0251 single round) | `6d1df6b50c20d02db715fd14b12b9134623739375c44a029dd15b6b30bd9344c`, 512 elements |

The new gpt2 anchor differs from the 0251 one (`88012868…`), as expected for
a fresh commit with fresh table salts; the policy's limited-table root and
record are this round's values. The model identity is unchanged.

## Results (single round each; wall ms, peak memory KiB, bytes B)

| case | commit wall / HWM | prove wall / HWM | verify wall / HWM | proof artifact | proof.bin (deleted after measurement) | output |
|---|---:|---:|---:|---:|---:|---|
| gpt2-seq512 | 243,792 / 27,295,588 | 165,428 / 27,346,148 | 103,699 / 14,477,400 | 6,794,079,509 | 6,790,541,622 | PASS, `eac0949d…`, 512 |
| gemma3-seq512 | 9,292 / 1,086,116 | 87,500 / 40,278,696 | 68,368 / 22,044,924 | 10,316,824,499 | 10,316,814,558 | PASS, `6d1df6b5…`, 512 |

All six stage executions returned `PASS` with `error_code=null`; the verify
`authenticated_output_digest`/`authenticated_output_elements` equal the fixed
GT values, and `verify-run/authenticated-output-i32le.bin` (2,048 B) has the
same sha256 as the GT file in both cases. Progress ended at
`completed_units=total` (`assets` 199 / 245; `relation_steps` 2,798 / 1,506).

## Comparison with the W05-0251 single-round baseline (same case and stage)

| case | commit | prove | verify | artifact | proof.bin |
|---|---:|---:|---:|---:|---:|
| gpt2-seq512 baseline | 243,839 / 27,295,484 | 165,761 / 27,660,200 | 102,688 / 14,476,448 | 6,794,103,744 | 6,790,541,622 |
| gpt2-seq512 this run | 243,792 / 27,295,588 (−0.019% / +0.000%) | 165,428 / 27,346,148 (−0.201% / −1.135%) | 103,699 / 14,477,400 (+0.985% / +0.007%) | 6,794,079,509 (−24,235 B) | 6,790,541,622 (same byte count) |
| gemma3-seq512 baseline | 9,305 / 1,086,100 | 87,421 / 40,278,252 | 68,609 / 22,043,232 | 10,316,824,499 | 10,316,814,558 |
| gemma3-seq512 this run | 9,292 / 1,086,116 (−0.140% / +0.001%) | 87,500 / 40,278,696 (+0.090% / +0.001%) | 68,368 / 22,044,924 (−0.351% / +0.008%) | 10,316,824,499 (exact) | 10,316,814,558 (exact) |

These are single-run values against a single-run baseline, not warm medians;
3% is used only as an anomaly hint (the 0251 record has no thread record and
is a single round). No anomaly was observed; no regression, optimization or
stability claim is made.

## Byte accounting (artifact vs proof.bin vs separately billed key)

- gpt2-seq512: artifact 6,794,079,509 B vs baseline 6,794,103,744 B
  (−24,235 B), itemized as: proof.bin same byte count 6,790,541,622 B
  (content differs; randomized proof; manifest digest differs), statement
  4,183 B (same), public-inputs 2,056 B (same), COMPLETE 65 B (same),
  consumer sidecar 3,527,432 B vs 3,551,656 B (−24,224 B; fresh per-request
  consumer seed), manifest.json 4,151 B vs 4,162 B (−11 B; sidecar entry
  bookkeeping). The key bundle is separately billed in the commit public
  bundle: 4,547,680 B vs 4,549,808 B (−2,128 B; fresh key); the other commit
  bundle files have the same sizes.
- gemma3-seq512: artifact 10,316,824,499 B exact; proof.bin 10,316,814,558 B
  exact with identical manifest digest; statement 4,187 B, public-inputs
  2,056 B, manifest 3,633 B, COMPLETE 65 B all exact; no key and no consumer
  sidecar.
- No byte-identical requirement is claimed for randomized proofs; the
  separately billed key and the artifact are not mixed.

## Judgment and boundaries

- Candidate judgment: both cases pass all three stages, the outputs match the
  fixed 512-element GT, and the identity/log/byte accounting is closed; the
  per-stage 30-minute timeout was not hit.
- Coverage status: this NOTES is the final-round candidate validation record.
  Per the task, TL self-check and expert review are still required; this file
  does not declare the long-sequence coverage complete.
- Boundaries: single round per case; no warm median, no stability claim;
  seq512 is a sequence length, not 512 decode steps; the historical 0251
  record and this run are single-round records of their own versions; no
  performance sign-off, no all-model or full-decode claim, and no
  weight-batch result.

## Evidence files

- `driver.log` (header plus the full run console), `driver-console.log`,
  `monitor.log` (disk and process sampling), `analysis.txt` (derived data),
  `cases/<case>/summary.json`, and `cases/<case>/round-1/{commit,prove,verify}-run/{result.json,progress.jsonl,metrics.json}`.
- Policies, commit/proof manifests and stage artifacts under
  `cases/<case>/round-1/`. Large `.bin` artifacts stay on disk and are
  gitignored; `proof.bin` files are deleted after measurement. Committed
  texts are in git; keys and binaries remain only on the measurement
  machine; no preservation (D01) claim is made here.
