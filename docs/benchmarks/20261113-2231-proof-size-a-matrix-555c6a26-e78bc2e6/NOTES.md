# Five-case acceptance matrix for the shared-object work package

## Scheme and baseline

- Scheme: `proof-size-a-matrix`.
- Code: `555c6a26`; parent `e78bc2e6`; branch `refactor/v2`. The commits above
  the interval-2 code are trace-only: `git diff --name-only e78bc2e6..HEAD`
  lists only files under `docs/benchmarks/`.
- Before baseline: the five-case matrix at `d82425ae` in
  `docs/benchmarks/20261112-2107-rework18-five-case-d82425ae-aee16caf`
  (same machine, same command, same case files, 1 cold + 3 warm per case).
  The seq16 interval-1 and interval-2 before/after numbers were measured
  separately on the same basis and are recorded in their own traces.
- Verification paths: before is the CLI matrix verify at `d82425ae`; after is
  the CLI matrix verify at `555c6a26` (the decode budget is sized from the
  proof file) plus the external harness. The harness uses the same-run cold-0
  public bundle, policy and proof; it is a separate timing path from the CLI.
- Command:
  `target/release/zkie matrix --cases models/gpt2/case.toml
  models/gemma3/case.toml models/timesfm/case.toml --cold-runs 1
  --warm-runs 3 --out <trace>/matrix`
- Machine: 64 logical CPUs (32 cores x 2 threads), 495 GiB RAM, no GPU,
  `/data` 2.0T. Thread basis: the matrix entry runs the CLI with the default
  rayon pool (64 logical CPUs); no `--threads` argument and no
  `RAYON_NUM_THREADS` override. The baseline matrix used the same basis.
- Disk cleanup: before this matrix, 20 `proof.bin` files from the previous
  scheme (`20261112-2107-rework18-five-case-d82425ae-aee16caf`, total
  642,392,163,856 B) were deleted to free space. Every textual `result.json`,
  progress log, summary and note was retained. No textual trace was deleted.

## Results

All 20 runs (5 cases x 1 cold + 3 warm) pass commit, prove and verify. The
proof bytes column is the proof artifact directory total (proof.bin plus
statement, public inputs, manifest and completion marker); the parsed
proof.bin is reported separately for the independent verifier.

| Case | Commit cold ms | Prove cold ms | Prove warm median ms | Verify cold ms | Verify warm median ms | Proof bytes | proof.bin bytes | Prove HWM warm median KiB | Verify HWM warm median KiB | Output elements | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gpt2-seq16 | 3,373 | 14,301 | 14,286 | 71,627 | 71,609 | 2,190,159,492 | 2,190,155,518 | 4,496,672 | 4,103,228 | 16 | 4/4 PASS |
| gpt2-seq512 | 3,372 | 60,973 | 60,963 | 318,670 | 313,535 | 11,127,652,972 | 11,127,643,038 | 27,377,872 | 18,868,356 | 512 | 4/4 PASS |
| gemma3-seq16 | 9,308 | 35,302 | 35,244 | 182,674 | 183,594 | 6,223,297,337 | 6,223,293,366 | 12,718,464 | 10,951,568 | 16 | 4/4 PASS |
| gemma3-seq512 | 9,529 | 113,996 | 114,221 | 487,948 | 486,512 | 16,578,332,025 | 16,578,322,086 | 52,503,080 | 30,509,692 | 512 | 4/4 PASS |
| timesfm-16patch | 5,390 | 23,738 | 23,668 | 117,847 | 116,243 | 3,824,905,648 | 3,824,637,660 | 8,208,312 | 4,185,692 | 32,768 | 4/4 PASS |

Authenticated output digests (all four runs per case bind the same value):

| Case | Authenticated output digest |
| --- | --- |
| gpt2-seq16 | `135c4d1c47aee4fd1985fa862afefbedde9d59beaf112bd2cf0dcfff03339280` |
| gpt2-seq512 | `eac0949d6463927eb9a961d61d618fb703e4b02169e89db10dbada94a5f0dfdd` |
| gemma3-seq16 | `8683f5e22e51d1405fe3efcc7487a9a4df2e694c5d9f0bed64e921810d77c562` |
| gemma3-seq512 | `6d1df6b50c20d02db715fd14b12b9134623739375c44a029dd15b6b30bd9344c` |
| timesfm-16patch | `cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3` |

Every digest equals the expected output digest bound by that case's policy and
the integer ground-truth file digest. Per-run cold and warm rows are in
`results.md`.

## Before and after (same machine, same command)

Before is `d82425ae` (pre shared-object work package); after is `555c6a26`.

| Case | Proof bytes before | Proof bytes after | Delta | Prove warm median ms before -> after | Verify warm median ms before -> after | Prove HWM KiB before -> after | Verify HWM KiB before -> after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 4,161,786,553 | 2,190,159,492 | -1,971,627,061 (-47.37%) | 21,858 -> 14,286 (-34.6%) | 25,339 -> 71,609 (+182.6%) | 8,316,884 -> 4,496,672 (-45.9%) | 5,389,032 -> 4,103,228 (-23.9%) |
| gpt2-seq512 | 38,992,550,944 | 11,127,652,972 | -27,864,897,972 (-71.46%) | 167,634 -> 60,963 (-63.6%) | 232,400 -> 313,535 (+34.9%) | 78,570,244 -> 27,377,872 (-65.2%) | 47,018,204 -> 18,868,356 (-59.9%) |
| gemma3-seq16 | 8,766,306,050 | 6,223,297,337 | -2,543,008,713 (-29.01%) | 45,153 -> 35,244 (-21.9%) | 46,312 -> 183,594 (+296.4%) | 17,523,476 -> 12,718,464 (-27.4%) | 13,341,848 -> 10,951,568 (-17.9%) |
| gemma3-seq512 | 101,812,159,615 | 16,578,332,025 | -85,233,827,590 (-83.72%) | 417,925 -> 114,221 (-72.7%) | 830,363 -> 486,512 (-41.4%) | 207,848,936 -> 52,503,080 (-74.7%) | 120,676,876 -> 30,509,692 (-74.7%) |
| timesfm-16patch | 6,865,533,616 | 3,824,905,648 | -3,040,627,968 (-44.29%) | 35,584 -> 23,668 (-33.5%) | 43,141 -> 116,243 (+169.4%) | 13,823,488 -> 8,208,312 (-40.6%) | 7,320,824 -> 4,185,692 (-42.8%) |

- Proof bytes and prove wall time and prove/verify peak memory decreased for
  all five cases.
- The CLI verify wall time increased for four of five cases and decreased for
  `gemma3-seq512`. This column is not attributable to the shared-object work
  alone: the bounded trace decoder merge and the proof-sized decode budget
  repair are in the same interval. At intermediate commits the CLI verify
  reported `resource_limit` on proofs above the default budget; those records
  are not counted as PASS and not used as valid timings. The independent
  interval 1 -> interval 2 harness verify (no CLI budget) decreased
  (gpt2-seq16 21.32 s -> 18.18 s; gemma3-seq16 31.65 s -> 30.80 s). The
  numbers here are the measured values; no timing is claimed to be noise-free.
- The proof artifact total and the parsed proof.bin differ by the artifact
  metadata: 3,971-3,974 B for seq16, 9,934-9,939 B for seq512 and 267,988 B
  for timesfm (the timesfm statement carries the 32,768-element output
  binding). The before and after artifact totals include their own metadata,
  so a small part of the byte delta is metadata rather than proof payload.

## Seq512 estimates replaced by measurement

The interval-1 and interval-2 estimates were computed by parsing the pre-work
proof.bin. The measured deltas below are on the matrix artifact basis; the
metadata difference between the two versions is a few bytes and is included.

| Case | Interval-1 estimate B | Interval-2 estimate B | Input root entries B | Total estimate B | Measured delta B | Difference B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 597,405,696 | 1,373,766,752 | 454,608 | 1,971,627,056 | 1,971,627,061 | +5 |
| gpt2-seq512 | 17,615,339,520 | 10,249,223,648 | 334,800 | 27,864,897,968 | 27,864,897,972 | +4 |
| gemma3-seq16 | 1,995,972,608 | 546,827,232 | 208,872 | 2,543,008,712 | 2,543,008,713 | +1 |
| gemma3-seq512 | 76,639,961,088 | 8,593,699,680 | 166,824 | 85,233,827,592 | 85,233,827,590 | -2 |
| timesfm-16patch | no estimate | no estimate | no estimate | no estimate | 3,040,627,968 | measured only |

The seq512 estimates are within 4 B of the measurement; they are replaced by
the measured values in the delivery record. `timesfm-16patch` had no prior
byte-decomposition estimate; only the measured before/after is reported.

## Independent verifier (after proofs)

The verifier lives outside the repository. It reads the public commitment
bundle, policy, proof, token input and ground truth and calls the same
verification entry points as production: `verify_model_anchor` plus
`verify_shared_trace` for the sequence models and `verify_shared_trace_output`
for the patch model. Each case uses the same-run cold-0 pairing: the cold-0
public bundle, cold-0 proof, cold-0 policy and the cold-0 input. Warm runs
reuse the cold-0 commit directory in production, so the cold bundle is the
one against which the run's proofs are verified.

| Case | proof.bin bytes | Steps | Checked elements | Elapsed | VmHWM KiB | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gpt2-seq16 | 2,190,155,518 | 4,462 | 16 | 19.96 s | 4,316,692 | PASS, argmax = integer GT |
| gpt2-seq512 | 11,127,643,038 | 2,798 | 512 | 48.97 s | 22,311,068 | PASS, argmax = integer GT |
| gemma3-seq16 | 6,223,293,366 | 2,090 | 16 | 30.72 s | 12,237,868 | PASS, argmax = integer GT |
| gemma3-seq512 | 16,578,322,086 | 1,506 | 512 | 77.53 s | 33,990,708 | PASS, argmax = integer GT |
| timesfm-16patch | 3,824,637,660 | 6,683 | 32,768 | 22.71 s | 7,574,476 | PASS, integer output = integer GT |

- The ground truth is the integer ground truth bound by the policy
  (`gt_int_argmax_*` for the sequence models, `gt_int_output_i32.bin` for the
  patch model). The original ground-truth file for gemma3 is a different
  file and is not the policy binding.
- The first-pass verifier logs are preserved as `verify-after-*.first-pass.txt`
  and `negatives-*.first-pass.txt`. They are driver errors, not proof
  rejections: cross-run mixing of the commitment and proof paths, the
  original gemma3 ground truth instead of the integer ground truth, the
  missing patch-token planning parameter, and an opening-group selection bug
  in the negative driver. The corrected runs above use the same-run pairing
  and the policy-bound integer ground truth.
- Before independent-verifier references exist only for the seq16 cases and
  only for the interval-1 state: gpt2-seq16 21.32 s / 8,282,372 KiB and
  gemma3-seq16 31.65 s / 17,959,744 KiB. No pre-work-package harness run
  exists. seq512 and timesfm have after-only harness results. CLI verify and
  harness timings are separate measurement paths and are not mixed.

## Negative matrix

The same external verifier was run with `--negative` on the same-run cold-0
proofs of three plan families. Each mutation is applied to the decoded trace,
the real verification entry point must reject it, and the mutation is
reverted. All five classes are rejected on every case.

| Class | gpt2-seq16 | gemma3-seq16 | timesfm-16patch |
| --- | --- | --- | --- |
| Changed object | `invalid_proof` | `invalid_proof` | `invalid_proof` |
| Changed root | `invalid_proof` | `invalid_proof` | `invalid_proof` |
| Changed position | `invalid_index` | `invalid_index` | `invalid_index` |
| Changed mapping | `statement_mismatch` | `statement_mismatch` | `statement_mismatch` |
| Missing authentication | `statement_mismatch` | `statement_mismatch` | `shape_mismatch` |

## Hard gate

The norm-free toy proof is 45,080 B with sha256
`7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`. The test
`proof::output_head_lin::tests::merged_request_accepts_and_round_trips` passed
and a temporary local write in the test produced the proof file; the write
was removed with `git checkout` afterwards. `cmp` is identical against both
gate references (`/tmp/zkie-proof-after.bin` and `/tmp/zkie-proof-before.bin`).
The source tree is clean apart from this trace directory. Evidence is in
`proof-bytes.txt`.

## Workspace

`cargo test --workspace`: 48 suites, 973 passed, 0 failed, 17 ignored, exit 0
(`workspace.txt`).

## Wording correction

The interval-2 delivery record said "plan digest version upgrade". The
accurate statement for the current code is: the shared input openings are
removed; a proof that still carries an input opening is rejected by
`plan::check_shared_input_form` semantics (any non-empty input group returns
`UnsupportedVersion`); the plan digest is not upgraded and the model anchor
is unchanged. `plan::SHARED_TRACE_INPUT_VERSION = 2` is declared but not read.
This correction changes no code and no model identity.

## Boundaries and gaps

- Scope: interval 1 (shared authentication paths for fully opened oracles and
  asset rows) and interval 2 (relation inputs materialized from the trusted
  plan source). The asset plaintext channel and the segmented folding work
  are not in this package and are not claimed.
- The matrix covers the production shared trace. The hidden norm chain from
  the parallel input-binding line is not exercised here; the hard gate covers
  the norm-free toy path only.
- Verify wall time increased versus the pre-work baseline on four of five
  cases. The cause is not isolated by this matrix and is reported as measured.
- Missing baselines: no pre-work independent-verifier run; no harness before
  for seq512/timesfm; no byte estimate for timesfm. These are listed, not
  backfilled.
- The CLI `resource_limit` records at intermediate commits are excluded from
  PASS and from timing.
- No push.

## Artifacts

`matrix/matrix.json`, `matrix.log`, `identity.txt`, `results.md`,
`verify-after-*.txt` (plus `*.first-pass.txt`), `negatives-*.txt` (plus
`*.first-pass.txt`), `proof-bytes.txt`, `workspace.txt`, `NOTES.md`. The large
proof binaries stay on disk in `matrix/` and are not committed.
