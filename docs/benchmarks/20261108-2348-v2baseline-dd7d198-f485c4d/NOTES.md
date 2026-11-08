# v2baseline - dd7d198 / f485c4d

- Scheme: `v2baseline`
- Matrix code base: `dd7d1986`
- Previous baseline commit: `f485c4d4`
- Verification runs executed on: `acda8a63`
- Date: 2026-11-08
- Purpose: full matrix baseline on the cleaned implementation. `dd7d1986`
  differs from `acda8a63` only by removing the three `executor_smoke` example
  files and syncing the conventions entry table; the benchmark code path has
  zero differences.

## Gate

- `cargo test --workspace`: **91 passed, 0 failed** (confirmed on `acda8a63`
  and re-confirmed after rebasing onto `dd7d1986`).
- Both verification runs returned `PASS` for commit, prove and verify.

## Previous verification data

The two verification runs are archived both in this directory
(`verify-seq16/`, `verify-seq512/`, full run directories including
`proof.bin`) and in the earlier trace directory
`docs/benchmarks/20261108-2339-v2baseline-acda8a6-f485c4d/`.

### Minimum case three-stage validation - GPT-2 seq16

| stage | wall ms | VmHWM KiB | proof bytes |
| --- | ---: | ---: | ---: |
| commit | 3,333 | 355,528 | - |
| prove | 17,630 | 8,315,544 | 4,161,786,553 |
| verify | 24,539 | 5,402,436 | - |

### seq512 scaling measurement - GPT-2 seq512

| stage | wall ms | VmHWM KiB | proof bytes |
| --- | ---: | ---: | ---: |
| commit | 3,348 | 356,032 | - |
| prove | 123,986 | 78,569,796 | 38,992,550,944 |
| verify | 230,146 | 47,018,064 | - |

Scaling vs seq16: proof 9.37x, prove wall 7.03x, prove VmHWM 9.45x,
verify wall 9.38x, verify VmHWM 8.70x.

## Matrix - dd7d1986

Running against `models/gpt2/case.toml`, `models/gemma3/case.toml` and
`models/timesfm/case.toml`, cold 1 + warm 3. All five cases passed.

| case | proof bytes | commit wall (cold) | prove wall (warm med) | verify wall (warm med) | prove VmHWM | verify VmHWM | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gpt2-seq16 | 4,161,786,553 | 3,344 | 21,576 | 24,360 | 8,315,824 | 5,385,024 | PASS |
| gpt2-seq512 | 38,992,550,944 | 3,327 | 165,973 | 230,274 | 78,569,540 | 47,016,984 | PASS |
| gemma3-seq16 | 8,766,306,050 | 10,496 | 45,076 | 44,401 | 17,523,028 | 13,357,036 | PASS |
| gemma3-seq512 | 101,812,159,615 | 9,313 | 417,040 | 835,721 | 207,848,492 | 120,670,388 | PASS |
| timesfm-16patch | 6,865,533,616 | 5,356 | 35,308 | 42,356 | 13,821,988 | 7,329,656 | PASS |

### Artifact sizes

`proof_bytes` is the value recorded in `result.json`; `proof.bin` length is the
actual file size of the cold run; the case directory total covers all four runs
and their text artifacts. The two proof-number columns differ slightly and are
recorded as-is.

| case | proof_bytes (result.json) | proof.bin length (cold-0) | case directory bytes |
| --- | ---: | ---: | ---: |
| gpt2-seq16 | 4,161,786,553 | 4,161,782,574 | 16,647,339,772 |
| gpt2-seq512 | 38,992,550,944 | 38,992,541,006 | 155,970,449,721 |
| gemma3-seq16 | 8,766,306,050 | 8,766,302,078 | 35,065,410,339 |
| gemma3-seq512 | 101,812,159,615 | 101,812,149,678 | 407,249,074,575 |
| timesfm-16patch | 6,865,533,616 | 6,865,265,628 | 27,462,530,276 |

- Matrix directory total: 642,394,831,501 bytes (598.3 GiB).
- Whole trace directory total: 685,549,447,423 bytes (638.5 GiB), including the
  full verification archive under `verify-seq16/` and `verify-seq512/`.
- All matrix outputs are on `/data` (ext4).

Matrix prove timings include serializing `proof.bin` to `/data` (ext4). The
standalone verification runs wrote to `/tmp` (tmpfs), so their prove wall times
are lower for the same proof bytes; the matrix warm medians are the ledger
numbers.

- Matrix output: `docs/benchmarks/20261108-2348-v2baseline-dd7d198-f485c4d/matrix/`
- Matrix log: `docs/benchmarks/20261108-2348-v2baseline-dd7d198-f485c4d/matrix.log`
- Matrix was executed on `dd7d1986` with the binary `target/release/zkie-bench`.
- A later commit `bc958d86` renames the crate/binary to `zkie` (directory
  `tools/zkie/`, binary `target/release/zkie`) with no proof-code change.
  Profile runs will use the current pulled HEAD with the `zkie` binary (see the
  Profile section).
- Historical trace artifacts in
  `docs/benchmarks/20261108-2339-v2baseline-acda8a6-f485c4d/` retain the
  original `zkie-bench` strings and are not modified.

## Artifact layout

- `verify-seq16/`: complete run directory, including `proof.bin` (4.16GB) and
  all text results/logs. Binary files are gitignored; text artifacts are
  committed.
- `verify-seq512/`: complete run directory, including `proof.bin` (38.99GB) and
  all text results/logs. Binary files are gitignored; text artifacts are
  committed.
- `matrix/`: matrix run directories and their text results; `proof.bin` files
  stay on disk and are gitignored.
- Original verification run directories remain intact under `/tmp/v2smoke` and
  `/tmp/v2smoke512` (tmpfs, volatile).

## Acceptance checks

- The matrix ran on `dd7d1986` with `target/release/zkie-bench`.
- After the matrix, `git pull --rebase origin refactor/v2` landed on
  `9cf05f89`. The range `773aa21a..9cf05f89` changes only `.gitignore`,
  `AGENTS.md` and `CLAUDE.md` (44 insertions, 2 deletions); there is no code
  diff.
- `cargo build --release -p zkie -p zkie-check` succeeded on `9cf05f89`;
  binaries are `target/release/zkie` and `target/release/zkie-check`.
- `zkie-check assets --root /data/zkie-assets --cases` (three case files):
  exit 0, `checked=720`, `unique_files=715`, `missing=0`, `mismatched=0`,
  `bad_patch_tokens=0`. Outputs: `assets-check.json`, `assets-check.log`.
- `zkie-check report --input <matrix> --require-profile cpu-original-cases`:
  exit 0, all five cases `PASS` (`passed=5`, `required=5`, `blocked=0`,
  `incomplete=0`, `state_conflict=0`, `budget_exhausted=0`). Output:
  `matrix/acceptance-report.json`, `report-matrix.log`.
- Invoking `zkie-check report` on an individual case directory with the same
  profile returns exit 1 (`case cold-0 is outside the cpu-original-cases
  profile`): that input layout treats the run subdirectories as case ids. The
  per-case logs are archived as `report-<case>.log`. The matrix-level report is
  the supported input and is the acceptance result; this is a tool-input naming
  issue, not a proof failure.

## Storage context

- Before cleanup: `/data` 2.0T, 12G free; `target` 28G; `zkie-assets` 769G;
  `zkie-refactor-runs` 1.1T.
- The approved cleanup removed `/data/zkie-refactor-runs` only.
  `/data/zkie-assets`, `/data/zkie-live-20261024` and the repository were not
  modified by the cleanup.
- At matrix completion: `/data` 2.0T total, about 1.5T used, 435G available;
  the matrix directory is 642,394,831,501 bytes and the whole trace directory
  is 685,549,447,423 bytes.
- `/tmp` is tmpfs (memory-backed, volatile); `/dev/shm` is also tmpfs.

## Profile

_pending - profile matrix remains on hold. When approved, it will run on the
current pulled HEAD (`9cf05f89`) with the `zkie` binary. `bc958d86` introduced
the CLI rename; later commits move the CLI to the repository root and change
documentation and collaboration rules. None of them changes the proof path._

## Conclusions and open items

- Proof sizes: GPT-2 seq16 4.16GB, GPT-2 seq512 38.99GB, Gemma 3 seq16 8.77GB,
  Gemma 3 seq512 101.81GB, TimesFM 16patch 6.87GB.
- All five matrix cases and all acceptance checks passed.
- Matrix storage must be watched per case; stop if `/data` free space falls
  below 2G.
- No benchmark artifact has been deleted.
