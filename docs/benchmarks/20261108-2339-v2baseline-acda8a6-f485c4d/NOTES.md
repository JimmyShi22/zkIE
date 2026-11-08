# v2baseline — acda8a6 / f485c4d

- Scheme: `v2baseline`
- Current commit: `acda8a63`
- Previous baseline commit: `f485c4d4`
- Date: 2026-11-08
- Machine: `/data/zkIE` test machine, 495 GiB RAM, `/data` 2.0T (12G free at start)
- Purpose: first full baseline after the single-implementation cleanup; GPT-2,
  Gemma 3 and TimesFM across the five supported cases.

## Gate

- `cargo test --workspace`: **91 passed, 0 failed** (31 test binaries, exit 0).
- Release build: `target/release/zkie-bench` rebuilt before the first run.
- GPT-2 seq16: all three stages returned `PASS`.
- GPT-2 seq512: all three stages returned `PASS`.

## Minimal-case three-stage validation — GPT-2 seq16

| stage | wall ms | VmHWM KiB | proof bytes |
| --- | ---: | ---: | ---: |
| commit | 3,333 | 355,528 | - |
| prove | 17,630 | 8,315,544 | 4,161,786,553 |
| verify | 24,539 | 5,402,436 | - |

- `/tmp/v2smoke` total: 3.9G.
- Commit artifact: ~100KB. `proof.bin`: 4.16GB. `statement.bin`: 214B.
  `public-inputs.bin`: 72B.
- `proof.bin` is archived on disk in this directory (gitignored, not committed):
  `docs/benchmarks/20261108-2339-v2baseline-acda8a6-f485c4d/mincase-seq16/proof/proof.bin`.
- A copy also remains under `/tmp/v2smoke` (volatile tmpfs) and may disappear;
  the on-disk copy above is the durable archive.

## seq512 scaling measurement — GPT-2 seq512

| stage | wall ms | VmHWM KiB | proof bytes |
| --- | ---: | ---: | ---: |
| commit | 3,348 | 356,032 | - |
| prove | 123,986 | 78,569,796 | 38,992,550,944 |
| verify | 230,146 | 47,018,064 | - |

- `/tmp/v2smoke512` total at completion: 37G.
- Scaling vs seq16: proof 9.37x, prove wall 7.03x, prove VmHWM 9.45x,
  verify wall 9.38x, verify VmHWM 8.70x.
- `proof.bin` is archived on disk in this directory (gitignored, not committed):
  `docs/benchmarks/20261108-2339-v2baseline-acda8a6-f485c4d/seq512/proof/proof.bin`.
- A copy also remains under `/tmp/v2smoke512` (volatile tmpfs) and may
  disappear; the on-disk copy above is the durable archive.

## Matrix

_TBD — on hold pending scope/storage decision._

## Profile

_TBD — on hold pending scope/storage decision._

## Storage context

- `/data` root: 2.0T total, 12G free at start.
- `/data` usage: 1.1T `/data/zkie-refactor-runs`; 769G
  `/data/zkie-assets`; 37G `/data/zkIE`; 5.6G `/data/zkie-live-20261024`;
  the rest is below 1G.
- `/tmp` is tmpfs: 182G available at start, memory-backed and volatile.
- `/dev/shm` is tmpfs: 173G available at start.

## Conclusions and open items

- The current proof is 4.16GB for GPT-2 seq16 and 38.99GB for GPT-2 seq512.
- A full matrix (20 runs) and profile matrix (45 runs) cannot fit in the
  available storage: at 4-39GB per run they would exceed the 12G `/data` and
  182G `/tmp` budget together.
- Full runs are on hold pending a TL decision on the run subset and storage
  layout. No artifacts are deleted.
