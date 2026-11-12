# rework18: same-version five-case matrix

## Scheme and baseline

- Scheme: `rework18-five-case`.
- Code: `d82425ae`; previous scheme head `aee16caf`.
- Command:
  `target/release/zkie matrix --cases models/gpt2/case.toml
  models/gemma3/case.toml models/timesfm/case.toml --cold-runs 1
  --warm-runs 3 --out docs/benchmarks/
  20261112-2107-rework18-five-case-d82425ae-aee16caf/matrix5`.
- Same machine; one cold and three warm runs per case.

## Purpose

Rerun the five-case matrix on the frozen rework18 version, replacing the
rework16 snapshot as the same-version regression evidence. The interval
planner, the nonce-search tail and the harness changes are in this binary;
the hidden fixture plan is separately exercised by the interval seq16 trace.

## Binary cleanup record

Before this matrix, 20 `proof.bin` artifacts (642,392,163,856 bytes) from
`docs/benchmarks/20261112-1427-rework16-loop-wiring-1ca158c6-44a7a019/matrix5`
were deleted to free space; every textual `result.json`, progress log and
summary of that trace was retained. No textual trace was deleted.

## Results

All five cases PASS commit/prove/verify (one cold and three warm runs).

| Case | Proof bytes | Commit cold ms | Prove cold ms | Prove warm median ms | Verify cold ms | Verify warm median ms | Prove HWM median KiB | Verify HWM median KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-seq16 | 4,161,786,553 | 3,417 | 21,809 | 21,858 | 25,325 | 25,339 | 8,316,884 | 5,389,032 |
| gpt2-seq512 | 38,992,550,944 | 3,399 | 167,561 | 167,634 | 233,576 | 232,400 | 78,570,244 | 47,018,204 |
| gemma3-seq16 | 8,766,306,050 | 9,387 | 45,046 | 45,153 | 46,184 | 46,312 | 17,523,476 | 13,341,848 |
| gemma3-seq512 | 101,812,159,615 | 9,390 | 420,191 | 417,925 | 855,864 | 830,363 | 207,848,936 | 120,676,876 |
| timesfm-16patch | 6,865,533,616 | 5,478 | 35,744 | 35,584 | 43,144 | 43,141 | 13,823,488 | 7,320,824 |

The proof byte counts equal the rework16 matrix on this basis; the warm
medians are within the same-machine range. The matrix is a production-entry
regression, not the hidden fixture interval plan.

## Open items

- The matrix does not exercise the 252-position hidden fixture plan; that
  plan is covered by the interval seq16 trace.
- No push in this round.

## Artifacts

- `matrix5/matrix.json`, per-run `result.json` and `progress.jsonl`, the
  `matrix.log`, and this note. The large proof binaries stay on disk in the
  same tree and are gitignored.
