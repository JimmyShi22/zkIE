# rework19: interval boundary evidence and trusted-plan fault checks

## Scheme and baseline

- Scheme: `rework19-boundaries`.
- Code: `TBD`; previous scheme head `d801ec02`.
- Fixture: `/data/lin-fixture`, GPT-2 seq16 formal hidden session.
- Command (three runs): the interval formal command; run 1 adds
  `ZKIE_SESSION_KEY_MAP=1` to print the real ledger key/source map.
- Same machine; runs 1–3 back to back, no other heavy process.

## Purpose

Collect the TURN 457 S1 engineering evidence that remains in scope: interval
rounding/overflow/search boundary tests, trusted-plan `planner_checked`
enforcement evidence, and the real key-to-source map for S2 raw evidence.
The S2 joint-hiding math proof, the S4 composition theorem and the S1 shared
inheritance proof are deferred by the TL §14 scope; this trace does not claim
them. S3 code/config consistency and S5 log mapping are in the bridge notes.

## Results

| Run | hidden prove ms | hidden verify ms | proof bytes | test s | wall | VmHWM KiB |
|---|---:|---:|---:|---:|---:|---:|
| r1 (key map) | 386,319 | 64,768 | 65,432,832 | 1,194.05 | 20:06.14 | 169,428,876 |
| r2 | 387,126 | 64,598 | 65,432,832 | 1,194.83 | 19:56.32 | 169,415,396 |
| r3 | 385,997 | 64,753 | 65,432,832 | 1,194.69 | 19:56.16 | 169,403,640 |

All three PASS. The proof file is overwritten by each run; the interval
proof sha256 from the preceding clean run remains
`6d38a1e34f898a05090d9cf261f0eb2bc3ade27d782425a8021b0749412ec3c4`
for the same code path and plan. Ledger totals are unchanged:
139/18,129/18,058 before LastBase and 145/18,912/18,841 after, with one
exhausted last-switch key (rank 259, unique 259).

## Key map

r1 printed 83 oracle keys: 74 with a non-zero tail rank; the rest are the
activation key and rank-zero star/star-mask placeholders. The grouped
kind/tree/run/configuration table is in the bridge `r2-hiding-budget.md`;
the complete log is `key-map.log` (and `run1-key-map.txt`).

## Trusted-plan fault assertions

The formal test now builds the same hidden plan before proof and asserts that
a changed own stage-0 position and a changed shared join position each fail
`HiddenProximityPlan::fault`. All three runs pass those assertions, so the
checks run before proof work. The non-planner mode is explicitly the
toy/three-stage comparison mode, not an acceptance path.

## Tests

`cargo test --workspace`: 48 suites, 802 passed, 0 failed, 17 ignored; `cargo check --workspace`: clean; focused interval 4/4, planner 4/4.

## Open items

- S2 joint-hiding math proof, S4 composition theorem, S1 shared inheritance
  proof: deferred to the expert notes per TL §14.
- Overall provable union bound remains 2^-123.69; no whole-chain 128-bit
  claim.
- No push in this round.

## Artifacts

- `run1-key-map.txt`, `run2.txt`, `run3.txt`, `workspace.txt`, `check.txt`,
  `focused-*.txt`, `summary.txt`, `NOTES.md`.
