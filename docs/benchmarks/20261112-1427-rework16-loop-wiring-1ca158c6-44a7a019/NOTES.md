# rework16 loop wiring: trusted stage shapes and acceptance runs

## Scheme and baseline

- Scheme: `rework16-loop-wiring`.
- Code: `1ca158c6` (loop fix `64731c9d`, document correction `985c1945`,
  final-verifier negatives `1ca158c6`); previous round head `44a7a019`.
- The three seq16 runs were taken on this tree after the loop wiring and the
  measured stage-shape correction. The production proof code is `64731c9d`;
  `985c1945` changes documentation and `1ca158c6` adds test-only negatives, so
  neither changes the release binary measured here.
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head,
  16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  /usr/bin/time -v cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.
- Performance baseline: `20261112-0914-rework10-final-473e395d-10cf4df3`
  (same fixture and machine, single-stage path, code `473e395d`).

## Purpose

Record the TURN 451 P1-P7 loop wiring and the local corrections it needed: the
own BAT member bound, the measured own stage-0 fold `k=14` (8,258 main rows),
the diagnostic single-stage plan and the staged final-verifier negatives. The
trace carries the same-machine cost of replacing the single-stage proximity
end with the loop.

## Validation basis

`seq16x3` means three reruns of the same GPT-2 formal fixture, not three cases;
only `gpt2-seq16` is covered. The five-case set is the full acceptance basis.
The multi-stage loop changes the structural proof path, so the conservative
reading requires the five-case run; the scope decision is pending with the TL
and no five-case result is claimed here. Evidence is the three run logs, the
workspace log, the focused test logs and this note.

## Runs

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof bytes | own FOLD rounds | result |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 744,204 | 51,142 | 1,035.04 | 213,833,272 | 9,527,120 | 14 | PASS |
| 2 | 736,296 | 51,681 | 1,027.94 | 213,775,000 | 9,527,120 | 14 | PASS |
| 3 | 740,925 | 51,428 | 1,028.43 | 213,788,360 | 9,527,120 | 14 | PASS |
| baseline `473e395d` | 359,802 | 50,538 | 646.80 | 169,394,600 | 9,939,144 | 14 | PASS |

The loop proof is 412,024 bytes smaller than the single-stage baseline
(-4.15%). Hidden prove wall time is 2.068x the baseline (+106.8%), hidden
verify is +1.20%, maximum RSS is 1.262x (+26.2%) and the whole test is +60.0%.
The bytes do not grow; wall time and peak memory do. No optimization gain is
claimed; the numbers are the measured cost of the multi-stage scheme.

## Workspace

`cargo test --workspace`: 48 test binaries, 794 passed, 0 failed, 17 ignored.
The complete log is `workspace.txt`; `check.txt` is `cargo check --workspace`
and `focused-tests.txt` is the loop and staged-final-verifier run.

## Security and asset documents

- `docs/lin-bind-closure-security.md` v13: own main-row total 8,258, stage-0
  `k=14`, direct FOLD term `28/p^3`, multi-stage candidate merge `58/p^3`;
  the earlier `4,162`/`k=13`/`26/p^3` and `56/p^3` figures are withdrawn.
- `docs/hidden-proximity-loop.md`: trusted stage list, old-mask sources,
  switch semantics, envelope/compatibility, measured cost and open items.

## Compatibility

The hidden output-head proof envelope is version `5` and the session layout is
`14`. A previous envelope is rejected with an unsupported-version error and an
unknown version with an invalid-encoding error; a staged plan rejects the
single-stage tail and a plan without stages rejects a loop tail. The
diagnostic plan carries no stage list because its group drops the Dot,
index-mask and closure members; the formal hidden plan carries the loop.

## Five-case matrix (TL decision)

The TL ruled that the loop is a structural proof-path change and hit the full
case structure, so the five-case set (`gpt2-seq16`, `gpt2-seq512`,
`gemma3-seq16`, `gemma3-seq512`, `timesfm-16patch`) is run with the standard
matrix (one cold run and three warm runs per case):

`target/release/zkie matrix --cases models/gpt2/case.toml
models/gemma3/case.toml models/timesfm/case.toml --cold-runs 1
--warm-runs 3 --out <trace>/matrix5`.

Hit basis recorded as ruled: the loop changes the structural proof path, so
the full five-case set is the validation basis. The hidden loop itself is
exercised by the seq16 formal runs above; this matrix is the production
commit/prove/verify regression over all five cases. Results are in
`matrix5/matrix.json` and the per-run `result.json` files.

All five cases completed commit, prove and verify with `PASS` (one cold run
and three warm runs each; the warm runs reuse the cold commit). Reference
proof bytes are unchanged from the `473e395d` baseline:

| case | commit ms (cold) | prove ms cold | verify ms cold | prove ms warm median | verify ms warm median | proof bytes | prove VmHWM KiB | verify VmHWM KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-seq16 | 3,640 | 21,773 | 25,515 | 21,750 | 25,392 | 4,161,786,553 | 8,315,400 | 5,387,880 |
| gpt2-seq512 | 3,406 | 166,992 | 232,866 | 166,710 | 231,372 | 38,992,550,944 | 78,570,344 | 47,036,760 |
| gemma3-seq16 | 11,201 | 44,967 | 45,828 | 45,212 | 46,366 | 8,766,306,050 | 17,523,660 | 13,346,648 |
| gemma3-seq512 | 9,359 | 418,122 | 820,862 | 416,845 | 822,000 | 101,812,159,615 | 207,848,748 | 120,674,800 |
| timesfm-16patch | 5,676 | 35,524 | 43,154 | 35,484 | 43,046 | 6,865,533,616 | 13,824,216 | 7,322,336 |

The production commit/prove/verify entries do not call the hidden
`hidden_send_session` proximity loop; the matrix is a full-path regression,
and the loop's own real-shape acceptance is the seq16 formal set above.


The previous scheme's gitignored `proof.bin` files under
`docs/benchmarks/20261110-0547-verify-bytes-109934bd-670a98c9/prod/` were
deleted to free 159 GB before this matrix; every textual `result.json` and
`NOTES.md` in that trace was kept.

## Open items

- The planner-derived stage positions, work and error budget; `positions=2`
  and `work=0` remain candidates, and a `2^-128` per-stage target needs much
  larger counts.
- A two-stage own chain (stage 0 `k=14,h=2`, last `ell=65,536/mask 16,384,
  k=h=0`) is shape-legal under the same conditions and may avoid the
  8192x256 intermediate switch; it is not adopted because the stage list is a
  trusted planner parameter.
- The joint security bound and the loop openings in the external ledger.
- The outer-envelope negative matrix for every source type (the loop-level
  matrix is broader than the final-verifier matrix).

## Artifacts

- `run1.txt`, `run2.txt`, `run3.txt`: complete logs with `/usr/bin/time -v`.
- `workspace.txt`, `check.txt`, `focused-tests.txt`: complete test logs.
