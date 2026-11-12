# rework17: own two-stage candidate, trusted planner and the 2-vs-3 comparison

## Scheme and baseline

- Scheme: `rework17-own-two-stage`.
- Code: `b6cd5403` (planner and two-stage wiring `7778a5f4`); previous
  scheme head `27b07870` (the five-case trace from rework16).
- Fixture: `/data/lin-fixture`, GPT-2 formal hidden session, 16 positions.
- Command, two-stage:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  /usr/bin/time -v cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.
- Command, three-stage comparison:
  the same command with `ZKIE_SESSION_THREE_STAGE=1`; both use the same
  stage-0 positions/work and the same last-tail/position values.
- The two runs were taken back to back after the five-case matrix finished;
  no other heavy process ran. Table commit was 29.6 s and 29.0 s, and the
  diagnostic prove/verify numbers were 17.4/8.1 s and 17.3/8.2 s, so the load
  and cache state of the pair were comparable.

## Purpose

Record TURN 453 Q1/Q2/Q5: the own stage list is the two-stage candidate,
the trusted planner re-derives its positions/work/last tail before the first
query, and the same-parameter three-stage chain is measured for comparison.
Q4's five-case matrix results are in
`docs/benchmarks/20261112-1427-rework16-loop-wiring-1ca158c6-44a7a019/`.

## Plan values

- Stage 0: frozen group codes `(17,65,536,8,192)/(17,512,256)`, `k=14,h=2`;
  `positions=254`, `samples=1`, `work_bits=24`. The cap is the group mask
  tail 256 minus the two SEND openings.
- Last: `(17,65,536,259)/(17,16,384,259)`, `k=h=0`, `positions=259`.
- Planner target 128 bits with a 0.2-bit planning margin; the module re-derives
  the Johnson `m`, list, MCA/FOLD/BASE, sample/field/batching terms and the
  positions/work from the trusted codes. The values match a candidate
  derivation recorded in `bridge/tasks/evidence/rework17/planner-candidate.md`.
  The interval arithmetic of the reference planner is not ported yet.

## Runs

| metric | two-stage | three-stage | change |
|---|---:|---:|---:|
| hidden prove ms | 372,116 | 768,920 | -396,804 (-51.6%) |
| hidden verify ms | 64,736 | 75,151 | -10,415 (-13.9%) |
| hidden proof bytes | 66,135,784 | 214,228,928 | -148,093,144 (-69.1%) |
| max RSS KiB | 169,961,812 | 212,843,964 | -42,882,152 (-20.1%) |
| whole test s | 674.20 | 1,083.66 | -409.46 (-37.8%) |
| wall clock | 11:17.70 | 18:05.11 | -6:47.4 |

Both runs report `hidden_own_fold_rounds=14` and PASS. The two-stage proof is
larger than the old positions=2 proof (9.5 MB) because the candidate raises the
security parameters; the comparison above holds positions/work fixed.

## Tests and evidence

- `cargo test --workspace`: 48 suites, 796 passed, 0 failed, 17 ignored.
- `cargo check --workspace`: clean.
- Focused: `proximity_planner` 2/2, `staged_final_verifier` and the
  `multi_stage_iopp` tests pass.
- The staged final-verifier toy test now also rejects shared-zero/own-Member
  old-mask openings and a changed fresh opening; missing switch, changed
  target/root/LastBase are covered as before.
- The five-case matrix (gpt2/gemma3 seq16+seq512, timesfm-16patch) passed with
  proof bytes unchanged from the previous baseline; see the rework16 trace.

## Open items

- Q3 (loop openings in the external ledger) remains open: switch/FOLD/star
  trees are not reserved and their actual openings are not recorded. See
  `bridge/tasks/evidence/rework17/q3-ledger-gap.md`.
- Q2's planner uses floating-point logs and a 0.2-bit margin; the reference
  interval arithmetic and the exact `largest_m` rule are not ported.
- The outer-envelope negative matrix for every source type is not complete.
- Two-stage is the candidate, not a final parameter; the proof is 66 MB and
  peak RSS is 170 GB. Whether segmentation or a technical condition is needed
  is for the boss to decide under TURN 453.

## Artifacts

- `q1-two-stage.txt`, `q5-three-stage.txt`: complete logs with `/usr/bin/time`.
- `workspace.txt`, `check.txt`, `focused-tests.txt`: complete test logs.
