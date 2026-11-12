# rework17 Q3: loop opening ledger and staged outer negatives

> Superseded as primary evidence by `20261112-1925-rework17-q3-ledger-clean-fce27884-81e55d23`,
> the clean final-code run. This trace is retained; the run 3 binary predates
> the final test-only ledger-consistency assertions.

## Scheme and baseline

- Scheme: `rework17-q3-ledger`.
- Code: `fce27884` (loop ledger wiring, LastBase position return, CS joint-hiding
  update, real outer negative matrix); previous scheme head `81e55d23` (own
  two-stage plan, planner and Q1/Q5 trace).
- Fixture: `/data/lin-fixture`, GPT-2 formal hidden session, 16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  /usr/bin/time -v cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.
- One formal run on an otherwise idle machine; no other heavy process.

## Purpose

Close TURN 453 Q3: reserve the loop switch/FOLD/star trees from the trusted
plan before proof work, record their actual switch/fresh/LastBase openings in
the external ledger, return the LastBase positions from the verifier, include
the CS switch random material and public values in the joint-hiding note, and
run the real staged outer-envelope negative matrix. Q2 arithmetic material and
the Q4 matrix are in the same batch.

## Plan values

- Own stage 0: frozen group codes `(17,65,536,8,192)/(17,512,256)`,
  `k=14,h=2`, positions 254, samples 1, work 24.
- Own last: `(17,65,536,259)/(17,16,384,259)`, `k=h=0`, positions 259;
  `own_base_positions=259`.
- Shared: stage 0 `(17,65,536,8,192)/(17,512,256)`, `k=2,h=0`, positions 2;
  last `(17,65,536,256)/(17,16,384,256)`, `k=h=0`, positions 2.
- The planner re-derives the own values before the first query. The f64
  planner and its 0.2-bit margin are candidate arithmetic; the reference
  interval port remains open.

## Results

- hidden prove 377,899 ms; hidden verify 64,966 ms; proof 66,135,784 bytes;
  process VmHWM 169,442,488 KiB; Rust test 1,192.65 s; wall 19:54.17.
- The test wall includes the enlarged outer negative matrix; the pre-matrix
  run 2 was 683.96 s / 11:35.95 with hidden prove 378,325 ms and verify
  65,024 ms, so the proof path is unchanged by the matrix.
- `hidden_own_fold_rounds=14`; the run reports PASS.

## Ledger totals

Before LastBase: 139 events, 18,267 logical, 18,265 unique; table 603/601;
closure 12,544; total tail rank 558,595; minimum remaining 0; exhausted 2.
After LastBase: 145 events, 19,050 logical, 19,042 unique; minimum remaining 0;
exhausted 2. The final six events are three records per group (last switch
tree, star, star-mask). The unique delta is 777: each own key adds 257 unique
positions from 259 draws, each shared key adds 2. Exact exhaustion is the
reserved rank reached, not an overrun; `record_openings` rejects the next one.

## Tests and evidence

- `cargo test --workspace`: 48 suites, 796 passed, 0 failed, 17 ignored.
- `cargo check --workspace`: clean.
- Focused: `multi_stage_iopp` 27/27, `hidden_send_session` 45/45,
  `output_head_lin` 10/10, `base_last_session` 15/15.
- The real outer matrix rejects v4/v6 versions, truncation, trailing bytes,
  shared/own declared roots, wrong target, member/fresh openings, shared-zero
  and own-member old-mask openings, LastBase bytes, missing/duplicate switch,
  final seal, cross-group tail swap and a changed W root.
- Q2 / Q4 materials: `bridge/tasks/evidence/rework17/q2-arithmetic.md`,
  `q4-outer-matrix.md`, `q3-ledger-closed.md`.

## Provenance note

Run 3 was launched immediately before the final test-only ledger-consistency
assertions were added to the harness; the proof, ledger recording and outer
negative matrix code are otherwise the committed source. The three-stage
comparison run compiled and exercised those final assertions. No binary proof
artifact was deleted in this round; the textual logs above are committed.

## Open items

- The switch sample answer is a single evaluated row with the switch pad, not
  a tree-opening position-union event; the evidence note states this.
- A future dedicated old-mask tree source needs its own ledger key.
- The planner interval arithmetic and largest-m fixed point are not ported.
- Two-stage is a candidate parameter set, not a final security acceptance; no
  push in this round.

## Artifacts

- `run3-two-stage.txt`: complete formal log with `/usr/bin/time`.
- `workspace.txt`, `check.txt`, `focused-*.txt`: complete test logs.
