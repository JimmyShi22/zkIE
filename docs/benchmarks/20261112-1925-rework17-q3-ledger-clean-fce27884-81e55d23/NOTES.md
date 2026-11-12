# rework17 Q3: clean final-code two-stage seq16 trace

## Scheme and baseline

- Scheme: `rework17-q3-ledger-clean`.
- Code: `fce27884` (loop ledger wiring, LastBase position return, CS joint-hiding
  update, real outer negative matrix); trace commit `f1e5abdc`; previous scheme
  head `81e55d23`.
- Fixture: `/data/lin-fixture`, GPT-2 formal hidden session, 16 positions.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  /usr/bin/time -v cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.
- One formal run on an otherwise idle machine, from a clean working tree at
  `f1e5abdc`. This run includes the final ledger-consistency assertions and is
  the primary Q3 evidence; the earlier run 3 trace is retained as the
  pre-assertion companion and is not deleted.

## Purpose

Produce a clean two-stage Q3 trace on the final committed code, including the
loop ledger totals and the hidden proof sha256, so the batch evidence has no
launch-order caveat.

## Plan values

- Own stage 0: frozen group codes `(17,65,536,8,192)/(17,512,256)`,
  `k=14,h=2`, positions 254, samples 1, work 24.
- Own last: `(17,65,536,259)/(17,16,384,259)`, `k=h=0`, positions 259;
  `own_base_positions=259`.
- Shared: stage 0 positions 2; last `(17,65,536,256)/(17,16,384,256)`,
  `k=h=0`, positions 2.

## Results

- hidden prove `372,961` ms; hidden verify `64,519` ms; proof `66,135,784`
  bytes; process VmHWM `169,968,960` KiB; Rust test `1,179.74` s; wall
  `19:41.21`.
- hidden proof sha256:
  `47e7fe20a618f1c64f2f77414a33b70bdbb0dc8c1a5ba872ff65143036ebecd5`
  (file `gpt2-seq16/output-head-hidden-request.bin`, mtime 19:35:28).
- `hidden_own_fold_rounds=14`; the run reports PASS.

## Ledger totals

Before LastBase: 139 events, 18,267 logical, 18,265 unique; total tail rank
558,595; minimum remaining 0; exhausted 2. After LastBase: 145 events, 19,050
logical, 19,042 unique; minimum remaining 0; exhausted 2. The final six events
are the last switch tree, star and star-mask records for the two groups. Exact
exhaustion is the reserved rank reached, not an overrun.

## Tests and evidence

- Workspace on the same commit: 48 suites, 796 passed, 0 failed, 17 ignored
  (see the earlier Q3 trace workspace log; code unchanged).
- Focused tests: `multi_stage_iopp` 27, `hidden_send_session` 45,
  `output_head_lin` 10, `base_last_session` 15, all green.
- Q2/Q4 materials in `bridge/tasks/evidence/rework17/`.

## Open items

- Switch sample answer is a single evaluated row, not a position-union event.
- A future dedicated old-mask tree source needs its own ledger key.
- The planner interval arithmetic / largest-m fixed point remains open.
- No push in this round; expert review precedes any push.

## Artifacts

- `clean-two-stage.txt`: complete formal log with `/usr/bin/time`.
- `summary.txt`: key metric/ledger lines.
