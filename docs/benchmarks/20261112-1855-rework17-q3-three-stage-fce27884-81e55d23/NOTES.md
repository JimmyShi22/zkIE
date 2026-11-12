# rework17 Q3: same-parameter three-stage ledger comparison

## Scheme and baseline

- Scheme: `rework17-q3-ledger-three-stage`.
- Code: `fce27884`; previous scheme head `81e55d23`.
- Fixture: `/data/lin-fixture`, GPT-2 formal hidden session, 16 positions.
- Command: the two-stage Q3 command with `ZKIE_SESSION_THREE_STAGE=1`; the
  same stage-0 positions/work and the same last positions/tail values are
  used, only an extra middle stage `(17,8,256)/(17,16,256)`, `k=13,h=13`,
  positions 254, work 24 and the last codes `(17,8,259)/(17,1,024,259)` are
  inserted. The middle stage is a comparison basis, not planner-checked.
- One formal run on an otherwise idle machine after the two-stage run.

## Purpose

Re-run the 2-vs-3 comparison after the opening-ledger wiring and exercise the
later-switch member/fresh and stage-FOLD ledger branches, which the default
two-stage plan does not reach. This run also compiles and exercises the final
ledger-consistency assertions added to the harness.

## Results

- hidden prove 771,636 ms; hidden verify 76,442 ms; proof 214,228,928 bytes;
  process VmHWM 213,944,084 KiB; Rust test 1,622.63 s; wall 27:04.12.
- Two-stage reference from the Q3 run: prove 377,899 ms, verify 64,966 ms,
  proof 66,135,784 bytes, VmHWM 169,442,488 KiB, Rust test 1,192.65 s.
- `hidden_own_fold_rounds=14`; the run reports PASS.

## Ledger totals

Before LastBase: 141 events, 18,775 logical, 18,772 unique; total tail rank
559,107; minimum remaining 0; exhausted 2. After LastBase: 147 events, 19,558
logical, 19,555 unique; minimum remaining 0; exhausted 3. The extra events
are the stage-1 switch member and stage-FOLD fresh records plus the last-stage
LastBase records. Exact exhaustion is the reserved rank reached, not an
overrun.

## Conclusions

The later-switch and stage-FOLD ledger branches accept the three-stage proof,
and the final consistency assertions pass. The two-stage candidate remains
better on prove, verify, proof bytes and VmHWM under the same positions/work.
The three-stage middle stage is not planner-checked and is not the default.

## Open items

- The three-stage list is a comparison basis; the planner does not size the
  middle stage.
- No push in this round.

## Artifacts

- `three-stage.txt`: complete formal log with `/usr/bin/time`.
- The workspace/check/focused logs and the two-stage Q3 log are in
  `docs/benchmarks/20261112-1834-rework17-q3-ledger-fce27884-81e55d23/`.
