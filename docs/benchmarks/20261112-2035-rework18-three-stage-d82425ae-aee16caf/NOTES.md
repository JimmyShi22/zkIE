# rework18: same-parameter three-stage reorder evidence

## Scheme and baseline

- Scheme: `rework18-three-stage`.
- Code: `d82425ae`; previous scheme head `aee16caf`.
- Fixture: `/data/lin-fixture`, GPT-2 seq16 formal hidden session.
- Command: the interval two-stage command with `ZKIE_SESSION_THREE_STAGE=1`;
  the extra middle stage is `(17,8,256)/(17,16,256)`, `k=13,h=13`,
  positions 252, work 24; the last stage is `(17,8,259)/(17,1,024,259)`,
  `k=h=0`, positions 259. The middle stage is a comparison basis, not
  planner-checked.
- One formal run on an otherwise idle machine after the interval run.

## Purpose

Re-run the 2-vs-3 comparison on the interval plan and exercise, on the real
final verifier, the reorder of two actual own switch records. The test swaps
`switches[0]` and `switches[1]` before `verify_output_head_lin_hidden` and
requires rejection; the default two-stage proof has one switch and cannot
express this row.

## Results

- hidden prove 768,897 ms; hidden verify 74,595 ms; proof 212,523,920 bytes
  with sha256
  `5f5cc74f4a24c1c2e7e89ca97953267c65385dd0ff12166aa66fc422b193f782`;
  VmHWM 213,916,484 KiB; Rust test 1,667.56 s; wall 27:55.57.
- Two-stage interval reference: prove 389,493 ms, verify 64,434 ms, proof
  65,432,832 bytes, VmHWM 169,342,676 KiB.
- The run reports PASS, including the real two-switch swap rejection.

## Ledger totals

Before LastBase: 141 events / 18,633 logical / 18,631 unique; total tail
rank 559,107; minimum remaining 2; exhausted 0. After LastBase: 147 /
19,416 / 19,408; minimum remaining 2; exhausted 0. The later-switch member
and stage-FOLD events and the last-stage LastBase records are booked; the
last switch has two repeated BASE positions, so it does not exhaust its
rank.

## Conclusions

The real final verifier rejects the swapped three-stage switch records, and
the later-switch ledger branches accept the full three-stage proof. The
two-stage interval candidate remains better on prove, verify, proof bytes
and VmHWM under the same positions/work. The three-stage middle stage remains
a comparison basis.

## Open items

- The three-stage middle stage is not planner-checked.
- No push in this round.

## Artifacts

- `three-stage.txt`, `summary.txt`, `proof.sha256`.
- The two-stage interval trace and workspace/check/focused logs are in
  `docs/benchmarks/20261112-2014-rework18-interval-d82425ae-aee16caf/`.
