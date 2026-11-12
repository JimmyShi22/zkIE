# rework15 M2 part 1: trusted multi-stage IOPP stage lists in the plan

## Scheme and baseline

- Scheme: `rework15-m2-stage-list`.
- Code: `947ad2a8` (plan stage lists, validation, tests); document `6f38bc50`
  (v13 update); previous round head `016b2411`.
- This round adds plan data and validation only. The BAT -> (FOLD -> CS)* ->
  BASE prover/verifier loop is not wired; the proof format and the existing
  single-stage path are unchanged. No protocol run was performed.

## Stage lists

- Shared: stage0 `(17, 65,536, 8,192)/(17, 512, 256)`, `k=2,h=0`, positions 2,
  samples 1, work 0; stage1 (last) `(17, 65,536, 256)/(17, 16,384, 256)`,
  `k=h=0`, positions 2, samples 1.
- Own: stage0 `(17, 65,536, 8,192)/(17, 512, 256)`, `k=13,h=2`, positions 2,
  samples 1, work 0; stage1 `(17, 8, 256)/(17, 16, 256)`, `k=13,h=13`,
  positions 2, samples 1; stage2 (last) `(17, 8, 256)/(17, 1,024, 256)`,
  `k=h=0`, positions 2, samples 1.
- `HiddenProximityPlan` carries `shared_stages`/`own_stages`
  (`IoppStagePlan`); `iopp_stage_fault` checks non-last `k>=1`, `h<=k`,
  power-of-two lengths, adjacent `ell_i = 2^{k_{i+1}} ell_{i+1}`, next mask
  room, and last `k=h=0`. The first-stage codes equal the group codes and the
  stage0 `k/h` equal the actual member counts (shared 2/0, own 13/2).
- Unit tests: `actual_shared_and_own_stage_lists_are_legal` and
  `bad_stage_chains_are_rejected` (wrong first code, wrong folded length,
  non-last last stage).

## Workspace

`cargo test --workspace` at `947ad2a8`: 48 test binaries, 753 passed (751
previous + 2 stage-list tests), 0 failed, 17 ignored. The complete log is
`workspace-test.txt`.

## Not run

- No seq16x3: the multi-stage prover/verifier loop is not wired, so the proof
  format and the measured single-stage run are unchanged. When the loop is
  implemented, the protocol changes and the full seq16x3 + 5-case decision
  must be run (TL decides the 5-case scope and records it).

## Open items

- `GAP`: implement the BAT -> (FOLD -> CS)* -> BASE loop; per-stage old/new
  row authentication, independent verifier reconstruction, failure/rejection
  matrix, and planner-sized positions/work/error parameters. The stage lists
  are present; the execution is not.
- The proximity `[GAP]` (main miss 0.3163976670) remains.
- No push: origin remains `831e7134`.

## Ledger

- Tests passed: 753 workspace tests, 0 failed, 17 ignored.
- Proof bytes/wall time/VmHWM: not re-measured; the existing single-stage path
  is unchanged.
