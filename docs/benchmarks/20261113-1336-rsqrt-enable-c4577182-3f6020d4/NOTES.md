# Rsqrt relations enabled in the hidden fixture

- Scheme: `rsqrt-enable`
- Change: code commit `c4577182` (relation 24 target fix, fixture member block
  order, index mask block shape, targeted negatives).
- Baseline: previous commit `3f6020d4` (rsqrt relations trace).
- Purpose: enable relations 16..25 for the hidden fixture and pass the real
  hidden prover/verifier, following the TURN 494 ruling that the earlier
  failure was not evidence of a FOLD/code capacity shortage.

## Root cause and fixes

The earlier BASE `TargetLinear` rejection at 25/26 relations was a
declaration-level defect, not a capacity limit:

- Relation 24 (`index = fine*raw + mid*(FINE+q_mid) +
  coarse*(COARSE_END+q_coarse)`) built its right map with a unit entry at
  position zero, but the BAT row target omitted `eq[0]`. The PSC map equation
  passed; the BAT/FOLD/BASE linear identity was then off by `eq[0]`, which is
  exactly the deterministic `BaseFinalRejection { layer: TargetLinear }` at
  N=25. The constant is now included; N=24, 25 and 26 all pass the full chain.
- The fixture index mask message used three rows. The isolated FOLD fixture
  places member blocks as power-of-two blocks in registration order, so the
  three-row mask made the lookup member set fall outside the fixture subset.
  The index mask now declares four rows (its tree symbol count follows).
- The own member list is now ordered by descending main block and, among
  mask-only members, by descending mask block, matching the BAT block order
  that the isolated FOLD layout check compares against. This replaces a
  sort that only considered the main rows, under which the lookup mask
  (16 rows) preceded the norm mask (32 rows).
- The masked instance count assertion derives from the plan rather than a
  fixed 16.

No FOLD `k`/`h` or code capacity parameter was changed for this step; the
fixture code parameters stay as committed. The engine layout fixture is
unchanged (the alignment attempt was reverted because its own tests pin the
fixture subset).

## Evidence

- Hidden chain: N=24 (`planned` 170), N=25 (176), N=26 (182) all pass the
  real `prove_hidden_lin_hidden_with_norm` + `verify_hidden_lin_hidden` on
  the same fixture (see `focused.txt`).
- Targeted negatives through the real prover path, all rejected:
  - a changed rsqrt `fine_slack` scalar (the committed round digits are
    re-committed consistently, so the relation itself rejects);
  - a changed trusted `table_real_len` in the plan (relation 25 constant);
  - a changed mask entry of relation 24's instance.
- Plan/actual counts: `planned = 6N + 26` = 170/176/182; norm mask entries
  `2N*11 + 1` = 529/551/573; both within the mask plan capacity
  (32 rows x 32 columns) and the declaration set is unchanged per relation.
- Workspace: exit 0, 48 suites ok (see `workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (see `proof-bytes.txt`).

## Four-column mapping

| TURN 494 item | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| (i) per-declaration target/coefficient check, focus on the first new declarations | relation 24 right-hand constant now includes its unit entry; the six declarations added at N=25 are the index-formula rows | N=24 passes, N=25/26 now pass; the rejection was exactly the base target linear identity | relation 25's own target is covered by the plan-length negative |
| (ii) per-member block/layout check; capacity change only with evidence | index mask block is a power-of-two block; own members ordered by descending block sizes; no k/h/code change | lookup carrier and full chain pass with unchanged code parameters; engine fixture untouched | no group split needed |
| (iii) trusted plan declaration total vs runtime | the instance assertion derives from the plan; planned and mask entry counts recorded | counts above; no planning-failure claim | model-level counts come later |
| Mask/declaration tampering rejected | witness slack, plan table length, relation-24 mask entry | three negatives rejected by the real prover path | more negatives in the acceptance batch |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: norm suite 28 passed; N sweep 24/25/26 all passed.
- Workspace: 48 suites ok.
- No wall time or memory claim is made.

## Open items

- Wire the frozen rstd table view into the lookup key plan (column 0 =
  committed index, column 1 = committed table value, table identity and real
  length bound, index < N checked).
- Complete the TURN 492 acceptance batch and the remaining negative matrix.
- No push in this round.
