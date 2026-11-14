# Stage-list and run-size checks for the whole-chain budget

## Scheme and baseline

- Scheme: `plan-iopp-checks` (structural part of the whole-chain budget).
- Code: `a43cbffc`; previous commit `2aa1b24b`; branch `refactor/v2`.

## What changed

- `proximity_planner.rs`: `stage_list_within(list_lengths, sized_positions)`
  rejects a planned list longer than the stage's sized random coordinates and
  names the stage; `RunStage` and `iopp_sized_fault(run, solved)` compare the
  run's stage sizes with the solved stages (stage count, fold bits, main and
  mask lengths, code lengths) and name the stage and the mismatching field.
- Unit tests: lists at and above the sized counts, the stage-count mismatch,
  the row mismatch and the code mismatch with their named failures.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| A list longer than the sized stage is rejected | within check | `stage_list_within` | unit test |
| The run's stages must match the solved stages | sized-fault check | `iopp_sized_fault` | unit test |
| Mismatch names stage and field, not accepted | named failure | `StageFailure` | unit test |
| Per-round round counts and the total budget | next step | **not yet** | gap |

## Evidence

- `cargo test -p zkie-ops --lib plan_iopp_tests`: 2 passed.
- `cargo test --workspace`: 51 suites, 1005 passed, 0 failed, 19 ignored.
- Hard gate: 45,080 bytes, sha256 `7a5da6eb…de4cfb`, unchanged.
- No push.

## Boundaries

- The per-round round counts (BAT, the stage folds, the stage switches and the
  base) and the total budget over the real group and claim counts are not yet
  implemented; only the within and sized-fault structural checks are. The
  hiding-conditions recheck is not wired.
- The production session still uses the diagnostic fixture parameters; the
  planner check stays disabled; fixture constants remain test-only.
- No protocol object, message or verification condition changed; all existing
  full-weight checks remain; no plaintext is removed.

## Artifacts

`identity.txt`, `change.diff`, `workspace.txt`, `proof-bytes.txt`,
`proof-45080.bin`, `iopp-tests.txt`, `NOTES.md`.
