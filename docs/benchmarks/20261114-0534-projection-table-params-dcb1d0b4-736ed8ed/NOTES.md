# Production projection table parameters

## Scheme and baseline

- Scheme: `projection-table-params` (first sub-step of the production
  consumer wiring).
- Code: `dcb1d0b4`; previous commit `736ed8ed`; branch `refactor/v2`.
- Scope: the trusted plan expression of the encoded table for the first K
  projection weight. The commit binding, the proof field and the verifier
  wiring are the next sub-steps.

## What changed

- `models/gpt2/src/plan.rs`: `projection_table_params(path)` returns the B1
  `AssetTableParams` for a square projection weight whose specification is a
  1024 by 1024 element matrix. The geometry is plan data: table rows 1024
  (`v` 10), table code `log_n` 12 / `ell` 2048 / `r` 8, one answer, 40-byte
  salts, zero distances; the valid range is the full i32 range and the layout
  is 1024-element rows in a 2048-entry message with an eight-entry tail, three
  pads and zero padding beyond the valid elements.
- A unit test validates the parameters against the real asset specification
  of `gpt2/weights/L0_k_w_i32.bin`.
- No commit artifact, anchor, policy or proof path changes in this step.

## Four-column mapping

| Requirement | Anchor | Change | Evidence |
| --- | --- | --- | --- |
| Trusted table expression for one production consumer | B1 `AssetTableParams`; gpt2 asset specs; the fixture geometry validated at 1024 x 1024 | `projection_table_params` in the gpt2 plan | unit test and crate tests |
| Parameters describe the real asset | `validate_asset_table_params` | validated against the L0 K weight spec | test |
| Commit and anchor binding | B1 `commit_assets_with_tables`, anchor leaf | **not yet** | gap |
| Proof field and verifier wiring | hidden LIN session entries | **not yet** | gap |
| Hidden-input mode, mask slot and bias | TURN 524 items 2-4 | **not yet** | gap |

## Evidence

- `cargo test -p zkie-models-gpt2`: 21 passed, 0 failed (2 suites).
- The full workspace was not re-run for this additive plan-only change; the
  previous run was 50 suites, 984 passed, 0 failed, 17 ignored.
- Hard gate: no hidden path touched; not re-run in this step.
- No push.

## Boundaries

- This is the trusted plan expression only. The encoded table is not yet
  committed for the production asset, the anchor and model identity are
  unchanged, and no proof or verification path was modified.
- The production consumer closure (commit binding, proof field, hidden-input
  mode, index-mask slot, bias check, negatives) remains the next work.
- The expert acceptance is not claimed. No plaintext was removed.

## Artifacts

`identity.txt`, `change.diff`, `model-tests.txt`, `NOTES.md`.
