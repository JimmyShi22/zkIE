# Norm mask row derivation

## Scheme and baseline

- Scheme: `norm-mask-rows-derivation`.
- Code: `8d86836d`; previous commit `3ec7d339`.
- This is partial progress on the norm plan. The norm mask row count is now
  derived from the relation count, the mask code answers and the per-instance
  allocation instead of being a fixed number, and the proximity own-mask row
  count is derived by one shared function from the index-mask rows and the norm
  mask rows. The per-instance mask slot meaning, the hiding and shown counts
  and the mask ledger accounting are unchanged.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Single mask-size derivation | rows must hold one mask block per PSC instance plus the claims | `NormMaskPlan::required_rows`, `free_entries_for` | Plan validation and tests use this function |
| Row validation | the plan's rows must be a power of two, no larger than the tree columns and at least the derived count | `NormMaskPlan::validate` | A too-small row count is rejected |
| Shared own-mask rows | the proximity own mask rows are the power-of-two sum of the index-mask and norm-mask rows | `norm_own_mask_rows` used by `HiddenProximityPlan::fault` | One derivation instead of an inline duplicate |
| Default path | no norm mask means no change | absent norm mask contributes zero rows | Norm-free bound and bytes unchanged |

## Results

- `cargo test --workspace`: 48 suites, 849 passed, 0 failed, 17 ignored.
- Focused `norm`: 25 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run.

## Coverage

- The derivation test checks 4 relations at 2 positions (4 rows), 8 relations
  (8 rows) and 17 relations (16 rows), plus the own-mask combinations 4/8/16/1.
- Existing norm positives and negatives still pass; a mask plan whose rows are
  below the derived count is rejected by the plan validation.
- No per-instance mask value, hiding/shown count or claim slot changes.

## Scope and next

- The relation list is still `[sum, squares, mean, variance]`; relations 4–7
  and 8–15 are next, now with a derived mask size.
- No lookup forest and no BAT/FOLD binding for the new relations yet.
- No claim that integer rounding is proven.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
