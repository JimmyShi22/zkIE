# Norm round group wiring

## Scheme and baseline

- Scheme: `norm-round-group-wiring`.
- Code: `31f57a99`; previous commit `77a45b97`.
- This is partial progress on the norm integration. The norm group enumeration is
  now one list of the five committed groups plus the optional round group. The
  round group takes part in the sample-row construction, the retained-tree
  lookup and the proximity bound, so it is counted by its physical oracle and
  tree like the other groups. No relation is added in this step.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| One group list | five groups plus the optional round group | `norm_group_plans`, `norm_group_witnesses` return one vector | No five-element duplicate path remains |
| Sample rows | every committed member gets a sample row | `own_sample_rows` iterates the same list and includes the round message | Round members are sampled like the rest |
| Retained trees | every member reads its own committed tree | `witness_tree_for` iterates the same list | Round members open from their retained trees |
| Physical bound | oracle and tree identifiers enter the bound | `proximity_bound_ids` counts `positions * 6` when the round group is present and pushes its oracle and tree ranges | The round identity is a real bound entry |
| Default path | the absent round group is equivalent to the five-group plan | the list length is five when `round` is absent | Norm-free and five-group behavior unchanged |

## Results

- `cargo test --workspace`: 48 suites, 848 passed, 0 failed, 17 ignored.
- Focused `norm`: 24 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run.

## Coverage

- The bound test now checks that the round group's oracle and tree numbers enter
  the bound and were absent before.
- All existing norm positives and negatives still pass with the round group in
  the plan, sample rows and tree lookup.
- A missing or extra round group still fails closed through the plan and first
  message checks.

## Scope and next

- Relations remain `[sum, squares, mean, variance]`; the new round-slot
  relations are next.
- No lookup forest and no BAT/FOLD binding for the new relations yet.
- No claim that integer rounding is proven.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
