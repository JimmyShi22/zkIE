# Shared authentication paths for fully opened oracles

## Scheme and baseline

- Scheme: `proof-size-shared-paths`.
- Code: `38ae6a25`; previous commit `5459838e`.
- Flag: `[proof-size]`. This is interval 1 of the A work package: remove
  duplicate authentication bytes for fully opened oracles; interval 2
  (object sharing for relation inputs) follows separately.

## What changed

- Relation-step oracle openings no longer carry Merkle paths. Every oracle is
  opened at all of its leaves, and the verifier already checks position order
  and padding, so it rebuilds each root from the decoded leaves. One root
  recomputation replaces one Merkle path per opening.
- Full asset openings follow the same rule and carry no row path; partial asset
  openings keep one path per row, and both forms check the same anchor-bound
  root.
- The session baseline verifier delegates to the shared group verifier instead
  of keeping a second copy; the pathless form is accepted only when the opening
  count equals the committed leaf count and every leaf is present and ordered.
- Negatives: a flipped leaf, a reordered group, a missing leaf, a non-zero
  padding row, a changed root, a changed asset row, a changed port mapping and
  a missing external/edge binding are all rejected (existing tests plus new
  pathless cases).

## Results

Same machine, production entry (`zkie matrix`), one cold and three warm runs
per case. The matrix verify stage is documented separately below.

| Case | Proof bytes before | Proof bytes after | Delta | Prove warm median ms before → after | Prove HWM KiB before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | `4,161,786,553` | `3,564,380,854` | `-597,405,699 (-14.35%)` | `21,919` → `20,402` | `8,316,648` → `7,693,320` |
| gemma3-seq16 | `8,766,306,050` | `6,770,333,437` | `-1,995,972,613 (-22.77%)` | `45,289` → `39,739` | `17,523,776` → `15,372,840` |

- The before binary is built from `5459838e` in a detached worktree; the after
  binary from the delivered commit. Both use the same case files (the
  production case files with their policies, trimmed to the seq16 entries) and
  the same matrix command.
- Independent verifier (same entry points as production, external harness):
  gpt2-seq16 before 25.66 s / 9,453,500 KiB, after 21.32 s / 8,282,372 KiB;
  gemma3-seq16 before 45.89 s / 21,928,048 KiB, after 31.65 s / 17,959,744 KiB.
  Both before and after proofs are accepted and the recomputed argmax equals
  the ground truth (`verify-before.txt`, `verify-after.txt`).
- Matrix verify stage: at this HEAD the CLI builds `RuntimeConfig::default()`
  (1 GiB decode budget) while the merged bounded trace decoder rejects any
  proof file larger than the budget, so the matrix verify stage reports
  `resource_limit` for both the before and the after proof. This is a
  pre-existing integration state of the merged decoder, identical on both
  sides and not caused by this change; the independent verifier above exercises
  the same verification code without the CLI budget.
- `cargo test --workspace`: 48 suites, 969 passed, 0 failed, 17 ignored,
  exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  `cmp` identical (`proof-bytes.txt`).

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| Remove duplicate authentication bytes | full-open step oracles and full asset openings carry no paths; verifier rebuilds roots from all leaves | before/after proof bytes and prove timings; independent verifier accepts both forms | every leaf must be present and ordered; partial asset openings keep paths |
| Every tree is checked, not only the first | every group in every step goes through the shared verifier; roots are rebuilt for each one | `any_opening_position_is_checked`, group reorder/coverage tests, step root tamper | fixture and production traces |
| Missing authentication rejected | pathless form requires count == leaf count and ordered leaves | coverage/pop/reorder/value/root negatives | an empty path with partial coverage is rejected |
| Hidden budget does not grow | no tree is opened at new positions; sharing removes trees, does not add openings | by construction; same proof semantics | interval 1 only removes paths |
| Hard gate and workspace | no change to the norm-free toy path | 45,080 bytes identical; 48 suites green | — |

## Scope and next

- Interval 2 (shared input objects for previous outputs and plan-materialized
  inputs) is not in this commit; its savings are estimated in the statistics
  file and will be measured the same way.
- The seq512 cases are not re-run for interval 1; the parsed statistics give
  the expected interval-1 savings there, and the final A acceptance runs the
  full matrix.
- No push until TL review.

## Artifacts

- `workspace.txt`, `before-matrix.json`, `after-matrix.json`, `verify-before.txt`,
  `verify-after.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
