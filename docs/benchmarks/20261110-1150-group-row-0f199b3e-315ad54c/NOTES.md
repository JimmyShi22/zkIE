# group-row - 0f199b3e / 315ad54c

Trace of the isolated GroupRow source and FOLD component comparison.

## Change

Code commit `0f199b3e` ("Add the isolated GroupRow source and FOLD comparison"),
based on `315ad54c`. The production verify entry is unchanged.

- New isolated `group_row_session` module builds a candidate GroupRow source
  from the complete authenticated asset opening, independently reconstructs the
  fixed main slice and its commitment on the verifier side, and compares them
  with the same source the FOLD component consumes.
- The FOLD component runs the fixed order: commit the fresh mask and fix the
  start sum before drawing epsilon; then send one fixed round message and draw
  that round's gamma.
- Main messages, main tails, old-mask messages, old-mask tail, fresh-mask rows
  and fresh tail are retained separately.
- The 3b projection handoff continues to reject; the comparison does not
  authenticate the fresh term.
- The complete asset opening is resident; this is an intentional isolation
  control, not a production authentication or memory path.

This is not a real GroupRow, G_u or BAT verification, not a valid 3b handoff and
not a production proof acceptance.

## GroupRow source table

| component | source | shape |
| --- | --- | --- |
| main messages | authenticated asset opening rows, fixed slice and encoding | `2^k * ell = 16` |
| main random tails | fixed zero tail fixture (retained, not read by the FOLD component) | `2^k * r = 4` |
| old-mask messages | fixed old-mask fixture | `2^h * mask_ell = 6` |
| old-mask random tail | fixed old-mask tail fixture | `2^h * mask_r = 1` |
| fresh-mask rows | derived from the fixed fresh-mask coefficients | `2^h * mask_ell = 6` |
| fresh-mask random tail | fixed fresh-mask tail fixture | `2^h * mask_r = 1` |
| main claim coefficients | fixed plan values | `2^k * ell = 16` |
| old-mask claim coefficients | fixed plan values | `2^h * mask_ell = 6` |
| kept forms | none (no-kept subset) | `0` |
| target | main and old-mask claim dots | scalar |

The layout has two non-kept main members; no kept rows are present. The fresh
mask is committed with `MaskTree::commit` from the fresh rows, fresh tail and
fixed salts. The start sum is the prover's fixed mask sum.

## Independent comparison

The verifier path:

1. verifies the trusted anchor, asset metadata and complete opening;
2. rejects a changed opening at the authentication layer;
3. reconstructs the expected main messages and tails directly from the opening
   rows with the fixed shape and slice, independently of the source's reported
   values;
4. re-commits the expected table and compares the commitment with the source's
   table root;
5. re-commits the fresh mask and compares its root;
6. verifies that the source passed to the FOLD component is the same `Arc` as
   the source the comparison checked.

A prover-side recomputation is not used as independent authentication. The
comparison result has separate flags for the opening, main rows, table
commitment and fresh commitment; all are true in the real fixture.

## Order and component separation

| step | retained data |
| --- | --- |
| commit fresh mask and fix start sum | fresh root, fresh rows/tail, start sum |
| draw epsilon | epsilon |
| round 1 fixed message, then gamma | round message, gamma |
| round 2 fixed message, then gamma | round message, gamma |
| final FOLD output | main/old/fresh folded components and target |

The outcome keeps `main_messages`, `main_tails`, `old_mask_messages`,
`old_mask_tail`, `fresh_mask` and `fresh_tail` as separate vectors. The round
components are never merged for tracking. Challenges are used by their roles:
not every witness entry depends on epsilon, and the main, old-mask and fresh
terms enter the round and final equations through their own coefficients.

Fixed coins are a test-only numbered source. They are not a random mask and are
not evidence of hiding or a soundness bound.

## Negative coverage

| negative | layer | observed result |
| --- | --- | --- |
| opening value changed | opening authentication | `InvalidProof` / `StatementMismatch` |
| valid re-commitment with different main data | independent comparison | `StatementMismatch` |
| round drawn before epsilon | transcript order | `InvalidSessionState` |
| main vs old-mask vs fresh rows or tails mixed | source shape and separate fields | unit assertions |
| 3b projection handoff attempt | handoff coverage | rejected (`UnsupportedPrimitive`) |

The tests name the rejecting layer; they do not treat a random sample as a
must-reject condition. The valid-recommitment negative builds a table with a
valid root but different main data and shows that the independent comparison,
not the commitment path, rejects it.

## Tests and measurements

- GroupRow unit: 4 passed.
- Fold projection unit: 4 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 38, column 9, edge 2.
- Real gpt2-seq16 GroupRow: 1 passed (57.31 s test, 58.15 s wall, VmHWM
  4,759,564 kB; GroupRow session 737 ms; comparison all true; same source
  true).
- Workspace: 447 passed, 0 failed, 9 ignored.
- No proof bytes were produced; no performance gain, hiding or soundness bound
  is claimed.

Real roots:

| root | value |
| --- | --- |
| asset root | `9a8be4edbbc390fd6844710399441788506c2f1f3a1fc9c9c95e19e067971832` |
| table root | `83a62c3995b33823d29b5465f7a271d69b2cc9a4f73d4c403ee9e96c01f623b3` |
| fresh mask root | `75087d1c1bcf8dd963dcd5f44c799e0ebb709c6a4c804bf627cb3344a4547342` |

The roots are local component commitments, not a real GroupRow/G_u/BAT
authentication.

## Validation scope

Only `gpt2-seq16` is used. The candidate main rows come from the complete asset
opening; old-mask and fresh-mask data are fixed fixtures. G_u and BAT are
skipped. The three seq16 cases beyond the dependency case were not run because
this step does not change the shared three-stage path or seq512 logic. No
five-case three-stage run was made.

## Unclosed items

- Real GroupRow, G_u and BAT verification.
- Real random mask and fresh-mask authentication; the current mask is a fixed
  fixture.
- Complete authentication of sampled columns.
- A valid 3b FOLD-to-CS/BASE handoff; `handoff_statement` still rejects.
- Real SEND sample declaration and follow-up checks.
- Pending CS claims are registered but not proved by a later proximity test.
- Request identity/source authentication.
- Full graph and all upstream/downstream relations.
- Proximity, hiding and single-shot semantics.
- Production wiring: `verify_request` unchanged; this is not an accepted proof
  format.

## Artifacts

- `tests/group-row-unit.txt`
- `tests/fold-handoff-unit.txt`
- `tests/proximity-session-unit.txt`
- `tests/group-row-real.txt` (local fixture path replaced with `<fixture>`)
- `tests/regression-foldeq.txt`
- `tests/regression-column.txt`
- `tests/regression-edge.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.

The earlier 3b r3 package remains retained unchanged. This is a new frozen
evidence set; old packages are not deleted.
