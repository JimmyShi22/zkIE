# Norm proximity members

## Scheme and baseline

- Scheme: `norm-proximity-members`.
- Code: `9bf823bd`; previous commit `fe0089c0`.
- This is partial progress. It adds the norm groups to the hidden proximity
  member list and to the proximity identifier bound. It does not add the
  product-sumcheck endpoints, BAT/FOLD/CS/BASE rows, committed mask entries or
  the negative matrix, and it does not change the existing output-head proof.

## Changes

- `NormWitness` and `NormGroupWitness` carry the prover-side message rows and
  encoding tails of the five norm groups and the optional mask rows.
- `HiddenSendWitness` carries an optional norm witness.
- `own_members` appends one member per position for the trusted x group and the
  sum, squares, mean and variance witness groups, plus the optional mask
  member. A member uses the plan's main code, mask code, row counts, answer
  count and tree plan.
- `proximity_bound_ids` counts `5 * positions` norm members plus one mask
  member, and registers the norm oracle ranges, mask oracle, group tree numbers
  and mask tree number.
- `HiddenProximityPlan::fault` checks that every norm group shares the
  proximity main code and that the norm mask shares the proximity mask code.
- The norm-free path is unchanged: every call site passes the plan's optional
  norm, which is absent for the existing output-head proofs.

## Results

- `cargo test --workspace`: 48 suites, 824 passed, 0 failed, 17 ignored.
- Focused `hidden_norm_psc`: 8 passed, 0 failed.
- Focused proximity bound test: 1 passed, 0 failed.
- Proof-byte hard gate: the toy output-head proof with the norm absent is
  45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.

## Test coverage

- Trusted x root: a changed x group root is rejected; the plan-pinned x roots
  are compared with the first message.
- Witness roots: changed sum/squares/mean/variance roots are accepted by the
  first-message trusted check, because those roots are proof-fixed values and
  not plan inputs.
- Mask root: a zero mask root, a changed mask tree and a changed opening
  position list are rejected.
- Opening positions and bound: each group's opening count is bounded by its
  code tail in plan validation; the proximity bound counts one member and one
  tree per position for each norm group and one member and one tree for the
  mask, and the integrated plan with the norm present validates.

## Explicitly deferred

- Product-sumcheck endpoints and row binding of the norm relations.
- BAT/FOLD/CS/BASE rows for the norm members.
- Committed mask entries and zero-target claims.
- The negative matrix for wrong rows, coordinates, remainders, endpoints,
  masks, challenges and source mapping.
- Mean/variance relation close.
- No soundness, privacy or complete-norm claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
