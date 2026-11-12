# Norm row-binding Stage A: plan and message envelope

## Scheme and baseline

- Scheme: `norm-rowbinding-a` (T-003 batch 2b.1 row-binding, Stage A).
- Code: `22c7dc28`; previous commit `69cad655`.
- This is partial progress only. It adds the integrated norm plan and message
  envelope; it does not add members, rows, masks or negative coverage, and it
  does not change the existing output-head hidden proof.

## Changes

- Integrated `NormPlan`: the trusted x group roots are plan values; the
  sum/squares/mean/variance groups are proof-fixed witness groups whose shapes,
  identifiers and opening counts are plan values; the mask plan is optional in
  this stage.
- `NormFirstMessages` and `NormMaskMessage` encode/decode and trusted-x checks;
  the second-message decode path for `NormPlan`.
- `HiddenLinPlan`, `HiddenSendMessages` and `HiddenSecondMessages` carry an
  optional norm envelope. The layout is conditional on the plan carrying the
  norm; the norm-free path writes no new bytes.
- Constructors and external plan literals set the norm to absent.

## Results

- `cargo test --workspace`: 48 suites, 821 passed, 0 failed, 17 ignored.
- Focused `hidden_norm_psc`: 6 passed, 0 failed.
- Proof-byte hard gate: the toy output-head proof with the norm absent is
  45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- The integrated first-message round-trip test covers trusted x roots, witness
  group declarations and the second-message decode path.

## Explicitly deferred

- Adding the norm groups as proximity members and opening them (Stage B).
- Product-sumcheck endpoints and BAT/FOLD/CS/BASE rows (Stage C).
- Committed mask entries and the negative matrix (Stages C/D).
- Mean/variance and the four-relation close (Stage E).
- No soundness, privacy or complete-norm claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
