# shared-fold-binding

- Scheme: shared-fold-binding (B3-2: the shared fold and Omega relations and
  their shared proximity test, with the shared point drawn only after the
  binding).
- Based on: shared-column-auth (cd6d3a42); previous baseline commit cd6d3a42.
- Purpose: record the coverage of the shared folding relations and the shared
  test on both sides, and pin the binding order: the request first part ends in
  its seal, the shared chain is bound to that seal and to the plan statement,
  and only then is the shared point drawn.
- Change: tests only, no protocol change.
  - The Fold row is tested directly: the folded `G_u` term and the `Omega`
    term both enter it, the target is zero, another shared point or relation
    point changes the coefficients, and a short point is refused. The shared
    point, `Omega`'s relation point and both messages are therefore one
    relation, not a caller-supplied value.
  - Shared transcript negatives: a changed first seal (the binding), a
    changed fold-start value, a changed fold round, a changed shared
    proximity tail and a changed shared seal are refused. The fold and Omega
    relations and the shared proximity test were already executed on both
    sides; this group records their coverage.
- Binding note: the single-request session binds the shared chain to the one
  request seal and to the plan statement in `shared_start_chain`; the batch
  binding tree over several request seals and its place count are a later
  batch feature, not part of this single-request session.
- Numbers:
  - focused: `the_shared_` 4 passed; chain segment 1 passed; norm shared
    carrier 1 passed
  - workspace: 1114 passed / 0 failed over 51 suites
  - hard gate: 45,080 B, sha256
    7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
- Boundaries and memo items (not claimed closed): B3-2 is the second group of
  B3; the ledger and budget wiring, the message order, and the parameters and
  negatives groups are pending. The full diagnostic entry still fails with the
  placeholder claim and zero maps; real claim/maps derivation, kept owner,
  the weight unit class and the shared-forest lookup shape keep their expert
  memo boundaries. Local evidence only, not R5. Not pushed.
