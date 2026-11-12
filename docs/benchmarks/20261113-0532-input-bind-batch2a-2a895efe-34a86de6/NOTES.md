# input-bind batch 2a: hidden norm core interface and integer checks

## Scheme and baseline

- Scheme: `input-bind-batch2a` (internal split 2a of batch 2).
- Code: `2a895efe` (parent `34a86de6`), branch `refactor/v2`.
- Scope: the externally trusted input commitment and output interface
  commitment of one norm, plus the exact integer core of every position:
  valid-column sum, round-half-up mean and remainder, centered squares,
  round-half-up variance and remainder, and canonical base-256 digit rows
  for the sum magnitude, squares and variance.
- This slice is prover/fixture side. It does not yet prove the relation to
  the committed input in a hidden verifier, does not carry the full carry
  equations, and does not implement the rsqrt table, weight product,
  shift/bias/clamp or the MatMul consumer. Those are later slices; 2a is not
  a batch-2 acceptance result.

## Changes

- `crates/zkie-ops/src/proof/hidden_norm.rs`: `NormShape`,
  `NormCommitments`, `NormCorePlan`, `NormCoreRow` and `NormCoreWitness`;
  exact integer checks and toy fixtures.
- `crates/zkie-ops/src/proof/mod.rs`: module registration.
- Tests: positive exact core, wrong input/output commitment, tampered mean,
  digits, remainder and sum digits, invalid shape/root.

## Validation

- Command: `cargo test --workspace`.
- Result: 48 test suites, 808 passed, 0 failed, 17 ignored.
- Text result: `workspace.txt`; summary: `test-summary.txt`; change stat:
  `change.stat`.
- No proof artifact, no benchmark and no performance claim in this slice.

## Boundaries

- No hidden verifier binding of the integer core; the witness self-check is
  prover/fixture side.
- No full carry equations, rsqrt index/table lookup, weight product,
  shift/bias/clamp or MatMul consumer.
- No push; origin remains `4959cae2`.
