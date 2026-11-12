# input-bind batch 1: plan-level previous object

## Scheme and baseline

- Scheme: `input-bind-batch1`.
- Code: `12f655cc` (parent `4959cae2`), branch `refactor/v2`.
- Scope: the trusted plan now declares a previous object for the output-head
  segment: the input matrix of the selected final norm, produced by an earlier
  plan step and consumed by the output-head path. The object carries an owner
  identity derived from the producer plan step, a positions x hidden shape and
  a flat row/column layout. The verifier derives the same object from the plan
  and compares the declaration; no proof message contributes.
- No proof relation, commitment object, challenge order or public/private
  boundary is changed. The upstream subproof is not hidden in this batch; the
  object is an interface-level plan value.

## Changes

- `crates/zkie-ops/src/proof/plan.rs`: `PreviousObjectPlan` and its owner,
  shape and layout parts; plan-derived owner/object identities; plan
  declaration and validation.
- `crates/zkie-ops/src/proof/output_head_lin.rs`: the selected output-head
  step carries the validated previous object; selection derives and checks it
  before returning.
- `models/gpt2/src/plan.rs`: the gpt2 trace plan declares the object for the
  final norm, for both 16- and 512-position plans.
- Tests: the toy output-head plan builds a producer, final norm and MatMul
  chain and declares the object; negatives reject a wrong owner, shape,
  layout, object identity, consumer port, missing declaration and a swapped
  owner step. The real gpt2 plan test derives and checks the object for
  `trace_plan(16)` and `trace_plan(512)`.

## Validation

- Command: `cargo test --workspace`.
- Result: 48 test suites, 804 passed, 0 failed, 17 ignored.
- Text result: `workspace.txt`; summary: `test-summary.txt`; change stat:
  `change.stat`.
- No benchmark and no proof artifact are part of this batch; no performance or
  proof-byte number is claimed.

## Boundaries

- The upstream subproof is still plaintext; the previous object is not yet a
  hidden commitment or a proof-carrying object.
- The final norm hidden relation and the same-segment consumer connection are
  later batches.
- No push in this batch.
