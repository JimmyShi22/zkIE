# relation-session - 642aacb3 / cc6bd9b0 (rework 1486e2d3)

Trace of the isolated selected-step MatMul session baseline.

## Rework after review

Rework code commit `1486e2d3`, based on `1acfb0ec`. Expert review returned two
items; this section supersedes the matching text below.

- F1: the isolated entry now takes trusted model metadata (anchor id, model id,
  parameters digest and asset specifications) and calls the existing
  `verify_model_anchor` before selecting the step or using any asset opening.
  The trusted values come from the same policy and static plan source used by
  production. A forged `meta.root` with intact leaf/link records is rejected.
  This establishes internal consistency with the chosen trust assumption; it is
  not the deferred external weight authorization decision.
- F2: parameters, statement and record now come from the existing binding
  candidate. `BindingParameters::with_plan_parts` receives the proof-header
  version word, a candidate protocol identity constant, a candidate request
  parameter set and the selected-step plan digest as the plan identifier.
  `BindingStatement::new` receives the same version and protocol identity, the
  plan digest bytes as the case id, the plan digest as the request id, and the
  trusted anchor id as the table identity. `assemble` produces the parameters
  digest, statement bytes and record bytes; the existing absorption,
  `Chain::begin`, `step` and `draw` wiring then uses those outputs. Tests assert
  that the outcome equals the assembled fields and that the assembled first
  query equals a fresh `Chain::begin` over those fields.

Rework logs: `tests/r2-session-baseline-unit.txt`,
`tests/r2-session-baseline-real.txt`, `tests/r2-injection-*.txt` and
`logs/cargo-test-r2.txt`.

## Change

Code commit `642aacb3` ("Verify one selected MatMul step through an absorbed
session baseline"), based on `cc6bd9b0`. The production verify entry is
unchanged. This is a replaceable selected-step baseline, not a production
verifier, not a compression or challenge-driven proof, and not a relation
soundness claim.

- New isolated module `crates/zkie-ops/src/proof/session_baseline.rs` and one
  real-artifact integration test in `zkie/tests/session_baseline.rs`.
- The trusted plan selects one MatMul step by index. The verifier checks the
  step opcode, config, input/output port counts, source edges and external
  asset maps against that plan; a forged selection or a different step is
  rejected. Only flat previous outputs and flat asset inputs are supported by
  this baseline.
- The entry first verifies the trusted model anchor with the existing
  `verify_model_anchor`; only then does it select the step and use asset
  openings. Trusted anchor/model values come from the policy and the parameters
  digest and asset specifications come from the static plan, the same source
  used by production.
- Parameters, statement and record come from the existing binding candidate:
  the plan identifier is the selected-step plan digest, the proof-header
  version word and a candidate protocol identity are used, and `assemble`
  produces the three values consumed by `Chain::begin`. The verifier then
  absorbs the step's commitment references and full plaintext opening content
  through the existing chain message and `step`/`draw` interface. The challenge
  is drawn only after absorption. No bare prefix call is used.
- The same immutable step and asset opening references then feed Merkle
  authentication and the full recomputation through
  `verify_op_relation_with_external`. Values are not re-read or replaced after
  absorption.
- The selected step's oracle openings and external asset openings are checked
  for identity, root, start, row count, row order, width, coverage and exact
  correspondence to the required ports. Every registered root must be used.
- Malformed group counts, dimensions, row ranges, bias shape, intermediate
  lengths and trailing values are rejected before the relation helper is
  called; the verifier does not panic on these inputs.
- The field order, messages, challenge position and plaintext encoding are
  candidate-internal and replaceable. A later production closure will change
  them.

## Selected step and ports

The real-artifact test selects GPT-2 sequence-16 plan step `4444`:

| field | value |
| --- | --- |
| opcode | MatMul |
| config | `[16, 65536, 1024, 0, 0]` |
| input 0 | previous step `4443`, output `0`, flat |
| input 1 | asset `gpt2/weights/lm_head_w_i32.bin`, flat, external pool `192`, asset id `192` |
| output 0 | one tensor |
| relation lengths | inputs `[16384, 67108864]`, outputs `[1048576]`, raw `[1048576]` |
| roots | 3 |

The positive run takes 1,728 ms inside the release test; the whole test,
including the fast real-artifact negatives, takes 11.23 s.

## Negative coverage

All negative cases return `Err`; none panic.

| negative | coverage |
| --- | --- |
| input value with valid Merkle path | `authenticated_wrong_input_or_output_or_raw_is_rejected` |
| output value with valid Merkle path | same test, output target |
| raw value with valid Merkle path | same test, raw target |
| asset value with valid Merkle path | `authenticated_wrong_asset_value_is_rejected` |
| forged `meta.root` with intact leaf/link records | `forged_asset_meta_root_is_rejected_before_use` |
| forged trusted anchor identity | `forged_trusted_anchor_identity_is_rejected` |
| root | `plan_port_asset_and_root_tampering_is_rejected` |
| plan config, opcode, bias and dimensions | `malformed_plan_dimensions_are_rejected`, `plan_port_asset_and_root_tampering_is_rejected` |
| source edge and external map | `plan_port_asset_and_root_tampering_is_rejected` |
| opening index and order | `opening_position_and_malformed_shapes_are_rejected` |
| every checked position, first and last | `any_opening_position_is_checked` |
| quantization and rounding | `rounded_output_mismatch_is_rejected` |
| truncated or extra openings and values | `opening_position_and_malformed_shapes_are_rejected` |
| malformed counts, lengths and element windows | `opening_position_and_malformed_shapes_are_rejected`, `malformed_plan_dimensions_are_rejected` |
| real artifact config, source, opening swap, root, meta root and asset value | real-artifact integration test |

The real-artifact test also rejects a forged selection with a changed config
and rejects selecting a non-MatMul step or an unsupported non-flat input. The
binding test asserts that the outcome parameters, statement bytes and record
bytes equal the assembled candidate fields, and that the candidate first query
equals a fresh chain start over those fields.

## Transcript checks and injections

`prover_and_verifier_derive_the_same_challenge` derives the transcript twice
from the same absorbed words and compares the result. The two tamper tests
change an absorbed opening value and an absorbed asset value, assert that the
absorbed words differ, and assert that both derivations of the tampered message
agree. They do not assert that every discrete tamper changes the final
challenge, because a hash collision is possible.

| removed absorption | failing test | log |
| --- | --- | --- |
| oracle opening values | `changing_an_absorbed_input_value_changes_the_words` | `tests/injection-opening-values-fail.txt` |
| external asset row values | `changing_an_absorbed_asset_value_changes_the_words` | `tests/injection-asset-values-fail.txt` |

Restore logs use the same names with `-restore.txt`. The injection only
exercises the transcript side: the arithmetic recomputation is decoupled, and
this trace does not claim that removing absorption makes the arithmetic check
accept a wrong relation.

## Boundary

The selected-step checks cover only this step. The anchor check establishes
internal consistency with the trusted anchor, model, parameters and asset
specification metadata; it does not decide external weight authorization, which
remains deferred. Upstream step outputs, the complete model graph and other
steps are not connected. A successful run does not show that a real challenge
used this candidate binding, does not authenticate the source of the previous
graph edge, and does not prove relation soundness, hiding or single-shot
semantics. Hash authentication and the surrounding system keep their own
assumptions, so no probability-free security claim is made.

## Validation scope

Only the `gpt2-seq16` case is used. The object is one GPT-2 model and one
relation; the selected MatMul logic is generic in `m` and has no
sequence-512-specific branch. Sequence 512 only scales the same dimensions and
output size, and the production path is unchanged, so the single sequence-16
case is sufficient for this isolated wiring baseline. No five-case three-stage
run was made.

## Workspace gate

`cargo test --workspace` on the rework head: 399 passed, 0 failed, 3 ignored
(the two multi-GiB byte-consumption tests and the real-artifact session test
above). Logs: `logs/cargo-test-r2.txt`,
`tests/r2-session-baseline-unit.txt` (14 tests) and
`tests/r2-session-baseline-real.txt`.

## Artifacts

- `tests/session-baseline-unit.txt`: round 1, eleven synthetic module tests.
- `tests/session-baseline-real.txt`: round 1 real selected-step test.
- `tests/injection-*.txt`: round 1 fail and restore logs.
- `logs/cargo-test.txt`: round 1 workspace log.
- `tests/r2-session-baseline-unit.txt`: rework, fourteen synthetic module
  tests.
- `tests/r2-session-baseline-real.txt`: rework real selected-step test.
- `tests/r2-injection-*.txt`: rework fail and restore logs.
- `logs/cargo-test-r2.txt`: rework workspace log.
- Round 1 logs are retained; the r2 logs are the current evidence.
- No binary artifacts were produced or deleted.
