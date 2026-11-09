# relation-session - 642aacb3 / cc6bd9b0

Trace of the isolated selected-step MatMul session baseline.

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
- The verifier builds candidate parameters from the public model identity and a
  candidate statement from the trusted plan, starts the existing chain, and
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

The positive run takes 1,749 ms inside the release test; the whole test,
including the fast real-artifact negatives, takes 11.29 s.

## Negative coverage

All negative cases return `Err`; none panic.

| negative | coverage |
| --- | --- |
| input value with valid Merkle path | `authenticated_wrong_input_or_output_or_raw_is_rejected` |
| output value with valid Merkle path | same test, output target |
| raw value with valid Merkle path | same test, raw target |
| asset value with valid Merkle path | `authenticated_wrong_asset_value_is_rejected` |
| root | `plan_port_asset_and_root_tampering_is_rejected` |
| plan config, opcode, bias and dimensions | `malformed_plan_dimensions_are_rejected`, `plan_port_asset_and_root_tampering_is_rejected` |
| source edge and external map | `plan_port_asset_and_root_tampering_is_rejected` |
| opening index and order | `opening_position_and_malformed_shapes_are_rejected` |
| every checked position, first and last | `any_opening_position_is_checked` |
| quantization and rounding | `rounded_output_mismatch_is_rejected` |
| truncated or extra openings and values | `opening_position_and_malformed_shapes_are_rejected` |
| malformed counts, lengths and element windows | `opening_position_and_malformed_shapes_are_rejected`, `malformed_plan_dimensions_are_rejected` |
| real artifact config, source, opening swap, root and asset value | real-artifact integration test |

The real-artifact test also rejects a forged selection with a changed config
and rejects selecting a non-MatMul step or an unsupported non-flat input.

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

The selected-step checks cover only this step. Upstream step outputs, the
complete model graph, other steps and external weight authorization beyond the
selected asset opening are not connected. A successful run does not show that a
real challenge used this candidate binding, does not authenticate the source of
the previous graph edge, and does not prove relation soundness, hiding or
single-shot semantics. Hash authentication and the surrounding system keep
their own assumptions, so no probability-free security claim is made.

## Validation scope

Only the `gpt2-seq16` case is used. The object is one GPT-2 model and one
relation; the selected MatMul logic is generic in `m` and has no
sequence-512-specific branch. Sequence 512 only scales the same dimensions and
output size, and the production path is unchanged, so the single sequence-16
case is sufficient for this isolated wiring baseline. No five-case three-stage
run was made.

## Workspace gate

`cargo test --workspace` on `642aacb3`: 396 passed, 0 failed, 3 ignored (the two
multi-GiB byte-consumption tests and the real-artifact session test above).
Logs: `logs/cargo-test.txt`, `tests/session-baseline-unit.txt` (11 tests) and
`tests/session-baseline-real.txt`.

## Artifacts

- `tests/session-baseline-unit.txt`: the eleven synthetic module tests.
- `tests/session-baseline-real.txt`: the real gpt2-seq16 selected-step test.
- `tests/injection-*.txt`: two fail logs and two restore logs.
- `logs/cargo-test.txt`: full workspace log.
- No binary artifacts were produced or deleted.
