# send-flow - fcdacbb1 / 3162adef

Trace of the single-oracle fixture SEND answer data flow.

## Change

Code commit `fcdacbb1` ("Add the fixture SEND answer data flow"), based on
`3162adef`. The production verify entry is unchanged.

- New `send_session` module provides a fixture SEND answer provider. It computes
  the complete ordered answers from the actual session G_u shape, the
  independent seed set and a plaintext witness, using the sample-form coefficient
  row and pad positions over the extension field.
- The verifier receives those answers and registers them as pending sample
  claims. It has no standard-answer comparison path and does not call the
  provider's plaintext oracle.
- The pre-FOLD flow accepts a caller-provided answer function after the seed
  draw. No challenge is drawn between the seed and the answers.
- The fixed root cannot be replaced after the seed; the session stage rejects a
  second `send_gu`.
- The same continuous chain is transferred to BAT; the sample declaration is in
  the BAT frozen input. BAT does not judge truth.
- The existing 6-to-8 mask adapter and adapter-only isolation flags remain.

All earlier packages remain retained unchanged.

## Sample form and answer construction

| item | source |
| --- | --- |
| oracle shape | the actual session plan (`GuShape`) |
| seed | the session's independent `shared_seeds` draw after the fixed root |
| sample form | `sample_row` coefficient row with main/mask cube and pad positions |
| witness | local plaintext fixture G_u table (auxiliary test oracle) |
| answer | evaluation of the sample-form row on the fixture witness, in extension-field arithmetic |

The deterministic public pads are test-only values and do not imply hiding. The
plaintext witness and its arithmetic are an auxiliary test oracle, not the
verification path.

## Verifier behavior

The verifier calls `receive_gu_answer` with each provided answer and registers
the corresponding pending sample claim. It does not compute or compare a
standard answer. The outcome records:

- `sample_answers_provider_constructed = true`;
- `verifier_standard_answer_comparison = false`;
- the sample declaration target equals the provider answer;
- the sample declaration appears in the BAT frozen input and is still pending.

A syntactically valid wrong answer is registered as pending and enters BAT; it
is not rejected or self-certified. Without a later proximity/relation check no
truth decision is made.

## Order and commitment

The fixed order is:

1. fixed oracle commitment;
2. independent seed draw;
3. complete ordered answer list, absorbed in order;
4. sample claims registered;
5. record/column joins;
6. same continuous chain transferred to BAT;
7. BAT group frozen and `x` drawn.

The engine test `send_root_cannot_be_replaced_after_seeds_are_drawn` shows a
second `send_gu` after the seed returns `InvalidSessionState`. Truncated or
over-long answer lists fail closed before registration (`ShapeMismatch`). The
answer order is the session's fixed answer index order.

## BAT input

The real frozen BAT input is:

| item | value |
| --- | --- |
| order | plain, plain, plain, plain, plain, sample |
| count | `6` |
| oracle/group | `0` / `0` (G_u `11/5` retained) |
| sample answer | `Fp3([5024530528705913004, 14485199157231542173, 15335975921221294098])` |
| sample target in BAT | the same value, last frozen target |

BAT only combines this pending declaration. The result is not a truth check and
is not connected to a real FOLD/proximity path.

## Tests and measurements

- G_u combine unit: 7 passed.
- GroupRow unit: 6 passed.
- Fold projection unit: 4 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 40, column 9, edge 2.
- Real gpt2-seq16 SEND data flow: 1 passed (55.95 s test, 56.79 s wall, VmHWM
  4,239,348 kB; send session 366 ms).
- Workspace: 458 passed, 0 failed, 11 ignored.
- No proof bytes were produced; no performance gain, hiding or soundness bound
  is claimed.

The real test also checks:

| case | result |
| --- | --- |
| provider math vs auxiliary plaintext oracle | equal |
| sample declaration target vs provider answer | equal |
| sample declaration in BAT frozen input | present, last target |
| syntactically valid wrong answer | registered pending; combined target changes; no rejection |
| truncated answer list | `ShapeMismatch` before registration |
| commitment replacement after seed | `InvalidSessionState` (engine test) |

The full asset opening verification and table build are included in the test
cost; this is a single `gpt2-seq16` case and is not a production performance or
memory measurement.

## Validation scope

Only `gpt2-seq16` is used. The G_u root, witness and SEND answers are fixtures.
The three seq16 cases beyond the dependency case were not run because this step
does not change the shared three-stage path or seq512 logic. No five-case
three-stage run was made.

## Unclosed items

- Real G_u authenticity and a real SEND provider.
- Multi-oracle SEND.
- Random pads, hiding and a security bound; current pads are deterministic test
  values.
- Complete FOLD and proximity checks for the pending declarations.
- A real FOLD/proximity connection for the adapter-only BAT result.
- A valid 3c FOLD-to-CS/BASE handoff; `handoff_statement` still rejects.
- Fresh-mask handoff and folding-tail derivation.
- Single-shot semantics.
- Production wiring and performance acceptance; `verify_request` unchanged.
- Any declaration not listed above.

## Artifacts

- `tests/send-flow-real.txt` (local fixture path replaced with `<fixture>`)
- `tests/gu-combine-unit.txt`
- `tests/group-row-unit.txt`
- `tests/fold-handoff-unit.txt`
- `tests/proximity-session-unit.txt`
- `tests/regression-foldeq.txt`
- `tests/regression-column.txt`
- `tests/regression-edge.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.

The 3e package remains retained unchanged. This is a new frozen evidence set;
old packages are not deleted.
