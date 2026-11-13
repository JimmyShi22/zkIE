# Shared relation inputs materialized from the trusted plan

## Scheme and baseline

- Scheme: `proof-size-shared-inputs`.
- Code: `e78bc2e6`; previous commit `740505eb`.
- Flag: `[proof-size]`. Interval 2 of the A work package: relation inputs are
  no longer committed and opened per step; the verifier materializes every
  input from the trusted plan source (previous verified output, verified asset
  opening or public token/layout) and rejects the old input-opening form.

## What changed

- `prove_op_relation_shared` (the trace entry) commits no input oracle and
  writes an empty input group per port; lengths still describe the inputs.
  `prove_op_relation_with_inputs` keeps the committed-input form for the
  request path and the selected-step session, and `prove_op_relation` uses it.
- `plan::verify_step` now returns the materialized input values per port and
  keeps the plan identity checks: a flat previous edge must name the same
  producer and output, an asset port must match the planned asset identity,
  and other sources carry no edge or external pool. No prover-supplied input
  value enters the equation.
- The shared-trace verifier feeds the materialized values into the relation
  equation as the per-port input vector; the old input-opening comparison is
  gone.
- Version gate: `plan::SHARED_TRACE_INPUT_VERSION = 2`; a step that still
  carries an input opening is rejected with `unsupported_version` (version 1
  form). The wire format of `SharedTrace` does not carry a version field, so
  the verifier semantics are the gate; the plan digest is not changed because
  it feeds the model parameters digest and the anchor identity.
- The selected-step session (`session_baseline`) keeps its committed-input
  form and now builds its fixture with `prove_op_relation_with_inputs`; it is
  a separate session, not the shared production trace.

## Results

Same machine, `zkie matrix` 1 cold + 3 warm, production seq16 cases. Before is
the interval-1 state (`740505eb`, accepted), after is this commit.

| Case | proof bytes before → after | Delta | Prove warm median ms before → after | Prove HWM median KiB | Verify before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 3,564,380,854 → `2,190,159,492` | `-1,374,221,362 (-38.56%)` | 20,402 → `14,349` | 7,693,320 → `4,488,832` | harness 21.32 s / 8,282,372 KiB → 18.18 s / 6,237,928 KiB; CLI 109,295 ms / 4,782,100 KiB → 72,047 ms / 4,118,628 KiB |
| gemma3-seq16 | 6,770,333,437 → `6,223,297,337` | `-547,036,100 (-8.08%)` | 39,739 → `35,391` | 15,372,840 → `12,718,828` | harness 31.65 s / 17,959,744 KiB → 30.80 s / 17,047,776 KiB; CLI (before not measured) → 182,921 ms / 10,966,596 KiB |

- Delta against the estimates: gpt2 -1,374,221,362 vs estimate 1,373,766,752
  plus the input root entries 454,608 (2 B round-off); gemma3 -547,036,100 vs
  estimate 546,827,232 plus 208,872 (4 B round-off). The measured bytes include
  the removed input root entries that the estimate listed separately.
- The before verify column is the interval-1 proof under the interval-1
  verifier (harness) and the budget-fixed CLI; the after verify column is the
  interval-2 proof under the interval-2 verifier.
- Independent verifier (same entry points, external harness): the new proofs
  are accepted and the argmax equals the ground truth; the interval-1 proof is
  rejected with `unsupported_version` (old-form gate) through both the harness
  and the CLI.
- `cargo test --workspace`: 48 suites, 973 passed, 0 failed, 17 ignored,
  exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  `cmp` identical (`proof-bytes.txt`).

## Negative matrix

- Old-form proof (input opening present) rejected: `unsupported_version`.
- Materialized input differs from the step's input: relation equation fails.
- Changed previous edge, missing edge, changed external port and extra port:
  rejected by the plan identity checks.
- Changed asset opening row, step root, opening position/order, output value:
  rejected by the existing authentication and equation checks (plan check and
  session baseline suites).

## Four-column mapping

| Requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| Shared input references the same authenticated object | inputs materialized from verified previous outputs, verified asset openings and public tokens; plan identity checks kept | positive matrix; wrong-input, wrong-edge, asset and position negatives | the plan is trusted; the selected-step session keeps its own committed-input form |
| Old proof form rejected | input groups must be empty; version constant 2 | harness and CLI reject the interval-1 proof with `unsupported_version` | the plan digest/anchor is not changed (it feeds model identity) |
| Every step and port checked | per-port materialization and equation for every step | full matrix and plan-check negatives | — |
| Bytes do not grow, HWM and wall time within bounds | input openings removed | before/after table | — |
| Hard gate and workspace | no change to the norm-free toy path | 45,080 bytes identical; 48 suites green | — |

## Scope and next

- The seq512 cases are not re-run for interval 2; the parsed statistics give
  the expected interval-2 savings there, and the final A acceptance runs the
  full matrix.
- No push until TL review.

## Artifacts

- `workspace.txt`, `after-matrix.json`, `verify-after.txt`, `old-form.txt`,
  `proof-bytes.txt`, `change.stat`, `NOTES.md`; the matrix proof tree stays on
  disk with binaries gitignored.
