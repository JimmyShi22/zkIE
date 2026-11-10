# lin-session - 66b5a30a / f71cf391

Trace of the isolated candidate LIN claim session wiring.

## Rework after review

Rework code commit `9839be1b`, based on `6da2ba4c`. Expert review
returned three corrections; this section supersedes the matching text below.

- F1: the component Eval and Dot checks now receive the fixed, authenticated W
  vector. `lin_eval_holds_with` and `lin_dot_holds_with_w` are added as variants;
  the existing helpers are unchanged. The verifier passes
  `messages.w.values` directly. Tests show that a W message with a valid
  commitment but wrong values fails the component Eval and Dot checks
  themselves; `control_w` is run separately and remains the full-value
  protection.
- F2: the invalid bit-bound constants and outcome fields are removed. The
  documentation records only that the base field has `p < 2^64`, `Fp3` is a
  cubic extension and `|Fp3| = p^3 < 2^192`. No soundness bound is given: a
  multivariate nonzero-polynomial bound needs its degree and challenge
  distribution, and Omega is fixed after the shared challenge, so no
  before-fixed bound applies. The trace records only the field description and
  the message order.
- F3: the component and control layers are separate functions
  (`check_lin_components`, `check_lin_control`) so tests can see which layer
  rejects a tamper before the combined entry returns. The negative table below
  separates component relation checks, the deterministic full-value control,
  and authentication/wiring checks.

Rework logs: `tests/r2-lin-session-unit.txt`,
`tests/r2-lin-session-real.txt` and `logs/cargo-test-r2.txt`.

## Change

Code commit `66b5a30a` ("Wire the candidate LIN claim through existing session
primitives"), based on `f71cf391`. The production verify entry is unchanged.

- New isolated module `crates/zkie-ops/src/proof/lin_session.rs` and one
  real-artifact integration test in `zkie/tests/lin_session.rs`.
- This is component-level LIN claim wiring. It is not a proof verifier and its
  result must not be used to accept a proof.
- The authenticated full-recompute isolation control from the selected-step
  baseline is retained and runs alongside this demonstration in the integration
  test.
- The table is built only after the trusted model anchor and the asset opening
  are verified. The candidate Eval and Dot checks receive the submitted,
  authenticated W vector directly; the candidate Fold check receives the
  submitted Omega and folded vectors. A separately named row-wise control
  recomputes W and Omega and compares every value.
- The binding is assembled by the existing binding candidate. The candidate
  protocol identity and request parameters are demonstration placeholders, not
  real protocol values.
- The upstream graph source `x` is not closed: the openings show membership in
  the commitments absorbed here, not that those commitments are the output of a
  trusted graph predecessor.

The FOLDEQ `draw_queries` helper was reviewed. It draws column positions and
requires a full FOLDEQ plan with a reader, a verified statement and runtime
oracles; this isolated claim has no such plan. The shared and later extension
points therefore use the existing chain draw interface, and no parallel
generator is introduced.

## Selected object

The real-artifact test uses the GPT-2 sequence-16 final MatMul step `4444` and a
table slice of the authenticated `lm_head_w_i32.bin` asset:

| field | value |
| --- | --- |
| table shape | `matrix_rows=1024`, `k_lo=0`, `k_col=6`, `table_rows=64`, `table_columns=1024` |
| table columns | first 64 output columns of the asset |
| weights | step input row 0, 1024 values |
| W message | step output row 0, first 64 values |
| claim point | 6 fixed extension-field values |
| target | multilinear evaluation of the W message at the claim point |
| Omega | 1024 extension-field values, length `2^k_row` |
| folded rows | set equal to Omega in this fixture |
| later challenge | 10 extension-field values, length `k_row` |

## Causal stages

| stage | implementation |
| --- | --- |
| 1. bind parameters and W | the existing binding candidate starts the chain; the claim digest, shape and W commitment are absorbed; W is fixed before the shared challenge |
| 2. shared challenge | a real extension-field point over the six column variables, drawn from the chain after stage 1 |
| 3. Omega and folded message | Omega is computed from the authenticated table and shared point, then Omega and folded commitments and values are fixed and absorbed |
| 4. later challenge | the ten-coordinate extension point is drawn after stage 3; the Fold and Dot checks use it and the earlier shared point |

The prover fixes W before stage 2 and Omega/folded before stage 4. The verifier
reconstructs the same stages from the same binding and messages. The fixed test
point values are test-only and are not a production seed.

The base field has `p < 2^64` and `Fp3` is a cubic extension, so
`|Fp3| = p^3 < 2^192`. This demonstration gives no soundness bound: a
multivariate nonzero-polynomial detection bound needs its degree and the
challenge distribution, and Omega is fixed after the shared challenge, so a
bound fixed before that point does not apply. The trace records only the field
description and the message order: W is fixed before the shared challenge;
Omega and folded are fixed before the later challenge.

## Checks and value sources

| check | call | value source |
| --- | --- | --- |
| authenticated table | trusted anchor verification plus asset-opening Merkle check, row order and width | authenticated asset opening |
| candidate Eval | `lin_eval_holds_with` | submitted W is used directly; weights, claim point and target are self-reported |
| candidate Dot | `lin_dot_holds_with_w` | submitted W and Omega are used directly; weights are self-reported |
| candidate Fold | `lin_fold_holds` | Omega and folded rows are self-reported; folded equals Omega in this fixture, so this demonstrates wiring and a local equality, not an independent fold authentication |
| row-wise control | exact vector comparison of recomputed W and Omega against the self-reported messages | deterministic full-value comparison |

The W, Omega and folded openings only show membership in the absorbed
commitments. They do not show that the commitments are outputs of a trusted
graph step. The claim point, target, weights and messages are self-reported.

## Negative coverage and rejection sources

Every negative case returns `Err`; none panic. The source of the rejection is
recorded so a random relation check is not presented as a deterministic full
protection.

| negative | component Eval/Dot/Fold | control layer | other source |
| --- | --- | --- | --- |
| table opening root, row value or path | not reached | not reached | trusted anchor / Merkle opening check, deterministic |
| W message with a valid commitment | Eval and Dot fail on the submitted W | `control_w` fails | component relation checks; control is a separate deterministic full-value check |
| Omega message with a valid commitment | Dot and/or Fold fail for the tested change | `control_omega` fails | component relation checks plus deterministic control |
| folded message with a valid commitment | Fold fails for the tested change | control passes | component relation check only; no claim that every folded tamper is always rejected |
| claim target | Eval fails | control passes | component Eval check |
| claim weights | Eval passes because submitted W is unchanged; Dot fails | `control_w` fails | component Dot plus deterministic control |
| W commitment root replacement | not reached as a valid message | not reached | commitment and challenge wiring, deterministic |
| message field order | not reached as a valid message | not reached | opening order and commitment checks, deterministic |

The tests do not claim that all tampering is necessarily rejected. Component
Eval, Dot and Fold are relation checks; their detection probability depends on
the degree, the challenge distribution and the fixed message order, and this
trace does not derive a soundness bound. The row-wise control and the
Merkle/anchor checks are the deterministic full-protection parts.

## Tests

- 9 synthetic unit tests: reproducible fixed test values, independent prover
  and verifier agreement, table opening root/row/path, and layered
  component/control assertions for W, Omega, folded and claim tampering.
  The W test shows the component Eval and Dot checks fail on the submitted W
  with a valid commitment; the folded test shows the component Fold check fails
  while the control passes.
- Real gpt2-seq16 test: authenticated full-recompute control passed in 1,677 ms;
  the isolated LIN session passed in 129 ms; the whole test took 8.83 s.
- Workspace: 408 passed, 0 failed, 4 ignored (the earlier multi-GiB tests and
  the two real-artifact session tests).

## Validation scope

Only the `gpt2-seq16` case is used. The object is one GPT-2 model and one
relation; the selected MatMul logic is generic in the sequence dimension and
the production path is unchanged. No five-case three-stage run was made. No
claim is made about the full model or performance from this single case.

## Unclosed items

- x upstream source: the commitments are not tied to a trusted previous graph
  step.
- y downstream graph binding: no downstream edge is bound.
- FOLDEQ authenticated openings: folded rows are self-reported.
- Commitment consistency: W, Omega and folded commitments are only locally
  absorbed.
- Proximity: not connected.
- Real prefix: no real full protocol prefix is used.
- Graph edges: not connected.
- Hiding: not claimed.
- Single-shot: not claimed.
- Production wiring: the production verify entry is not switched.

## Artifacts

- `tests/lin-session-unit.txt`: the nine module tests.
- `tests/lin-session-real.txt`: the real gpt2-seq16 test.
- `logs/cargo-test.txt`: full workspace log.
- No binary artifacts were produced or deleted.
