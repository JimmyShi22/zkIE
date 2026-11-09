# accum bounds - bfb0db46 / 38bf8b21

Arithmetic-bound closure in the relation verifier, with a three-case repeat
verification.

## Change

- ProfileRmsNorm and ProfileLayerNorm now check every intermediate square and
  sum. ProfileStandardize, LayerNorm and RmsNorm use the same checked pattern.
- BatchedMatMul, GatherMatMul and CausalAttention check every product term and
  accumulator; MoeDown checks both routed sums and their total. Matrix
  multiplication already used the checked shared kernel.
- RopeProjection now requires the decoded raw length to equal the input
  length.
- New boundary fixtures use valid openings and assert `bound_violation` or
  `shape_mismatch`: ProfileRmsNorm with two `i64::MIN` squares, ProfileLayerNorm
  with centered squares that exceed the accumulator, RopeProjection with a
  short raw group, and the BatchedMatMul, GatherMatMul and CausalAttention
  accumulators.

## Gate

`cargo test --workspace`: 143 passed, 0 failed. Log: `logs/cargo-test.txt`.

## Audit and reachability

The trusted bounds are the plan-provided opcode, config, port counts and
dimensions; proof lengths are checked against dimensions derived from them.
Values are proof-supplied i64 with no magnitude promise, so value products and
accumulations are checked and fail closed.

A scan of the model plan modules shows the opcodes that the three models
actually build steps for:

- gpt2: Add, AffineScale, Argmax, EmbeddingIndex, MatMul, ProfileLayerNorm,
  Projection, Softmax, SoftmaxRows, TableIndex, TableLookup.
- gemma3: Add, AffineScale, Argmax, GatedFfn, MatMul, ProfileRmsNorm,
  ProfileRope, Projection, RowEmbeddingIndex, Softmax, SoftmaxRows, TableIndex,
  TableLookup.
- timesfm: Add, AffineScale, Hadamard, MatMul, ProfileLayerNorm,
  ProfileRmsNorm, ProfileStandardize, Projection, Relu, Softmax, TableIndex,
  TableLookup.

RopeProjection, CausalAttention, BatchedMatMul, GatherMatMul, LayerNorm,
RmsNorm, Layout and the other generic branches are not built by any of the five
plans; they are exercised only through the generic relation entry and the
fixture-level tests. A name in a configuration or executor module does not make
a branch reachable. A gap in a generic-entry branch would not by itself
demonstrate an attack on the five registered cases, and this trace does not
claim such an attack.

## Case selection

The change is arithmetic hardening in the shared relation verifier. The three
seq16 cases exercise every changed branch that the plans can reach: gpt2 uses
ProfileLayerNorm, gemma3 uses ProfileRmsNorm, and timesfm uses ProfileLayerNorm,
ProfileRmsNorm and ProfileStandardize. No sequence-512-specific logic or
template is changed. Under the review-set rule this is not a hit, so the two
sequence-512 cases were not run; their plans use the same reachable opcode set
with longer lengths, so they add no changed-branch coverage. This is the
recorded basis and the reason the sequence-512 cases are absent.

## Same-proof repeat verification

Each case reused the existing commit artifacts, ran a single prove, and
verified that one proof three times with the registered policy. This is not a
cold three-stage rerun: no new commit was produced and the prove is a single
run. Baseline prove values are the baseline warm medians.

| case | prove ms | baseline prove ms | verify runs ms | verify median ms | baseline verify ms | delta | proof bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 21,651 | 21,576 | 25,311 / 25,487 / 25,504 | 25,487 | 24,360 | +4.6% | 4,161,786,553 |
| gemma3-seq16 | 45,162 | 45,076 | 45,025 / 45,392 / 45,750 | 45,392 | 44,401 | +2.2% | 8,766,306,050 |
| timesfm-16patch | 35,579 | 35,308 | 43,242 / 43,253 / 43,221 | 43,242 | 42,356 | +2.1% | 6,865,533,616 |

All outcomes PASS; every verify binds the expected output; proof bytes are
identical to the baseline for every case. Comparison basis: the baseline
values are the warm medians recorded in the baseline trace on the same
machine; the baseline proof files and the earlier proofs have been deleted, so
the baseline verify could not be re-measured in this session, and the deltas
are against the recorded baseline numbers rather than a fresh side-by-side
run. Cases outside the 3% band: gpt2-seq16 +4.6%. They are recorded as-is and are pending a boss decision. Proofs were deleted immediately after the three verifies per
case (19,793,350,280 bytes across the three `proof.bin` files; the
`proof_bytes` metric in the table also counts the statement, public inputs and
manifest). Commit directories remain on disk.

## Artifacts

- `prod/<case>/`: prove and verify stage records; `verify-2/` and `verify-3/`
  hold the repeat verifies; `policy.toml`, `policy-source.json` and
  `parameters.toml` per case.
- `warm3.log`: harness log with stage records and proof sizes.
- `logs/cargo-test.txt`: full workspace test log.
- Commit directories are reused from
  `20261109-0908-oracle-ports-b4af4698-eaae60b0/prod/<case>/commit`.

## Open items

- Cases outside the 3% band: gpt2-seq16 +4.6%. They are recorded as-is and are pending a boss decision.
