# oracle ports warm - 35b624ae / aacae4e4

Final-version warm verification on reused commits plus the row-length audit.

## Change

- The Layout branch now requires the input length to equal `from_x * from_y`
  and the output length to equal `to_x * to_y`, both products checked. A short
  input and output pair is rejected before any indexed access.
- The Softmax branch now requires the decoded raw length to equal the input
  length; its shift uses checked shifting and its denominator uses checked
  addition.
- Arithmetic and index bounds were reviewed across every operator branch:
  matrix and embedding length products, profile row products, attention
  sequence products, MoE table products, KV cache capacity products, top-k
  output lengths and division output lengths use checked arithmetic; every raw
  group is checked for presence before its rows are read; quotient and
  auxiliary lists have length gates before indexing.
- The shared-trace verifier now also requires `trace.steps.len()` to equal
  `plan.steps.len()` instead of relying on the final-output structure.
- New tests: a Layout fixture with a valid four-cell root and a three-cell short
  proof, and a Softmax fixture whose raw group is rebuilt with one fewer value
  while keeping a valid root and paths. Both are rejected with
  `shape_mismatch`; the corresponding positive proofs verify.

## Gate

`cargo test --workspace`: 139 passed, 0 failed. Log: `logs/cargo-test.txt`.

## Audit and reachability

A scan of the three model plan modules shows the opcodes that the five
registered cases actually build steps for:

- gpt2: Add, AffineScale, Argmax, EmbeddingIndex, MatMul, ProfileLayerNorm,
  Projection, Softmax, SoftmaxRows, TableIndex, TableLookup.
- gemma3: Add, AffineScale, Argmax, GatedFfn, MatMul, ProfileRmsNorm,
  ProfileRope, Projection, RowEmbeddingIndex, Softmax, SoftmaxRows, TableIndex,
  TableLookup.
- timesfm: Add, AffineScale, Hadamard, MatMul, ProfileLayerNorm,
  ProfileRmsNorm, ProfileStandardize, Projection, Relu, Softmax, TableIndex,
  TableLookup.

Names that appear only in configuration or executor modules do not make a
branch reachable: Layout, Requantize, TopK, RouteGate, Scatter, Gather,
BatchedMatMul, GatherMatMul, IndexGather, Pad, Range, DivRound, IntSqrt,
MoeDown, KvCache, CausalAttention, LayerNorm, RmsNorm and RopeProjection are
not built by any of the five plans and are exercised only through the generic
relation entry and fixture-level tests. A gap in a generic-entry branch would
not by itself demonstrate an attack on the five registered cases, and this
trace does not claim such an attack. The Layout negative is a generic-entry
guard; the Softmax negative targets a branch reachable by all three models.
The empty-port, oversized-external and missing-output negatives run through
the shared-trace entry. Every reachable branch is exercised end to end by the
prove and verify runs below.

## Case selection

All five cases ran. The change touches the shared relation verifier and the
row-length checks used by all three models, and it touches the SoftmaxRows and
longer-template paths exercised by the sequence-512 cases. Under the review-set
rule this is a hit case, so the full five-case set applies here; no case was
skipped, and a sequence-512 not-run reason does not apply. Routine checks return
to the three seq16 cases unless a later change hits sequence-512 logic or shared
structure.

## Same-proof repeat verification

Each case reused the existing commit artifacts from the previous trace, ran a
single prove, and verified that one proof three times with the registered
policy. This is not a cold three-stage rerun: no new commit was produced and
the prove is a single run. The table reports the three verify runs, their
median, the baseline warm median and the proof size. Baseline prove values are
the baseline warm medians.

| case | prove ms | baseline prove ms | verify runs ms | verify median ms | baseline verify ms | delta | proof bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-seq16 | 21,705 | 21,576 | 25,649 / 25,607 / 25,280 | 25,607 | 24,360 | +5.1% | 4,161,786,553 |
| gpt2-seq512 | 167,361 | 165,973 | 233,637 / 233,632 / 231,596 | 233,632 | 230,274 | +1.5% | 38,992,550,944 |
| gemma3-seq16 | 44,974 | 45,076 | 45,955 / 45,308 / 46,028 | 45,955 | 44,401 | +3.5% | 8,766,306,050 |
| gemma3-seq512 | 418,062 | 417,040 | 827,289 / 824,477 / 823,716 | 824,477 | 835,721 | -1.3% | 101,812,159,615 |
| timesfm-16patch | 35,211 | 35,308 | 42,320 / 42,939 / 42,884 | 42,884 | 42,356 | +1.2% | 6,865,533,616 |

All outcomes PASS; every verify binds the expected output; proof bytes are
identical to the baseline for every case. Comparison basis: the baseline
values are the warm medians recorded in the baseline trace on the same machine;
the baseline proof files and the earlier final-code proofs have been deleted,
so the baseline verify could not be re-measured in this session, and the
deltas are against the recorded baseline numbers rather than a fresh
side-by-side run. Cases outside the 3% band: gpt2-seq16 +5.1%, gemma3-seq16 +3.5%. They are recorded as-is and are pending a boss decision. timesfm is +1.25% on the final version. Proofs were
deleted immediately after the three verifies per case (160,598,040,964
bytes across the five `proof.bin` files; the `proof_bytes` metric in the table
also counts the statement, public inputs and manifest). Commit directories
remain on disk.

## Artifacts

- `prod/<case>/`: prove and verify stage records; `verify-2/` and `verify-3/`
  hold the repeat verifies; `policy.toml`, `policy-source.json` and
  `parameters.toml` per case.
- `warm3.log`: harness log with stage records and proof sizes.
- `logs/cargo-test.txt`: full workspace test log.
- The commit directories are reused from
  `20261109-0908-oracle-ports-b4af4698-eaae60b0/prod/<case>/commit`.

## Open items

- Cases outside the 3% band: gpt2-seq16 +5.1%, gemma3-seq16 +3.5%. They are recorded as-is and are pending a boss decision.
- timesfm is +1.25% on the final version.
