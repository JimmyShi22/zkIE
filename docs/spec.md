# zkIE proof specification

This document describes the implementation that is currently exercised by the
three-stage pipeline. It covers the data model, the commit / prove / verify
stages, the relation receipts and the verification checks.

## 1. Scope and security model

zkIE proves the result of fixed-point inference for a committed model. The
statement fixes the model commitment, the input and the expected output (or
expected token choice). The prover supplies the execution trace; the verifier
checks the trace against the public commitment.

The proof is **not zero-knowledge**. Weight values, lookup tables, activations
and intermediate vectors are part of the public proof material. The verifier
does not re-run the model and does not read the original weight files.

All witness arithmetic is integer or fixed-point. No host floating point
participates in the proved computation.

## 2. Encoding and statements

Model assets are little-endian integer tensors with one of two encodings:

- `i32-le`: four bytes per element;
- `i64-le`: eight bytes per element.

Each asset declares `asset_id`, `path`, `element_count`, `row_width` and
`encoding`. An asset is addressed by `(asset_id, row range)`; rows past the
declared element count are zero-filled by the reader.

A statement contains the case identifier, the model name, the model commitment
digest, the input tokens or values, and the expected argmax or output tensor.
The verifier binds the statement to the public commitment digest and to the
expected public inputs supplied with the proof.

## 3. Commit stage

For each asset the committer computes a table identity:

```text
table_identity = H(asset_id, path, element_count, row_width, encoding)
```

Rows are hashed into leaves using the domain-separated oracle leaf hash. The
row count is padded to the next power of two; padding rows are zero rows. The
leaves form a Merkle tree whose root is the asset root.

The commit produces:

- one `LeafRecord` per asset: `{leaf_id, generation, table_identity,
  leaf_root}`;
- an `AnchorRecord`: `{anchor_id, model_commitment, parameters_digest,
  registry_root, leaf_count}`. The registry root commits to all leaf records;
- `ModelCommitmentPublic`: the anchor record plus per-asset metadata
  (`asset_id`, path, element count, row width, start digest, root, padded row
  count, `LeafRecord`). It contains no weight values.

The commit artifact layout is:

```text
<commit-dir>/public/model-commitment.bin   serialized ModelCommitmentPublic
<commit-dir>/public/provider-config.bin    case/provider configuration
<commit-dir>/public/protocol.bin           protocol description
<commit-dir>/private/parameters.bin        private parameters
```

Only the public directory is needed for verification.

## 4. Prove stage

The prover consumes an existing commit artifact, re-opens the committed assets
by mmap, and executes the model's fixed-point executor. The executor emits a
`SharedTrace`:

```text
SharedTrace {
    tokens,
    assets: [AssetOpening],
    steps:  [SharedTraceStep],
    argmax,
    output,
}
```

An `AssetOpening` opens one asset interval once per trace. It contains the
asset id, the link proof to the anchor, the start/root digests, the row count
and one `RowOpening` per row. Each `RowOpening` contains the row index, the row
values and the Merkle path.

A `SharedTraceStep` contains:

```text
SharedTraceStep {
    message:  RelationReceiptMessage,
    external: [Option<usize>],   // input -> shared asset-opening pool index
    sources:  [Option<(usize, usize)>], // input -> (earlier step, output)
}
```

`RelationReceiptMessage` contains the opcode, the relation configuration, the
oracle registry roots, the declared external inputs and an `OpRelationProof`.

`OpRelationProof` commits the vectors used by the operator relation:

- inputs and outputs;
- raw accumulators (for example full-width products before a shift);
- quotients and range helpers used by rounding checks;
- auxiliary vectors specific to the operator.

Each vector is committed as rows with Merkle openings. Inputs that come from a
committed model asset carry no local opening; they are mapped through
`external` to an `AssetOpening`. Inputs produced by earlier steps are mapped
through `sources` to an earlier step output.

The proof artifact layout is:

```text
<proof-dir>/statement.bin      serialized statement
<proof-dir>/proof.bin          serialized SharedTrace
<proof-dir>/public-inputs.bin  expected argmax or output tensor
```

## 5. Verify stage

Verification proceeds as follows.

1. Load `ModelCommitmentPublic`; check that its `model_commitment` equals the
   expected model id. Load the statement and check the same binding.
2. Check that the trace tokens equal the statement tokens.
3. For every asset opening, check that the start and root match the public asset
   metadata, verify the link proof from the leaf record to the anchor registry,
   and verify every row opening against the asset root.
4. For every step, rebuild the oracle registry from the receipt roots and
   recompute the input and output oracle ids. Fill external inputs from the
   referenced asset openings and call the relation verifier with those values.
5. Check every `sources` edge: the local input values must equal the referenced
   output values of the earlier step.
6. Check the final result. For token models, the recorded argmax must equal the
   public expected argmax; for TimesFM, the final output tensor must equal the
   public expected output, and the trace's `output` field must also match the
   final step's output tensor.

Any mismatch rejects the proof. The verifier does not call the operator's
forward routine.

## 6. Relation equations

Each live operator has a relation configuration and an equation family over the
opened vectors. The configuration carries the dimensions, shift amounts,
clamps and table descriptors needed by the equation. Representative checks:

- `add`: each output value is the sum of the corresponding inputs.
- `affine_scale`: the output is the shifted, rounded and optionally clamped
  affine transform of the input.
- `matmul` / `projection`: the raw accumulator equals the full-width dot
  product, and the output equals the shifted and rounded raw value.
- `relu`: the output is the clamped non-negative part of the input.
- `argmax`: the output index is the deterministic argmax with the declared tie
  rule.
- `softmax` / `softmax_rows`: the exponential values come from the committed
  LUT and the division/rounding follows the relation configuration.
- `profile_norm` / `profile_rms`: the reciprocal-square-root values come from
  the committed LUT; the normalization and clamping are checked on the opened
  vectors.
- `table_lookup`: the output row is the selected row of the committed table;
  indices must be in range.
- `index_gen`: the output is the selected table or embedding row.
- `hadamard`, `gated_ffn`: the elementwise and gated feed-forward equations are
  checked on the opened raw, quotient and output vectors.

The exact configuration layout and equations are implemented in
`crates/zkie-ops/src/proof/relations.rs` and the operator implementations.

## 7. Public LUTs

Lookup tables are model assets. They are committed with the same table
identity, leaf records and anchor path as weights. A trace opens a LUT interval
once and steps reference that opening as an external input. The verifier checks
the opening against the public anchor; it never accepts a table supplied only
by the prover.

## 8. Rejection behavior

The verifier returns `accepted = true` only after all checks pass. A failure in
any of the following rejects the proof:

- model commitment or statement binding;
- token/input binding;
- asset link proof or Merkle path;
- relation configuration or equation;
- external-input mapping;
- producer/consumer edge;
- final argmax or output equality.

## 9. Command-line interface

The `zkie-bench` commands `commit`, `prove` and `verify` implement the three
stages as separate processes. `run` performs all three in one invocation.
`matrix`, `decode-matrix` and `profile-matrix` repeat the stages over case
lists and collect per-stage measurements. `zkie-check` validates asset
manifests, contracts and run reports.

See `README.md` for command examples and artifact paths.

## 10. Performance

Current commit/prove/verify timings, peak resident memory and proof sizes are
**待测**. They must be measured with the current code and the documented
benchmark method; no historical figures are part of this specification.
