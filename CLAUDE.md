# zkIE project instructions

## What this is

zkIE proves fixed-point ML inference with a three-stage **commit → prove →
verify** pipeline over a shared model commitment. Weight and public LUT assets
are committed as Merkle-backed rows; each operator step emits a relation
receipt; the verifier checks openings, relation equations and data-flow edges
without re-running the model. Proofs are non-ZK: witness data is public.

## Layout

- `crates/zkie-core`: shared types, errors, artifact handling, field/codec and
  fixed-point helpers, claims/oracles, memory and storage traits, profiling.
- `crates/zkie-engine`: runtime context, deterministic CPU kernels, mmap and
  scratch storage, Merkle/oracle commitments and link proofs, transcript and
  field arithmetic.
- `crates/zkie-ops`: live operators, registry, three-stage pipeline, relation
  receipts and shared model commitment.
- `models/common`: asset and ground-truth helpers shared by model crates.
- `models/{gpt2,gemma3,timesfm}`: fixed-point configuration and shared-trace
  executor for each supported model.
- `tools/zkie-bench`: commit / prove / verify CLI and batch runners.
- `tools/zkie-check`: asset, contract and run-report validation.
- `scripts/`: ONNX extraction and fixed-point simulation helpers.
- `docs/`: current specification, benchmark method and roadmap.

## Build / test

```bash
cargo build --workspace
cargo test --workspace
```

## Live path

1. **commit** builds `ModelCommitmentPublic` from the model's weight and LUT
   assets and publishes the public bundle.
2. **prove** re-opens the committed assets, runs the model executor and writes a
   `SharedTrace` of per-step relation receipts.
3. **verify** checks asset openings, relation equations, producer/consumer edges
   and the final output against the statement.

## Assets

Model weights and lookup tables are external, gitignored assets under
`models/*/weights/`; run the extraction scripts before proving. Committed asset
files are read-only inputs: do not rewrite or regenerate them as part of code
changes. Case files and ground truth live outside the repository; `zkie-bench`
takes their paths from the case configuration.
