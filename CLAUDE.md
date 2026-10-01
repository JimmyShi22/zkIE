## What this is

GKR / sum-check route for verifiable ML inference (non-ZK). The proof stack is
Goldilocks + WHIR/FRI + LogUp lookup; the public interface is the unified op
layer in crates/zkie-ops/src/compose.rs. See README.md.

## Layout

- crates/zkie-core: proving substrate (common reductions + WHIR/FRI PCS, optional CUDA)
- crates/zkie-ops: op-level proof primitives (compose / projection / norms / softmax / par)
- crates/zkie-engine: autotune engine + per-model builders + examples
- scripts/: ONNX extraction / fixed-point simulation / fidelity checks
- docs/: spec, benchmarks, roadmap

## Build / test

    cargo build --workspace
    cargo test --workspace
    cargo run --release -p zkie-engine --example prove_gpt2

The GPT-2 512 example reads extracted tensors from models/gpt2/weights/ (gitignored);
run the extract scripts first to populate it.
