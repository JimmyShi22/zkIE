# zkIE — Project Notes

## What this is

GKR / sum-check route for verifiable ML inference (non-ZK). The proof stack is
Goldilocks + WHIR/FRI + LogUp lookup; the public interface is the unified op layer
in crates/zkie-gkr/src/ops.rs. See README.md.

## Layout

- crates/zkie-gkr: proving primitives + op interface + examples
- crates/zkie-cuda: optional CUDA backend (NTT + Merkle)
- scripts/: ONNX extraction / fixed-point simulation / fidelity checks
- docs/superpowers/specs/: fixed-point semantics contract

## Build / test

    cargo build --workspace
    cargo test -p zkie-gkr
    cargo build --release --example prove_200m_ops

The TimesFM 200M example reads extracted tensors from models/ (gitignored); run
the extract scripts first to populate it.
