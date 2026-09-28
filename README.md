# zkIE

Verifiable inference for large models, built on the GKR / sum-check route (non-ZK).

zkIE proves that an inference result was actually produced by a given model, and
keeps the proof succinct, without hiding the witness. Weights and activations are
public; private-model proving is a separate future topic.

## Why not ZK

GKR / sum-check / WHIR / FRI are inherently non-ZK proof protocols. For public
models the witness is public anyway, so zkIE targets correctness + succinctness,
not zero-knowledge.

## Architecture

Bottom-up:

    GKR proving primitives (sumcheck, WHIR/FRI, LogUp lookup)
      echo"LICENSE written"; wc -l LICENSEunified op interface (crates/zkie-gkr/src/ops.rs)
      echo"LICENSE written"; wc -l LICENSEper-model program (crates/zkie-gkr/examples/prove_200m_ops.rs)

Key choices:

- Field: Goldilocks (64-bit)
- Fixed-point: int16/i32/i64 at scale 2^16 / 2^32 / 2^48, round-half-up
- Ops: MatMul, Add, Affine, Scale, ReLU, Softmax, LayerNorm, RMSNorm, Lookup

The fixed-point semantics contract is in
docs/superpowers/specs/2026-09-28-zkie-op-fixed-point-semantics.md.

## Status

Proves the full TimesFM 1.0 200M forward pass (prologue + 20 layers + output head).

Measured on a 64-thread machine:

    wall time    ~6.5 min
    peak memory  ~1.4 GB

## Crates

- zkie-gkr: GKR/WHIR proving primitives + unified op interface
- zkie-cuda: optional CUDA backend (NTT + Merkle), enabled with --features cuda

## Build & run

    cargo build --release --example prove_200m_ops
    ./target/release/examples/prove_200m_ops

    cargo test -p zkie-gkr

## License

MIT OR Apache-2.0
