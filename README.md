<div align="center">
    <img width="3265" height="994" alt="e89e95aa8ed6a99833394682c6632cb7" src="https://github.com/user-attachments/assets/9ed2ab1a-aab5-4133-a5e2-5bcbdad0ffea" />
</div>

# zkIE

> **Under development.** zkIE is not a usable release. The proof primitives, the
> op layer and the shard layer are implemented and tested; the AI-assisted
> autotuning built on top of them is still being completed.

**zkIE** — a Zero-Knowledge Inference Engine for verifiable AI inference.

Under verifiable AI inference, a result can be trusted without trusting whoever
produced it. Three things this unlocks:

- **Trustless work and delivery.** A deliverable — including human work done
  off-chain — can be judged on-chain and settled automatically: the evaluator of
  a job becomes a contract rather than a trusted party, which is what agentic
  commerce requires.
- **Trusted escrow for work.** Funds are released only once the work has been
  proven, e.g. an automated trading bot that can prove the strategy and the
  trades it actually executed before any payout.
- **On-chain governance with verifiable AI.** Decisions can be settled by a
  proven model output — an objective evaluation — instead of relying on
  multi-party human voting.

The obstacle is cost, not expressiveness: proving a model today is dominated by
time and memory, both still far above the cost of the inference itself.

**How it is used.** Export the model to ONNX — architecture, weights and op
graph — and zkIE composes the proving logic for that model's inference circuit
out of op-level proof primitives. For any input, that circuit then yields a
succinct proof that the output is what the model computes; verifying it is cheap
and does not re-run the model. Weights and activations are public: the current
route targets correctness and succinctness rather than witness hiding.

## How it works

1. **Algorithms.** The stack is built on established high-performance proving
   algorithms: **GKR / sum-check** for the linear-algebra reductions, **WHIR /
   FRI** for polynomial commitments and openings, and **LogUp** for the lookup
   arguments that implement the non-linear operators, over the 64-bit
   Goldilocks field.
2. **Primitives as the stable interface.** Proof primitives are designed per
   ONNX op type and implemented with the algorithms above, and that
   implementation is currently competitive. Because the primitive is the
   interface and the algorithm is only its internals, a better algorithm can be
   integrated later without touching the proving circuits already built for
   existing models — only the primitive's implementation changes.
3. **Autotuning.** An AI-assisted autotune loop decides how the circuit is split
   into shards and how each proof stage is scheduled across CPU and GPU. There
   is no need to hand-tune the low-level circuit composition, which makes
   adapting to a new model much faster.

### Architecture

Three layers, plus autotuning on top. Layers 1-3 are code; the autotuning in 4
is a process that works through their interfaces, not a component that optimises
by itself.

1. **Proof primitives** — matmul GKR reduction, LogUp fractional lookup, the
   rounding range check, and WHIR/FRI commitments and openings.
2. **Op layer** — one proof primitive per ONNX op type. This is the deliverable
   and it does not change: a new model is wired on top of it, never into it.
3. **Shard layer** — consecutive ops fold into one sumcheck, and only the shard
   boundary is committed and opened. How many ops make a shard is a public
   parameter: one op, a few ops, one transformer layer, several layers, or the
   whole model — the same code path.
4. **AI-assisted autotuning** — an AI agent searches `shard granularity ×
   per-stage CPU/GPU schedule × layout` through the interfaces exposed by 1-3:
   measure, change, measure again, keep the fastest. It is not a self-contained
   optimiser inside the prover.

CPU/GPU is not a per-shard switch. Each stage inside a shard dispatches on its
own — forward matmuls, commitment (FFT + Merkle), the sumchecks, and the FRI
openings — so a large codeword can go to the GPU while a small, launch-bound one
stays on the CPU.

### Op → primitive

| ONNX op | Proof primitive |
| --- | --- |
| MatMul | GKR reduction over the contraction index, leaving input/output claims |
| Add / elementwise | elementwise constraint |
| Softmax | exp lookup + row sum + division |
| LayerNorm / RMSNorm | normalization + affine |
| GELU / non-linear | LogUp fractional lookup |
| Lookup | LogUp fractional sumcheck |
| Affine (fixed-point rounding) | rounding range check (LogUp) |

## Usage

Build and run the real GPT-2 512 end-to-end proof (weights/tables are under
`models/gpt2_stack/`, expected at the crate working directory):

```bash
# full GPT-2 512 proof: argmax sanity check + prove/verify time
cargo run --release --example prove_gpt2_full

# shard-granularity sweep (1 shard vs 13 shards)
cargo run --release --example bench_gpt2_sharded

# autotune over shard granularities
cargo run --release --example bench_gpt2_autotune

# library tests
cargo test --lib
```

## Benchmarks

Measured on a 64-thread CPU. Full detail and the reasoning behind the numbers
is in [`docs/benchmarks.md`](docs/benchmarks.md).

| Model | Setup | Wall time | Peak RSS |
| --- | --- | --- | --- |
| TimesFM 1.0 200M | op granularity | ~302 s | ~1.4 GB |
| GPT-2 124M | seq=16 | ~1 min 53 s | ~5.6 GB |
| GPT-2 124M | seq=512, 1 shard (whole model) | ~4.8 min | ~43 GB |
| GPT-2 124M | seq=512, 13 shards (per layer, parallel) | **~0.92 min** | ~43 GB |
| Gemma 3 270M | adapting | - | - |

The GPT-2 512 rows use real weights, multi-head attention and pre-norm; argmax
matches ground truth 511/512. The 13-shard configuration is what the autotuner
picks (`bench_gpt2_autotune`) and beats DeepProve (~7.6 min) by roughly 8x. The
Gemma 3 row is the reuse target — the same op primitives and the same autotune
flow.

Status: GPT-2 512 proves end to end with a measured time (~0.92 min at 13
shards), and after autotuning is faster than the layer-granularity pipeline and
faster than DeepProve (~7.6 min) by roughly 8x.

## Repository layout

- `crates/zkie-gkr/` — the proving engine:
  - `src/compose.rs` — op primitives (`Op`, `prove_shard`, `prove_shard_dag`,
    cross-shard `same_poly` binding, weight batch commit).
  - `src/engine.rs` — shard granularity, per-stage schedule, and the autotune
    loop.
  - `src/projection.rs`, `src/layer_norm_centered.rs`, `src/softmax_scaled.rs`,
    `src/layernorm_chain.rs` — op-level proof primitives.
  - `src/sumcheck.rs`, `src/matmul.rs`, `src/logup_gkr.rs`, `src/same_poly.rs`,
    `src/mle.rs` — the reduction substrate.
  - `src/whir.rs`, `src/batch_open.rs`, `src/committed.rs` — WHIR/FRI
    polynomial commitments.
  - `src/par.rs` — the i64 fixed-point forward matmul (cache-friendly, with
    field fallback).
  - `examples/` — `prove_gpt2_full`, `bench_gpt2_sharded`,
    `bench_gpt2_autotune`, and micro-benchmarks.
- `crates/zkie-cuda/` — optional CUDA backend (Sppark DFT + Poseidon2 Merkle)
  behind the `cuda` feature.
- `models/` — GPT-2 512 weights and lookup tables (i32 fixed-point).
- `docs/` — `spec.md` (design), `benchmarks.md` (measurements), `roadmap.md`.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
