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
proof that the output is what the model computes. Today the verifier still
recomputes the forward pass to obtain its interior evaluations (`verify_shard` /
`verify_shard_dag`, `compose.rs`), so verifying costs about as much as proving
(measured: verify ~90% of prove at seq=512, CPU); making verification succinct,
with no re-run, is the open design item tracked in #15. Weights and activations
are public: the current route targets correctness and succinctness rather than
witness hiding.

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

Each model has its own crate under `models/`. Weights and lookup tables are
gitignored and expected under each model's `weights/` directory (export steps
and sha256 are in the per-model `README.md`).

```bash
# GPT-2 124M - end-to-end proof, shard sweep, and autotune
cargo run --release -p zkie-models-gpt2 --example prove
cargo run --release -p zkie-models-gpt2 --example bench_sharded
cargo run --release -p zkie-models-gpt2 --example bench_autotune

# TimesFM 1.0 200M
cargo run --release -p zkie-models-timesfm --example prove
cargo run --release -p zkie-models-timesfm --example bench

# Gemma 3 270M (sequence length via GEMMA_SEQ, default 16)
GEMMA_SEQ=16 cargo run --release -p zkie-models-gemma3 --example prove
GEMMA_SEQ=16 cargo run --release -p zkie-models-gemma3 --example bench

# DeepSeek-V2-Lite (sequence length via M, default 16)
M=16 cargo run --release -p zkie-models-deepseek-v2-lite --example prove
M=16 cargo run --release -p zkie-models-deepseek-v2-lite --example bench

# library tests
cargo test --workspace
```

To add a new model, see [docs/adding-a-model.md](docs/adding-a-model.md).

## Benchmarks

Measured on a 64-thread CPU. Full detail and the reasoning behind the numbers
is in [`docs/benchmarks.md`](docs/benchmarks.md).

| Model | Active / Total params | Setup | Prove | Verify | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| TimesFM 1.0 200M | 200M / 200M | seq=16, 20 shards (per layer) | ~29.9 s | ~11.1 s | ~27.7 GB |
| GPT-2 124M | 124M / 124M | seq=16, 13 shards (per layer) | ~20.0 s | ~8.8 s | ~36.9 GB |
| GPT-2 124M | 124M / 124M | seq=512, 13 shards (per layer) | **~34.1 s** | **~19.5 s** | ~39.3 GB |
| Gemma 3 270M | 270M / 270M | seq=16, 22 shards (per layer) | ~33.4 s | ~16.6 s | ~25.1 GB |
| Gemma 3 270M | 270M / 270M | seq=512, 22 shards (per layer) | ~56.5 s | ~36.7 s | ~27.4 GB |
| DeepSeek-V2-Lite | 2.4B / 15.7B | seq=16, 21 shards (lazy-open MoE, mmap weights) | ~516.5 s | ~270.8 s | ~124.5 GB |
| DeepSeek-V2-Lite | 2.4B / 15.7B | seq=512, 28 shards (lazy-open MoE, mmap weights) | ~1552.5 s (~25.9 min) | ~1081.4 s (~18.0 min) | ~372.2 GB |
| DeepSeek-V4.1-Flash | 8B/16B* / 552B | MoE + CSA2 + vision (planned) | - | - | - |

* DeepSeek-V4.1-Flash active params: 8B prefill / 16B decode.

## Repository layout

- `crates/zkie-core/` — the proving substrate:
  - `src/common/` — Goldilocks field, fixed-point embedding, MLE evaluation,
    sumcheck, the matmul / LogUp / `same_poly` reductions, and the claim type.
  - `src/pcs/` — the WHIR/FRI polynomial commitment scheme (`whir.rs`),
    batch openings (`batch_open.rs`), committed tensors (`committed.rs`), and
    the optional CUDA backend (`dft_cuda.rs`, `merkle_cuda.rs`, `cuda_ffi.rs`,
    `cuda_buffer.rs`) behind the `cuda` feature.
- `crates/zkie-ops/` — op-level proof primitives, one per ONNX op type:
  - `compose.rs` — `Op`, `Store`, `prove_shard`, `prove_shard_dag`, and the
    cross-shard `same_poly` binding.
  - `projection.rs`, `layer_norm_centered.rs`, `softmax_scaled.rs`,
    `layernorm_chain.rs` — the matmul/normalization/softmax primitives.
  - `par.rs` — the i64 fixed-point forward matmul (cache-friendly, with field
    fallback).
- `crates/zkie-engine/` — the autotune engine:
  - `src/engine.rs` — shard granularity, per-stage CPU/GPU schedule, and the
    autotune loop.
  - `benchmark/` — generic (non-model) micro-benchmarks of the primitives and
    the WHIR/FRI PCS.
- `models/gpt2/`, `models/timesfm/`, `models/gemma3/`,
  `models/deepseek-v2-lite/` — one crate per model; each holds the op-graph
  builder (`src/lib.rs`), its own `examples/` (prove + benchmarks), and a
  gitignored `weights/` (export steps + sha256 in the per-model `README.md`).
- `docs/` — `spec.md` (design), `benchmarks.md` (measurements),
  `adding-a-model.md` (how to add a new model), `roadmap.md`.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
