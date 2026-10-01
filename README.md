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

## Benchmarks

Measured on a 64-thread CPU. Full detail and the reasoning behind the numbers
is in [`docs/benchmarks.md`](docs/benchmarks.md).

| Model | Setup | Wall time | Peak RSS |
| --- | --- | --- | --- |
| TimesFM 1.0 200M | op granularity | ~302 s | ~1.4 GB |
| GPT-2 124M | seq=16 | ~1 min 53 s | ~5.6 GB |
| GPT-2 124M | seq=512, op granularity | ~19.6 min | ~30 GB |
| GPT-2 124M | seq=512, layer granularity + parallel | ~32 s | - |
| Gemma 3 270M | adapting | - | - |

The ~32 s row is not like-for-like: it uses synthetic weights, single-head
attention, post-norm and a plain model (no committed boundary or weights).
Closing that gap is the remaining work. The Gemma 3 row is the reuse target —
the same op primitives and the same autotune flow.

Target: GPT-2 512 proves end to end with a measured time, and after autotuning
is faster than the current layer-granularity pipeline (no regression) and faster
than DeepProve (~7.6 min).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
