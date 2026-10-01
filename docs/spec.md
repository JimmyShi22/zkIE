# zkIE — Design Specification

Living document. It describes what is implemented today, plus the design that is
still to be built. Measurements live in [`benchmarks.md`](benchmarks.md), model
progression in [`roadmap.md`](roadmap.md).

## 1. Scope

zkIE proves AI inference: for a model and an input, a succinct proof that the
output is what the model computes. It is not zero-knowledge — weights and
activations are public — so the targets are correctness and succinctness.

The engine turns a model into a proving workload:

    ONNX op graph -> op primitives (IR) -> shard DAG
                  -> per-stage CPU/GPU schedule -> autotune

Design rule: **the op layer is the stable interface.** A better proving
algorithm changes the internals of a primitive, not the circuits already built
for existing models.

## 2. Field and fixed-point

- Field: Goldilocks, `p = 2^64 - 2^32 + 1` (`field.rs`, `P`).
- Signed embedding: `x < 0` maps to `p - |x|` (`fixed_point.rs`:
  `from_i16` / `from_i32` / `from_i64`, and the `to_*` inverses).
- Scales: `2^16` for activations, weights and biases; `2^32` for matmul
  accumulation; `2^48` for raw norm output (three `2^16` factors). Rounding is
  round-half-up everywhere.
- Padding: every tensor length is a power of two, as WHIR's multilinear
  extension requires. The real dimension is passed separately as `n_real`, and
  reductions sum only the first `n_real` entries.

## 3. Proof substrate (`zkie-core`)

`crates/zkie-core/src/common`:

- **`sumcheck.rs`** — degree-2 sumcheck, a degree-3 variant, sum-of-products
  batching, and the virtual-polynomial sumcheck `prove_virtual` /
  `verify_virtual`, which proves `sum_x coeff_i * prod_j f_j(x)` for an
  arbitrary term list. This is the substrate for both the op constraints and
  logUp.
- **`matmul.rs`** — one contraction, `C(u,v) = sum_k A(u,k) * B(k,v)` with `A`
  stored transposed; it leaves claims on `A`, `B` and the scalar `C(u,v)`.
  Chained matmul (attention `Q -> scores = Q*K^T`) keeps the intermediate
  virtual.
- **`logup_gkr.rs`** — LogUp as a fractional sumcheck: the fraction-addition
  tree is proven layer by layer with eq-weighted virtual sumchecks, ending in a
  single numerator/denominator claim. `prove_lookup_fractional` /
  `verify_lookup_fractional` wrap it as a table lookup. There is no standalone
  grand-product argument.
- **`same_poly.rs`** — claim merging. Given claims `(r_i, y_i)` on the same MLE,
  it proves they are evaluations of one polynomial and merges them into a single
  claim at a fresh point. This binds the claim produced by one sumcheck to the
  one consumed by the next, and a shard's output to the next shard's input.
- **`field.rs`, `fixed_point.rs`, `mle.rs`, `claim.rs`** — Goldilocks field,
  signed fixed-point embedding, MLE evaluation, and the shared claim type.

`crates/zkie-core/src/pcs`:

- **`whir.rs`** — WHIR multilinear PCS over Goldilocks: `commit` /
  `commit_batch`, `open` / `open_batch` / `open_batch_multi` and their
  verifiers. `Whir::new` uses a 90-bit security level; `Whir::new_testing` is
  the fast, low-security instance used for local iteration.
  `global_open_count()` is the process-wide FRI opening counter used by the
  benchmarks.
- **`batch_open.rs`** — opening reduction (many claims to few openings).
- **`committed.rs`** — committed tensor wrappers over `Whir`.
- CUDA (feature `cuda`): `dft_cuda.rs` (Sppark Goldilocks radix-2 NTT) and
  `merkle_cuda.rs` (Poseidon2 Merkle commit on GPU). Openings and verification
  stay on the CPU code, so a GPU-built tree is bit-identical to a CPU-built
  one. Backend dispatch is per engine: `ZKIE_CUDA` sets the default and
  `ZKIE_CUDA_DFT` / `ZKIE_CUDA_MMCS` override the DFT and Merkle paths
  individually.

## 4. Op layer — the stable interface (`zkie-ops`)

The op layer is a set of proof primitives, one per ONNX op type. The canonical
entry point is `compose.rs`:

- **`compose.rs`** — `Op` (the op-graph IR), `Store` (witness / tensor
  storage), `prove_shard` / `verify_shard` (fold a group of ops into one
  `g`), and `prove_shard_dag` / `verify_shard_dag` (chain shards with one
  `same_poly` binding per boundary).
- **`projection.rs`, `layer_norm_centered.rs`, `softmax_scaled.rs`,
  `layernorm_chain.rs`** — the matmul+bias, normalization, and softmax
  primitives.
- **`par.rs`** — the row-parallel i64 fixed-point forward matmul
  (cache-friendly, with field fallback).

How each op is built:

| ONNX op | Construction |
| --- | --- |
| MatMul | GKR reduction over the contraction index, leaving input/output claims |
| Add | elementwise constraint |
| Affine | constraint on `out = round(in / 2^shift) + bias`, plus a LogUp range check on the rounding remainder |
| Softmax | exp lookup + row sum + division |
| LayerNorm / RMSNorm | integer mean / mean-square reduction over `n_real`, rsqrt table lookup, then affine |
| GELU / non-linear | LogUp fractional lookup against a table |

## 5. Shard composition

**Shards** (`compose.rs`):

- `prove_shard` folds a group of ops into one `g` at a random point.
- `prove_shard_dag` chains shards and emits one `same_poly` binding per boundary.
- Verification holds iff every shard is sound on its own AND every boundary is
  consistent: a shard's input commitment must equal the upstream shard's output
  commitment, forced inside the aggregation rather than chosen by the shard's
  own prover.

## 6. Engine (`zkie-engine`)

`engine.rs`:

- `Granularity::{Ops(n), Layers(n), WholeModel}` — a public parameter, not a
  hardcoded layer.
- `Stage::{Forward, Commit, Sumcheck, Open}` and `StageSchedule` — each stage
  dispatches to CPU or GPU independently, so backends can be mixed inside one
  shard.
- `compile_shard_dag(op_count, layers, granularity, schedule)` returns a
  `ShardDag { shards, boundaries }`.
- `Model { op_count, layers, costs, gpu, max_parallel_shards }`, `estimate()`,
  `autotune()`, and `autotune_with()` — the search over
  `granularity x schedule` against the cost model, or measured by the caller.

The engine does not optimise by itself. It is the search over the interfaces the
layers expose, run by an AI agent. Tune once per model, cache the result.

## 7. Implementation status

- Lib tests split across `zkie-core`, `zkie-ops`, and `zkie-engine`.
- End-to-end program: `prove_gpt2` (GPT-2 124M, 12 layers, seq=512, 12 heads,
  real weights) proves end to end via the reusable `models::gpt2` builder,
  argmax 511/512.
- Shard DAG with `same_poly` boundary binding, and the engine / autotune
  skeleton.
- CUDA backend behind the `cuda` feature, opt-in and still being wired up.

## 8. What remains

1. **Quantization**: 12-bit / adaptive bit width.
2. **GPU tuning** for the large-codeword FRI commit.
3. **Remove the remaining claim-chaining overhead**: `same_poly` binding is
   parallelized per shard, but cross-shard bindings and per-op logUp lookups
   still dominate the GKR phase.
4. **Autotune for real** over granularity x per-stage backend x layout using
   `autotune_with`, with the winning configuration cached per model.
5. **A new model end to end** (Gemma 3 first) through the same primitives and the
   same autotune flow.

Open items:

- **Soundness parameters.** Today's benchmarks use `Whir::new_testing`
  (32-bit); production runs need `Whir::new` (90-bit) or equivalent, and the
  fixed-point tables (exp / gelu / rsqrt) are sized for the test instances.
- **Layer parallelism does not scale linearly.** the 13-shard GPT-2 512 prove is
  ~34 s (forward ~5.5 s + parallel GKR ~27 s) against a ~6 s ideal: the layer
  loop and the rayon pool still contend.

## 9. Module map

| Crate / module | Responsibility |
| --- | --- |
| `zkie-core::common::field`, `fixed_point`, `mle`, `claim` | Goldilocks, signed fixed-point embedding, MLE evaluation, claim type |
| `zkie-core::common::sumcheck` | degree-2 / degree-3 / virtual-polynomial sumchecks |
| `zkie-core::common::matmul`, `logup_gkr`, `same_poly` | the three reduction primitives |
| `zkie-core::pcs::whir`, `batch_open`, `committed` | WHIR PCS, openings, committed tensors |
| `zkie-ops::compose` | op IR, shard folding, shard DAG with boundary binding |
| `zkie-ops::projection`, `layer_norm_centered`, `softmax_scaled`, `layernorm_chain`, `par` | op-level proof primitives + forward matmul |
| `zkie-engine::engine` | granularity, per-stage schedule, autotune |
| `zkie-engine::models` | per-model op-graph builders |
