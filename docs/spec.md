# zkIE — Design Specification

Living document. It describes what is implemented today, plus the design that is
still to be built. Measurements live in [`benchmarks.md`](benchmarks.md), model
progression in [`roadmap.md`](roadmap.md).

## 1. Scope

zkIE proves AI inference: for a model and an input, a succinct proof that the
output is what the model computes. It is not zero-knowledge — weights and
activations are public — so the targets are correctness and succinctness.

The engine turns a model into a proving workload:

    op list (program IR) -> per-layer claim chains -> shard DAG
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

## 3. Proof primitives

`crates/zkie-gkr/src`:

- **`sumcheck.rs`** — degree-2 sumcheck, a degree-3 variant, sum-of-products
  batching, and the virtual-polynomial sumcheck `prove_virtual` /
  `verify_virtual`, which proves `sum_x coeff_i * prod_j f_j(x)` for an
  arbitrary term list. This is the substrate for both the layer circuit and
  logUp.
- **`matmul.rs`** — one contraction, `C(u,v) = sum_k A(u,k) * B(k,v)` with `A`
  stored transposed; it leaves claims on `A`, `B` and the scalar `C(u,v)`.
  `prove_chain` / `verify_chain` chain matmul into matmul (attention
  `Q -> scores = Q·Kᵀ`) so the intermediate is never committed.
- **`logup_gkr.rs`** — LogUp as a fractional sumcheck: the fraction-addition
  tree is proven layer by layer with eq-weighted virtual sumchecks, ending in a
  single numerator/denominator claim. `prove_fractional` / `verify_fractional`
  are the raw form; `prove_lookup_fractional` / `verify_lookup_fractional` wrap
  it as a table lookup. There is no standalone grand-product argument.
- **`layer_circuit.rs`** + **`layer.rs`** — fold a list of elementwise
  constraints into ONE eq-weighted sumcheck ("one `g` per layer"): each op's
  arithmetic constraint is a virtual polynomial that must vanish, combined with
  random coefficients. `layer::compile` turns `ElemOp` (Add / Mul / Affine) into
  those constraints.
- **`same_poly.rs`** — claim merging. Given claims `(r_i, y_i)` on the same MLE,
  it proves they are evaluations of one polynomial and merges them into a single
  claim at a fresh point. This binds the claim produced by one sumcheck to the
  one consumed by the next, and a shard's output to the next shard's input.
- **`whir.rs`** — WHIR multilinear PCS over Goldilocks: `commit` /
  `commit_batch`, `open` / `open_batch` / `open_batch_multi` and their
  verifiers. Opening points are base-field coordinates embedded into the
  degree-2 extension field. `Whir::new` uses a 90-bit security level;
  `Whir::new_testing` is the fast, low-security instance used for local
  iteration. `global_open_count()` is the process-wide FRI opening counter used
  by the benchmarks.
- **`zkie-cuda`** — optional backend: `CudaDft` (Sppark Goldilocks radix-2 NTT)
  and `CudaMerkleTreeMmcs` (Poseidon2 Merkle commit on GPU). Openings and
  verification stay on the upstream CPU code, so a GPU-built tree is
  bit-identical to a CPU-built one. Both fall back to CPU when no device is
  present or a kernel errors.

Backend dispatch is per engine: `ZKIE_CUDA` sets the default and
`ZKIE_CUDA_DFT` / `ZKIE_CUDA_MMCS` override the DFT and Merkle paths
individually, which is what the hybrid backend sweeps in the benchmarks.

## 4. Op layer — the stable interface

Two entry points over the same op set:

- **`ops.rs`** — the committed interface. A `Ctx` owns per-size `Whir` instances
  and a `BatchBuilder`; the ops are `matmul`, `add`, `affine`, `scale`, `relu`,
  `lookup`, `rms_norm_rows`, `layer_norm_rows`, `softmax_rows`. Each op computes
  its output natively, commits the involved tensors, and proves the
  relationship, returning a committed `Tensor`.
- **`ir.rs`** — the program IR and the two-phase executor. `Op` is MatMul / Add /
  Affine / Scale / Relu / RmsNorm / LayerNorm / LayerNormRows / Lookup /
  Softmax. `Exec` builds a program imperatively (`matmul`, `affine`, …,
  `finish_layer`), records every tensor into a `BatchBuilder` keyed by
  (size, group), and `Exec::prove` then walks the same op list and runs the
  batch proof. `layer_spans()` is what the shard layer consumes.

How each op is built:

| ONNX op | Construction |
| --- | --- |
| MatMul | GKR reduction over the contraction index, leaving input/output claims |
| Add | elementwise constraint |
| Affine | constraint on `out = round(in / 2^shift) + bias`, plus a LogUp range check on the rounding remainder |
| Scale | `out = round(in * scale / 2^16) + bias` |
| ReLU | affine plus zero-bits range check |
| Softmax | exp lookup + row sum + division: `out = round(e * 2^16 / sum)` |
| LayerNorm / RMSNorm | integer mean / mean-square reduction over `n_real`, rsqrt table lookup, then affine |
| Lookup | LogUp fractional lookup against a table |

Implementations live in `committed.rs` (committed and batch forms); `ops.rs` and
`ir.rs` are the call sites a per-model program compiles against.

## 5. Layer and shard composition

**Per-layer chains** — one claim chain per transformer layer, with intermediates
kept virtual:

- `layernorm_chain.rs`, `attention_chain.rs`, `ffn_chain.rs`,
  `softmax_scaled.rs` — the sub-chains.
- `transformer_chain.rs` — `prove_transformer_layer` / `verify_transformer_layer`
  chain them together; the residual stream is claimed by both halves at
  different points and bound with `same_poly`. `transformer_layer_forward`
  exposes the plain forward pass for witness generation.

**Shards** (`shard.rs`):

- `prove_shard` folds a group of ops into one `g` at a random point.
- `prove_shard_dag` chains shards and emits one `same_poly` binding per boundary.
- Verification holds iff every shard is sound on its own AND every boundary is
  consistent. In the committed model the boundary is a single commitment shared
  by the two shards.

## 6. Engine

`engine.rs`:

- `Granularity::{Ops(n), Layers(n), WholeModel}` — a public parameter, not a
  hardcoded layer.
- `Stage::{Forward, Commit, Sumcheck, Open}` and `StageSchedule` — each stage
  dispatches to CPU or GPU independently, so backends can be mixed inside one
  shard. `ALL_STAGES` and `all_schedules()` enumerate the 16 combinations.
- `compile_shard_dag(op_count, layers, granularity, schedule)` returns a
  `ShardDag { shards, boundaries }`; boundary `i` means shard `i`'s output
  commitment must equal shard `i+1`'s input commitment.
- `Model { op_count, layers, costs, gpu, max_parallel_shards }`, where
  `StageCosts` are per-op (forward, sumcheck) and per-boundary (commit, open).
  `estimate()` turns a (granularity, schedule) pair into a time breakdown.
- `autotune(model, granularities)` searches granularity × schedule against the
  cost model.
- `autotune_with(granularities, schedules, measure)` is the real loop: the
  caller runs the actual proof for each configuration and returns the measured
  result, and the engine keeps the lowest-total one. Tune once per model, cache
  the result.

The engine does not optimise by itself. It is the search over the interfaces the
layers above expose, run by an AI agent.

## 7. Implementation status

- 106 lib tests in `zkie-gkr`; 35 examples.
- End-to-end programs: `prove_200m_ops` / `prove_200m_ir` (TimesFM 1.0 200M,
  20 layers) and `prove_gpt2_ops` / `prove_gpt2_ir` (GPT-2 124M, 12 layers,
  seq=512, 12 heads).
- Layer-granularity chain, shard DAG with `same_poly` boundary binding, and the
  engine / autotune skeleton.
- CUDA backend behind the `cuda` feature, opt-in.

## 8. What remains

1. **Real GPT-2 is DONE**: multi-head attention, pre-norm, real weights
   lm_head, seq=512; `prove_gpt2_full` / `bench_gpt2_sharded` / `bench_gpt2_autotune`
   prove the full model end to end, argmax 511/512. WHIR commitments for the
   boundary and weights are not yet wired into this path.
2. **Quantization**: 12-bit / adaptive bit width.
3. **GPU tuning** for the large-codeword FRI commit.
4. **Remove the remaining claim-chaining overhead**: `same_poly` binding is
   parallelized per shard, but cross-shard bindings and per-op logUp lookups
   still dominate the GKR phase.
5. **Autotune for real** over granularity × per-stage backend × layout using
   `autotune_with`, with the winning configuration cached per model.
6. **A new model end to end** (Gemma 3 first) through the same primitives and the
   same autotune flow.

Open items:

- **Soundness parameters.** Today's benchmarks use `Whir::new_testing`
  (32-bit); production runs need `Whir::new` (90-bit) or equivalent, and the
  fixed-point tables (exp / gelu / rsqrt) are sized for the test instances.
- **Layer parallelism does not scale linearly.** the 13-shard GPT-2 512 prove is ~34 s (forward ~5.5 s + parallel GKR ~27 s)
  against a ~6 s ideal: the layer loop and the rayon pool still contend.

## 9. Module map

| Module | Responsibility |
| --- | --- |
| `field.rs`, `fixed_point.rs`, `mle.rs` | Goldilocks, signed fixed-point embedding, MLE evaluation |
| `sumcheck.rs` | degree-2 / degree-3 / virtual-polynomial sumchecks |
| `matmul.rs`, `logup_gkr.rs`, `same_poly.rs` | the three reduction primitives |
| `layer.rs`, `layer_circuit.rs` | elementwise constraint folding ("one g") |
| `committed.rs` | every op, committed and batch forms |
| `ops.rs`, `ir.rs` | the stable op interface and the program IR / executor |
| `*_chain.rs`, `softmax_scaled.rs` | per-layer claim chains |
| `shard.rs` | shard folding and shard DAG with boundary binding |
| `whir.rs`, `batch_open.rs` | WHIR PCS, openings, opening reduction |
| `engine.rs`, `autotune.rs` | granularity, per-stage schedule, autotune |
| `par.rs` | row-parallel CPU helpers (rayon) |
