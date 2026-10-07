## GPT-2 512 full model (op-primitive shard DAG, GKR + logUp) (2026-10-01)

Real GPT-2 (12 layers, 12 heads, pre-norm, real weights, lm_head), sequence
length 512, 2114 ops, proven through the op-primitive shard-DAG composer
(`compose::prove_shard_dag`) with `same_poly` cross-shard binding. Release build,
64-thread CPU. This path commits only shard boundaries; intermediate activations
are virtual MLE and nonlinearities use logUp lookups (no WHIR commit in this
path).

| seq | shards | prove | verify | total | argmax |
| --- | --- | --- | --- | --- | --- |
| 16 | 13 (per layer) | 20.0 s | 8.8 s | ~28.9 s (0.48 min) | 16/16 |
| 512 | 13 (per layer) | 34.1 s | 19.5 s | ~53.6 s (0.89 min) | 511/512 |

Peak host memory: **~36.9 GB RSS** (seq=16), **~39.3 GB RSS** (seq=512). The memory
is dominated by the lm_head weights/tables and the verifier's store clone, not
by the sequence length.

13-shard prove breakdown: forward witness ~5.5 s + parallel GKR ~27 s.

Notes:

- The autotuner (`bench_gpt2_autotune`) picks `Layers(1)` (13 shards) as the
  fastest granularity.
- This beats DeepProve (~7.6 min) by roughly **8x**.
- The forward matmul uses the i64 fixed-point path with cache-friendly (ikj)
  ordering and overflow-aware dispatch (`par::mm_par_fixed`).
- The remaining bottleneck is the logUp lookups (exp / rsqrt / gelu tables) in
  the GKR sumcheck; those tables are already at their minimum lossless size for
  16-bit fixed point.

## TimesFM 1.0 200M full model (op-primitive shard DAG) (2026-10-01)
Real TimesFM 1.0 200M (20 layers, 16 heads x 80, H=1280), sequence length 16,
4316 ops, proven through `compose::prove_shard_dag` with `same_poly`
cross-shard binding (RMSNorm / LayerNorm / SiLU / ReLU / causal attention).
Release build, 64-thread CPU.

| seq | shards | prove | verify | total |
| --- | --- | --- | --- | --- |
| 16 | 20 (per layer) | 29.9 s | 11.1 s | ~40.9 s (0.68 min) |

Peak host memory: **~19.5 GB RSS**. Finer granularity (40 shards) is tied
(~37.6 s); coarser granularity is slower (4 shards ~68.7 s), so the autotuner
picks per-layer. Output matches the ONNX float reference to ~1 LSB.

# Benchmarks

Baseline measurements for the TimesFM 200M end-to-end proof (prologue + 20 layers + output head) through the unified op interface (prove_200m_ops).

Machine: 64-thread CPU, 3x NVIDIA L20 (46 GB each), 495 GB RAM.

| Backend | Wall time | Peak host memory |
|---|---|---|
| CPU (rayon, 64 threads) | ~404 s (6.7 min) | ~1.4 GB |
| GPU (CUDA, L20) | ~560 s (9.3 min) | ~1.65 GB |

Notes:

- GPU is currently slower than CPU (about 39 percent) because the model has about 29k small tensor commitments that are launch/transfer-bound; the GPU parallelism does not pay off at this granularity.
- The GPU run used one L20 and added about 0.9 GB of VRAM on top of the host memory reported above.
- These are pre-autotuning baselines. Autotuning targets coarser shards so commitments become fewer and larger, which is where the GPU is expected to win.

## IR batch-commit profiling (2026-09-28)

Per-op profiling of the IR two-phase executor (`prove_200m_ir`), release build,
1 layer, to locate the end-to-end bottleneck.

| Backend | commit (1 layer) | GKR sumcheck (k=2048) | WHIR open/verify (k=2048) |
|---|---|---|---|
| CPU (rayon, 64 threads) | 1.89 s | 24.0 ms | 345 ms |
| GPU (ZKIE_CUDA=1, L20) | 2.07 s | 23.3 ms | 397 ms |

Key finding:

- The dominant cost is **WHIR opening/verification** (the FRI opening proof),
  not the GKR sumcheck. Per 2048-dim matmul the open/verify is ~345 ms while
  the GKR sumcheck is ~24 ms (~14x smaller).
- The current CUDA backend (Sppark DFT + Poseidon2 Merkle) only accelerates
  the **commit** (building the Merkle tree); the opening/verification path is
  the upstream p3-merkle-tree CPU code, so the GPU does not accelerate the
  actual bottleneck.
- The GPU is slightly *slower* than the 64-thread CPU on both commit (2.07 s
  vs 1.89 s) and open/verify (397 ms vs 345 ms). The workload is launch-bound
  on the GPU, and the CPU AVX2 Poseidon2 is already competitive.

Implication: batching commitments (the earlier "29x batch commit" microbench)
targets the wrong stage. The real cost is the many small FRI opening proofs.
A meaningful speedup needs either (a) aggregating openings across ops, or
(b) a PCS with cheaper/fewer openings, rather than further GPU-izing the
commit.

## Hybrid backend autotuning (2026-09-29)

Decoupled the WHIR DFT and Merkle backends (`ZKIE_CUDA_DFT` / `ZKIE_CUDA_MMCS`,
commit `0330656`) and swept all four combinations. 1 layer, release build,
`cuda` feature enabled.

| config | commit | ops (GKR + open/verify) |
|---|---|---|
| cpu_cpu | 1.865 s | **121.1 s** |
| gpu_gpu | 2.064 s | 157.6 s |
| gpu_cpu (DFT=GPU, Merkle=CPU) | 1.853 s | 124.7 s |
| cpu_gpu (DFT=CPU, Merkle=GPU) | 2.047 s | 161.7 s |

Conclusion:

- **DFT is backend-neutral.** GPU DFT gives no measurable benefit and is ~3%
  slower on the ops phase.
- **Merkle (Poseidon2) is decisively CPU.** GPU Merkle is ~33% slower on ops
  and ~10% slower on commit.
- The autotuner converges to **all-CPU**. The GPU provides no speedup for this
  GKR + WHIR(Poseidon2) proof.
- The ops phase is ~121 s per layer, so the full 20-layer 200M proof is
  **~40 min**, correcting the earlier ~9-10 min estimate (which extrapolated
  from an uncompleted run).

The only GPU-friendly stage left is the forward `dense_m` matmul, which is
~3% of the total (~75 s of ~40 min), so even a perfect GPU matmul would cap
the hybrid ceiling at ~3%. The dominant cost is the WHIR opening proofs
(Poseidon2 re-commits), which can only be reduced by batching/aggregating
openings, not by backend selection.

## Opening-cost optimizations (2026-09-29)

The IR batch executor was initially ~6x slower than the per-tensor baseline
because `open_batch` padded every opening with `num_tables - 1` dummy points,
so each opening paid `num_tables` FRI evaluations instead of one. Three fixes
brought it back past parity (all tests green, 42/42):

| step | full 20-layer 200M |
|---|---|
| per-tensor baseline | ~404 s |
| IR before fixes | ~42 min |
| + avoid double-opening bit columns (`2641af7`) | ~31 min |
| + batch bit-column openings (`b2ebab2`) | ~18.6 min |
| + isolate default tensors to single-point opens (`f61c6a8`) | **~302 s (5.0 min)** |

Per-layer `prove()` ops dropped from 121 s to ~13.7 s (layer_norm 4.0 s,
rms_norm 4.0 s, matmul 2.8 s, affine 2.2 s, everything else sub-second).

Remaining bottleneck: the forward `dense_m` matmul is serial and now dominates
(~250 s of the 302 s). Peak RSS is also still ~34 GB vs the baseline ~1.4 GB
because the two-phase executor holds `self.plain` and the BatchBuilder copy
simultaneously; both are independent follow-ups.

## GPT-2 124M (2026-09-29)

First decoder-only Transformer through the unified ops interface
(`prove_gpt2_ops`). Full forward pass: token+positional embeddings (precomputed),
12 transformer blocks (causal attention + GELU FFN), final LayerNorm, and the LM
head (tied embeddings, 50257 vocab padded to 65536). seq=16.

Machine: 64-thread CPU, 495 GB RAM.

| metric | value |
|---|---|
| wall time | ~1 min 53 s |
| CPU time | ~4380 s (~42 threads) |
| peak RSS | ~7.6 GB |
| correctness | argmax matches onnxruntime ground truth on all 16 tokens |

Adaptations vs TimesFM:

- GELU (`gelu_new`, tanh form) is a single LogUp lookup; its input range reaches
  ~[-64, 64], so the table spans [-128, 128] (offset 2^23, size 2^24).
- Causal self-attention with a fused QKV projection split into q/k/v matmuls,
  and per-head head_dim=64. heads=12 is padded to HEADS_PAD=16 so the all-heads
  softmax batch is a power-of-two size.
- Attention scale 1/sqrt(64) is folded into the QK^T `affine` shift (19 = 16 + 3).
- LayerNorm `rsqrt` lookup table must cover per-position variance up to ~13294
  (GPT-2 outlier dimensions), so it is widened to 2^28 entries (max var 16384);
  this is backward-compatible with the TimesFM 2^19 table.
- `lookup::prove` now sums over the *distinct* indices instead of the whole
  table, so a single-element LayerNorm lookup is O(1) and the large tables do
  not slow the proof.
### GPT-2 124M per-op timing + memory (post rsqrt optimization)

seq=16, 64-thread CPU. Same run after switching the LayerNorm `rsqrt` table from
uniform 2^28 entries to piecewise 2^21 entries (1 GB -> 8 MB on disk).

| metric | value |
|---|---|
| wall time | ~1 min 53 s |
| peak RSS | ~5.6 GB (was ~7.6 GB before the rsqrt shrink) |

WHIR timing (aggregate across all ops):

| stage | count | seconds | share |
|---|---|---|---|
| commit (Merkle) | 23800 | 14.0 | 12% |
| open (FRI) | 46123 | 70.7 | 63% |
| verify (FRI) | 46123 | 18.5 | 16% |
| GKR + lookup + forward (rest) | - | ~9.5 | 8% |

Conclusion: WHIR FRI opening dominates (~63%), exactly as in TimesFM. GKR
sumcheck is not the bottleneck, so the GPU (which only accelerates
commit/forward) does not help this proof.
## Cross-model comparison (TimesFM 200M vs GPT-2 124M)

Both measured on the same 64-thread CPU / 495 GB RAM machine, seq=16, through
the same GKR/WHIR/LogUp op interface.

| model | layers | params | wall time | peak RSS |
|---|---|---|---|---|
| TimesFM 200M | 20 | 200M | ~5 min (302 s) | ~1.4 GB |
| GPT-2 124M | 12 | 124M | ~1 min 53 s | ~5.6 GB |

GPT-2 is ~2.7x faster, but the comparison is not like-for-like:

- GPT-2 has fewer layers (12 vs 20) and a smaller hidden dim (768 vs 1280), so
  it does less work per layer.
- GPT-2 additionally proves a 50257 -> 65536 (padded) vocab LM head, which
  TimesFM does not have; that head is the main reason GPT-2's memory is higher
  (~5.6 GB vs ~1.4 GB).
- Both are dominated by WHIR FRI opening (GPT-2: 63% of wall time), so the
  speedup comes from model size, not a faster proof system. A fair comparison
  needs matched sequence length and parameter count.
## GPT-2 124M @ seq=512 (2026-09-29)

Scaled context from 16 to 512 tokens.

| metric | value |
|---|---|
| wall time | ~43 min 43 s |
| peak RSS | ~8.4 GB |
| correctness | argmax matches onnxruntime on 511/512 tokens |

seq=512 fix:

- Causal mask deepened from -2^21 (-32) to -2^30 (-16384). GPT-2's QK^T/8
  attention scores reach ~304 (layer 4), so the old -32 additive mask let
  future tokens leak into the softmax and flip argmax. The deeper mask fully
  suppresses them (the exp table clamps at -32).
- One remaining mismatch (pos 172: 262 vs 257) is a close call, consistent with
  2^16 fixed-point quantization accumulating over 512 positions (a precision
  artifact, not a logic bug).

The wall-clock bottleneck is WHIR FRI opening (60% of wall time), the same
bottleneck as TimesFM, not the GKR sumcheck.

## GPT-2 124M opening reduction (2026-09-30)

Optimized the GPT-2 124M seq=16 proof (IR two-phase executor, `prove_gpt2_ir`)
by reducing the number of WHIR opening proofs. Results verified against the
reference argmax.

Machine: 64-thread CPU, 3x NVIDIA L20, 495 GB RAM, release build.

| Metric | Before | After | Change |
|---|---|---|---|
| WHIR openings | 5659 | 3835 | -32.2% |
| CPU wall time | 53.1 s | 48.5 s | -9% |
| Peak host memory | 16.2 GB | 6.1 GB | -62% |
| GPU wall time | - | 72.9 s | launch-bound |

Two kinds of opening reduction, with very different payoff:

- **Same-point merge** (different tables opened at one point via
  `open_batch_multi`): affine in/out, lookup x/y/a, softmax steps. Saves only
  the per-proof fixed overhead, so the wall-time gain is small.
- **Single-table multi-point reduction** (the same table opened at several
  different points, reduced to one FRI proof plus one sum-of-products
  sumcheck): the real win. LayerNorm opens `x` at three points per row; merging
  them cut 800 openings (-17%). Implemented as `open_table_multi_point` and
  `batch_open_committed` in `zkie_core::pcs::batch_open`.

Other changes:

- Streaming per-layer proof (`prove_gpt2_ir`): forward + prove each layer with
  a fresh executor, dropping it before the next layer, cutting peak memory from
  16.2 GB to 6.1 GB.
- Fixed a double-count in `GLOBAL_OPEN_COUNT` (was reporting 2x the real opens).
- `CudaDft` now falls back to the CPU DFT for codewords larger than the fixed
  twiddle buffer (MAX_LG), instead of panicking (e.g. the 2^26 lm_head weight).

Key findings:

- At seq=16 the codewords are small, so the GPU is launch/transfer-bound and
  slower than the 64-thread CPU (72.9 s vs 48.5 s). GPU acceleration is expected
  to win at larger sequence lengths / models where the codewords are bigger.
- The remaining ~1200 openings are the matmul A/B/C evaluations: three tensors
  of three different sizes, each opened once per matmul, which cannot be merged
  within a single matmul and are mostly unique across ops.

## GPT-2 124M seq=512 LayerNorm opening aggregation (2026-09-30)

Batched LayerNorm rows into per-chunk groups (16 rows per chunk) so each
LayerNorm's many per-row openings collapse into a handful of multi-point
batch openings (`prove_layer_norm_rows_batch` + `batch_open_committed`).

seq=512, 64-thread CPU, `Whir::new_testing`:

| metric | before | after |
| --- | --- | --- |
| global_open_count | 53435 | 29435 |
| wall time | 21.85 min | 33.8 min (incl. ~4 min manual suspend/resume) |
| peak RSS | 30.1 GB | 30.1 GB |
| argmax vs ground truth | - | 511/512 (1 near-tie, fixed-point) |

Notes:

- Opening count drops ~45% (LayerNorm rows 2x12800 -> 32 chunks x 2), which
  lowers verification cost, but prover wall time does not improve: the CPU
  prover is dominated by the FRI commit over the total witness plus the GKR
  matmul sumchecks, neither of which the opening count changes.
- A single all-rows batch (512 tables, 2^19 codeword) was ~10x slower on CPU;
  chunk=16 keeps each batch at a 2^14 codeword.
- GPU (`ZKIE_CUDA=1`) is slower still here: the matmul/affine commitments are
  many and small, so the CUDA Merkle/DFT path is launch-bound.

## GPT-2 124M seq=512: LogUp limb range check (2026-09-30)

Replaced the softmax 31-binary-column range check with two 16-bit limbs plus a
LogUp lookup per limb (`prove_limbs_range`), reusing the existing
`prove_lookup` grand-product argument. This drops the softmax range-check
witness from `31*n` to roughly `8*n` per side and removes the per-bit sumchecks
(62 -> ~6 per range check).

64-thread CPU, `Whir::new_testing`:

| metric | before (31-bit) | after (LogUp limbs) |
| --- | --- | --- |
| layer 0 wall time | 125.2 s | 99.7 s |
| full seq=512 wall time | ~27 min | 19.6 min (1175 s) |
| global_open_count | 29435 | 29747 |
| commit time (all batches) | - | 112.3 s |

The commit (Merkle over the total witness) is now only ~10% of wall time; the
rest is GKR/sumcheck work plus the FRI opens. Next levers: replace the
affine/scale bit decomposition (`prove_round_batch`) the same way, then attack
the matmul/sumcheck hot path.

## Layer-granularity GPT-2 512: per-block costs (2026-10-01)

Layer circuit, plain model, single-threaded. Examples in
`crates/zkie-engine/examples/`:

| block | dims | time |
| --- | --- | --- |
| projection (matmul + affine + logUp) | 512x1024x1024 | 2.06s |
| full FFN (2 projections + gelu) | 512x1024x4096 + 512x4096x1024 | 18.55s |
| softmax (exp lookup + row-sum) | seq=512, table 2^18 | 0.14s |

Findings:

- softmax is cheap (0.14s); the dominant cost is the projection affine + logUp
  range check (O(m*n) elementwise sumcheck + fraction tree), not the matmul GKR
  (O(contraction dim), ~0.03s).
- Softmax rescale is now O(m*n) (NOT the O(m*n*n) flattened product):
  `softmax_scaled::prove_softmax_scaled` proves exp lookup + row-sum + rescale
  (field-inverse `out*sum_broadcast = e`, sum as a separate tensor) in 0.16s at
  seq=512. This removes the 2^27 (infeasible) flattened rescale.
- Full transformer layer (single-head attention + FFN + 2 residual adds, plain
  model, no layernorm yet): 27.57s/layer proof. 12-layer end-to-end (incl.
  witness generation) = 624s (~10.4 min) via `bench_gpt2_12layer`; proof-only is
  ~5.5 min. Witness gen (forward matmul) is ~half the wall time and is a
  separate optimizable axis (GPU/parallel), not part of the proof. Multi-head
  (12x the softmax/matmul, ~+2s) and layernorm were still to be wired on top.
- Full layer extrapolation: attention (~6s projections + ~1.7s softmax) + FFN
  (18.55s) + layernorm (~1s) ~= 27s/layer, ~5.4 min for 12 layers, ~3.5x faster
  than the op-granularity 19.6 min.
- Committed cost (testing Whir params): committing + opening one projection's
  5 tensors (x,w,bias,out,rem) = 0.30s vs 2.06s proving, so WHIR commitment is
  ~15% overhead. The affine + logUp range check (O(m*n)) dominates, not the
  openings; aggregate-openings is a second-order win on CPU, and layer-boundary
  commitment (one WHIR per layer) removes most of the per-op commit cost anyway.
- Caveats: plain model (affine base tensors not WHIR-committed, only the logUp
  fraction tree is); single-threaded; real gelu/exp tables are 2^21..2^23 (the
  bench used 2^16..2^18). N-ary sharding + layer parallelism + GPU were still to
  be added on top.

## Layer-granularity GPT-2 512: performance journey (2026-10-01)

GPT-2 512 scale, synthetic weights, single-head, post-norm, plain model.

| step | per-layer | notes |
| --- | --- | --- |
| op granularity (old baseline) | ~19.6 min total | every op committed/sumchecked/opened |
| layer granularity (claim-chained, single-threaded) | 58.9s | one g per layer, virtual intermediates |
| + row-parallel matmul (64 cores, `par::mm_par`) | 6.55s | 9x |
| + parallel sumcheck (`prove_virtual` hot loops) | 3.62s | 1.8x |
| 12-layer end-to-end (layer-parallel proof) | ~32s | witness 9.4s + proof 22.6s |

Key findings:

- The bottleneck is elementwise work (forward matmul and the affine+logUp
  sumcheck), not the matmul GKR reduction or the WHIR commit/open. Both are
  embarrassingly parallel across rows/elements, so 64 CPU cores give ~30x.
- WHIR commit+open is ~0.30s/projection (testing params) and GPU commit is only
  ~1.1-1.4x (launch/transfer-bound) — the real lever is CPU parallelism, not
  the GPU commit path.
- Claim chaining (same_poly) adds ~7s/layer over independent proofs — the
  "eliminate claim merge overhead" target.
- The 12-layer proof does not scale to the theoretical ~6s (measured 22.6s):
  nested parallelism (12 std threads x rayon pool) contends; switching the outer
  loop to `rayon::par_iter` is applied but not yet confirmed effective.
## Gemma 3 270M full model (op-primitive shard DAG) (2026-10-01)

Real Gemma 3 270M (18 layers, GQA 4 heads x head_dim 256, QK RMSNorm, RoPE,
post-norm sandwich, gated-GELU MLP), sequence length 16, 1154 ops, proven through
`compose::prove_shard_dag` with `same_poly` cross-shard binding. Release build,
64-thread CPU. This is the first model exercising the RoPE primitive and the
Gemma "1 + gamma" RMSNorm convention.

| seq | shards | prove | verify | total | argmax | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 22 (per layer) | 33.4 s | 16.6 s | ~50.0 s (0.83 min) | 16/16 | ~25.1 GB |
| 512 | 22 (per layer) | 56.5 s | 36.7 s | ~93.1 s (1.55 min) | 505/512 | ~27.4 GB |

The per-layer granularity (22 shards) is the fastest of the swept
configurations; coarser granularities (fewer, larger shards) are slower because
each shard's same-poly binding cost grows with shard size.

## DeepSeek-V2-Lite full model (op-primitive shard DAG) (2026-10-02)

DeepSeek-V2-Lite (27 layers, MLA attention + DeepSeekMoE: 64 routed experts top-6
+ 2 shared experts), proven through `compose::prove_shard_dag` with `same_poly`
cross-shard binding. Release build. The MoE is assembled lazily (only routed
experts are opened), and the top-k routing is proven in-circuit: a `TopKSelect`
op enforces the gate equals the softmax scores on the selected experts, the
selection is binary and threshold-consistent (`sel*(x-thr)` and
`(1-sel)*(thr-x)` are non-negative), and exactly `k=6` experts are selected per
row, so the gate is no longer a trusted constant.

| seq | shards | prove | verify | total | argmax | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 21 (per layer) | 516.5 s | 270.8 s | ~13.1 min | 16/16 | ~124.5 GB |
| 512 | 28 (per layer, mmap weights) | 1552.5 s | 1081.4 s | ~43.9 min | 512/512 | ~372.2 GB |

Lazy expert opening drops the op count from 20604 (dense) to 14638 at seq=16
(19854 at seq=512) and peak memory from ~272 GB to ~242 GB at seq=16. At seq=512
the top-6 routing spans 58 of 64 experts, so lazy opening approaches dense; the
MoE expert weights are memory-mapped (file-backed, i32 read + converted on
demand), so the kernel pages them in/out instead of keeping them resident. That
lets the full 64-thread pool run without OOM, cutting prove from ~77 min (8
threads) to ~26.5 min. `verify` recomputes the witness in-place (no store clone), so it no longer
doubles memory.
