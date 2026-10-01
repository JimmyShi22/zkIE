# Adding a new model

How to bring a new model into zkIE end to end: get the weights, extract
fixed-point weights and lookup tables, map the op graph onto the op primitives,
prove + verify, then autotune the shard granularity. The op primitives and the
engine are model-agnostic; only the extractor and the op-graph assembly are new
per model.

## Pipeline at a glance

```
HuggingFace / torch model
   |  (1a) safetensors (preferred)          (1b) or export to ONNX
   v                                              v
model.safetensors                              model.onnx
   |  (2) extractor script
   v
models/<model>/weights/*  (i32 fixed-point weights + tables + ground truth)
   |  (3) build_* fn in models/<model>/src/lib.rs
   v
Vec<compose::Op>  (op-primitive graph)
   |  (4) prove_shard_dag + argmax sanity
   v
verified proof
   |  (5) shard-granularity sweep (tune once, reuse)
   v
tuned granularity
```

## 1. Get the weights

Two equivalent paths. For HuggingFace models the safetensors path is simpler
(no ONNX export step and no float32 round-trip) and is what `extract_gemma3.py`
uses.

### 1a. Safetensors (preferred for HuggingFace models)

Download the checkpoint and parse it directly. `scripts/extract_gemma3.py` is a
full, stdlib+numpy example:

- read the 8-byte header length, then the JSON header, then slice the tensor
  byte ranges;
- bf16 -> f32 is `(uint16_bits << 16).view(float32)` (numpy);
- f32 weights are already `float32`.

### 1b. ONNX

Export a single ONNX file (graph + initializers):

```python
import torch
model = load_your_model()
dummy = torch.zeros((1, seq_len), dtype=torch.int64)  # match the model input
torch.onnx.export(model, dummy, "model.onnx", opset_version=14)
```

`scripts/export_gpt2.py` and `scripts/export_timesfm_200m.py` are existing
examples of this path.

## 2. Extract weights and tables

The extractor quantizes every weight/bias to i32 fixed point, pads each tensor
to a power of two, and writes under `models/<model>/weights/`:

- per-layer weights/biases — `*_w_i32.bin`, `*_b_i32.bin`
- the embedded input for a fixed input sequence — `x0_i32.bin`
- nonlinearity lookup tables — `gelu_table_i32.bin`, `exp_table_i32.bin`,
  `rsqrt_table_i32.bin` (these are shared; GPT-2's tables under
  `models/gpt2/weights/` are reused by other models)
- RoPE cos/sin tables (if the model uses rotary embeddings) —
  `rope_cos_*_i32.bin`, `rope_sin_*_i32.bin`
- ground-truth logits / argmax — `gt_logits_*`, `gt_argmax_*_i32.bin`

References: `scripts/extract_gpt2.py`, `scripts/extract_timesfm_weights.py`,
`scripts/extract_gemma3.py`.

Conventions that must be kept:

- **scale** `SCALE = 65536.0` (2^16); quantize with `round(x * SCALE)` and clip
  to i32.
- **pad** every dimension to a power of two (the multilinear extension requires
  it); pass the real dimension separately as `n_real`/`H` (not `H_PAD`).
- **weight layout is `[in, out]`**. HuggingFace stores linear weights as
  `[out, in]`, so transpose them (`.T`) before writing. Getting this wrong is
  the most common silent bug — the proof still verifies but argmax is garbage.
- **biases**: many modern models are `bias=False`. The `Projection` op still
  needs a bias tensor, so broadcast a zero vector.
- **RMSNorm weights in Gemma-family models are `1 + gamma`**. The checkpoint
  stores the zero-centered `gamma`; the effective weight is `1.0 + gamma`.
  Using the raw weight silently scales every norm output and the residual
  stream blows up. This is not universal — GPT-2/TimesFM LayerNorm/RMSNorm use
  the stored weight directly — so always check the model's RMSNorm forward.
- **RoPE convention**: `apply_rotary_pos_emb` uses rotate-half, i.e.
  `out[:d/2] = x[:d/2]*cos - x[d/2:]*sin` and
  `out[d/2:] = x[d/2:]*cos + x[:d/2]*sin`, with `cos`/`sin` of length `d/2`
  shared by both halves. The angle for pair `i` is `pos * base^(-2i/d)`.
- **rsqrt coverage**: if a model's residual stream grows (Gemma does), the
  piecewise `rsqrt_table_i32.bin` coarse range must cover the largest
  `mean(x^2)`; `extract_gemma3`'s run needed a wider coarse step (see
  `rsqrt_index` in `compose.rs`). Reuse the existing table unless the model
  overflows it.

Fixed-point scales: activations/weights 2^16, matmul accumulation 2^32, and
norm/affine raw products 2^48 before the `2^32` (or `2^shift`) round back to
2^16.

## 3. Map the op graph onto primitives

Write `models/<model>/src/lib.rs` with a `build_*` function that pushes
`compose::Op` primitives into a `Store`. The op set lives in
`crates/zkie-ops/src/compose.rs`:

| model op | `compose::Op` |
| --- | --- |
| linear (matmul + bias + round) | `Projection` |
| bare matmul | `MatMul` |
| transpose | `Transpose` |
| elementwise add (residual) | `Add` |
| elementwise scale | `Scale` |
| elementwise scale by a vector | `ScaleVec` |
| elementwise multiply (gated activation) | `ScaleVec` with `shift=16` |
| ReLU | `Relu` |
| GELU / SiLU / other nonlinearity | `GeluIndex` + `Lookup` (into a table) |
| softmax | `StableSoftmaxIndex` + `Softmax` (or `SoftmaxIndex` + `Softmax`) |
| RMSNorm | `RmsNorm` |
| LayerNorm (centered) | `LayerNormCentered` |
| LayerNorm | `Layernorm` |
| rotary embedding | `RoPE` |
| table gather / embedding lookup | `Lookup` |

Notes:

- **gated activations** (`gelu(gate) * up`, SwiGLU `silu(gate) * up`) are just
  `ScaleVec { x: gelu_output, scale: up_output, shift: 16 }` — an elementwise
  multiply with a fixed-point round. No dedicated multiply op is needed.
- **GQA / MHA** is assembled per head: slice the Q projection per head, run
  per-head attention, then slice the `o_proj` rows per head and `Add` the head
  contributions (this is how GPT-2 and Gemma 3 are built; no concat op exists).
- multiply-consumed tensors (residual stream, shared K/V) are bound
  automatically via `same_poly`.

References: `models/gpt2/src/lib.rs`, `models/timesfm/src/lib.rs`,
`models/gemma3/src/lib.rs`. The GPT-2 and Gemma 3 builders are the two
representative shapes (pre-norm vs post-norm sandwich, MHA vs GQA+RoPE).

## 4. Prove, verify, and autotune

Prove + verify the assembled graph and check argmax against ground truth:

```rust
let proof = prove_shard_dag(&mut store, &ops, ops_per_shard, &mut rng);
assert!(verify_shard_dag(&store, &ops, ops_per_shard, &proof));
// argmax over the first VOCAB logits, compared to gt_argmax_*_i32.bin
```

Then sweep shard granularity (per-layer is usually fastest; coarser = fewer but
larger shards = slower same-poly binding) and record prove/verify time + peak
RSS. `models/gpt2/examples/bench_sharded.rs` and
`models/gemma3/examples/bench.rs` are the reference harnesses; the generic
autotuner lives in `zkie-engine::engine`.

## What is model-specific vs reused

Model-specific (write once per model):

1. the extractor script (safetensors/ONNX -> i32 `.bin` + tables + ground truth);
2. the `build_*` function (op-graph assembly);
3. the argmax / ground-truth validation.

Reused unchanged:

- all op primitives (`zkie-ops`: `compose`, `projection`, `rms_norm`,
  `layer_norm_centered`, `softmax_scaled`, `layernorm_chain`, `rope`);
- the shard-DAG composer and cross-shard `same_poly` binding;
- `zkie-engine::engine` autotune;
- the forward matmul (`zkie-ops::par`) and WHIR commitments
  (`zkie-core::pcs::whir`, `committed`).
