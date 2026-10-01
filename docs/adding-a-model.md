# Adding a new model

How to bring a new model into zkIE end to end: export to ONNX, extract
fixed-point weights and lookup tables, map the op graph onto the op primitives,
prove + verify, then autotune the shard granularity. The op primitives and the
engine are model-agnostic; only the extractor and the op-graph assembly are new
per model.

## Pipeline at a glance

```
HuggingFace / torch model
   |  (1) export to ONNX
   v
model.onnx
   |  (2) extractor script (extract_gpt2.py pattern)
   v
models/<model>/...  (i32 fixed-point weights + lookup tables + ground truth)
   |  (3) build_* fn in a Rust example
   v
Vec<compose::Op>  (op-primitive graph)
   |  (4) prove_shard_dag / prove_shard + argmax sanity
   v
verified proof
   |  (5) engine::autotune_with sweep
   v
tuned granularity (tune once, reuse)
```

## 1. Export to ONNX

Export the model to one ONNX file (graph + initializers).

```python
# torch
import torch
model = load_your_model()
dummy = torch.zeros((1, seq_len), dtype=torch.int64)  # match the model input
torch.onnx.export(model, dummy, "model.onnx", opset_version=14)
```

For a HuggingFace model:

```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("your/model")
# then torch.onnx.export(m, ...) with the right input spec
```

The GPT-2 example in this repo uses `models/gpt2/gpt2.onnx` (already exported).

## 2. Extract weights and tables

The extractor reads the ONNX, quantizes every weight/bias to i32 fixed point
(scale `2^16`), pads each tensor to a power of two, and writes:

- per-layer weights/biases — `models/<model>/*_w_i32.bin`, `*_b_i32.bin`
- the embedding for a fixed input — `embedding_i32.bin`
- nonlinearity lookup tables — `gelu_table_i32.bin`, `exp_table_i32.bin`,
  `rsqrt_table_i32.bin`
- ground-truth logits / argmax for validation — `gt_logits_*`, `gt_argmax_*_i32.bin`

Reference: `scripts/extract_gpt2.py` (run via `scripts/run_extract_gpt2.sh`,
which uses a `python:3.12` Docker image with `onnx onnxruntime numpy`). The
conventions to keep:

- `SCALE = 65536.0`; quantize with `round(x * SCALE)` and clip to `i32`.
- pad every dimension to a power of two (WHIR's multilinear extension requires
  it); pass the real dimension separately as `n_real`.
- run onnxruntime once and save ground-truth logits / argmax for the prover to
  validate against.

Fixed-point scales (see `docs/spec.md`): activations/weights `2^16`, matmul
accumulation `2^32`, LayerNorm raw output `2^48`.

## 3. Map the op graph onto primitives

Write a Rust example that walks the model op graph and pushes `compose::Op`
primitives into a `Store`. The op set lives in `crates/zkie-ops/src/compose.rs`:

| ONNX op | `compose::Op` |
| --- | --- |
| MatMul | `MatMul` (GKR over the contraction index) |
| Add / elementwise | `Add` |
| multiply by a constant | `Scale` |
| Transpose | `Transpose` |
| MatMul + bias + round | `Projection` |
| Softmax | `StableSoftmaxIndex` + `Softmax` |
| GELU | `GeluIndex` + `Lookup` |
| LayerNorm (centered) | `LayerNormCentered` |
| table lookup | `Lookup` |

Reference: `crates/zkie-engine/examples/prove_gpt2.rs` — `build_layer`
assembles the full 12-layer GPT-2 graph. The residual stream and the multi-head
attention show that multiply-consumed tensors are bound automatically via
`same_poly`.

Then prove and verify the assembled graph, and check argmax against the
ground-truth file:

```rust
let proof = prove_shard_dag(&mut store, &ops, ops_per_shard, &mut rng);
assert!(verify_shard_dag(&store, &ops, ops_per_shard, &proof));
// argmax over the first VOCAB logits, compared to gt_argmax_*_i32.bin
```

## 4. Autotune

Sweep shard granularity with `engine::autotune_with` and keep the lowest-total
configuration:

```rust
use zkie_engine::engine::{autotune_with, Backend, Granularity, StageSchedule, TuningResult};

let granularities = [
    Granularity::WholeModel,
    Granularity::Layers(3),
    Granularity::Layers(1),
    Granularity::Ops(88),
];
let schedules = [StageSchedule::uniform(Backend::Cpu)];

let best = autotune_with(&granularities, &schedules, |g, _sched| {
    let ops_per_shard = g.ops_per_shard(ops.len(), LAYERS);
    // run prove_shard_dag + verify_shard_dag, return a TuningResult whose
    // total_s is the measured prove + verify wall time
});
```

Reference: `crates/zkie-engine/examples/bench_gpt2_autotune.rs`. Tune once per
model and cache the result.

## What is model-specific vs reused

Model-specific (write once per model):

1. the extractor script (ONNX → i32 `.bin` + tables);
2. the `build_*` function (op-graph assembly);
3. the argmax / ground-truth validation.

Reused unchanged:

- all op primitives (`zkie-ops::compose`, `projection`, `layer_norm_centered`,
  `softmax_scaled`, `layernorm_chain`);
- the shard-DAG composer and cross-shard `same_poly` binding;
- `zkie-engine::engine` autotune;
- the forward matmul (`zkie-ops::par`) and WHIR commitments (`zkie-core::pcs::whir`, `committed`).
