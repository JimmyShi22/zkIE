# DeepSeek-V2-Lite

`zkie-models-deepseek-v2-lite` builds the DeepSeek-V2-Lite forward (MLA
attention + MoE FFN) as an op-primitive shard DAG.

The MoE is currently assembled densely (all 64 routed experts are computed and
scaled by a precomputed top-6 gate); the top-k routing proof is not yet wired,
so the gate is a trusted constant for now.

## Weights

`deepseek-ai/DeepSeek-V2-Lite` is public on Hugging Face (4 bf16 shards,
~31 GB). Download them into `models/deepseek-v2-lite/raw/`, then run:

```sh
DS_SEQ=16 /usr/bin/python3.11 scripts/extract_deepseek.py
DS_SEQ=16 /usr/bin/python3.11 scripts/ref_deepseek.py
```

`extract_deepseek.py` writes the padded int32 weights + RoPE/SiLU tables + `x0`
under `models/deepseek-v2-lite/weights/` (~84 GB). `ref_deepseek.py` writes the
per-layer routing gate and ground-truth argmax/logits.

## Prove / benchmark

```sh
cargo run --release -p zkie-models-deepseek-v2-lite --example prove
cargo run --release -p zkie-models-deepseek-v2-lite --example bench
```

## Benchmark (seq=16, release, 64-thread CPU)

| shards | prove | argmax | peak RSS |
| --- | --- | --- | --- |
| 29 (per layer) | 806.7 s (~13.4 min) | 16/16 | ~272 GB |

The dense-MoE assembly (all 64 routed experts) drives the op count (20604 ops)
and memory. The top-k routing proof and lazy expert opening are the next steps.
