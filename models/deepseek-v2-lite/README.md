# DeepSeek-V2-Lite

`zkie-models-deepseek-v2-lite` builds the DeepSeek-V2-Lite forward (MLA
attention + MoE FFN) as an op-primitive shard DAG.

The MoE is assembled lazily (only routed experts are opened) and the top-k
routing is proven in-circuit: a `TopKSelect` op enforces the gate equals the
softmax scores on the selected experts, so the gate is not a trusted constant.

## Weights

`deepseek-ai/DeepSeek-V2-Lite` is public on Hugging Face (4 bf16 shards,
~31 GB). Download them into `models/deepseek-v2-lite/raw/`, then run:

```sh
DS_SEQ=16 /usr/bin/python3.11 scripts/extract_deepseek.py
DS_SEQ=16 /usr/bin/python3.11 scripts/ref_deepseek.py
```

`extract_deepseek.py` writes the padded int32 weights + RoPE/SiLU tables + `x0`
under `models/deepseek-v2-lite/weights/` (~84 GB). `ref_deepseek.py` writes the
per-layer routing mask and ground-truth argmax/logits.

## Prove / benchmark

```sh
cargo run --release -p zkie-models-deepseek-v2-lite --example prove
cargo run --release -p zkie-models-deepseek-v2-lite --example bench
```

## Benchmark (seq=16, release, 64-thread CPU)

| shards | prove | verify | argmax | peak RSS |
| --- | --- | --- | --- | --- |
| 21 (per layer) | 486.2 s (~8.1 min) | 270.5 s (~4.5 min) | 16/16 | ~242 GB |

Lazy expert opening (only routed experts) brings the op count down to 14638 and
prove time to 486.2 s; the top-k routing is proven in-circuit.
