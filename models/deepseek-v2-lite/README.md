# DeepSeek-V2-Lite

`zkie-models-deepseek-v2-lite` builds the DeepSeek-V2-Lite forward (MLA
attention + MoE FFN) as an op-primitive shard DAG.

The MoE is assembled lazily (only routed experts are opened) and the top-k
routing is proven in-circuit: a `TopKSelect` op enforces the gate equals the
softmax scores on the selected experts, the selection is binary and
threshold-consistent, and exactly `k=6` experts are selected per row, so the
gate is not a trusted constant.

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

## Benchmark (release)

| seq | shards | prove | verify | argmax | peak RSS |
| --- | --- | --- | --- | --- | --- |
| 16 | 21 (per layer) | 489.8 s (~8.2 min) | 268.5 s (~4.5 min) | 16/16 | ~242 GB |
| 512 | 28 (per layer, mmap weights) | 1588.1 s (~26.5 min) | - | 512/512 | ~377 GB |

Lazy expert opening (only routed experts) brings the op count down to 14638 at
seq=16 and 19854 at seq=512; the top-k routing is proven in-circuit. At seq=512
the top-6 routing spans 58 of 64 experts, so lazy opening approaches dense; the
MoE expert weights are memory-mapped (file-backed, i32 read + converted on
demand) so the full 64-thread pool runs without OOM (26.5 min instead of 77 min
at 8 threads). `verify` is still skipped (it clones the full store, ~2x
memory).
