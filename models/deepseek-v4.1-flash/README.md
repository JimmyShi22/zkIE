# DeepSeek-V4.1-Flash text-only

`zkie-models-deepseek-v41-flash` 目前是 layer0 纵向切片，先把量化 codec、MLA
投影和输出投影逐段和 `scripts/v41_ref_cpu.py` 对齐，再扩展到 40 层全图。

## 已完成并验证

- `zkie-core` 支持 FP8-E4M3/E8M0、packed FP4-E2M1 的流式反量化，并按二维
  rows/cols 分别补零到 2 的幂。
- `zkie-ops` 增加 `BlockQuantizedMatMul` op/proof。
- layer0 `wq_a/wq_b/wkv` 投影、RoPE + attention、`wo_a/wo_b` 输出投影均与
  CPU 参考逐项一致。
- range-based committed prove/verify API 可用。

## 运行

```bash
# 单个量化 matmul committed 切片
cargo run --release -p zkie-models-deepseek-v41-flash --example bench_committed

# layer0 投影对拍
cargo run --release -p zkie-models-deepseek-v41-flash --example layer0_proj_ref
python3 scripts/v41_ref_cpu.py --shard models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors --mode proj

# 其它参考模式
python3 scripts/v41_ref_cpu.py --shard models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors --mode attention
python3 scripts/v41_ref_cpu.py --shard models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors --mode oproj
python3 scripts/v41_ref_cpu.py --shard models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors --mode moe_gate
python3 scripts/v41_ref_cpu.py --shard models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors --mode shared_expert
python3 scripts/v41_ref_cpu.py --shard models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors --mode routed_expert --expert-id 240
python3 scripts/v41_ref_cpu.py --shard models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors --mode mlp

# 生成 token / RoPE 表
python3 scripts/extract_v41.py --raw models/deepseek-v4.1-flash/raw --out /tmp/cache --tokenizer-json models/deepseek-v4.1-flash/raw/tokenizer.json --tokens-out /tmp/tokens.bin --prompt "The sky is" --seq 16
python3 scripts/extract_v41.py --raw models/deepseek-v4.1-flash/raw --out /tmp/cache --rope-out /tmp/rope --seq 16
```

## 待接

- `SparseAttention` / `SparseIndexer` / `HyperConnection` / `EngramLookup` 的
  op/proof。
- 40 层 CSA2/CED 全图、MoE、真实 seq16/seq512 端到端。
