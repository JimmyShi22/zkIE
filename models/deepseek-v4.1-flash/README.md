# DeepSeek-V4.1-Flash text-only

`zkie-models-deepseek-v41-flash`：text-only 的 committed store-free 端到端证明。
已实现完整 40 层 Hyper-Connection + MLA（滑动窗口 + 共享压缩 KV + sink）+
SwiGLU MoE（384 routed + 1 shared）+ Engram 前向与输出头，并接入流式全量权重
承诺（同一时间只物化一个 batch）。

## 当前状态

- 40 层前向（`build_v41_forward`）端到端可跑，`m=16` 真实 prompt 全程无定点溢出。
- committed prove/verify 全链路验证通过（1 层）：
  `prove≈1h42m verify≈3.6s RSS≈110GB proof≈180MB`（10-bit 测试 WHIR）。
- 40 层 seq16 committed 证明可后台启动（约 68 小时，按层流式提交）。
- 真实 prompt 数据修复：`The sky is` 只有 3 token，之前用 BOS 补 16 会让模型
  真实放大重复 BOS 位置（float64 参考第 8 层约 300 倍），导致 Q16 定点 i64 溢出。
  现改用真实 16-token prompt，由 `scripts/v41_make_prompt.py` 生成 embed/hash id。
- 定点累加已从 i64 升到 i128（`mm_par_fixed` / `Scale` / `ScaleVec`），按字段阶
  取模，避免大值溢出；对 Q16 模型结果不变。

## argmax 状态

修复 Engram 哈希桶 bug（桶模数从 `engram_compressed_vocab_size=99092` 改成
`engram_vocab_size=16000000`）后，seq16 的 Q16 argmax 与 float64 参考 **16/16
完全一致**。此前 13/16 的偏差正是这个 Engram 哈希错误造成的，不是 Q16 精度问题。

## 运行

```bash
# 40 层前向 + argmax（真实 prompt）
ZKIE_REAL=1 ZKIE_LAYERS=40 ZKIE_M=16 \
  ./target-native/release/examples/v41_forward_ops \
  /data/zkIE/models/deepseek-v4.1-flash/raw

# 每层前向对拍（打印 per-layer first8 / tokmax）
ZKIE_LAYERS=3 ZKIE_M=16 \
  ./target-native/release/examples/v41_dump_diag \
  /data/zkIE/models/deepseek-v4.1-flash/raw

# 完整 committed 端到端（1 层或 40 层）
ZKIE_REAL=1 ZKIE_LAYERS=1 ZKIE_M=16 RAYON_NUM_THREADS=64 \
  ./target-native/release/examples/bench_v41_committed \
  /data/zkIE/models/deepseek-v4.1-flash/raw

# float64 argmax 参考（40 层）
python3 scripts/v41_ref_argmax.py \
  --raw models/deepseek-v4.1-flash/raw --m 16 --layers 40 \
  --embed models/deepseek-v4.1-flash/cache/engram/embed_prompt_i32.bin \
  --hash-ids models/deepseek-v4.1-flash/cache/engram/hash_ids_i32.bin

# 生成真实 prompt 的 embed + hash id（默认 16-token prompt）
python3 scripts/v41_make_prompt.py
```

## 待接

- 90-bit WHIR 已就位：`weight_commit_plan` 内部 `SECURITY=90`，边界承诺用
  `with_params(arity, 90, 17)`；bench 里的 `new_testing(10)` 是未使用的遗留参数。
- seq512 真实数据 + 参考 + 证明。
- Engram 嵌入表的完整承诺：目前 engram 的 `embed.weight`（约 384M×256×2 ≈ 196B
  参数）只物化被选中的行作为 witness，未把完整表提交进承诺，属于对 engram 表
  的信任假设。提交完整表（反量化后约 1.5TB）需要按 chunk 流式提交或改为提交
  原始 FP8 字节（约 196GB）再公开反量化，是后续的 soundness 补全项。
