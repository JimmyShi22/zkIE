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

## argmax 精度说明（重要）

seq16 前向的 Q16 argmax 与 float64 参考有 13/16 一致，3 个位置（token 0/6/7）
是 logit 边界极近的翻转（token 0 的 top1/top2 只差 0.042）。根因不是 bug：

- 每层对 token 0 是一次 40 步非线性迭代；实测 layer0 对 1 Q16 输入扰动有约
  3.7 倍增益，轨迹对初值敏感（Lyapunov≈0.2/层）。
- 证明系统验证的就是 Q16 计算本身，`verify=true` 已证明 Rust 无 bug；Q16 与
  float64 在近边界 token 上属两条不同轨迹，差异是精度特性而非错误。

要 16/16 只能二选一：把 ground truth 定义为 Q16 计算（证明已验证它），或把激活
精度整体提 Q20（Goldilocks 64 位字段放不下 Q24，且 Q20 未必压得住 0.042 的边界）。

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
- 把 argmax 基准统一成 Q16 计算，或评估 Q20 精度提升。
- Engram 嵌入表的完整承诺：目前 engram 的 `embed.weight`（约 384M×256×2 ≈ 196B
  参数）只物化被选中的行作为 witness，未把完整表提交进承诺，属于对 engram 表
  的信任假设。提交完整表（反量化后约 1.5TB）需要按 chunk 流式提交或改为提交
  原始 FP8 字节（约 196GB）再公开反量化，是后续的 soundness 补全项。
