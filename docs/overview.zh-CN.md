# zkIE 项目介绍（中文总结）

> 本文是 [`README.md`](../README.md) 与 [`CLAUDE.md`](../CLAUDE.md) 的中文归纳，用于快速了解项目定位、技术路线、代码结构与当前进度。英文原文仍以 `README.md` 为准。

## 1. 一句话定位

**zkIE = Zero-Knowledge Inference Engine**，一个面向「可验证 AI 推理」的零知识推理证明引擎：把模型导出为 ONNX 后，由 zkIE 用算子级的证明原语拼出该模型前向计算的证明电路，对任意输入生成一份证明，证明「输出就是该模型真正算出来的结果」。

验证方不再重跑前向推理，只需拿着陈述（statement）、查找表（lookup tables）和承诺根（commitment roots）校验证明即可。

> 状态：**开发中，尚不可作为发行版使用**。证明原语层、算子层、分片（shard）层已实现并有测试；其上基于 AI 辅助的自动调优（autotuning）仍在完善。

## 2. 解决的问题与价值场景

当前证明 AI 推理的瓶颈是**成本**——时间和显存/内存开销都远高于推理本身；表达力并不是瓶颈。zkIE 的全部设计围绕降低这个成本。

可验证推理可解锁的场景：

- **免信任的交付与结算**：交付物（包括链下完成的人力工作）可以在链上被判定并自动结算，任务的「评估者」由可信第三方变成合约——这是 agentic commerce 的前提。
- **工作的可信托管（escrow）**：只有工作被证明后才放款，例如交易机器人先证明其策略与真实执行的交易，再获得收益。
- **带可验证 AI 的链上治理**：由可证明的模型输出做客观判定，取代多方人工投票。

## 3. 使用方式（工作流）

1. 将模型导出为 ONNX（包含架构、权重、算子图）。
2. zkIE 依据算子图，用算子级证明原语组合出该模型推理电路。
3. 对任意输入生成证明。
4. 验证方用 `verify_shard_dag` / `verify_committed_shard_dag`（见 [`compose.rs`](../crates/zkie-ops/src/compose.rs)）校验：
   - 朴素路径：只读陈述与查找表，不读 witness，也不需要 `Store`；
   - 完备可信路径（fully-sound）：额外校验权重与分片边界的 WHIR opening。

已测数据：seq=512、CPU 上验证耗时约为证明耗时的 **~25%**。
当前权重与激活值都是公开的——本阶段目标是**正确性与简洁性**（succinctness），暂不做 witness hiding，因此严格说是「非 ZK 的可验证推理」。

## 4. 技术路线

底层算法栈（域与协议）：

- **Goldilocks 64 位素数域**作为工作域；
- **GKR / sum-check**：做线性代数归约（尤其是 matmul 的收缩维归约）；
- **WHIR / FRI**：多项式承诺与 opening；
- **LogUp**：实现非线性算子所需的查找论证（lookup argument）。

设计原则：**原语即接口，算法只是其内部实现**。证明原语按 ONNX 算子类型设计，若日后有更好的算法，只替换原语实现，已为现有模型搭好的证明电路不用改。

## 5. 架构：三层代码 + 一层调优流程

| 层 | 内容 |
| --- | --- |
| 1. 证明原语（Proof primitives） | matmul 的 GKR 归约、LogUp 分数查找、舍入范围检查、WHIR/FRI 承诺与 opening |
| 2. 算子层（Op layer） | 一种 ONNX 算子对应一个证明原语。这是**交付接口，保持稳定**：新模型接在它之上，而不是改它 |
| 3. 分片层（Shard layer） | 连续多个算子折叠进同一个 sumcheck，只对分片边界做承诺与 opening。一个分片里放几个算子是**公开参数**：单算子、若干算子、一层 Transformer、多层乃至整模型，走同一条代码路径 |
| 4. AI 辅助自动调优 | 一个 AI agent 通过 1–3 暴露的接口搜索 `分片粒度 × 各阶段 CPU/GPU 调度 × 数据布局`：测量 → 改动 → 再测量 → 保留最快者。它不是 prover 内部的自优化器，而是使用这些接口的一个流程 |

CPU/GPU **不是按分片一刀切**：分片内每个阶段各自调度——前向 matmul、承诺（FFT + Merkle）、各个 sumcheck、FRI opening 都能独立选择设备，因此大码字可以走 GPU，小而受启动开销限制的仍留在 CPU。

### 算子 → 原语 对照

| ONNX 算子 | 对应证明原语 |
| --- | --- |
| MatMul | 沿收缩维的 GKR 归约，留下输入/输出 claim |
| Add / 逐元素 | 逐元素约束 |
| Softmax | exp 查找 + 行求和 + 除法 |
| LayerNorm / RMSNorm | 归一化 + 仿射 |
| GELU / 非线性 | LogUp 分数查找 |
| Lookup | LogUp 分数 sumcheck |
| Affine（定点舍入） | 舍入范围检查（LogUp） |

## 6. 仓库结构

- [`crates/zkie-core/`](../crates/zkie-core) — 证明底座：
  - `src/common/`：Goldilocks 域、定点嵌入、MLE 求值、sumcheck、matmul / LogUp / `same_poly` 归约、claim 类型；
  - `src/pcs/`：WHIR/FRI 承诺方案（`whir.rs`）、批量 opening（`batch_open.rs`）、已承诺张量（`committed.rs`），以及 `cuda` feature 后的 CUDA 后端（`dft_cuda.rs`、`merkle_cuda.rs`、`cuda_ffi.rs`、`cuda_buffer.rs`）。
  - `cuda/`：`goldilocks_ntt.cu`、`poseidon2_merkle.cu` 内核（`build.rs` 用 nvcc 编译，依赖 Sppark）；
  - `vendor/p3-merkle-tree`：vendored 的 Plonky3 Merkle 树，加了 doc-hidden 的 `MerkleTree::from_parts`，用于在 GPU 上算好摘要后复用上游 open/verify 逻辑（由根 `Cargo.toml` 的 `[patch.crates-io]` 接入）。
- [`crates/zkie-ops/`](../crates/zkie-ops) — 算子级证明原语（每种 ONNX 算子一个）：
  - `compose.rs`：`Op`、`Store`、`prove_shard`、`prove_shard_dag`、跨分片 `same_poly` 绑定；
  - `projection.rs`、`layer_norm_centered.rs`、`softmax_scaled.rs`、`layernorm_chain.rs`：matmul / 归一化 / softmax 原语；
  - `par.rs`：i64 定点前向 matmul（缓存友好，带域运算回退）。
- [`crates/zkie-engine/`](../crates/zkie-engine) — 自动调优引擎：
  - `src/engine.rs`：分片粒度、逐阶段 CPU/GPU 调度、autotune 循环；
  - `benchmark/`：与原语、WHIR/FRI PCS 相关的通用微基准。
- [`models/`](../models) — 每个模型一个 crate（`gpt2`、`timesfm`、`gemma3`、`deepseek-v2-lite`）：算子图构建（`src/lib.rs`）、各自的 `examples/`（prove + benchmark），以及被 gitignore 的 `weights/`（导出步骤与 sha256 见各模型 README）。
- [`scripts/`](../scripts) — ONNX 抽取、定点模拟、保真度（fidelity）校验脚本。
- [`docs/`](../docs) — `spec.md`（设计）、`benchmarks.md`（实测）、`adding-a-model.md`（如何接新模型）、`roadmap.md`。

## 7. 已测性能（64 线程 CPU）

| 模型 | 激活/总参数 | 配置 | Commit | Prove | Verify | argmax | 峰值 RSS | 证明大小 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimesFM 1.0 200M | 200M / 200M | seq=16，108 ops/分片（committed） | ~25.0 s | ~141.4 s | ~4.3 s | - | ~21.7 GB | ~114.6 MB |
| GPT-2 124M | 124M / 124M | seq=16，13 分片（逐层，committed） | ~11.7 s | ~100.1 s | ~0.8 s | 16/16 | ~24.7 GB | ~22.4 MB |
| GPT-2 124M | 124M / 124M | seq=512，13 分片 | ~18.6 s | ~863.0 s | ~3.8 s | 512/512 | ~45.4 GB | ~96.1 MB |
| Gemma 3 270M | 270M / 270M | seq=16，22 分片 | ~21.7 s | ~299.6 s | ~2.0 s | 16/16 | ~60.7 GB | ~42.6 MB |
| DeepSeek-V2-Lite | 2.4B / 15.7B | seq=16，12 分片（expert-major MoE，mmap 权重） | ~79.3 s | ~651.8 s | ~3.6 s | 16/16 | ~156.6 GB | ~96.0 MB |
| DeepSeek-V4.1-Flash | 8B/16B* / 552B | MoE + CSA2 + 视觉（规划中） | - | - | - | - | - | - |

\* DeepSeek-V4.1-Flash 激活参数：prefill 8B / decode 16B。Gemma3 与 DeepSeek 的 seq=512 数据仍待测。

## 8. 常用命令

```bash
# 编译与测试
cargo build --workspace
cargo test --workspace

# 带承诺的路径（commit + prove + verify，真实输入/输出）
M=16 cargo run --release -p zkie-models-gpt2 --example bench_committed
cargo run --release -p zkie-models-timesfm --example bench_committed
cargo run --release -p zkie-models-gemma3 --example bench_committed
M=16 cargo run --release -p zkie-models-deepseek-v2-lite --example bench_committed

# 朴素（无承诺）参考路径
cargo run --release -p zkie-models-gpt2 --example prove
```

注意：

- 权重与查找表不入库，需先跑对应模型的导出脚本填充 `models/<model>/weights/`。
- 需要 **Rust ≥ 1.85**：workspace 各 crate 是 edition 2021，但 vendored 的 `p3-merkle-tree`（及其依赖的 p3-* 0.7.0）是 edition 2024。本仓库已用 [`rust-toolchain.toml`](../rust-toolchain.toml) 固定为 `stable`，避免旧默认工具链（如 1.79）在解析 manifest 时直接报 `feature 'edition2024' is required`。
- 详细性能数据与解读见 [`benchmarks.md`](benchmarks.md)；接新模型的步骤见 [`adding-a-model.md`](adding-a-model.md)。

## 9. 关键设计取舍（阅读代码时的注意点）

- **非 ZK**：目标是把成本和简洁性做下来，权重/激活公开；witness hiding 不在本阶段范围内。
- **原语是稳定接口**：算子层是对外交付物，模型接在其上。因此优化通常落在 `zkie-core` 的归约与 PCS 实现，以及 `zkie-engine` 的调度上。
- **分片粒度是公开参数**：同一套代码支持「一算子一分片」到「整模型一分片」，autotune 只是在这个参数空间里搜索。
- **端到端可复现**：每个模型 crate 自带导出说明与 sha256，`scripts/` 下有定点模拟与保真度校验脚本，用于把浮点参考实现和定点证明路径对齐。

## 10. 许可

Apache-2.0，见 [`LICENSE`](../LICENSE)。
