# zkIE 概览

**zkIE：面向定点推理的可验证计算。** 当前实现是一条三阶段流水线：
**commit → prove → verify**，三个阶段是独立的进程入口。

> 开发中。当前支持 GPT-2 124M、Gemma 3 270M、TimesFM 1.0 200M。证明不是零知识
> 证明：权重、查找表和中间激活都属于公开见证数据。

## 证明什么

对一条公开陈述 `(模型承诺, 输入, 期望输出或期望 token)`，证明者要证明：

- 陈述与公开的模型承诺、输入绑定；
- 执行中读到的每一段权重和查找表都来自已承诺模型的开区间；
- 每个算子步骤在打开后的输入、输出和中间向量上满足该算子的代数关系；
- 步骤之间的数据流与声明的生产者/消费者边一致；
- 最终输出（或最终 token 选择）与陈述中的期望值一致。

验证方只检查公开承诺、陈述和证明，不重新运行模型，也不读取权重文件。

## 三阶段

### commit

读取模型的权重和查找表资产，把每个资产按行哈希成叶子并构建 Merkle 树：

- 资产由 `asset_id`、路径、元素数、行宽和编码（`i32-le` 或 `i64-le`）描述；
- 叶子绑定表身份，资产树根写入 `LeafRecord`；
- `AnchorRecord` 绑定模型承诺、参数摘要和全部叶子记录；
- `ModelCommitmentPublic` 只包含锚点和资产元数据，不包含权重值。

commit 产物包含公开模型承诺、provider 配置、协议描述和私有参数文件；验证只需要公开部分。

### prove

prove 读取已有的 commit 产物，按 `(asset_id, 行区间)` 重新打开已承诺资产，执行模型的定点 executor。每个算子步骤产生一条关系回执：

- 算子码及其关系配置（维度、移位、截断范围、表描述）；
- 该步骤使用的 oracle 注册表根；
- 以 oracle 形式提交的输入、输出、原始累加器、商、范围辅助量和辅助向量，每段都带 Merkle 路径；
- 外部输入到共享资产池的映射：同一段权重或查找表在一次 trace 中只打开一次；
- 本地输入到更早步骤输出的映射。

序列化后的 `SharedTrace`（token、资产打开、步骤、最终 argmax 和最终输出）就是证明。

### verify

verify 读取公开模型承诺、陈述、证明和公开输入：

1. 校验模型承诺和陈述绑定；
2. 校验 token/输入绑定；
3. 对每个资产打开，校验 link proof 到锚点，并校验每一行的 Merkle 路径；
4. 对每个步骤，重建 oracle 注册表，从打开值校验该算子的关系方程；
5. 校验生产者/消费者边；
6. 校验最终 argmax 或输出张量。

任何一项不一致都会拒绝。验证方不会调用算子的 forward 实现。

## 共享模型承诺与公开查找表

权重和查找表走同一条承诺路径。查找表也是模型资产：commit 时按行承诺，prove 时一次打开，相关步骤把它作为外部输入引用。验证方用与权重打开相同的锚点和 Merkle 根校验，证明者无法替换成另一张表。

## 支持的算子

`add`、`affine_scale`、`argmax`、`gated_ffn`、`hadamard`、`index_gen`、
`matmul`、`projection`、`profile_norm`、`profile_rms`、`profile_rope`、
`profile_standardize`、`relu`、`softmax`、`softmax_rows`、`table_lookup`。

每个算子都有定点 forward 实现和验证方检查的代数关系；整数舍入使用显式移位和数学 floor，见证计算中不使用宿主浮点。

## 模型

| 模型 | 输入形态 | executor |
| --- | --- | --- |
| GPT-2 124M | 16 或 512 token | `models/gpt2` |
| Gemma 3 270M | 16 或 512 token | `models/gemma3` |
| TimesFM 1.0 200M | 512 个原始点切成 16 个 patch | `models/timesfm` |

模型权重和查找表是外部资产，位于各模型的 `weights/` 目录（已被 gitignore）；`scripts/` 提供导出和定点模拟脚本。模型 crate 只包含定点配置和共享 trace executor。

## 运行

```bash
cargo build --workspace
cargo test --workspace

# 构建并发布共享模型承诺
cargo run --release -p zkie-bench -- commit \
  --case <case.toml> --assets-manifest <manifest> --parameters <parameters> \
  --out <commit-dir> --run-dir <run-dir>

# 执行模型并写出证明
cargo run --release -p zkie-bench -- prove \
  --commit <commit-dir> --input <input> --model-id <model-id> \
  --out <proof-dir> --run-dir <run-dir>

# 验证公开承诺、陈述和证明
cargo run --release -p zkie-bench -- verify \
  --public <commit-dir>/public --statement <proof-dir>/statement.bin \
  --proof <proof-dir>/proof.bin --public-inputs <proof-dir>/public-inputs.bin \
  --model-id <model-id> --policy <policy> --run-dir <run-dir>
```

`zkie-bench run` 一次完成 commit、prove、verify；`matrix`、`decode-matrix`、`profile-matrix` 在 case 列表上重复同样的阶段并记录分阶段测量。`zkie-check` 用于资产、contract 和 run report 校验。

## 验证边界

- **非零知识**：权重、查找表、激活和中间向量在证明中公开。
- **哈希承诺**：承诺节点和 transcript 使用域分离哈希；字段层提供 Goldilocks 基域和三次扩域运算供 kernel 使用。
- **公开陈述**：验证方信任输入的 case 配置和 policy，并通过陈述把它们与承诺、证明绑定。
- **不重跑模型**：验证只读取公开承诺、陈述、证明和公开输入。
- **性能**：当前 commit/prove/verify 时间、峰值内存和证明大小均为**待测**；本文档不记录历史数字。

## 许可

Apache-2.0，见 [`LICENSE`](../LICENSE)。
