# zkIE Sub-Project 5 (Design) — DAG 分片 + 并行证明 + 聚合链接框架

**日期**: 2026-07-27
**状态**: 设计已通过 brainstorming 讨论确认,待写实现计划
**动机**: 参考 SP1(zkSTARK 证明 + SNARK 递归聚合)的架构思路——电路可以拆开并行计算,最后再聚合。本设计只解决"怎么拆、怎么并行、怎么聚合校验"这个编排层问题,STARK 证明后端本身留到后续 sub-project 再选型/实现。

## 0. 这次不做什么(明确排除)

- 不选定/实现真实的 STARK 证明库(Plonky3/Winterfell/Stwo 等)——`Prover` 是可插拔接口,这次用 `MockProver`(真实计算 commitment,但不生成真实证明)。
- 不实现真正的递归/聚合 SNARK——`Linker` 这次是一个"检查 commitment 是否对得上"的校验流程,不是一个简洁证明。把它换成真实的 SNARK-of-STARKs 递归聚合是未来工作。
- 不做"自动检测重复子图"的通用算法——分片边界由人工审核的静态文件定义,不是运行时算法推断出来的。
- 不做分布式调度——`Prover` trait 的设计不排斥"以后把某个 shard 的证明任务发给远程节点",但这次只做进程内并行(rayon)。

## 1. 背景:真实模型的图结构(不是假设,是读文件读出来的)

分析对象:`models/timesfm_1_0_200m.onnx`(TimesFM 1.0 200M 真实导出,1.4MB)。

- 全图共 **778 个 ONNX 节点**。`op_type` 统计中 `Softmax`/`LayerNormalization`/`Relu`/`ReduceMean` 等均恰好出现 **20 次**,与 TimesFM 20 层 transformer block 的真实层数吻合(FinText 8M 模型对应文件里同一批 op 出现 7 次,对应它是 7 层)。
- 节点名字是自动生成的(`node_sub`、`node_lt` 之类),**不保留** PyTorch 模块路径(不是 `/layers.0/attn/...` 这种命名),所以"第几层"这个边界不能靠读名字,只能靠数重复出现的 op 序列。
- 用 `LayerNormalization` 节点下标定位:`[108, 144, 178, 212, ..., 756]`,层与层之间间隔几乎全部是 **34**(只有第 0 层因为紧挨着 prologue 多出 2 个节点,是 36)。逐节点核对后确认:**层的真实边界是"残差 Add → 下一层 RMSNorm 起点(`Pow`)",不是 `LayerNormalization` 节点本身**(`LayerNormalization` 其实在层的中段,对应 FFN 前的 norm)。
- 图的真实形状是 **prologue(节点 0–81,输入归一化/mask/patch embedding/位置编码)+ 20 个结构同构的 layer block + epilogue(输出投影头 + 反归一化)**,不是从头到尾均匀重复的一条链。
- 发现两处**广播型依赖**(不是沿链传递,而是从 prologue 直接扇出给多个下游):
  1. attention mask(`minimum_1`)——prologue 算一次,20 层各自的注意力步骤都直接引用它。
  2. 反归一化统计量(`div`/`clamp_2`,即输入的均值/标准差)——prologue 算一次,epilogue 最后一步直接引用,跳过全部 20 层。

这些发现直接决定了下面的 DAG 设计:不能假设"纯链式"依赖,必须显式支持"一对多"的广播边。

## 2. 整体分层架构

```
Model（ONNX 文件，如 TimesFM / 未来的 Gemma）
   ↓ zkie-compiler 编译（现有流程，完全不变）→ CompiledProgram
   ↓
engines/<model>/   —— 模型专属的 "IE" 层：
   · partition.toml —— 人工审核过的静态分片边界定义
   · PartitionPolicy 实现 —— 加载 partition.toml，不跑检测算法
   · 组装 Dag、调用 Prover、跑 Linker 的胶水代码
   ↓ 调用
zkie-core（新增 dag 模块，模型无关，通用）：
   · Shard / EdgeKind（Sequential | Broadcast）数据结构
   · build_dag：对 CompiledProgram 做寄存器级 def-use 分析，自动推导 shard 之间的依赖边
   · Prover trait + MockProver（真实 hash commitment，假 validity）
   · Linker：校验每条边的 commitment 一致性 + 每个 shard 自身 validity
   · ISA（现有 chips，不变；以后这里是接入 zkGPT/DeepProve 式电路优化、真实 STARK 后端的地方）
```

**目录/crate 布局**:新增顶层目录 `engines/`(`models/` 已用于存放 `.onnx` 文件,不重用),`engines/timesfm/` 是独立 crate(包名 `zkie-ie-timesfm`),加入 workspace `members`,只依赖 `zkie-core` 和 `zkie-compiler`。这样设计是为了以后可以把 `engines/timesfm/` 整个拆成独立 repo(它对 zkie 的唯一依赖就是公开的 ISA/Dag/Prover/Linker 接口)。未来加 Gemma 就是新增 `engines/gemma/`,复用同一套 `zkie-core` 机制。

## 3. `zkie-core::dag` 模块设计

- **`Shard`**:`{ id, instruction_range: Range<usize>, name: String }`——对应 `CompiledProgram` 里一段连续指令区间。
- **`EdgeKind`**:`Sequential`(上一个 shard 直接产出,主链顺序传递)或 `Broadcast`(更早的 shard 产出、被 2 个以上 shard 各自独立引用,如 attention mask)。
- **`build_dag(program: &CompiledProgram, boundaries: &[usize]) -> Dag<Shard>`**:给定切分点,扫描每条指令读写的 `RegisterRef`,做标准的 def-use 分析——谁写了某个寄存器、后面哪些 shard 读了它——从而自动推导出跨 shard 的依赖边和边的种类。这一步是纯粹基于 `CompiledProgram` 现有信息的机械分析,不需要回头解析原始 ONNX 图,也不含任何模型特定的知识。
- **`Prover` trait**:`fn prove(&self, shard: &Shard, witness: &ShardWitness) -> ShardProof`。`ShardWitness` 是明文推理跑完后,该 shard 指令区间涉及的全部寄存器具体值(含它引用的 `Sequential`/`Broadcast` 输入值)的一个只读切片。`ShardProof` 包含每个输入/输出寄存器的 **commitment**(真实计算,如 blake3 哈希)和一个 `validity` 标记。
- **`MockProver`**:老实计算 commitment,但 `validity` 恒为真、不生成任何真实证明。这是本次唯一的 `Prover` 实现。以后换真实 STARK 后端只需要新增一个 `Prover` 实现,`Linker`/`Dag` 完全不用改。
- **`Linker`**:`fn link(dag: &Dag<Shard>, proofs: &[ShardProof]) -> Result<(), LinkError>`。遍历每条边(`Sequential` 或 `Broadcast`),检查生产者的输出 commitment 是否等于消费者声明的对应输入 commitment,并检查每个 `ShardProof` 自身的 `validity`。`LinkError` 是类型化的(如 `LinkError::CommitmentMismatch { edge, expected, actual }`、`LinkError::InvalidShardProof { shard_id }`),不是字符串,方便测试精确断言"到底是哪条边错了"。

## 4. 分片策略:人工审核的静态文件,不是运行时算法

- `PartitionPolicy` trait 保留(不同 engine 未来可能有不同加载方式),但 `engines/timesfm` 的具体实现就是**加载一份静态 TOML 文件**,不跑检测算法:

```toml
[[shard]]
name = "prologue"
start = 0
end = 82

[[shard]]
name = "layer_0"
group = "layer"
start = 82
end = 118

# ... 20 条 layer 记录 ...

[[shard]]
name = "epilogue"
start = 764
end = 778
```

(以上下标是示例;真实数值由草稿工具生成、人工核对后写入 `engines/timesfm/partition.toml`,与第 1 节的实际分析一致——层 19 结束于残差 Add 节点 763,`horizon_ff_layer` 输出头从节点 764 开始。)

- 配一个**草稿生成辅助工具**(example 二进制,类似现有的 `bench_rowcount.rs`):对着 `CompiledProgram` 找重复的锚点指令(如每次 RMSNorm 起点),打印候选边界和间隔,人工照着草稿去编辑最终的 `partition.toml`,不用从零手数。这个工具不追求完全准确,因为最终把关的是人工审核。
- **不做**运行时"指纹校验"机制(曾在讨论中提出,后确认过度设计):ONNX 文件里图结构(节点/算子/连接)只取决于模型架构和导出代码,和训练出来的权重数值(`initializer`)是分开的两件事——重新训练/微调模型不会改变指令下标,只有我们自己改动导出代码/编译器逻辑时才需要重新生成 `partition.toml`。这属于人工纪律(改了编译器就重新跑草稿工具、人工核对),不需要额外的运行时基础设施。测试里最多加一句指令总数的简单断言即可。

## 5. 端到端数据流

1. `zkie-compiler` 编译 ONNX → `CompiledProgram`(现有流程,不变)。
2. `engines/timesfm` 加载 `partition.toml` 得到边界列表。
3. `zkie_core::dag::build_dag` 用边界列表 + `CompiledProgram` 生成 `Dag<Shard>`(含 Sequential/Broadcast 边)。
4. 完整跑一遍明文推理(已有能力),得到所有寄存器的具体值——这是每个 shard 并行证明所需的全部 witness,一次性就绪,不在关键路径上。
5. 对 `Dag` 中每个 `Shard` 并行调用 `Prover::prove`(rayon,和现有 `bench_rowcount.rs` 的并行方式一致)——因为 witness 已全部就绪,任何 shard 的证明都不需要等待其它 shard 的证明完成。
6. `Linker::link` 校验所有边的 commitment 一致性 + 每个 shard 自身 validity。

## 6. 测试策略

- **Golden path**:用真实 TimesFM 的 `CompiledProgram` 走完整流程,断言 `Dag` 节点数为 22(prologue + 20 层 + epilogue),`Linker::link` 返回 `Ok`。
- **仿照项目已有的 soundness 测试纪律**(`constrain_equal` 那次教训的延续):故意让某个 shard 的 `MockProver` 输出错误的 commitment,断言 `Linker::link` 精确报出是哪条边、哪个 shard 不一致,而不是笼统失败。
- **广播边专项测试**:构造一个"一个生产者对应多个消费者"的场景(对应真实的 mask/反归一化案例),只破坏其中一条消费边,确认 `Linker` 精确抓出那一条,不误报仍然一致的兄弟边。

## 7. 后续工作(不在本次范围内,但架构不应排斥)

- 选定真实 STARK 证明库,实现真实的 `Prover`。
- 把 `Linker` 换成真正的递归/聚合 SNARK(证明"我验证了 N 个 STARK 证明",产出一个小的、可上链验证的证明)。
- 分布式分片证明:`Prover` 换成"把 shard 证明任务发给远程节点、等结果回来"的实现,`Dag`/`Linker` 不需要改动。
- 自动检测重复子图的通用算法(如果人工维护 `partition.toml` 的成本后续变得不可接受)。
