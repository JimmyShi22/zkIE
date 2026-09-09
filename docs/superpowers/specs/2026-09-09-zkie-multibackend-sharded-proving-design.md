# zkIE 多后端分片证明与资源感知调度设计

**日期：** 2026-09-09
**状态：** brainstorming 已确认，实施计划已完成
**替代范围：** 本设计扩展并部分替代 `2026-07-27-zkie-dag-sharding-aggregation-design.md` 中的静态人工分片、进程内并行、Mock-only Prover 和 native-only Linker 方案。

## 1. 目标与结论

zkIE 应能在给定的绝对 CPU、系统内存、GPU 和显存预算下，将 ONNX 编译结果切分成独立可证明的 shard，通过可恢复队列完成 witness、证明、验证和聚合。CPU 与 GPU 只是执行后端；证明语义和 artifact 格式由 `proof_flavor` 决定。

第一条完整生产路径采用 Halo2 KZG CPU backend。接口从一开始保持证明系统中立，使后续能够加入兼容的 Halo2 CUDA backend，并允许用 STARK backend 做独立对照实验。STARK 与 Halo2 proof 不直接混入同一个 aggregation node，除非以后实现明确的 bridge/wrapper circuit。

本设计的核心不变量是：

1. 调度器只能执行前置任务全部为 `verified` 的任务。
2. shard 的输入 commitment 必须由电路约束，并在 aggregation 中等于上游输出 commitment，不能由受益方自由声明。
3. `execution_failed`、`verification_failed` 和 `resource_exceeded` 不得折叠成一个布尔值。
4. 调度器可以选择 CPU 或 GPU，但不得改变已固定并摘要绑定的分片与聚合计划。
5. 只有经过独立验证并原子发布的 artifact 才能被下游消费。

## 2. 范围

### 2.1 本期实现

- 证明系统中立的 `WitnessBackend`、`ProofBackend`、资源描述和 artifact 接口。
- Halo2 KZG CPU leaf backend，将真实 proof bytes、public statement 和验证结果接入生产路径。
- 确定性的 zkIE ISA CPU witness backend。
- 编译期自动分片、稳定的 shard DAG 和可审计 partition plan。
- 绝对资源预算、历史内存校准、依赖感知并发和独立 worker 进程。
- SQLite WAL 状态库、内容寻址 artifact/key store、崩溃恢复和重试。
- circuit-friendly boundary commitment 和严格的 native aggregation manifest。
- 可配置 N 叉 aggregation plan 与 `AggregationBackend` 接口。
- public model 模式的完整支持，并为 private model 保留类型与接口。

### 2.2 后续实现

- Halo2 CUDA proving backend。
- 真正的 N 叉递归 aggregation circuit 与最终 root proof。
- STARK/AIR leaf backend、GPU 基准和可选 STARK-to-SNARK wrapper。
- private model 权重 commitment 电路。
- 多节点远程调度和共享对象存储。

### 2.3 明确不做

- 修改 PSE Halo2，使单个 proof 支持 out-of-core FFT/MSM 或中途 checkpoint。
- 运行时临时改变 shard 边界。
- 将浮点 PyTorch/ONNX Runtime 输出当作 zkIE 定点语义的权威 witness。
- 在没有显式 bridge 的情况下混合不同 proof flavor。

## 3. 分层架构

```text
ONNX model
  -> ONNX importer
  -> backend-independent zkIE ISA / CompiledProgram
  -> PartitionPlanner(target shard memory)
  -> immutable ShardDag + PartitionPlan digest
  -> WitnessBackend
  -> leaf ProofBackend (CPU or GPU execution backend)
  -> independent verification
  -> N-ary AggregationPlan
  -> native verified manifest (phase 1)
  -> recursive AggregationBackend root proof (phase 2)
```

三个图结构必须分开：

- ONNX 计算 DAG 描述模型计算。
- Shard DAG 描述独立 proof 任务及 Sequential/Broadcast 依赖。
- Aggregation tree 描述 proof 如何收敛，N 是最大 fan-in，不是 ONNX 节点的 fan-in。

大型模型可以按 Transformer block 等语义分组，先生成 block-level aggregate，再生成全局 root。所有层级使用同一个 aggregation claim 规则。单个 shard 若仍超过资源目标，编译器继续将其拆成更小 shard，而不是在 shard 内隐藏另一套不可见调度。

## 4. 身份、摘要与公开声明

### 4.1 运行身份

每次 run 固定以下内容并计算 `run_plan_digest`：

- model graph digest；
- weights digest；
- compiler、ISA 和 quantization version；
- partition plan digest；
- proof flavor；
- model visibility；
- aggregation fan-in N 和 aggregation plan digest；
- 公开输入 schema version。

运行期间不得修改这些字段。恢复任务时任一摘要不一致都创建新 run，不能复用旧 artifact。

### 4.2 两类 commitment

- `BoundaryCommitment`：电路内部计算的 field-friendly hash。第一版采用带 domain separation 的 Poseidon-family commitment，具体参数集写入 `proof_flavor`。输入包括 tensor role、dtype、shape、quantization metadata 和按规范编码的 field elements。
- `ArtifactDigest`：BLAKE3，仅用于 proof/key/metadata 文件的内容寻址和损坏检测，不承担证明 soundness。

leaf circuit 必须约束自己真实读取或产生的 boundary values 与公开 `BoundaryCommitment` 一致。aggregation 不接受调用方另行提供的未约束标签。

### 4.3 模型可见性

第一版实现 `public_model`：结构和权重公开，weights digest 与 circuit/VK 绑定，私有内容为输入和中间 witness。

类型系统保留 `private_model`：权重作为 witness，根 statement 公开 `model_commitment`。该模式必须等到权重 commitment 电路完成后才能启用，不能只靠配置标签假装支持。

## 5. Backend 接口

### 5.1 共同类型

```rust
struct ProofFlavorId(String);
struct ExecutionBackendId(String);

struct ResourceRequest {
    cpu_cores: u32,
    ram_bytes: u64,
    gpu_count: u32,
    gpu_vram_bytes_per_device: u64,
}

struct BackendCapabilities {
    execution_backend: ExecutionBackendId,
    proof_flavors: Vec<ProofFlavorId>,
    supports_cpu: bool,
    supports_cuda: bool,
}
```

`execution_backend` 如 `halo2-cpu`、`halo2-cuda`；`proof_flavor` 如 `halo2-kzg-bn256-shplonk-v1`。CPU/GPU backend 只要输出相同 flavor，其 proof 可以混入同一 aggregation tree。backend ID 记录在运行元数据中，但 verifier 的密码学判断依据是 proof、public statement、VK digest 和 flavor。

### 5.2 WitnessBackend

```rust
trait WitnessBackend {
    fn capabilities(&self) -> BackendCapabilities;
    fn estimate_resources(&self, job: &WitnessJob) -> Result<ResourceRequest, BackendError>;
    fn generate(&self, job: &WitnessJob, output: &Path) -> Result<WitnessArtifact, BackendError>;
}
```

第一版 `ZkieIsaCpuWitnessBackend` 严格执行 zkIE 的定点数、舍入、范围和溢出规则。以后可加 CUDA witness backend。PyTorch/ONNX Runtime 只用于导入、交叉检查和基准，不是权威语义来源。

### 5.3 ProofBackend

```rust
trait ProofBackend {
    fn capabilities(&self) -> BackendCapabilities;
    fn estimate_resources(&self, job: &ProofJob) -> Result<ResourceRequest, BackendError>;
    fn prepare(&self, job: &PrepareJob, store: &KeyStore) -> Result<PreparedKeys, BackendError>;
    fn prove(&self, job: &ProofJob, output: &Path) -> Result<UnverifiedProof, BackendError>;
    fn verify(&self, proof: &UnverifiedProof) -> Result<VerifiedProof, VerificationError>;
}
```

接口不得向 scheduler、DAG 或 artifact store 暴露 Halo2 的具体 Rust 类型。Halo2 circuit lowering 位于 Halo2 adapter 内；未来 STARK backend 从同一 `CompiledShard` 降低为 AIR/trace，而不是复用 Halo2 chip 类型。

### 5.4 AggregationBackend

```rust
trait AggregationBackend {
    fn estimate_resources(&self, job: &AggregationJob) -> Result<ResourceRequest, BackendError>;
    fn prepare(&self, job: &PrepareAggregationJob, store: &KeyStore)
        -> Result<PreparedKeys, BackendError>;
    fn aggregate(&self, job: &AggregationJob, output: &Path)
        -> Result<UnverifiedAggregate, BackendError>;
    fn verify(&self, proof: &UnverifiedAggregate)
        -> Result<VerifiedAggregate, VerificationError>;
}
```

第一阶段提供 native manifest implementation，它执行与未来递归电路相同的关系检查，但明确标记为 `native_verified_manifest`，不得冒充 succinct root proof。

## 6. 自动分片

`PartitionPlanner` 在编译后、执行前生成 immutable plan。输入为 CompiledProgram、目标单 shard 内存、proof flavor 和静态 cost model。输出包含：

- shard ID、稳定名称、指令范围和 circuit digest；
- 输入/输出寄存器及 tensor metadata；
- Sequential/Broadcast edges；
- 估算行数、`k`、列数和首次资源预留；
- 人类可读理由，例如语义 block 边界或资源切分点。

算法优先在模型 block、残差边界和算子组边界切分，然后在估算超过目标时继续细分。它必须保留 def-use 依赖，并验证每个跨 shard register 有唯一生产者。计划写盘并摘要绑定；scheduler 只能调度它，不能动态改边界。

支持人工传入目标 shard 内存和经过审核的边界提示，但提示不能跳过结构校验。旧 TimesFM 静态 partition 可作为 hint 和 golden fixture，不再是唯一真实来源。

## 7. 资源模型与调度

### 7.1 默认绝对预算

当前 32 物理核/64 线程、495 GiB RAM、无 swap、3×NVIDIA L20 测试机使用：

```text
memory_budget_gib = 400
memory_hard_limit_gib = 440
cpu_budget_cores = 56
aggregation_fan_in = 4
```

`memory_budget_gib` 是 admission 上限；达到后停止启动新任务。`memory_hard_limit_gib` 是 worker 总内存的紧急线，不是额外可分配额度。所有 GiB 参数都按 `2^30` bytes 解释。GPU 尚未启用时 GPU 预算为零。

### 7.2 估算与校准

首次资源估算来自 `k`、列数、lookup/permutation 规模、SRS、PK 和 backend 静态模型。完成任务后记录整个 worker 进程树的 peak RSS。

历史键为：

```text
circuit_digest + k + proof_flavor + execution_backend + hardware_profile
```

下一次预留：

```text
max(static_estimate, max_observed_peak * 1.15)
```

硬件 profile 包括 CPU 型号/核心数、RAM、GPU 型号/数量、backend build digest 和关键 runtime 版本。样本不足时不使用跨硬件推断。

### 7.3 Ready queue

只有所有 required predecessors 均为 `verified` 的任务才进入 ready。scheduler 使用资源 best-fit：在剩余 CPU、RAM、GPU 和显存中选择能提高利用率的 ready tasks。每个 worker 获得明确的 Rayon 线程数；CUDA worker 通过 `CUDA_VISIBLE_DEVICES` 独占分配的设备集合。

防止大任务饥饿：等待超过 600 秒的任务进入 reservation mode，scheduler 暂停填入会阻塞它的小任务，直到资源可用或该任务被明确判定无法适配机器。

Linux 上优先将全部 worker 放入 scheduler 管理的 cgroup v2 子树，使用 `memory.high` 对应 admission budget、`memory.max` 对应 hard limit，并从 `memory.current` 读取不会重复计算共享页的总量。scheduler 自身位于该 cgroup 外，确保 worker 被内核限制时仍能记录状态。若主机没有委派 cgroup 控制权，启动必须明确报告 hard limit 只能 best-effort；只有用户传入 `--allow-best-effort-hard-limit` 才允许继续。

best-effort 模式每秒采样整个 worker process tree。超过 admission 预算时冻结新 admission；连续三次采样超过 hard limit 时终止最近启动、预期已完成工作最少的 worker。先发送 SIGTERM，10 秒未退出再强制终止。该 attempt 标记为 `resource_exceeded`，提高预留后重新排队。cgroup 的 `memory.max` 是轮询之外的最后保护，不替代 scheduler 的受控终止流程。

### 7.4 GPU 和未来 proof system

同一 run 固定一个 proof flavor，但允许兼容 flavor 的 CPU/GPU worker 混合。GPU 资源是独立维度，系统 RAM 与 GPU VRAM 不互相抵扣。

STARK 是独立 proof flavor。第一版不实现空壳 `StarkBackend`；通过无 Halo2 泄漏的公共接口保持可实现性。后续对同一 FinText shard 比较证明时间、peak RAM、proof size、验证时间、GPU 利用率和 aggregation/wrapper 成本后再选择。

## 8. 任务状态与错误语义

```text
pending -> ready -> witnessing -> witness_ready -> proving -> proved
       -> verifying -> verified

running stage -> execution_failed
running stage -> resource_exceeded -> requeued
verifying     -> verification_failed
dependency not verified -> blocked
```

- `execution_failed`：程序、CUDA、I/O、key load 或证明过程异常。默认最多重试 2 次。
- `resource_exceeded`：被资源保护策略终止。提高预留后最多重新调度 3 次。
- `verification_failed`：proof 已生成但独立验证失败。默认不自动重试，阻断依赖分支并保留证据供诊断。
- `blocked`：必要前置任务尚未 `verified`、永久失败或摘要不匹配。
- `interrupted`：scheduler/host 崩溃导致 attempt 未完成；恢复后转回 ready，不计入密码学失败。

最终 run 状态必须包含失败阶段、类型化错误码、可清洗错误摘要、attempt 次数和直接阻断原因。`false` 与异常不能互换。

## 9. Artifact、状态库与恢复

采用 SQLite WAL + 内容寻址文件目录：

```text
run.sqlite
artifacts/<artifact_digest>/
  proof.bin
  public_statement.bin
  metadata.json
keys/<key_digest>/
  key.bin
  metadata.json
```

scheduler 是 SQLite 状态转换的唯一写入者。worker 读取 immutable job spec，将输出写入同文件系统临时目录，并通过受控 IPC 返回结果；scheduler 校验大小和 BLAKE3 digest 后再提交状态。

proof 发布顺序：写临时文件、fsync、独立 verify、原子 rename、提交 SQLite `verified`。任何中间状态或临时文件都不能被下游读取。

恢复规则：

- `running` attempt 变为 `interrupted` 并重新排队；
- `proved` artifact 若完整则继续 verify；
- `verified` artifact 只有在全部 identity digest 匹配时复用；
- 损坏、缺失或摘要不匹配的 artifact 隔离并重新计算，不静默覆盖。

## 10. SRS 与 key lifecycle

运行分为 `prepare` 和 `prove`：

```text
prepare: validate SRS -> compile circuits -> generate/reuse PK/VK
prove:   read immutable prepared material -> prove -> verify
```

缓存身份：

- SRS：`proof_flavor + k + ceremony/source digest`；
- leaf PK/VK：`proof_flavor + circuit_digest + k`；
- aggregation PK/VK：`proof_flavor + aggregation_circuit_version + actual_arity + k`。

keygen 本身也是资源调度任务。key 文件使用锁、临时写和原子发布；verifier 校验真实 VK digest。production 缺少 SRS 时必须失败；只允许在显式 development 模式生成测试 SRS。

第一版每个 worker 的 reservation 包含各自加载 SRS/PK 的峰值。mmap、常驻 worker、host/GPU key cache 留作基准驱动的优化。

## 11. N 叉聚合

CLI 公开参数 `aggregation_fan_in = N`，第一版允许 `2 <= N <= 16`，默认 4。N 是 aggregation plan 和 root statement 的组成部分。

aggregation planner 优先按拓扑连续区间和模型 block 分组，降低 frontier 大小。每个 aggregation claim 包含：

- 有序 child IDs 和实际 arity；
- 覆盖的 shard IDs/范围；
- 已关闭的内部 DAG edges；
- 尚未关闭的 frontier inputs/outputs；
- model、compiler、partition、flavor 和 plan identity；
- public model input/output commitments。

节点验证每个 child proof，合并 claim，并对当前节点内首次同时出现生产者和消费者的 edge 强制 commitment equality。根节点必须关闭所有内部 DAG edge，只保留声明允许的模型公开输入和输出。

最后一组不足 N 时使用实际 m 叉节点，不使用未经验证的 dummy proof。递归阶段为实际 arity 生成并缓存对应 circuit/PK/VK，arity 与 VK digest 同时绑定。

第一阶段 native manifest 严格执行同一算法并输出 `native_verified_manifest`。第二阶段递归 backend 将相同 claim relation 证明化，输出单个 root proof。

## 12. CLI 与 worker 边界

预期命令：

```text
zkie plan         # 编译、估算、生成 immutable partition/aggregation plan
zkie prepare      # 准备 SRS 和 keys
zkie prove-queue  # 运行或恢复资源感知队列
zkie status       # 只读状态和失败原因
zkie verify       # 独立验证 manifest 或 root proof
zkie worker       # scheduler 启动的内部子进程入口
```

测试机默认示例：

```bash
zkie prove-queue \
  --memory-budget-gib 400 \
  --memory-hard-limit-gib 440 \
  --cpu-budget-cores 56 \
  --aggregation-fan-in 4
```

worker 每次只拥有一个 attempt。进程退出用于确定性释放 allocator、Halo2 polynomial 和 CUDA runtime 资源。scheduler 不把 proof 私有 witness 放入 SQLite；witness 临时文件默认在 proof 成功后删除。

## 13. 安全性与测试

### 13.1 Soundness tests

- 修改下游 input commitment，aggregation 必须拒绝。
- 替换 sibling shard、重排 child IDs 或改变 N，必须拒绝。
- 混用 model/weights/partition/compiler/flavor/VK digest，必须拒绝。
- Broadcast edge 任一消费者不匹配，必须精确指出对应 edge。
- BLAKE3 artifact digest 正确但 Poseidon boundary commitment 错误，必须拒绝。
- public model 与未实现的 private model 标志不得互换。

### 13.2 Backend conformance

- CPU backend proof 能由独立 verifier 验证。
- 未来 GPU backend 与 CPU backend 对同一 flavor 生成可互换验证的 proof。
- 同一 ISA input 在 CPU/GPU witness backend 下产生相同规范 boundary commitment。
- 不同 proof flavor 无 bridge 时不能进入同一 aggregation node。

### 13.3 Scheduler and recovery

- 依赖未 verified 时下游永不启动。
- 多 worker reservation 不超过绝对 CPU/RAM/GPU 预算。
- soft budget 停止 admission；hard limit 触发受控终止和重排。
- scheduler 在 witnessing、proving、proved 和 verifying 各阶段崩溃后均能恢复。
- verified artifact 可复用，临时或损坏 artifact 不可消费。
- 大任务经过 600 秒 aging 后不会被小任务无限饿死。
- execution、verification、resource 和 dependency failure 保持可区分。

### 13.4 性能门槛

在当前测试机上先用 FinText representative shards 校准；再逐步扩大到 TimesFM。每次记录 wall time、peak RSS、CPU 利用率、proof bytes、verify time、key load/keygen time 和可选 GPU 指标。性能结论必须来自相同 circuit digest、k、flavor 和硬件 profile 的重复测量。

## 14. 实施顺序

1. 固化公共 identity、状态、resource、artifact 和 backend 类型。
2. 把真实 Halo2 KZG leaf proving 从 benchmark 移入 CPU backend，并完成独立验证。
3. 实现确定性的 CPU witness backend 和 leaf public statement。
4. 将现有 def-use DAG 扩展为自动 partition plan，并加入 Poseidon boundary commitment。
5. 实现 SQLite/artifact/key stores、单 worker 队列和恢复。
6. 增加绝对资源 admission、进程树监控、并发 best-fit、aging 和重试。
7. 实现可配置 N 叉 aggregation plan 与严格 native verified manifest。
8. 在测试机实测 FinText，校准资源模型，再扩大模型和并发。
9. 实现递归 aggregation PoC，然后推广到全部实际 arity。
10. 对同一 shard 进行 Halo2 CUDA 与 STARK 后端评估。

## 15. 验收标准

- 一次 run 能从 immutable plan 恢复，不重复计算已验证 shard。
- scheduler 不会 admission 超过 400 GiB 或 56 个 CPU threads；在具备 cgroup v2 委派的 Linux 主机上由内核强制 worker 总内存不超过 440 GiB。
- 每个真实 leaf proof 独立验证后才发布；任何 dependency、commitment 或 identity 篡改都会失败。
- native manifest 明确区分于 recursive root proof，并完整报告所有阶段结果。
- backend 公共接口不泄漏 Halo2 类型，未来 CUDA 或 STARK adapter 无需修改 scheduler、artifact store 或 Shard DAG。
- 自动分片方案稳定、可摘要、可审计，scheduler 运行时不能改变切点。
