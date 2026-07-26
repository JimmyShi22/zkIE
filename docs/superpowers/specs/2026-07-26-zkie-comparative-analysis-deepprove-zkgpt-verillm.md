# zkIE 对比分析：DeepProve / zkGPT / VeriLLM

日期：2026-07-26
来源：`docs/usenixsecurity25-qu-zkgpt.pdf`（zkGPT，USENIX Security'25，全文已读）、`docs/VeriLLM.pdf`（arXiv:2509.24257，全文已读）、此前对 DeepProve（Lagrange Labs）的调研。

## 0. 三个项目的定位差异（先厘清，避免误比）

| 项目 | 证明后端 | 保证类型 | 与 zkIE 的可比性 |
|---|---|---|---|
| **zkIE（当前）** | Halo2 + KZG (SHPLONK)，逐标量门电路 | 密码学可靠性（soundness），无信任假设 | — |
| **DeepProve** | 自研 sumcheck/GKR + LogUp-GKR lookup + HyperKZG | 密码学可靠性 | 可比，同为 ZK 证明 |
| **zkGPT** | 自研 GKR 层电路 + sumcheck + Lasso/LogUp lookup，**Hyrax**（非 KZG，Pedersen+bulletproof，BN254） | 密码学可靠性 | 可比，同为 ZK 证明 |
| **VeriLLM** | **没有专用证明系统**——"经验重跑 + Merkle 承诺 + VRF 抽样 + 链上质押博弈"，仅在争议时兜底升级到通用 zkML/SNARK（未指定具体系统） | 概率性 + 经济博弈（one-honest-verifier 假设），**不是密码学可靠性** | **不可直接比开销**——VeriLLM 的"~1%开销"是"重跑+对比哈希"的开销，不是"生成证明"的开销，两者不是一回事 |

这一点很重要：VeriLLM 论文自己也在 Related Work 里把 zkML/zkGPT/DeepProve 列为"prohibitive prover cost, often hours per inference"的对比对象，说明它清楚自己走的是完全不同的路线（乐观博弈论 + 抽样审计，不是证明）。所以 VeriLLM 对 zkIE 的价值不在"抄证明技巧"，而在"抄计算结构分解的思路"（见第 3 节）。

## 1. DeepProve（已记录为参考项目）

- 仓库：`github.com/Lagrange-Labs/deep-prove`
- 架构：sumcheck/GKR + LogUp-GKR lookup argument + HyperKZG，**非 Halo2**
- 之前调研的 GPT-2 benchmark 数字（7.64 分钟量级，具体见此前报告）需要注意：和 zkGPT 的"~21.8s"数字**不是同一个 benchmark 设置**（序列长度、token 数、是否含完整 autoregressive decode 均可能不同），二者不能直接比"谁更快"，只能比"用了什么技巧"。

## 2. zkGPT 的具体技术点 → zkIE 可借鉴项

zkGPT 的核心思路：线性层（矩阵乘）用 GKR+sumcheck 证明（复杂度 O(nm+mk)，即证明代价与矩阵规模线性、验证对数），非线性层用"advice（见证结果）+ 单个 range relation 或 lookup table"证明，而不是模拟计算过程本身。以下是可迁移到 zkIE（Halo2/KZG，逐标量门架构）的点，按ROI排序：

### 2.1 除法/rsqrt：用单个 range relation 替代 quotient+remainder+双分解（高 ROI，可直接落地）

zkGPT 对 `z = round(C1/C2 * qx/qy)` 这类除法，不模拟除法计算过程，而是把结果 `z` 当作"advice"（证明者见证），只用**一个** range relation 校验：
```
C2*qy*(2z-1) <= 2*C1*qx < C2*qy*(2z+1)
```
对比 zkIE 现在 `crates/zkie-core/src/chips/div.rs` 的做法：quotient/remainder rescale gadget，需要 `r` 和 `slack=SCALE_18-1-r` 的双分解 range check（`RangeCheckChip`）—— 相当于至少 2-3 次独立 range check，而每次 range check 在我们自己的 `bench_rowcount.rs` benchmark 里都测出有约 184 行的固定开销（k 无关的常数项）。如果把 DivChip 重构成"witness 结果 + 单个 range relation"，可以把这块的固定开销砍掉一半以上。这是**不需要动证明后端、纯 Halo2 custom gate 层面就能做**的改动。

RmsNormChip/LayerNormChip 里的 `mean → diff → square → variance → epsilon add → rsqrt` 链条同理，目前每一步都有自己的 range check；zkGPT 的"constraint fusion"思路是代数替换合并多步，最终只做一次 range relation。这对我们收益可能更大，因为 RmsNorm/LayerNorm 目前是"多次小 range check 叠加固定开销"的重灾区（之前 FinText 8M attempt 中 `ReduceSumChip` i64 溢出、只能跑 8/264 channel 那次已经暴露了这条链路的脆弱性）。

### 2.2 GELU：zGeLU 多项式近似替代 lookup table（中 ROI，架构友好）

zkGPT 用 ReLU + 低阶多项式修正（仅在 |x| 小时生效）近似 GELU，避免了对 GELU 整体做 lookup table，只需要几次算术门 + 一次 `|x|>=threshold` 的 range check。这正好契合 sub-project 1 的设计原则（不用 lookup argument）——可以直接替换 `crates/zkie-core/src/chips/gelu.rs` 现在基于 lookup table 的实现，用多项式近似换掉整个 lookup argument（代价是精度略有取舍，论文报告 GPT-2 上 ΔPPL < 0.5%）。

### 2.3 矩阵乘法用 sumcheck/GKR 证明（高价值但侵入性最大，长期项）

zkGPT 对矩阵乘证明用 Thaler 的 sumcheck 协议，证明者开销 O(nm+mk)（渐进最优），而且论文专门优化了 bookkeeping table 的计算（"分组算法"，利用量化后数值范围小 + 权重矩阵零填充稀疏性，约 10x 加速）。这是 zkIE 当前"每个标量乘加对应一行电路"架构的根本性差异所在——但要落地需要在 Halo2 电路里嵌入一个 sumcheck 子协议（验证者做 O(log n) 工作，而不是逐行约束），属于架构级改动，不是简单加个 gate。建议列为路线图上的长期项，而不是马上动手。

### 2.4 Circuit squeeze（低直接相关性）

zkGPT 用"advice 化中间结果打破 GKR 层间依赖"来提升多线程并行度（14.7x 加速）。这个技巧是针对 GKR 分层电路结构的特有问题，Halo2 的 Plonkish 架构本身没有"层间顺序依赖"这个瓶颈（witness 生成本身可以并行），所以直接迁移价值不大，但下面第 3 节 VeriLLM 提供了一个对 Halo2 更直接适用的并行化思路。

## 3. VeriLLM 可抽取的并行化结构（用户重点关注项）

VeriLLM 不是证明系统，但它对"如何把一个天然串行的 LLM 推理计算，重新组织成可并行验证的形式"给出了两个非常具体、可直接迁移到 zkIE 证明流程的结构性技巧：

### 3.1 Decode 阶段的"causal mask 单次前向"重写（最高价值）

autoregressive decode 本质上是 T 步串行（每步依赖前一步的 KV cache），VeriLLM 的关键观察：**一旦完整输出序列已知**（对证明场景来说，这总是成立——我们是给定输入输出，事后证明这个计算是对的），验证者不需要重跑 T 步串行 decode，而是可以对拼接后的完整序列 `[prompt || output]` 在同一个 causal mask 下做**一次并行前向传播**，等价于把 T 个串行 token 位置的计算，摊平成一个"宽而并行"的电路。

这个思路可以**直接搬进 zkIE 的电路设计**：不用为 autoregressive 的每一步单独建一个依赖前一步输出的子电路，而是把整个 `prompt+output` 序列当作一次性输入，在 causal mask 约束下把所有 T 个 token 位置的 attention/FFN 计算铺开成同一个电路里的并行区域（每个 token 位置一组独立的 region，只有 attention 的 causal mask 引入跨位置依赖，但这不阻碍 witness 生成和证明的并行化，多核/多机可以按 token 位置分片）。这对我们目前"TimesFM autoregressive decode() loop 需要 horizon > output_patch_len 时逐步执行"的瓶颈是直接对症的——之前资源评估报告里全模型证明代价高，一部分原因正是把 autoregressive 循环按顺序展开证明；如果能重写成"已知完整输出序列后一次性并行证明"，理论上可以把证明工作量从"T 次独立的子证明"降到"一次覆盖全序列的并行证明"。

### 3.2 按 Transformer 层分段（segment）并行证明 + Merkle 边界拼接（中高价值，工程可行性高）

VeriLLM 把模型切成 L 个连续 segment（每个 segment 是若干层 Transformer block），不同 segment 的验证在不同节点上**独立并行**执行，segment 边界的 hidden state 用 Merkle root 承诺后传递，验证总时间 `Tverify ≈ (1/L) * Tprefill(全模型)`，接近线性于 segment 数量。

这个思路对 zkIE 的证明流水线是直接可用的工程模式：把整个模型的证明拆成"按层/按 block 分片的独立子证明"，每个子电路只需要知道"进入这个 segment 的边界值的承诺"（不需要重新证明上游 segment 的计算），跨 segment 靠一个哈希/承诺值（而不是完整的 witness）拼接，多台机器各自独立跑各自 segment 的 setup/keygen/prove，最后再做一次轻量的"边界一致性"验证或递归聚合。这正好对应我们资源评估报告里发现的"单机跑不动整个 TimesFM"的问题——分段并行证明是目前工程上最现实的缓解手段之一，且不需要更换证明后端。

## 4. 小结：优先级建议

1. **立即可做（不改架构）**：div/rsqrt 单 range-relation 重构（2.1）、GELU 多项式近似替代 lookup（2.2）——都是 Halo2 custom gate 层面的局部优化，直接降低我们自己 benchmark 已经证实的"固定开销主导小算子"问题。
2. **工程可行、收益明确**：按层/按 block 做分段并行证明 + 承诺拼接（3.2），直接对症"单机资源不够"的问题，且不需要换证明后端。
3. **架构级改动，收益最大但风险也最大**：autoregressive decode 的"causal mask 单次并行前向"重写（3.1）——需要重新设计 assembler 如何处理整段序列而非逐 token；sumcheck/GKR 矩阵乘证明（2.3）——需要在 Halo2 里嵌入一个 sumcheck 子协议或考虑混合架构。这两项建议作为路线图上的下一阶段讨论项，不是本轮直接实现。
