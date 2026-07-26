# ZKML/zk-LLM 电路设计业界调研（zkGPT 之外）

日期：2026-07-26
背景：在完成 zkGPT / VeriLLM / DeepProve 对比分析后（见 `2026-07-26-zkie-comparative-analysis-deepprove-zkgpt-verillm.md`），进一步调研业界还有哪些论文/项目在**电路设计**层面值得参考。本文由后台调研 agent 完成，部分一手来源（USENIX PDF、IACR ePrint PDF）被 403 拦截，只能靠搜索引擎摘要/代码仓库交叉验证，下面每条都标注了置信度：**[一手]**（直接读到 PDF/摘要原文或官方仓库）/ **[二手]**（只通过搜索片段确认）/ **[代码]**（直接读了 GitHub README）。

## 与 zkIE 现状的对照基准

zkIE 现在：Halo2（Plonkish）+ KZG（SHPLONK），sub-project 1 按设计**不用 lookup argument**，非线性算子（softmax/layernorm/rmsnorm/gelu/div）全靠 bit-decomposition range check，痛点是小算子的 range check 固定开销（约184行/次）占比过高。以下所有条目按"离这个现状的架构距离"排序标注可迁移性。

## 按谱系分类

### GKR/sumcheck 系（zkGPT 的同宗，架构距离远）

- **zkCNN**（Liu/Xie/Zhang, CCS 2021）**[二手]** — zkGPT 代码库的直接祖先。FFT 卷积的 sumcheck 证明（O(N) 而非 O(N log N)），非线性用 bit decomposition（和我们现在做法本质相同，只是嵌在 sumcheck/GKR 电路里）。对 zkIE 无直接可迁移点，仅作背景。
- **zkLLM**（Sun/Li/Zhang, CCS 2024）**[一手，arXiv PDF]** — **tlookup** 并行 lookup argument + **zkAttn**（利用 softmax 的平移不变性减少通信）。LLaMA-2 13B 全量推理证明 <15 分钟，proof <200KB，仅支持 CUDA。**对 zkIE 的启发**：softmax 减最大值再证明"平移不变残差"而非原始 softmax，这个思路和具体证明后端无关，理论上可以直接用在我们现在 bit-decomposition 风格的 Softmax chip 上，不需要引入 lookup argument。
- **Hao et al.**（USENIX Security'24，zkGPT 号称快 279x 的对比对象）**[二手]** — 代码显示实际是 **emp-zk（VOLE-based 交互式 ZK）**，不是 SNARK。这解释了部分"279x"差距的原因：VOLE 和 SNARK 不是同一类保证（VOLE 是交互式，牺牲了非交互/公开可验证性换取非线性算子更便宜）。对 zkGPT 289x的宣传需要打个折扣理解。

### Halo2/Plonkish 系（架构距离最近，最值得深挖）

- **ZKML**（Chen/Waiwitlikhit/Stoica/Kang, EuroSys 2024）**[二手，但多来源交叉一致]** — **和 zkIE 同为 Halo2，支持 KZG 和 IPA 两种后端**。43 个预制层电路 gadget，ReLU/softmax/division 用 Halo2 原生 plookup 实现——也就是说这篇论文已经在我们这套技术栈里把"要不要上 lookup argument"这件事真刀真枪做出来了。**这是本次调研里架构相关性最高的一篇**，如果 sub-project 1 未来要重新考虑"不用 lookup"这条设计约束，它的 43-gadget 划分（哪些算子用 custom gate、哪些用 lookup）是现成的设计参考。
- **EZKL**（github.com/zkonduit/ezkl）**[二手]** — **同样是 Halo2 + KZG**，走 `tract` 库压缩 ONNX 图（和我们自己的 ONNX→ISA 编译思路类似），ReLU/division-rescale 用 plookup，量化用显式 `out_scale` 列追踪定点缩放因子。**这是本次调研里唯一一个"技术栈完全相同的已发布真实系统"**——建议直接去读它的 rescale/lookup chip 源码（这次调研没做到代码级细读，值得单独花时间）。

### VOLE 系（架构距离远）

- **Mystique**（Weng/Yang/Xie/Katz/Wang, USENIX'21）**[一手摘要]** — VOLE-based，矩阵乘协议比之前快 7x（机制细节没能确认），定点量化在 ResNet-101/CIFAR-10 上只有 0.02% 精度损失——可以当作"定点量化精度损失能做到多好"的参考基准，验证我们自己 TimesFM 量化精度时用得上。

### Lookup argument 基础理论

- **Lasso**（Setty/Thaler/Wahby）**+ Jolt**（a16z）**[二手，高置信]** — Lasso 的核心洞察："可分解"的大 lookup table（例如 2^64 大小的表）可以拆成几个小很多的 subtable 来分别证明成员关系，不需要对整张大表承诺。这正是 zkGPT 用的 Lasso/LogUp 机制背后的通用原理。**关键点**：Lasso 本身不是 Halo2 原生的，但已经有 Halo2 兼容的 **LogUp（log-derivative lookup）**实现（DeepProve 用的"LogUp-GKR"也是这个族）——如果 zkIE 未来要加 lookup，LogUp 是比整体搬到 GKR/sumcheck 侵入性小得多的路径，可以只加在最痛的几个非线性算子（softmax exp、rsqrt）上。

### 纯 PCS 改进（架构距离：换掉 KZG 属于大改动）

- **Brakedown / Basefold / Orion****[二手]** — 域无关、线性证明时间的哈希/编码类 PCS，去掉了 KZG 的椭圆曲线 MSM/pairing 开销，代价是证明体积更大、且不能直接插进现有 Halo2（Halo2 的 custom gate/permutation argument 是针对 KZG/IPA 写的，换 PCS 相当于换底层框架）。**建议：观望，不要现在动**——只有当我们自己 profiling 证实 KZG 的 MSM 时间（而非非线性算子的 range check 开销）才是矩阵乘证明的主要瓶颈时，才值得认真评估。

### 折叠方案（Folding，架构距离：中到大）

- **Nova / HyperNova / ProtoStar****[二手]** — Nova 基于 R1CS 做增量计算折叠；**HyperNova 推广到 CCS**（可以表示 Plonkish/R1CS/AIR），所以比 Nova 更贴近我们的 Plonkish 架构。**对 zkIE 的启发**：TimesFM 的 autoregressive decode 循环里，每一步都是结构相同的电路实例（同样的前向计算，只有 KV 状态不同）——折叠方案可以把"证明这些结构相同的连续步骤"的开销摊薄，这和上次 VeriLLM 分析里"decode 阶段用 causal mask 单次并行前向"是同一个问题的另一种解法，只是这次是从 SNARK 证明侧（折叠）而不是"重跑对比"侧解决。zkGPT 的 circuit squeeze 针对的是"12 层 transformer block 重复"这个冗余，折叠方案针对的是"decode 步数重复"这个冗余——两者是互补的，不是竞争关系。这个是**独立的架构级研究项**，不是随手能加的一个 gate。

### 2025-2026 新论文（直接命中我们的痛点，值得优先读）

- **Range-Arithmetic**（Rahimi/Khalaj/Maddah-Ali, arXiv 2505.17623）**[二手]** — 把非算术操作（定点乘法后的舍入、ReLU）重新表述成可以用 **sumcheck + 拼接 range proof** 验证的算术步骤，明确避开 Boolean 编码、高次多项式、和大 lookup table 三者。**这是本次调研里和 zkIE 现在设计哲学（range-check、不用 lookup）最接近的一篇**，可能直接改进我们现有的 bit-decomposition 方案，而不需要引入 lookup argument——建议全文精读，优先级最高。
- **ZIP: Zero-Knowledge AI Inference with High Precision**（Riasi et al., CCS 2025）**[二手]** — commit-and-prove SNARK，支持完整 IEEE-754 双精度，用分段多项式编码成 lookup table（每一段多项式系数存成一条 lookup entry），号称把非线性层电路规模降低达 3 个数量级。**直接命中"小算子固定开销占比过高"这个痛点**，原理上和 Halo2 兼容，但需要引入一个受限范围的 lookup argument（不是全面推翻现有架构，只用在最痛的几个算子上）。
- **A Separation Principle for Lookup-Based zkML**（Jo, eprint 2026/1390）**[二手]** — 理论论文，核心论点：在 Shout 式（one-hot）lookup argument 里，单次 lookup 的证明代价只取决于**访问模式**，不取决于表的值/结构——意味着"设计更聪明的 lookup table"并不能把单次查表代价降到访问模式本身决定的下限以下。还论证了 pre-LN transformer 里唯一随深度放大误差的是 LayerNorm 的 1/σ 增益，所以整个模型可以用统一精度证明（证明代价对层数近似线性）。**建议在决定投入 ZIP/ZKML 那种"定制 lookup table"工程量之前先读这篇**，它给出了"lookup table 设计能带来多少收益"的理论天花板，帮助校准投入产出比。
- **GaugeZKP**（OpenReview）**[二手]** — 利用 attention 的对称群结构（GL(d_k)^h × GL(d_v)^h ⋊ S_h）把模型"规范化"一次（one-time canonicalization），之后每次推理证明都对着规范化后的模型证明，号称 Halo2 门数减少约 26%。**这是模型层面而非电路层面的优化，原理上可以叠加在任何 Halo2 电路上（包括 zkIE），不需要动电路架构**——但需要先确认 TimesFM 的 attention/patch 结构是否真的有可利用的对称性，这点还没验证。
- **NANOZK** **[二手偏代码]** — 显式 Halo2 IPA-based，把 transformer 推理拆成每层独立证明，每层证明体积恒定（约6.9KB，与模型宽度无关），支持并行证明。如果 zkIE 未来想做"按层拆分证明"（和上次 VeriLLM 分析里的 segment 并行思路一致，但这次是从 SNARK 侧而不是重跑验证侧），这篇是直接的参考。

### 其它背景/低直接相关性（仅记录，暂不深挖）

zkPyTorch（Polyhedra，非 Halo2）、Artemis（commit-and-prove 权重承诺开销优化）、Bionetta（Groth16/R1CS，移动端场景）、TeleSparse（模型剪枝降低证明成本，模型层面而非电路层面，67% 证明时间下降/46% 内存下降/约1%精度损失，可以低成本验证是否对 TimesFM 剪枝也有效）、Modular Sumcheck Proofs（GKR 族的证明组合框架）、"The Cost of Intelligence"（跨系统 benchmark 方法论，技术细节没能确认）。

## 优先阅读顺序（给定：继续用 Halo2/KZG，不打算换证明后端，痛点是小算子固定开销）

1. **Range-Arithmetic**（arXiv 2505.17623）——和我们现有设计哲学最接近，可能直接改进现有方案，不需要引入 lookup。
2. **ZIP**（CCS 2025）——分段多项式当 lookup table，直接命中痛点，但需要引入受限的 lookup argument。
3. **ZKML**（EuroSys 2024）——同技术栈的真实代码先例，43-gadget 划分是"要不要上 lookup"的现成参考。
4. **A Separation Principle for Lookup-Based zkML**（eprint 2026/1390）——在投入 2/3 的工程量之前，先读这篇校准"lookup table 到底能省多少"。
5. **EZKL 源码**——同技术栈的已发布系统，值得代码级细读 rescale/lookup chip。
6. **GaugeZKP**——模型层面，成本低，值得先确认 TimesFM attention 是否有可利用对称性。
7. **HyperNova/折叠方案**——长期项，专门针对 autoregressive decode 步骤重复的结构，需要单独立项研究，不是随手加的改动。
8. **Lasso/Jolt**——作为"为什么 LogUp 是对的选择"的理论支撑，如果未来决定加 lookup argument。
9. **Brakedown/Basefold/Orion**——观望，等自己的 profiling 证实 KZG MSM 是瓶颈再看。
10. 其余（zkCNN/zkLLM/Mystique/Hao et al./NANOZK/zkPyTorch/Artemis/Bionetta/TeleSparse/Modular Sumcheck/Cost of Intelligence）——背景参考，架构距离较远或细节确认不足，直接采用ROI较低。

## 方法论说明

多篇一手来源（USENIX PDF、IACR ePrint PDF、ddkang.github.io）被 403 拦截，只能靠搜索引擎摘要交叉验证，标记为**[二手]**；实际读到 PDF/摘要原文或官方仓库的标记**[一手]**/**[代码]**。凡是调研过程中没能确认的细节（如 zkLLM 具体用什么 PCS、Mystique 的 sVOLE 压缩机制、ZKML 的 GPT-2 vs ResNet-18 benchmark 数字矛盾）都如实标注未确认，没有编造。
