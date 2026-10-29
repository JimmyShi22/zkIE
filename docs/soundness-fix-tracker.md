# Soundness Fix Tracker

跟踪 committed-path soundness 修复进度。基于对 `9eaa8f12` 的安全审查（详见 gist: JimmyShi22/f9cda35dee7c949c2bcffac6f73ae98f）。

> 目标：先让错误计算被拒绝（修复验证关系），再谈性能优化。所有性能数字只在相同安全等级、相同真实输入、相同认证范围下重新比较。

## 修复顺序

按依赖 DAG 分阶段推进，每完成一个 S 更新状态并停下 review。

### 阶段 A · 入口（可并行）
- [x] **S1 · Sumcheck 挑战顺序（P0）** — batch_open / sumcheck / virtual / fractional GKR 改为逐轮 transcript 派生挑战，轮数来自 statement 维度。回归：`s1_forged_outer_sumcheck_rejected`（真实 WHIR opening + 伪造外层 sumcheck 拒绝）、`s1_round_count_rejected`、`transcript_challenges_bind_round_messages`、`virtual_transcript_binds_rounds_and_replays`、`s1_inner_fractional_layer_steer_rejected`（内层伪造拒绝，PR #45 review 反例）、`s1_inner_fractional_round_binds_challenge`（篡改内层轮消息改变该层派生挑战）。
  - 注（PR #45 review 修复后）：fractional GKR 改为树根→叶的分层 GKR reduction：每层 γ 批处理 num/den 恒等式为一个 eq 加权 virtual sumcheck（claim 0），挑战向量 s_k 由本层轮消息逐轮派生，层间不再共享嵌套前缀；子评估经两个 same_poly 合并（sound claim-combination）传入下一层；叶 claims 通过现有 batch-opening 路径（per-table 点）认证；根绑定 final_num = C·final_den 保持。S3 零分母拒绝在新结构留占位注释。
  - 注：`verify_lookup_split` 等函数的 alpha/beta 前导由调用方在 transcript 上重放（每个路径的前导不同）。
- [ ] **S8 · 防 panic 健壮性（P1）** — proof 结构校验，禁止对不可信数据 assert/unwrap，参数从受信 statement 选取。

### 阶段 B · 原语与框架（依赖 A）
- [ ] **S2 · Verifier 建立完整 claim 认证链（P0）** — 引入 VerifierStatement，每个 verify_reduce 返回 EvalClaim，verifier 自建所需 claims 集合。
- [ ] **S3 · LogUp 顶端绑定（P0）** — final_num/final_den 绑定归约树顶层，拒绝零分母退化。
- [ ] **S4 · same_poly 绑定（P0）** — 从输入 claims 派生批量系数，终点由 verifier 闭式计算。

### 阶段 C · 模型关系（依赖 B 框架）
- [ ] **S5 · op 关系补齐（P0）** — Add/Transpose/Scale/ScaleVec 等逐位置关系，未实现归约的 op 明确拒绝。
- [ ] **S6 · 公开序列接入（P0）** — 独立验证 API 传入 prompt，tokens/Argmax 绑定 embedding lookup。
- [ ] **S7 · MoE 专家绑定（P0）** — 路由 counts 归约到已认证路由，专家权重通过正确根 batch opening。

### 阶段 D · 安全论证（依赖 C 闭合）
- [ ] **S9 · 安全预算分析** — 建立端到端 ≥90-bit 参数依据，评估外层挑战迁扩域。

### 阶段 E · 性能优化（依赖 S1–S9）
- [ ] O1–O7（Argmax 公开选择器归约 / 边界 claims 合并 / 去堆分配 / eq 闭式求值 / 权重分组加载 / 拆 witness 与证明构建 / 同表 LogUp 共享）

## 反例回归

每个 S 修复后，把已复现攻击固化为应拒绝的回归测试：

| 反例 | 应拒绝项 | 对应 |
|---|---|---|
| 32×32 MatMul 空 sumcheck 无承诺 | 拒绝 | S2 |
| Add `[1,2]+[3,4]=[5,5]` | 拒绝 | S5 |
| Lookup `table[0]=99` 改 final 分数 | 拒绝 | S3 |
| 替换 tokens / argmax 长度 | 拒绝 | S6 |
| 同点 claim 1 和 2 | 拒绝 | S4 |
| 真实 WHIR opening + 伪造外层 sumcheck | 拒绝 | S1 |
| 缺 op proof 序列化合法 | 返回 false 不 panic | S8 |
| GatherMatMul fake counts/table | 拒绝 | S7 |
