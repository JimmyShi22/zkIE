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
- [x] **S2 · Verifier 建立完整 claim 认证链（P0）** — 引入 VerifierStatement，每个 verify_reduce 返回 EvalClaim，verifier 自建所需 claims 集合。
  - `EvalClaim{tensor_id,point,value}` 框架：`verify_shard_precomputed_inner` 从 op 校验的 final evals 自建 shard claims（proof.claims 字段已删除）；DAG/committed 层从本地 claims 重建 cross/boundary 集合并比对；committed 层用本地 claims 做 weight opening 点集多重集比对 + 四要素（张量/点/值/根）认证。bound 集合 = 本地 claims 中 ≥2 claim 的全体张量（不再排除 boundary）。
  - 4 个关键点：① leaf 单一来源——FractionalProof 的 leaf_num/leaf_den 顶层字段删除，`verify_fractional_chain_core` 重放后返回派生叶 claims，witness x/y 在派生 q_den 点 opening（den 关系 α+x+βy==den_leaf 在该点核对），num 在 q_num；② is_identity manifest 化——TableOpening 增 `table` 字段，identity 判定来自 statement tables，proof 无此标志；③ opening 四要素——weight/boundary/table/witness/perm/gather openings 的 tensor、point、value、root 全部核对，weight 点集多重集精确匹配（未使用权重允许空 opening，空列表不可跳过非空计算）；④ 空列表拒绝——32×32 MatMul 清空承诺/零轮 sumcheck、清空 weight batches 均拒绝。
  - committed 路径 op 缺口补齐：perm/gather fracs 改为 committed 专属 transcript 重证（shard transcript 保持 frac-free，`FracMode::Committed`）；gather 的 num/den 列用 committed FS alpha/gamma 重建并后置 commit（den 关系在 q_den 与索引公式核对，padding 连续延拓）；committed 路径补 lookup/perm/gather 的 split 交叉根关系；小权重（<2^5）零填充承诺（claim 点零延拓）；Layernorm out 关系 mles 加 w（degree-3，x·rsqrt·w+b−out）；RoPE out 在 r_f/r_s 两 half 点插位 claim（fe[4]），r_out 字段删除。`VerifierStatement{tables,ops,ops_per_shard}` 类型落地。
  - 回归：`s2_dropped_claim_binding_rejected`（删 bound 列表项）、`s2_empty_weight_batches_rejected`（清空 weight batches）、`s2_identity_flag_forgery_rejected`（伪造 identity 表）、`s2_table_leaf_point_swap_rejected`（opening 换点）、`s2_committed_lookup_roundtrip`、`s2_committed_scatter_gather_roundtrip`（诚实 committed 全链路）、`s2_matmul32_empty_commitments_rejected`（32×32 反例）、`s2_weight_opening_value_and_root_swap_rejected`（改值/换根）。
- [x] **S3 · LogUp 顶端绑定（P0）** — final_num/final_den 绑定归约树顶层，拒绝零分母退化。
  - `FractionalProof` 的 `final_num/final_den` 字段删除：树根由 verifier 从顶层 layer 的 child evals 派生（`N0·D1+N1·D0`、`D0·D1`，顶层 fe 由下层链 + same_poly 认证）；重放返回 `FractionalChainClaims{叶 claims + 派生根}`。修复前顶层 arity-0 恒等式对根对 (a,b) 只有一条标量约束（沿直线 a′+γb′=const 可任意改根），split 交叉关系可被同时改写通过——LogUp 最终分数与归约树断开。
  - 结构检查：`layers.len() == t`、layer li 的 `s.len()/rounds.len() == li`、`fe.len() == 6`（截断 fe 不再 panic）；t 由列长度/statement 派生（committed：batch arity；plain：proof 携带列长度 / statement 表长；op 级：`(m·n + 表长).next_pow2`）；t=0（零层）明确拒绝，t=1（单叶）合法。
  - eq 列闭式重算：`eq_poly(q, s)` 重算 fe[0]/fe[1] 并与存储值比对（顶层空点 eq≡1），不再接受 proof 自报；零分母：派生 `final_den == 0` 显式拒绝（0/0 不再让交叉乘法恒成立），split 两侧各自先查。
  - 调用点接线：`verify_fractional_chain_core/chain/storefree`、`verify_lookup_split(w,t_w,t_t)`、`verify_permutation(t)`、`verify_lookup_fractional_chain(t)` 全部带列深度参数；compose committed 各 frac 段（lookup/perm/gather）用 batch arity 并补零分母拒绝；projection/topk/range_decomp/gather/scatter 用 statement/列长。
  - 回归（10 个 `s3_*`，修复前 7 红 3 绿）：诚实多层/单叶 roundtrip、零层拒绝、根直线伪造拒绝、eq 列伪造拒绝、零分母（单侧/双侧 0/0）拒绝、删底层 layer 拒绝、畸形结构（fe 截断）拒绝不 panic、顶层 child eval 篡改拒绝、空层+claimed 绑定拒绝。
- [x] **S4 · same_poly 绑定（P0）** — 从输入 claims 派生批量系数，终点由 verifier 闭式计算。
  - 合并论证重写：`SamePolyProof { gamma, proof, merged_point }`（删 `coeffs` 与独立 `merged_eval` 字段）。批量系数 = 单个 transcript gamma 的幂 1, γ, γ², …（prove 与 verify 同侧派生，篡改 gamma 拒绝，γ∈{0,1} 退化拒绝）；verify 检查全部 claim 点维度一致、同点不同值快速拒绝，`W(r*) = Σ γ^i·eq_poly(r_i, r*)` 由 verifier 闭式计算（W=0 拒绝），**返回唯一输出 claim `(r*, final_claim·W(r*)⁻¹)`**——修复前 `verify_same_poly` 返回 prover 自由字段 `merged_eval`（篡改后被直接消费）、eq 值自报（S3 同类）。
  - soundness 结构：rounds 先承诺 G′ → r* 后派生 → 下游绑定（叶 opening / 下一层恒等式）迫使 G′(r*)=f(r*)·W(r*) → 低度 SZ 得 G′≡f·W → 随机 γ 迫使全部输入 claims 为真。输出 claim 被 frac 链消费（下一层 incoming claim，最终由 S2 叶 opening 认证）；compose 的 shard/DAG bind 与 committed cross-bind 的合并输出不被消费——其输入 claims 已各自被 committed openings（S2 点集）直接认证，same_poly 为附加一致性检查（代码注释说明）。
  - 回归（7 个 `s4_*`，修复前 3 红）：诚实不同点合并、**矛盾同点拒绝**（恶意 prover 自适应 f′ 构造，修复前接受）、**merged_eval 伪造拒绝**（修复前返回伪造值，现返回值必等于真实 opening 值）、gamma 篡改拒绝、merged_point 篡改拒绝、eq 值不再被读取（篡改 fe 返回值不变）、跨 tensor bind 互换拒绝。

### 阶段 C · 模型关系（依赖 B 框架）
- [x] **S5 · op 关系补齐（P0）** — Add/Transpose/Scale/ScaleVec 等逐位置关系，未实现归约的 op 明确拒绝。
  - 关系分两类：**线性关系在随机点 r 直接校验**（Add `a(r)+b(r)=c(r)`；Scale 除法截断 `x·factor = 2^shift·out + rho`、`rho = rem − 2^(shift−1)` 按远离零取整；CombineIdx `out = scale·a + b`；IdxBroadcast `out = b·base + jval(r)`；SoftmaxIndex 的 divmod `trunc = q·table_len + idx_f`）；**乘法关系必须 eq 加权为整列 virtual（乘积在项内）**——随机点上 MLE 的乘积 ≠ 乘积的 MLE：Relu `eq·out·(out−x)`、SoftmaxIndex 的 relu 段、GeluIndex 的 clamp `(idx−trunc)·(idx−(L−1))=0`、PadIdx `eq·(out − mask·base)`、BroadcastScale/ScaleVec/ScaleGate `eq·(x·scale − out) ± rem`。eq 加权模式：q 提前榨取（eq 点）、r 为 sumcheck 派生（claim 点），verify 检查 `fe[0]==eq_poly(q,r)` 与 `fc==virtual_eval(terms,fe)`。
  - topk 的 gate/d1/d2 与 norms（layernorm/layer_norm_centered/rms_norm）的 out、rope 的两个 half 关系全部改为 eq 加权（gist 点名 topk 逐位置约束，同类洞一并修）。
  - 范围检查接入**公共恒等表**：`ensure_public_tables` 在 prove 入口自动推 `[0,2^shift)`/`[0,2^22)`/`[0,table_len)` 恒等表，rem/idx/q 列经 LogUp lookup 绑定（committed 路径复用 S2 identity-table 机制）；SoftmaxIndex（relu+divmod：idx_f∈[0,L)、q∈[0,2^22)）、GeluIndex（relu+clamp：idx∈[0,L)、trunc−idx∈[0,2^22)）、StableSoftmaxIndex（公式 virtual 先行、行 virtual 后行，行点取公式派生点行位，sel∈{0,1}）同模板实现。**域契约**：`relu(x) < 2^22·table_len`（与真实模型一致——models/gpt2 先 `Op::Scale{shift:16}` 再喂 index op；build_gpt2_layer 测试 builder 同步修）。
  - Transpose 改为轴交换 claim 转发（out claims → x claims @ swap(p)，m/k 非二次幂拒绝）；PadIdx 增加 `base_len` 字段；未实现归约的 op 变体在 committed verifier 明确拒绝（不默默通过）。
  - 回归（15 个 `s5_*`）：add 逐位置伪造/元素对调拒绝、transpose 伪造 out claim 拒绝、scale 错误取整拒绝、scalevec/broadcastscale/scalegate 乘积伪造拒绝、relu 符号伪造拒绝、gelu/stable/softmax index 公式伪造拒绝、idx_broadcast/combine_idx/pad_idx 伪造拒绝、未实现变体拒绝。全量回归：ops 74 绿（15 s5 新增）、core 64、engine 3、models 全绿。
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
| 32×32 MatMul 空 sumcheck 无承诺 | 拒绝 | S2 ✅ `s2_matmul32_empty_commitments_rejected` |
| Add `[1,2]+[3,4]=[5,5]` | 拒绝 | S5 ✅ `s5_add_positionwise_rejected` |
| Lookup `table[0]=99` 改 final 分数 | 拒绝 | S3 ✅ `s3_root_steer_rejected` + 派生根 |
| 替换 tokens / argmax 长度 | 拒绝 | S6 |
| 同点 claim 1 和 2 | 拒绝 | S4 ✅ `s4_contradictory_same_point_rejected` |
| 真实 WHIR opening + 伪造外层 sumcheck | 拒绝 | S1 |
| 缺 op proof 序列化合法 | 返回 false 不 panic | S8 |
| GatherMatMul fake counts/table | 拒绝 | S7 |
