# zkIE Op Fixed-Point 语义契约（权威）

日期：2026-09-28
状态：权威定义，所有数值 op 必须遵守
范围：TimesFM 200M 需要的 7 个主干数值 op + 布局/头尾 op 的处置

## 1. 全局约定

- 域：Goldilocks（64 位）
- 有符号整数嵌入：负数 x 映射为 P-|x|（补码式映射）
- 舍入：全局统一 round-half-up（四舍五入，.5 向上），即 div_round 规则
- padding：所有 tensor 长度必须是 2 的幂（WHIR MLE 要求）；真实维度用 n_real 显式传入，norm 归约只算前 n_real 项、忽略 padding
- scale 三档：
  - 激活 / 权重 = 2^16（i32 嵌入）
  - matmul 累积输出 = 2^32（i64 嵌入，2^16 * 2^16）
  - norm 的 raw 输出 = 2^48（三个 2^16 因子相乘）

## 2. 各 op 契约

### MatMul
- 输入 A[2^16]、B[2^16]，输出 C = A@B[2^32]
- 无舍入（域上精确），累加在 i64（2^32），TimesFM 形状下不溢出
- m/k/n 必须 2 的幂

### Add
- 逐元素加，等长，scale 不变（2^16 + 2^16 = 2^16）

### Affine（rescale + bias，可选 ReLU）
- 输入 in[2^32]，输出 out = round(in / 2^shift) + bias[2^16]
- round-half-up；shift=16 把 matmul 输出降回激活 scale
- relu=true 时 out = max(out, 0)

### Scale（乘常数 + bias）
- 输入 in[2^16]、bias[2^16]、scale 整数[2^16]
- 输出 out = round(in * scale / 2^16) + bias
- round-half-up

### Softmax
- 输入 shifted logits[2^16]（scores - max，<= 0）
- exp 查表：idx = shifted + offset（clamp 到表范围），e = exp_table[idx]
- sum = 所有 e 之和；输出 = e / sum[2^16]
- 表：exp_table；offset 是每个 op 的公开参数

### LayerNorm
- 输入 x[2^16]、w[2^16]、n_real
- mean = sum(x) / n_real；var = sum((x-mean)^2) / n_real（整数归约，只算前 n_real 项）
- rstd = rsqrt_table[var_idx]（1/sqrt(var+eps) 的量化表）
- raw = (x - mean) * rstd * w[2^48]，之后需两次 rescale 回 2^16

### RMSNorm
- 输入 x[2^16]、w[2^16]、n_real
- s = mean(x^2)（整数归约，只算前 n_real 项）
- rstd = rsqrt_table[s_idx]
- raw = x * rstd * w[2^48]，之后需两次 rescale 回 2^16

### ReLU
- 输入 in[2^32]，输出 out = max(round(in / 2^16) + bias, 0)[2^16]
- round-half-up，SHIFT=16

## 3. 布局 op（不证明，只布局规则）

Reshape / Transpose / Split / Concat / Squeeze / Unsqueeze / Cast / Clip：
不改变数值，只重排 / 视图，证明层免费；编译时按布局规则处理（Split 切 QKV、Transpose 转 K 等）。

## 4. 头尾特殊 op（提取阶段预计算，不进证明）

Sin / Cos（位置编码）、Where / Less / GreaterOrEqual / Equal / Not / Min / Max（mask 逻辑）、
ArgMax / Gather / GatherND / GatherElements / Pad（索引 / 取值）、ReduceSum / ReduceMin / ReduceMax（归约）、
Sigmoid / Abs / Clip / Cast / Sub / Div / Mod（少量头尾逻辑）。
这些在 extract 脚本里预计算成常量，不进入 GKR 证明。
