# SparseIndexer proof design

Indexer 不是单个查找，而是一段可证明的计算。证明层按以下四块拆，每块都能映射到
已有或新增的原语。

## 1. indexer score

```
q        = wq_b(qr)                  # [s, H, D]
q        = rope_tail(q)              # [s, H, D]
k        = norm(wk(latent))          # [T, D]  (T = compressed positions)
weights  = weights_proj(x) * c       # [s, H]
score    = sum_h relu(q·k^T)_h * weights_h   # [s, T]
```

证明：

- `wq_b` / `wk` / `weights_proj` 都是普通投影，复用 `Projection`。
- `q·k^T` 复用 `MatMul`。
- `relu` 复用 `Relu`。
- 最后的按 head 加权求和，先用 `ScaleVec`/`BroadcastScale` 做 head 权重，
  再用 `MatMul` 对 head 维求和。

## 2. visibility mask

每个 query 只能看到它已越过的 compressed positions。用 `causal_mask` +
`StableSoftmaxIndex` 同款的 mask 逻辑，或者新增一个 `CompressMask` 原语。
不可达位置设为 `-(1<<30)`，保证后续 top-k 不会选中。

## 3. candidate pool（is_candidate_source）

`select_candidate_blocks` 分两级：

- 先按 `block_size` 把不可达 padding 成 `-inf`，对每个 block 取 `max`。
- 最新 block 固定 pin 为 `+inf`。
- 对 block score 做 top-k，选 `candidate_topk_blocks` 个 block。
- 展开回 position mask。

证明：block max 用 `TopKSelect`（k=1）取最大值；block top-k 用 `TopKSelect`；
展开 mask 用 `IdxBroadcast` / `CombineIdx`。

## 4. final top-k

对 masked 后的 score 做 top-k（`index_topk`），再按 position 排序，unreachable 填 -1，
最后加 `offset`。`TopKSelect` 已能证明 top-k；排序可用 `Argmax` 的排序路线，
或新增 `SortIndices` 原语。

## 落地顺序

1. 先用 builder 组合现有 op 实现 Full 模式（不新增证明原语）。
2. 对 Reindex 的 candidate pool 增加 `CompressMask`/`SortIndices` 原语。
3. 再决定是否把整段折叠成单个 `Op::SparseIndexer`。

# HyperConnection proof design

hc_mult=4，输入是 4 份残差流 `x: [s, 4, d]`。每个 sublayer 调用一次 `hc_mixes`
得到 pre/post/comb，再 `hc_pre` 塌缩成单份输入，sublayer 算完用 `hc_post` 扩回 4 份。

## hc_mixes

```
z = flatten(x, dim=(2,3))          # [s, 4*d]
z = z * rsqrt(mean(z^2)+eps)        # 归一化
mix = z @ hc_fn^T + hc_base          # [s, mix_hc], mix_hc=(2+4)*4=24
pre, post, comb = split_sinkhorn(mix)
```

证明：

- 归一化复用 `RmsNorm`。
- 线性投影复用 `Projection`。
- `split_sinkhorn` 是难点：`pre`/`post` 是 sigmoid 变换，`comb` 是 4x4 矩阵
  经过 Sinkhorn 迭代变成 doubly-stochastic。需要新增一个
  `SinkhornSplit` 原语，把每轮 row/col normalize 的算术约束逐轮证明。

## hc_pre / hc_post

```
y = sum_h pre_h * x_h                     # [s, d]
x'_h = post_h * y + sum_g comb[g,h] * x_g  # [s, 4, d]
```

这两个是线性组合，可以用 `MatMul`（把 hc 维当成归约维）证明，不需要新原语。

# EngramLookup proof design

layer14 的 Engram：

- 对 token n-gram 做确定性 hash 得到 `hash_ids`。
- 用 `hash_ids` 查 `embed.weight`（FP8, [384016682, 256]）得到 `e`。
- 门控 + `wkv` 输出。

证明难点：

1. 确定性 n-gram hash 是整数函数，可用 `Lookup` 或多轮 `CombineIdx` 表示。
2. `embed` 表有 3.84 亿行，需要一个支持大表的 streaming lookup 原语，并证明
   `table[idx] == e`。这是整个 Engram 的硬件和证明主压力。
3. 门控是 `e` 与 `q/k` 的内积/softmax，复用现有注意力算子。

建议先做 hash 的 CPU 参考和表读取路径，再决定 lookup 原语的 stream/batch 粒度。
