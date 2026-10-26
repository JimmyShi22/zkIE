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
