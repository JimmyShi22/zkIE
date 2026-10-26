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

## Engram 具体参数（已核对 config / 分片）

- 层：`engram_layer_ids = [1, 14]`；`max_ngram_size = 4`，`n_heads = 8`，
  `head_dim = 256`，所以 `n_hash_cols = (4-1)*8 = 24`。
- 表：`layers.{1,14}.engram.embed.weight` FP8 E4M3 `[384006168 | 384016682, 256]`，
  scale E8M0 `[N, 8]`（block=32）。`wkv.weight` FP8 `[25600, 6144]`，
  `q_weight/k_weight` BF16 `[4, 5120]`。layer 1 表在 `model-00047`，layer 14 表在 `model-00048`。
- hash（`NgramHashState.forward`，prefill）：
  1. `compressed = token_map[input_ids]`（token 归一化映射，需 tokenizer 预计算，
     目标是 `engram_compressed_vocab_size = 99092`）。
  2. 对 lookback `shift ∈ 0..4` 取 `source = compressed[pos-shift]`（越界/DEAD 用 `pad_id`）。
  3. `products = tokens * multipliers[layer, lookback]`；`rolling` 从 shift0 起逐位
     `xor`，每个 `i ∈ 1..4` 的 rolling 值 `% primes[layer][i-1][head]` 得到 8 个 head 的
     hash，再加 `offsets` 得到 24 个 `hash_ids`。
- 前向：`kv = wkv(flatten(embed[hash_ids]))`；split `key [s,4,5120]` / `value [s,5120]`；
  `weight = q_weight*k_weight`；`dot = (h*weight*key).sum(-1) * rstd(h)*rstd(key) / sqrt(d)`；
  `gate = sigmoid(sign(dot)*sqrt(|dot|))`；`out = h + gate*value`。

注意：`embed` 表有 3.84 亿行，完整承诺是硬目标；前向只需查 `m*24` 行，但证明要
streaming lookup 原语 + 分片提交，这块还没实现。

# CSA2 / CED 40 层接线设计

## 共享 KV / index

- `kv_source_layers = [2,8,14,20]` 各自压缩并写 `compress_kv_cache`。
- 非 source 层读共享 cache，不重复证明压缩。
- 证明时把 cache 作为边界张量跨层绑定，避免每个 layer 重复提交同一份 KV。

## CED（第 20 层全局 KV 投影）

第 20 层既是 kv source 又是 candidate source，需要：

1. 用 `Compressor` 产生 latent。
2. 用 `Indexer`（Full 模式）产生 candidate pool。
3. 写回共享 cache，供后续层 Reindex 使用。

## 层边界 shard

40 层按 layer 边界切 shard，已具备
`prove_committed_shard_dag_with_ranges`。每层内部的 attention / FFN / HC / Engram
各自是一个 shard 的 op 序列，跨层用 boundary commitment 绑定残差流。

# seq16 / seq512 端到端计划

1. seq16 先跑：层0 attention + FFN 的完整层输出和 `v41_ref_cpu.py` 对齐。
2. 再跑 40 层：先 Full-attention 的早期层，再接入 compressed/CSA2 层。
3. seq512 只是把 `m` 放大，`ops_per_shard` 和 proof size/RSS 重新测量。
4. 验收：`argmax=16/16`（seq16）、`verify=true`、记录 commit/prove/verify/RSS/proof size。
