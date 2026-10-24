# Streaming weight gather + gather-opening (seq512 enabler)

## The blocker

The expert-major MoE materializes `W_gathered = S_pad x H x MOE_PAD` via `Lookup` to
feed `BatchedMatMul`. For seq512 (S_pad = 4096, H = 2048, MOE_PAD = 2048) that is
17e9 field elements ~ 137 GB, so it OOMs. The dense per-expert weights are only
`N_ROUTED x H x MOE_PAD` = 64 x 2048 x 2048 ~ 2.1 GB (mmap'd).

## Key simplification

The `BatchedMatMul` eq-selector sumcheck leaves a weight claim
`h_eval = W~_gathered(ch_pos, ch_d, v)`. Expanding the MLE and substituting
`W_gathered[pos][d][j] = W_all[expert_of[pos]][d][j]`, then regrouping by expert:

    h_eval = sum_e counts_e(ch_pos) * table_eval[e]

where

    table_eval[e]     = MLE_eval(W_all[e], (ch_d, v))          (a K x N MLE at (ch_d, v))
    counts_e(ch_pos)  = sum_{pos: expert_of[pos]=e} eq(pos, ch_pos)

So the weight claim is a **linear combination over N_ROUTED=64 experts**, not a
recursive gather over S positions. `counts_e` is computed by the verifier from the
already-authenticated `expert_of` (carried by the Scatter permutation proof), and
`table_eval[e] = MLE of W_all at (e, ch_d, v)` is a single opening of the committed
`W_all` (n openings, batchable).

## What it buys

`w_v = partial_eval(W_gathered, v)` is the only thing `BatchedMatMul::prove` needs
from the weights, and it is `S_pad x H` (~67 MB). It can be computed on the fly:

    w_v[pos][d] = sum_j W_all[expert_of[pos]][d][j] * eq(j, v)

reading one `H x MOE_PAD` block of the mmap'd `W_all` at a time. So the 137 GB
materialization is replaced by a 67 MB `w_v` plus streaming reads; the wall-clock is
unchanged (same O(S*K*N) fold) but the peak memory drops ~2000x.

## To implement

1. Add a `gather_matmul`/`batched_matmul_streaming` prove that takes `(seg, expert_of,
   W_all)` instead of `(seg, W_gathered)`, computes `w_v` on the fly, and runs the
   existing eq-selector `prove_virtual`.
2. Emit the weight claim as `(counts_e, table_eval)`; the op layer binds it by opening
   `W_all` at the n points `(e, ch_d, v)` (batched) and checking the linear combination,
   instead of opening a virtual `W_gathered`.

No new sumcheck/permutation primitive is required: it is the existing eq-selector
sumcheck plus a linear-combination binding.

