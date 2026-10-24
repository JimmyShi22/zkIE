# Batched-matmul (per-position expert weight): resolution note

Status: resolved. `crates/zkie-core/src/common/batched_matmul.rs` now proves the
per-position matmul `g[pos][j] = sum_d seg[pos][d] * w[pos][d][j]` soundly, and
`crates/zkie-ops/src/gather.rs` (`moe_bmm::expert_major_bmm_closure`) validates the
full chain scatter -> weight-gather -> batched-matmul -> gather against a dense
per-expert reference, with tamper rejection.

## The problem

The expert-major FFN needs a matmul whose weight is per-position (depends on `pos`).
The ordinary single-contraction sumcheck identity `C~(u,v) = sum_d A~(u,d) B~(d,v)`
is multilinear in `u` only because `B~` does not depend on `u`. With a per-position
weight both `A~` and `B~` depend on `u`, so `sum_d S~(u,d) W~(u,d,v)` is degree-2 in
`u` and is not the MLE of `g`.

## The fix

Fix the output variable `j = v` and use an eq-selector virtual sumcheck over the
`(pos, d)` domain:

    G~(u,v) = sum_{pos,d} seg[pos][d] * w[pos][d][v] * eq(pos, u).

This is a single triple-product `prove_virtual` sumcheck over `S*K` (not `S*K*N`),
and it reduces to claims on `seg` (at the full challenge `ch`) and `w` (at `v || ch`),
plus the public `eq(pos, u)` selector. Domain `S*K` is ~67 MB for DeepSeek
(S=4096, K=2048), versus ~137 GB for the naive 3D version.

## Remaining

- Weight gather `W_gathered = Lookup(idx(expert_of), W_all)` reuses the existing
  `Lookup` op; no new primitive.
- Op-graph integration (`Op::BatchedMatMul` + `OpProof`), `build_moe` rewrite, and
  padding `S = 6m` to the next power of two.
- DeepSeek-V2-Lite seq512 end-to-end (commit/prove/verify/memory/proof-size/argmax).

