# Adapting a new model to the committed (root-only) path

The committed path proves a full model with 90-bit WHIR weight/boundary
commitments, root-only verification, and a one-time commit cached to disk. To
wire a new model (e.g. DeepSeek), follow this. No global config: the PoW budget
adapts per weight group automatically.

## 1. Entry points (all in `zkie-ops/src/compose.rs`)

- `commit_model_weights(store, ops, cache_dir) -> Vec<CachedWeightBatch>`
  commits every weight group once and streams the Merkle prover data to
  `cache_dir`. Call it once per model/weights (offline; it is not input-bound).
- `prove_committed_shard_dag(store, ops, ops_per_shard, whir, cached_weights, rng)
  -> CommittedShardDagProof` proves the shard DAG and opens weights/boundaries.
  `whir` is now ignored for boundaries (boundaries build a per-tensor Whir
  internally); pass `Whir::new_testing(1)` as a placeholder.
- `verify_committed_shard_dag_bytes(proof_bytes, tables, ops, ops_per_shard, whir)
  -> bool` verifies root-only.

`ops_per_shard` is the shard granularity (one transformer layer is the usual
choice). `tables` is `store.owned_slices()` (the public lookup tables).

## 2. Per-group PoW budget (automatic, do not hand-pick)

`commit_model_weights` groups weights by size, packs same-arity weights within
the PoW budget (horizontal merge), and for each group uses
`Whir::required_pow_bits(num_variables, 90)` as that group's `pow_bits`. This
lifts the old fixed `pow_bits=17` that broke on tensors larger than 2^26 (e.g.
Gemma 3's 2^28 lm_head).

- `Whir::required_pow_bits(num_variables, 90)` returns the minimum proof-of-work
  bits the folding needs to reach 90-bit security for `2^num_variables`
  evaluations (folding factor 5, capacity bound). It is derived from the
  `PowBitsExceedBudget { required }` error of `WhirConfig::new`.
- A larger tensor (more variables -> more folding rounds) needs a larger
  `pow_bits`; the cost is a one-time PoW grind inside `commit_model_weights`,
  amortized by the cache, not per proof.

## 3. Boundary commitment (automatic)

Boundary tensors (shard inputs/outputs) are committed with a per-tensor
`Whir::with_params(arity, 90, 17)`; non-uniform boundary sizes are handled
(grouped by arity). No manual sizing needed.

## 4. Checklist for a new model

1. Build the op graph into a `Store` + `Vec<Op>` (as the existing
   `models/{gpt2,timesfm,gemma3}` crates do).
2. Pick `ops_per_shard` = one layer's op count.
3. `commit_model_weights(&store, &ops, cache_dir)` -> `cached`.
4. `prove_committed_shard_dag(&mut store, &ops, ops_per_shard, &placeholder_whir,
   &cached, &mut rng)` -> `proof`.
5. `bincode::serialize(&proof)`, then
   `verify_committed_shard_dag_bytes(&bytes, &store.owned_slices(), &ops,
   ops_per_shard, &placeholder_whir)`.
6. Report commit / prove / verify / argmax / peak RSS / proof bytes separately
   (see `models/*/examples/bench_committed.rs`).
