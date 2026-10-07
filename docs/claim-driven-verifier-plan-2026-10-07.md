# Claim-driven verifier: integration plan (2026-10-07)

## The seam (precisely identified)

`verify_shard` (compose.rs) materialises the whole forward pass:

```
verify_shard(store, ops, proof)
  -> forward_ops(&mut ws, ops)          # recompute every tensor
  -> verify_shard_precomputed(&ws, ...) # read ws.get(tensor) per op
```

`verify_shard_precomputed` verifies each op relation correctly (matmul GKR,
projection, add, lookup fractional LogUp, layernorm, ...) but evaluates every
terminal claim by reading the materialised witness: `mle::eval(ws.get(t), pt)`.
That materialisation is the Layer-1 ~20x verify cost already noted in
`reedweave-assessment-2026-10-07.md`.

Claim-driven = replace `ws.get(t) + mle::eval(t, pt)` with
`open(commitment[t], pt)` (WHIR opening at the claim point). Claims then chain
across ops via shared commitments (same_poly / boundary binding).

## Building blocks already in tree (committed)

- WHIR PCS: commit/open, root-bound EF batch opening (`zkie-core/pcs/whir.rs`).
- Root-bound lookup LogUp (`extension_lookup_logup.rs`): `prove_op_lookup` /
  `verify`, verifier holds statement+proof only. Measured cheap: rows=2^15 /
  table=2^20 at 32-bit testing -> prove 2.9 s / verify 0.028 s.
- Root-bound Add->lookup shard (`claim_driven_shard.rs`), witness dropped.
- Cross-shard boundary binding (`extension_two_shard_chain_dag.rs`).
- Model/IO anchor aggregation envelope (`non_recursive_two_shard_aggregation.rs`).

## Remaining slices (the actual "finish the 4 steps" into production)

1. Committed-tensor claim type: `(root, arity, prover_data)` with
   `open_at(pt) -> EF`, the drop-in replacement for `ws.get(t) + mle::eval`.
2. Prover side: commit each shard tensor and record the claim-point openings.
3. Op-by-op replace `ws.get` in `verify_shard_precomputed` with openings, in
   order: MatMul (GKR already present) -> Projection -> Add -> Layernorm /
   RmsNorm / Softmax / RoPE -> Lookup (already done).
4. Claim chaining: keep the existing `claims` + `bound`/`binds` vectors, but
   evaluate via openings so shared tensors are bound by commitment, not
   recomputation.
5. Wire cross-shard binding + aggregation envelope around the claim-driven
   shard verifier.

## Separate track

90-bit `Whir::new` PoW grind (32-bit) is the dominant cost, not the op
relations. That is a PCS-level question (ReedWeave / accumulator / other
schemes), independent of the claim-driven refactor above.
