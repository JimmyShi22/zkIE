# Layer-granularity GKR migration (op -> layer)

Goal: change zkIE from op-granularity (each op commits + sumchecks + opens
separately) to layer-granularity: each transformer layer is one GKR sumcheck
over a unified tensor, one WHIR commit, intermediates not committed, logUp
folded into the sumcheck. Acceptance: GPT-2 512 runs and produces a time.

## DeepProve (zkml) findings
- Granularity is per-op: each op (einsum/softmax/layernorm/...) is one
  LayerProof (one sumcheck + one PCS open), chained via claims (same_poly
  merge). Intermediates are NOT committed; only the input + lookup tables are.
- logUp is a standard degree-2 sumcheck over the fraction-reduction polynomial
  g = num_low*denom_high + num_high*denom_low + c*denom_low*denom_high, run
  GKR over log2(n) layers; the division only appears in the final claim.
  (See zkml/src/lookup/logup_gkr/circuit.rs layer_proving_info.)
- Stack is BN254 + HKZG (ark-ff / dp-crypto / ceno). zkIE keeps Goldilocks + WHIR.

## zkIE migration steps
1. Virtual-polynomial sumcheck (DONE): sumcheck.rs prove_virtual/verify_virtual
   prove sum_i coeff_i * prod_j f_j(x) (arbitrary terms/degree). Substrate for
   both the layer circuit and the logUp GKR.
2. logUp GKR: express the logUp identity as the degree-2 virtual polynomial
   above and replace the standalone grand-product prove_product/prove_lookup.
3. Layer circuit: compiler from Op -> one virtual polynomial g per transformer
   block.
4. Claim chaining: stop committing intermediates; chain output claim -> input
   claim; only commit the layer boundary (unified tensor, one WHIR).
5. N-ary tree sharding (N public) + agg + layer parallelism + cross-shard
   binding.

## Claim chaining (step 4) design notes

Primitives are all built and unit-tested (54 lib tests green). The remaining
work is the integration.

- Arithmetic layers (add / hadamard-mul / affine): the intermediate can be
  substituted away, so "claim chaining" = write the composed constraint and
  prove with `prove_layer_circuit`, opening only the boundary
  (see `layer_circuit::tests::claim_chain_no_intermediate`).
- Lookup layers (softmax exp / gelu / layernorm rsqrt): the intermediate is a
  table lookup (not substitutable). Chain via `prove_lookup_fractional`, and
  pass the intermediate claim (a fraction at the random point) between layers
  with a same_poly / claim-merge step (consistent random point across layers).
- The eq-weighted sumcheck's final check carries an `eq(r,r)` factor (not 1 for
  non-boolean r); `verify_virtual` handles this because both sides use the same
  `eq(r,r)`. Soundness is the FIRST round check `p[0]+p[1] == claimed`, which is
  the MLE interpolation identity.

### Remaining (the big integration)
1. GPT-2 layer compiler: matmul (separate GKR reduction over the contraction
   index) chained with the elementwise/lookup layers into one per-layer circuit.
2. Unified tensor + one WHIR commit per layer; only the boundary is committed.
3. N-ary tree sharding (N public) + agg + layer parallelism + cross-shard
   binding (downstream input commitment == upstream output commitment).
