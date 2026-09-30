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
