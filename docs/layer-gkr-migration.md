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

## Reduction chaining (matmul / softmax-sum / layernorm-mean) — the hard part

Elementwise compiler is complete: Add/Mul/linear-Affine fold into
`prove_layer_circuit`; rounding-Affine = arithmetic constraint + logUp range
check (`layer::tests::affine_round_layer`). The remaining ops are the
REDUCTIONS (matmul contraction, softmax row-sum, layernorm mean), which are
structurally different.

Reduction chaining = the "virtual polynomial" approach (DeepProve/ceno):
- The reduction output (e.g. matmul C) is NOT committed. Its MLE is defined
  virtually as C(u,v) = sum_k A(u,k)*B(k,v).
- The elementwise layer on top (e.g. D = C + bias) is proven by an eq-weighted
  sumcheck over D's constraint, where C(x) is the virtual MLE, NOT an MLE in
  the witness list.
- The verifier chains the claim D(u,v) -> C(u,v) + bias(u,v) -> C(u,v), then
  verifies C(u,v) via the matmul GKR (reducing to A, B claims at the SAME
  point u,v).

Key missing primitive: `same_poly` (DeepProve zkml/src/iop/same_poly.rs) —
merge several claims on the same tensor at different points into one, so the
matmul's output point and the elementwise layer's input point are consistent.

Concrete next steps (in order):
1. `same_poly`: prove claims (r1,v1), (r2,v2) are evaluations of the same MLE,
   and merge to a fresh challenge point. (Two eq-weighted sumchecks + merge.)
2. Virtual reduction: extend `prove_virtual` (or a new helper) so a term can be
   a "sum over an index" of products (the matmul contraction), producing a
   virtual MLE without committing it.
3. Chain matmul -> affine -> add for a minimal m=k=n=2 case, only committing
   A, B, bias, D (not C).
4. Wire the whole layer compiler into the GPT-2 path (Exec -> layer circuit).

## Matmul chaining protocol (exact)

`matmul::prove` already returns `claimed = C(v,u)` plus the sumcheck proving
`C(v,u) = sum_k A(u,k)*B(k,v)`. The current `prove_matmul` then OPENS C at
`cp = v ++ u`. To avoid committing C, skip that opening and pass `C(v,u)` as a
claim to the downstream layer.

Chaining matmul -> add (`D = C + bias`), only committing A, B, bias, D:
1. matmul GKR: proves `C(v,u) = sum_k A(u,k)B(k,v)`; leaves claims on A at
   `ch ++ u`, B at `v ++ ch`, and the scalar C(v,u).
2. add eq-weighted sumcheck at the SAME point `p = v ++ u`:
   `sum_{i,j} eq(p,(i,j)) * (D(i,j) - C(i,j) - bias(i,j)) = 0`, reducing to
   `D(p) - C(p) - bias(p) = 0`. C here is the virtual MLE (prover computes
   C = A@B in plain, verifier never commits it).
3. binding: C(v,u) from step 1 == C(p) from step 2 (same point p = v ++ u).
4. open A, B, bias, D only.

Point order is the subtle part: C's MLE is indexed n-first (`c_point = v ++ u`),
so the add's eq selector must be `eq(v ++ u, (j,i))` (n-dim first). Keep this
convention throughout, or add a transpose layer.

## Wiring map (every GPT-2 op -> demonstrated primitive)

| GPT-2 op | mechanism | test |
| --- | --- | --- |
| matmul (projection) | bilinear reduction (virtual MLE, C not committed) | layer::matmul_add_chain_no_commit_c, matmul_affine_chain_no_commit_c |
| softmax exp / gelu / layernorm rsqrt | logUp fractional lookup | logup_gkr::lookup_fractional_roundtrip |
| softmax row-sum / layernorm mean | linear reduction (broadcast eq) | layer::row_sum_reduction_virtual |
| softmax rescale out=round(e*2^16/sum) | product with virtual reduction (broadcast over (i,j,j')) | layer::product_with_virtual_row_sum |
| affine rounding | arithmetic constraint + logUp range check | layer::affine_round_layer |
| add / residual | arithmetic constraint | layer::compile_add_mul_affine |
| claim merge | same_poly | same_poly::same_poly_roundtrip |

All of the above are unit-tested (61 lib tests green). No new primitive remains.
The only UN-demonstrated chaining pattern is matmul -> matmul (attention
Q -> scores = Q@K^T): the first matmul's output Q is a virtual MLE consumed by
the second matmul's bilinear reduction. This is a composed bilinear form
(degree 4 in A,B,D), and it is the last piece before wiring the full attention.

Remaining (mechanical, no new primitives):
1. matmul -> matmul chaining (nested virtual reduction).
2. Wire the full GPT-2 layer (q/k/v/o/fc/proj + attention softmax + residual
   adds) using the map above.
3. Rewrite Exec/committed.rs to use the layer circuit (integration).
4. N-ary tree sharding + agg + layer parallelism + cross-shard binding.

## Benchmarks at GPT-2 scale (layer circuit, plain model, single-threaded)

Examples in `crates/zkie-gkr/examples/`:

| block | dims | time |
| --- | --- | --- |
| projection (matmul + affine + logUp) | 512x1024x1024 | 2.06s |
| full FFN (2 projections + gelu) | 512x1024x4096 + 512x4096x1024 | 18.55s |
| softmax (exp lookup + row-sum) | seq=512, table 2^18 | 0.14s |

Findings:
- softmax is cheap (0.14s); the dominant cost is the projection affine + logUp
  range check (O(m*n) elementwise sumcheck + fraction tree), not the matmul GKR
  (O(contraction dim), ~0.03s).
- Full layer extrapolation: attention (~6s projections + ~1.7s softmax) + FFN
  (18.55s) + layernorm (~1s) ~= 27s/layer, ~5.4 min for 12 layers, ~3.5x faster
  than the op-granularity 19.6 min.
- Caveats: plain model (affine base tensors not WHIR-committed, only the logUp
  fraction tree is); single-threaded; real gelu/exp tables are 2^21..2^23 (the
  bench used 2^16..2^18). N-ary sharding + layer parallelism + GPU are still to
  be added on top.
