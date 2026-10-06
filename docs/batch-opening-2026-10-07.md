# batch-opening experiment — GPT-2 512 (2026-10-07)

Test machine, 13 shards / 12 cross-binds, `models/gpt2/examples/bench_committed.rs`
(Whir::new_testing(19), argmax against ground truth).

| variant | prove | verify | argmax | note |
|---|---|---|---|---|
| plain (`prove_shard_dag`, no WHIR) | 34.29 s | 19.41 s | 511/512 | verifier recomputes forward |
| committed naive (per-tensor commit, per-claim open) | 36.17 s | 183.06 s | 511/512 | adds WHIR commit + open |
| committed batch (`open_multi` multi-point open) | 36.09 s | 184.21 s | 511/512 | batching did NOT help |

## Diagnosis (from whir.rs stats)

```
open_stats   = (70, 0.5126 s)
verify_stats = (70, 0.0298 s)
global_open  = 70
```

WHIR open + verify together cost **~0.54 s** of the 184 s verify time (~0.3%).
The remaining ~183 s is the verifier **recomputing the forward pass**
(`verify_shard` + `forward_ops`), i.e. the Layer-1 succinctness gap.

## Conclusion

- Layer 2 (WHIR open cost) is NOT the bottleneck: 0.54 s total.
- The real bottleneck is Layer 1: the verifier recomputes the forward pass.
- `batch-opening` (and likewise `ReedWeave`, which also only cuts WHIR cost)
  cannot meaningfully reduce verify time while the verifier still recomputes.
- To reduce verify time: make the verifier claim-driven (open the boundary
  commitments instead of recomputing the forward), then wrap for constant EVM cost.
