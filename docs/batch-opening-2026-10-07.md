# batch-opening + parallel verify - GPT-2 512 (2026-10-07)

Test machine, 13 shards / 12 cross-binds, `models/gpt2/examples/bench_committed.rs`
(Whir::new_testing(19), argmax against ground truth).

| variant | prove | verify | argmax | note |
|---|---|---|---|---|
| plain (`prove_shard_dag`, no WHIR) | 34.29 s | 19.41 s | 511/512 | verifier recomputes forward once, verifies shards in parallel |
| committed naive (serial verify) | 36.17 s | 183.06 s | 511/512 | serial `verify_shard` clones the full store + re-forwards per shard |
| committed batch (`open_multi`) | 36.09 s | 184.21 s | 511/512 | batching did NOT help (bug masked the real cost) |
| committed batch + parallel verify (fix) | 36.33 s | **22.14 s** | 511/512 | ~8.3x faster verify |

## Root cause (why committed verify was 10x the plain verify)

`verify_committed_shard_dag` verified shards with a **serial** `for` loop calling
`verify_shard`, and `verify_shard` clones the **entire** witness store and re-runs
`forward_ops` for **each** shard. With 13 shards that is ~13 full-store clones plus
~13 forward passes, executed serially.

The plain `verify_shard_dag` already does it correctly: recompute the witness
**once**, then run `verify_shard_precomputed` over `ranges.par_iter()` in parallel.

The fix mirrors the plain path: one `forward_ops`, then a parallel
`verify_shard_precomputed` loop.

## Where the remaining 22 s goes

```
open_stats   = (70, 0.53 s)
verify_stats = (70, 0.03 s)
```

WHIR open + verify is ~0.56 s (~2.5% of verify). The rest (~20 s) is the single
`forward_ops` witness recompute - the Layer-1 succinctness gap, tracked in #15.

## Conclusion

- Layer 2 (WHIR open/verify, batch-opening, ReedWeave) is **not** the verify
  bottleneck: ~0.56 s total.
- The 10x committed-verify gap was a **parallelism bug**, now fixed: 184 s -> 22 s.
- The remaining verify cost is the **single forward recompute** (Layer 1), which
  a claim-driven verifier (#15) removes.
- `batch-opening` and `ReedWeave` only move the 0.56 s Layer-2 term; they cannot
  reduce verify meaningfully while the verifier still recomputes the forward.
