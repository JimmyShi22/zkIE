# LogUp batching (within-op fusion + cross-op batch by table) (2026-10-22)

Batch independent fractional-logUp sumchecks so that lookups into the same
table share one sumcheck, instead of one sumcheck per op.

Two changes:

1. **Within-op fusion** — RMSNorm's `rem_lo` + `rem_hi` range checks (both into
   the `0..2^16` limb table) are fused into one batched lookup, so RMSNorm goes
   from 3 logUp sumchecks (rsqrt + 2 range) to 2.
2. **Cross-op batch by table** — the rsqrt and softmax exp lookups are deferred
   out of the per-op proof functions and batched at the shard level:
   `prove_shard_precomputed` collects every op's lookup samples, groups them by
   table, and runs one `prove_lookup_fractional_batch` per table.

Primitive added in `zkie-core/common/logup_gkr.rs`:
`prove_lookup_fractional_batch` / `verify_lookup_fractional_batch`.

## Results (Gemma 3 270M, seq=512, 22 shards, 64-thread CPU)

| stage | prove | verify | total | argmax |
| --- | --- | --- | --- | --- |
| baseline (per-op logUp) | 73.85 s | ~36 s | 111.74 s | 505/512 |
| + within-op fusion | 71.92 s | 37.87 s | 109.79 s | 505/512 |
| + rsqrt cross-op batch | 61.14 s | 33.44 s | 94.58 s | 505/512 |
| + softmax cross-op batch | 59.05 s | 33.27 s | 92.32 s | 505/512 |

prove -20% (73.85 -> 59.05 s), total -17% (111.74 -> 92.32 s). argmax unchanged
at 505/512.

## Sumcheck count

Total `prove_lookup_fractional` calls drop 993 -> 639 (-36%):

- RMSNorm rsqrt: 163 -> 22 (one per shard)
- Softmax exp: 72 -> 22 (one per shard)
- RMSNorm range: 326 -> 163 (fused `rem_lo` + `rem_hi`)

## Which lookups benefit from batching

Batching helps only when the table is large relative to the sample count
(rsqrt 7M-entry table, softmax exp 2M-entry table). Lookups into small range
tables (projection / RMSNorm `0..2^16`) are sample-dominated, so batching them
gives no time win; GELU (2^24 table) has only ~1 lookup per shard, so there is
no within-shard opportunity.

## Memory fix

The batch initially raised peak RSS (~28.9 -> ~39.3 GB on Gemma 3 seq=512, and
~125 -> ~147 GB extrapolated on DeepSeek seq=16). Root cause: `prove_fractional`
copied the four half-slices at every folding layer (`to_vec()`), so the peak was
~7N per lookup. Fixed by taking ownership of `num`/`den` and folding via
`split_at` slices (peak ~3N), commit `e7948bc`.

### Results after the memory fix

| model | prove | verify | argmax | peak RSS |
| --- | --- | --- | --- | --- |
| GPT-2 seq=16 | 25.71 s | 5.81 s | 16/16 | - |
| GPT-2 seq=512 | 41.08 s | 17.04 s | 511/512 | - |
| TimesFM seq=16 | 67.40 s | 6.93 s | - | 19.19 GB |
| Gemma 3 seq=16 | 34.31 s | 11.46 s | 16/16 | - |
| Gemma 3 seq=512 | 58.17 s | 31.04 s | 505/512 | 28.62 GB |
| DeepSeek-V2-Lite seq=16 | 1627.21 s | 252.09 s | ok=true | 101.84 GB |

Gemma 3 seq=512 peak RSS is back to the pre-batch baseline (28.62 GB).
DeepSeek-V2-Lite seq=16 peak RSS -18.5% (124.95 -> 101.84 GB).

## Notes

- 37 zkie-ops unit tests pass; full workspace release build is green.
- DeepSeek-V2-Lite seq=512 pending (long run); will be added when it completes.
