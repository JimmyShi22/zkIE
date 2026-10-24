# Vertical weight batching — end-to-end (2026-10-21)

## What changed

The vertical-batching primitives now drive the model prove/verify path instead
of a sidecar micro-benchmark.

- `models/benchmarks/vertical_weights_e2e.rs` groups same-shape Projection
  weights into one MLE, commits each group once (cacheable), then opens all of a
  model's weight claims in a single batch per group. It is split into three
  phases so the harness times model commitment, prove, and verify separately.
- Prover data is streamed to a per-group cache and dropped after commit; the
  opening and verify phases lazily restore one group's Merkle tree at a time
  (lazy opening), so peak memory scales with a single group, not the whole model.
- Each model bench runs this three-phase end-to-end path and reports
  `commit` / `prove` / `verify` separately. The old baseline-vs-packed sidecar
  measurement is removed.

## Results

Security level 90, PoW bits 17, group size 4 (clamped to keep the packed
codeword within the PoW budget).

| Model | Config | Commit | Prove | Verify | Argmax | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | seq=16, 13 shards | 7.41s (first) | 34.59s | 9.46s | 16/16 | - |
| GPT-2 | seq=512, 13 shards | 2.05s (cached) | 47.75s | 20.01s | 511/512 | - |
| TimesFM 200M | seq=16, pps=108 | 26.01s (first) | 77.03s | 11.66s | - | 27.7GB |
| Gemma 3 | seq=16, pps=53 | 9.22s (first) | 48.91s | 16.76s | 16/16 | 25.6GB |
| Gemma 3 | seq=512, pps=53 | 2.56s (cached) | 72.21s | 34.92s | 505/512 | 28.2GB |
| DeepSeek-V2-Lite | seq=16 | 677.44s (first) | 1662.34s | 271.30s | ok | 124.95GB |

`Commit` is the model-commitment phase: pack + WHIR commit + cache write on the
first run, or cache load on later runs. GPT-2/Gemma 3 seq=512 are cached runs;
the others are first runs that write the cache.

Correctness holds: GPT-2 seq=512 argmax 511/512 (one near-tie fixed-point),
Gemma 3 16/16 (seq=512 505/512, fixed-point drift at longer sequences),
DeepSeek verify `ok=true open_ok=true`. The batch-opening proofs are part of the
prove/verify path (`verify_weight_openings` is asserted), not a sidecar.

## Horizontal merge

After the per-group (vertical) batching above, openings were still one FRI per
group: GPT-2 150, TimesFM 332, Gemma 3 59 independent WHIR/FRI proofs. The
opening phase now packs as many same-arity weight groups as the WHIR PoW budget
allows into a single `BatchCtx` (num_vars = arity + log2(group_size), capped at
26 for pow_bits=17), so one FRI opening covers every claim of that arity.

| Model | Config | FRI openings before -> after | Prove before -> after | Verify before -> after |
| --- | --- | --- | --- | --- |
| GPT-2 | seq=16 | 150 -> 3 | 34.59s -> 33.84s | 9.46s -> 8.93s |
| GPT-2 | seq=512 | 150 -> 3 | 47.75s -> 47.78s | 20.01s -> 19.37s |
| TimesFM 200M | seq=16 | 332 -> 9 | 77.03s -> 77.53s | 11.66s -> 10.66s (27.7 -> 24.1GB) |
| Gemma 3 | seq=16 | 59 -> 3 | 48.91s -> 50.24s | 16.76s -> 16.46s |
| Gemma 3 | seq=512 | 59 -> 3 | 72.21s -> 73.04s | 34.92s -> 35.74s |
| DeepSeek-V2-Lite | seq=16 | 1254 -> 189 | 1677.71s -> 1662.34s | 269.37s -> 271.30s (125.06 -> 124.95GB) |

Correctness is unchanged everywhere (GPT-2 16/16 and 511/512; Gemma 3 16/16
and 505/512; DeepSeek verify ok=true open_ok=true).

Effect: independent FRI proofs drop sharply (GPT-2 150->3, DeepSeek
1254->189), which is a
real soundness win (error no longer accumulates across hundreds of openings).
Prove time is essentially flat (within ~2%) because the Merkle/Poseidon work
saved by fewer openings is offset by the larger DFT of the now-packed codewords;
the next lever for prove time is the FRI codeword transform itself (GPU / lower
rate), not the number of openings.

## Bulk serialization

The Merkle prover data (leaves + digest layers) is now written as flat POD byte
streams instead of bincode, so restore is a single `read_exact`/`read_to_end`
plus a contiguous copy rather than a per-element decode. Opening restore dropped
~87%: TimesFM 28.3s -> 3.5s, GPT-2 -> ~1s. First-run commit time also dropped
(TimesFM 94.7s -> 26.0s, Gemma 3 28.9s -> 9.2s) because serialization is faster.

## Key finding: lazy opening is required for large MoE

DeepSeek-V2-Lite's 64-routed-expert MoE produces ~1254 same-shape projection
weight groups (173GB of cached WHIR prover data). Keeping all of them resident
for batch opening OOM-killed the 495GB machine (peak RSS past 415GB). Lazy
opening — restoring one group's Merkle tree at a time — brings peak RSS back to
124.96GB, matching the prior 124.9GB record. Memory scales with activated
parameters, not total parameters.

## Cost and effect

The prior DeepSeek seq=16 record (515s prove / 268s verify) had no weight PCS:
weights were memory-mapped and verified as plaintext during forward
recomputation, so there was no opening and no security parameter in play.

This run adds 90-bit weight commitments and verifies them end to end:
prove 515s -> 2411s, verify 268s -> 271s, memory 124.9GB -> 124.96GB.

- Verify: adding 90-bit weight-commitment verification costs only ~3s. That is
  the payoff of batch opening — without batching, per-weight openings at 90 bits
  would be far more expensive.
- Prove: the ~4.7x slowdown is the one-time pack/commit (~224s) plus lazily
  restoring ~1254 Merkle trees (173GB read back from disk) for opening.

## Notes

- Forward recomputation in `verify_shard` is unchanged; store-free verification
  is a separate concern and out of scope for this change.
- Batch-opening is now the verification relation for weight claims, so its cost
  is the online cost the batching optimization targets.

## How to run

- GPT-2: `cargo run --release -p zkie-models-gpt2 --example bench_sharded`
  (`ZKIE_SEQ_LEN=16` for seq=16).
- TimesFM: `cargo run --release -p zkie-models-timesfm --example bench`.
- Gemma 3: `cargo run --release -p zkie-models-gemma3 --example bench`
  (`GEMMA_SEQ` overrides seq).
- DeepSeek: `cargo run --release -p zkie-models-deepseek-v2-lite --example bench`
  (`M` overrides seq).
- Cache dir: `/data/jimmyshi/zkie-cache/{model}-model-batches-v2`.
