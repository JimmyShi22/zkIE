# zkIE Full-Model Resource-Requirement Estimate (FinText 8M / TimesFM 1.0 200M)

**Date**: 2026-07-26
**Status**: Analysis only — no `crates/zkie-core`/`crates/zkie-compiler` source changed. A throwaway
benchmark tool was added at `crates/zkie-core/examples/bench_rowcount.rs` (see "Files" section);
nothing was committed.

**Machine used for all measurements**: Apple M4 Pro, 14 cores, 48GB RAM (the same machine used in
the prior `2026-07-26-zkie-smaller-timesfm-attempt.md` session).

## Executive summary

- **FinText 8M, full model, context_len=512 (16 tokens)**: needs circuit `k=28` (2^28 = 268M row
  capacity for ~148M real rows), roughly **0.6–1 TB peak RAM**, roughly **17–34 GB** of SRS/proving-key
  material on disk, and roughly **2–3 hours** of total setup+keygen+proof wall-clock time on a
  14-core-class machine (of which proving itself, the per-inference marginal cost once
  setup/keygen are cached, is roughly **35–50 minutes**). **This is within reach of a single very
  large cloud memory-optimized instance** (1–2TB RAM SKUs exist today).
- **TimesFM 1.0 200M, full model, context_len=512 (16 tokens)**: needs circuit `k=32`, roughly
  **13–21 TB peak RAM**, roughly **275–550 GB** of SRS material, and on the order of **2–3.5 days**
  of total wall-clock time. At a more realistic longer context (context_len=2048, 64 tokens, an
  assumption — see Part 2), it needs `k=34`, roughly **47–84 TB** peak RAM, and **roughly 1–2.5
  weeks**. **Both exceed the practical ceiling of a single cloud instance today (~24TB is close to
  the largest currently-available single-box RAM SKU)** — full 200M-model proving at this
  circuit design would need a genuinely distributed/multi-machine proving setup, not just a
  bigger box.
- **Both models need two real code fixes before either estimate is even attemptable**, independent
  of hardware: (1) `ReduceSumChip`'s `i64` accumulator overflows for real hidden dimensions (would
  need widening to `i128`, mirroring `DotProductChip`'s already-`i128` accumulator), and (2)
  `Split`/multi-output-node support is missing from `zkie-compiler` (needed to compile a real
  fused-QKV projection node). These are correctness/capability blockers, not compute-resource
  problems, and are listed separately in Part 3.
- **A genuinely useful correction to the prior session's estimate**: the prior report guessed
  "16–30x rows-per-scalar-mult overhead" for this project's chip design. This session's real
  benchmark shows the *actual* multiplier for realistic transformer matmuls (contraction
  dimensions of 264–1280 elements) is only **~1.17x–1.7x** — the earlier 16–30x figure describes
  a different, narrower regime (`ReduceSumChip`-based reductions over very few elements, e.g. the
  prior session's `K=8`-channel RMSNorm proof, where `(8+184)/8 = 24x`), not `DotProductChip`-based
  matmuls, whose fixed per-instance overhead amortizes far better over a realistic contraction
  length. See Part 2 for the exact math.
- All of this is specific to **this project's current circuit design** (one `DotProductChip`
  region per scalar multiply-accumulate, no batched/vectorized matmul argument, no GPU
  acceleration). Production ZKML systems that use batched polynomial arguments and/or
  GPU-accelerated MSM/FFT can prove similarly-sized models with dramatically less time and memory
  — these numbers describe the cost of *this specific, deliberately simple, per-scalar-dot-product
  architecture*, not a fundamental limit on proving 8M/200M-parameter models in general.

---

## Part 1: Real, extended empirical benchmark

### Methodology

The prior session's benchmark (`crates/zkie-core/tests/scratch_bench_rowcount.rs`) was written, run,
then deleted without being committed — there is nothing to recover from git history (confirmed via
`git log --all --diff-filter=A --name-only | grep scratch_bench`, no hits). This session wrote a
new one at **`crates/zkie-core/examples/bench_rowcount.rs`**, deliberately placed under
`examples/` (not `tests/`) so `cargo test --workspace` never builds or runs it — examples are not
part of cargo's default test-target set, so this cannot pollute the normal test suite. It was not
committed; it is left in place for a supervisor to review or discard.

**Circuit shape**: `n_instances` independent `DotProductChip` regions, each performing a real
`DOT_LEN = 264`-length dot product (`264` = FinText 8M's real `hidden_size`, chosen specifically so
the measured rows-per-instance ties directly into Part 2's real row-count math), all sharing one
configured `DotProductConfig`. This exactly mirrors how `AssemblerChip::assign_dot_general`
instantiates one region per `(i, j)` output element of a real matmul — same accumulation gate,
same final-rescale gate, same three range-check sub-chips (`q`/`r`/`slack`) as every real
`DotGeneral` dispatch — so column count and gate degree are realistic, not a degenerate trivial
circuit.

**Exact row cost per instance** (read directly from `crates/zkie-core/src/chips/dot_general.rs`,
not guessed): `DotProductChip::configure` uses `K` rows for the running accumulation, plus a fixed
range-check overhead of **64 rows** (`q`, 64-bit range check) + **60 rows** (`r`,
`REMAINDER_BITS = 60`) + **60 rows** (`slack`, also 60 bits) = **184 fixed overhead rows per
instance, independent of K**. So `rows_per_instance = DOT_LEN + 184 = 448` for `DOT_LEN = 264`, and
`rows_per_mult = (K + 184) / K`. The benchmark's own printed output confirms this exactly:
`rows_per_mult=1.6970` at every tested `k` (`(264+184)/264 = 1.6970...`), which is real, measured
agreement with the source-level formula, not a coincidence.

At each `k`, `n_instances` was chosen to fill ~90% of `2^k` rows (leaving margin for halo2's small
blinding-row reservation), then: (1) `MockProver::assert_satisfied()` as a correctness gate, (2)
`ParamsKZG::<Bn256>::setup`, (3) `keygen_vk` + `keygen_pk`, (4) `create_proof`
(`ProverSHPLONK`), and (5) `verify_proof`, each timed separately via `std::time::Instant`. Peak
memory was measured by wrapping the whole binary in macOS's `/usr/bin/time -l`, which reports
"maximum resident set size" and "peak memory footprint" (the latter used as the primary figure
below, since it was available for every run; both are reported where available).

### Real measured results

| k | rows used | scalar mults | rows/mult | setup (s) | keygen (s) | prove (s) | **total** (s) | peak mem footprint |
|---|---|---|---|---|---|---|---|---|
| 10 | 896 | 528 | 1.697 | 0.044 | 0.024 | 0.055 | **0.124** | 11.9 MB |
| 12 | 3,584 | 2,112 | 1.697 | 0.124 | 0.060 | 0.102 | **0.286** | 24.4 MB |
| 14 | 14,336 | 8,448 | 1.697 | 0.407 | 0.121 | 0.276 | **0.805** | 82.8 MB |
| 16 | 58,688 | 34,584 | 1.697 | 1.550 | 0.396 | 0.808 | **2.755** | 402 MB |
| 18 | 235,648 | 138,864 | 1.697 | 6.802 | 1.490 | 3.200 | **11.492** | 1.505 GB |
| 20 | 943,488 | 555,984 | 1.697 | 24.883 | 6.364 | 12.863 | **44.109** | 5.699 GB |
| 22 | 3,774,848 | 2,224,464 | 1.697 | 97.884 | 27.182 | 52.122 | **177.188** | 20.96 GB |
| 24 | 15,099,392 | 8,897,856 | 1.697 | 404.965 | 126.178 | *(killed)* | *(>589)* | *(not captured)* |

**k=24 did not complete within the 10-minute (590s) budget.** `MockProver` check (14.4s), `setup`
(405.0s), and `keygen` (126.2s) all completed — a combined 545.5s — but `create_proof` was still
running when the wrapping `timeout 590` killed the process (exit code 124). Extrapolating from the
two completed phases' own scaling (setup: 4.14x time for 4.00x rows vs k=22; keygen: 4.64x time for
4.00x rows — both *slightly worse* than proportional), total time at k=24 is estimated at roughly
**700–800s (~12–13 minutes)** — i.e., not a dramatic cliff, but a real, measured sign that scaling
degrades slightly beyond pure-linear as row count grows into the tens of millions.

### Scaling-law analysis (real ratios, not assumed)

Every step above is a ~4.00x row-count increase (k→k+2). Fitting `time ∝ rows^p` from consecutive
points (`p = log(time_ratio) / log(rows_ratio)`):

| step | rows ratio | total-time ratio | **p** (time exponent) |
|---|---|---|---|
| k10→12 | 4.00 | 2.31 | 0.60 |
| k12→14 | 4.00 | 2.81 | 0.75 |
| k14→16 | 4.09 | 3.42 | 0.89 |
| k16→18 | 4.01 | 4.17 | 1.03 |
| k18→20 | 4.00 | 3.84 | 0.97 |
| k20→22 | 4.00 | 4.02 | **1.004** |
| k22→24 (setup only) | 4.00 | 4.14 | 1.02 |
| k22→24 (keygen only) | 4.00 | 4.64 | 1.11 |

**Interpretation**: at small `k` (≤16), fixed per-run overhead dominates and the apparent exponent
is well below 1 (sub-linear-looking, an artifact of the fixed cost, not a real efficiency gain). By
k=20→22 — the largest fully-measured, most reliable window — total time tracks almost *exactly*
linearly with row count (`p ≈ 1.004`, i.e., 4.02x time for 4.00x rows). The k=22→24 step shows
early signs of mild super-linearity creeping in for `setup` (p≈1.02) and more so for `keygen`
(p≈1.11), consistent with `O(N log N)` FFT/MSM-bound costs eventually showing through, but the
effect is modest, not a cliff. **This project's proving time is best described as close to linear
(O(N) to a mild O(N log N)) across the measured range**, not quadratic or worse.

Peak memory shows the same pattern (`p ≈ 0.52–1.15` across steps, converging to **~0.93–0.95** by
k=18→22 — i.e., also close to linear, perhaps very slightly sub-linear at these sizes as fixed
buffer overhead amortizes).

### Real SRS size on disk (measured, not theorized)

`ParamsKZG::write` (the `Params` trait's default, `SerdeFormat::RawBytes`, uncompressed) and
`ParamsKZG::write_custom(..., SerdeFormat::Processed)` (compressed) were both called and the
resulting files measured directly:

| k | raw (uncompressed) bytes | processed (compressed) bytes |
|---|---|---|
| 10 | 131,332 | 65,668 |
| 12 | 524,548 | 262,276 |
| 14 | 2,097,412 | 1,048,708 |

These three points fit an **exact** closed form (not approximate — verified bit-for-bit):
`SRS_raw(k) = 128·2^k + 260` bytes, `SRS_processed(k) = 64·2^k + 132` bytes. (`ParamsKZG` stores
`2·n` `G1Affine` points — `g` and `g_lagrange` — plus two `G2Affine` points; raw/uncompressed
`G1Affine` = 64 bytes, compressed = 32 bytes, matching BN256's standard point sizes exactly.) This
formula was used, not re-measured at huge `k` (a k=32 SRS alone would be hundreds of GB to write),
to extrapolate SRS disk size in Part 3.

### Core-count / parallelism check (real, measured)

`Cargo.lock` confirms `halo2_proofs`'s dependency tree includes `rayon`. To check whether this
project's actual proving path meaningfully uses multiple cores (not just in principle), k=18 was
run twice, back-to-back, with no other CPU-bound process competing for cores:

| | setup (s) | keygen (s) | prove (s) | **total** (s) |
|---|---|---|---|---|
| `RAYON_NUM_THREADS=0` (all 14 cores) | 6.309 | 1.450 | 3.127 | **10.885** |
| `RAYON_NUM_THREADS=1` (pinned to 1 core) | 64.045 | 8.752 | 14.181 | **86.978** |

**8.0x real speedup** from parallelism on a 14-core machine (setup benefits most, ~10.2x; proving
itself ~4.5x). This confirms the PSE halo2 fork's `ProverSHPLONK` path genuinely engages multiple
cores at these sizes (not just in theory) — but the realized 8.0x on 14 cores (~57% parallel
efficiency) is sub-linear, so more cores help materially but with diminishing returns; core-count
scaling beyond 14 was not tested.

---

## Part 2: Real row-count formulas for both models

### Real chip-level overhead formula

From Part 1: a single `DotProductChip`/`AssemblerChip`-dispatched dot product of contraction length
`K` costs `K + 184` circuit rows (the assembler's own dispatch, `assign_dot_general` in
`crates/zkie-core/src/assembler.rs`, adds one more row for its output register-bridge gadget, i.e.
`K + 185` — a <0.4% difference at the K values below, folded into the totals as noted). So:

`rows_per_mult(K) = (K + 184) / K`

This is **much tighter than a single flat multiplier** — it depends on which matmul's contraction
length `K` is used:

| matmul | K (contraction dim) | rows/mult |
|---|---|---|
| FinText QKV / AttnOut / FFN-up (K=hidden_size=264) | 264 | **1.697x** (matches the benchmark exactly) |
| FinText FFN-down (K=intermediate_size=1024) | 1024 | 1.180x |
| FinText attention Q·Kᵀ (K=head_dim=66) | 66 | 3.788x |
| FinText attention scores·V (K=context tokens, e.g. 16) | 16 | 12.5x |
| TimesFM QKV/AttnOut/FFN (K=hidden_size or intermediate_size=1280) | 1280 | 1.144x |
| TimesFM attention Q·Kᵀ (K=head_dim=80) | 80 | 3.30x |
| RMSNorm reduction, prior session's K=8-channel proof-of-concept | 8 | **24.0x** |

The prior report's "16–30x" estimate is the *reduction/small-K regime* (exactly matching the
K=8 RMSNorm case at 24.0x) — real matmuls with realistic contraction lengths (264–1280 elements)
amortize the fixed 184-row overhead far better, landing at **1.1x–1.7x**, not 16–30x. This is the
main correction this session's real measurement adds to the prior estimate.

### FinText 8M — architecture recap

`num_layers=7, hidden_size=264, num_heads=4, num_kv_heads=4, head_dim=66, intermediate_size=1024`.
Check: `num_heads·head_dim = 4·66 = 264 = hidden_size` ✓.

**Per-token, per-layer matmul mults** (four dominant projections):
- QKV: `hidden → (num_heads+2·num_kv_heads)·head_dim = (4+8)·66 = 792`; mults = `792·264 = 209,088`; rows = `792·(264+184) = 792·448 = 354,816`.
- AttnOut: `264 → 264`; mults = `264·264 = 69,696`; rows = `264·448 = 118,272`.
- FFN up: `264 → 1024`; mults = `1024·264 = 270,336`; rows = `1024·448 = 458,752`.
- FFN down: `1024 → 264`; mults = `264·1024 = 270,336`; rows = `264·(1024+184) = 264·1208 = 318,912`.
- **Sum (4 projections)**: mults = `819,456`, rows = `1,250,752`.

**Attention-score matmuls** (`Q·Kᵀ` and `scores·V`, accounted for explicitly, not waved away),
per query token attending over `L` context tokens, `num_heads=4, head_dim=66`:
- `Q·Kᵀ`: per head, output=`L`, K=`66`; mults=`L·66`; rows=`L·250`. Over 4 heads: mults=`4·66·L=264L`, rows=`4·250·L=1000L`.
- `scores·V`: per head, output=`66`, K=`L`; mults=`66·L`; rows=`66·(L+184)`. Over 4 heads: mults=`264L`, rows=`4·66·(L+184)`.
- Per query token: mults = `528·L`, rows depend on `L` (shown inline below).

#### Case A — FinText 8M, context_len=512 (patch_len=32 → **L=16** tokens), the "full small model" case

Per token/layer (4 projections + attention, `L=16`): mults = `819,456 + 528·16 = 827,904`; rows =
`1,250,752 + 68,800 = 1,319,552` (attention rows at L=16: `1000·16 + 4·66·(16+184) = 16,000 +
52,800 = 68,800`).

Per layer, **L=16 tokens** (4-projection cost scales linearly in L; attention-score cost scales as
`L²` since each of L query tokens attends over L keys):
- 4-projection rows/mults: `1,250,752·16 = 20,012,032` rows, `819,456·16 = 13,111,296` mults.
- Attention rows/mults: `68,800·16 = 1,100,800` rows, `8,448·16 = 135,168` mults.
- **Per layer total: 21,112,832 rows, 13,246,464 mults.**

**All 7 layers: 147,789,824 rows (~148M), 92,725,248 scalar mults (~92.7M).** Blended
rows/mult = `147,789,824 / 92,725,248 = 1.594x` — consistent with the per-matmul-type table above
(dominated by the 264/1024-length matmuls at ~1.18–1.70x, with the small-K attention terms pulling
the blend up slightly).

#### Case B — FinText 8M, context_len=32 (**L=1** token) — sanity check vs. the prior session

At `L=1`: attention rows/mults per token = `1000·1 + 4·66·(1+184) = 1,000 + 48,840 = 49,840` rows,
`528` mults. Per-layer total (single layer, since L=1 needs no further multiplication): **rows =
1,250,752 + 49,840 = 1,300,592; mults = 819,456 + 528 = 819,984.**

This **matches the prior session's own reported "~820K–1.1M mults/layer at minimal 1-patch
context" almost exactly** (819,984 sits right at the low end of that range) — a genuine,
independent cross-check that this session's from-scratch arithmetic agrees with the prior
session's. (Full 7-layer total at L=1, for completeness: 5,739,888 mults, 9,104,144 rows — not the
prior report's per-layer framing, but included here for completeness.)

### TimesFM 1.0 200M — architecture recap

`num_layers=20, hidden_size=1280, num_heads=16, num_kv_heads=16, head_dim=80, intermediate_size=1280`.
Check: `16·80 = 1280 = hidden_size` ✓ (no GQA — `num_kv_heads = num_heads`).

**Per-token, per-layer matmul mults** (four dominant projections):
- QKV: `1280 → (16+32)·80 = 3840`; mults = `3840·1280 = 4,915,200`; rows = `3840·(1280+184) = 3840·1464 = 5,621,760`.
- AttnOut: `1280 → 1280`; mults = `1,638,400`; rows = `1280·1464 = 1,873,920`.
- FFN up: `1280 → 1280`; mults = `1,638,400`; rows = `1,873,920`.
- FFN down: `1280 → 1280`; mults = `1,638,400`; rows = `1,873,920`.
- **Sum (4 projections)**: mults = `9,830,400`, rows = `11,243,520`.

**Attention-score matmuls**, `num_heads=16, head_dim=80`, per query token over `L` context tokens:
mults/token = `2·16·80·L = 2560L`; rows/token = `16·L·(80+184) + 16·80·(L+184) = 4224L +
1280·(L+184)`.

#### Case C — TimesFM 200M, context_len=512 (**L=16**), apples-to-apples with FinText

Per token/layer: attention rows = `16·264 + 80·(16+184) = 4,224·16⁄... ` — computed directly:
`Q·Kᵀ` rows (16 heads) = `16·16·264 = 67,584`; `scores·V` rows (16 heads) = `16·80·200 = 256,000`;
attention total = `323,584` rows, `40,960` mults (`2560·16`).

Per layer, L=16 tokens: 4-projection = `11,243,520·16 = 179,896,320` rows, `9,830,400·16 =
157,286,400` mults. Attention (L² scaling) = `323,584·16 = 5,177,344` rows, `40,960·16 = 655,360`
mults. **Per layer: 185,073,664 rows, 157,941,760 mults.**

**All 20 layers: 3,701,473,280 rows (~3.70B), 3,158,835,200 scalar mults (~3.16B).** Blended
rows/mult = `1.172x` (better than FinText's 1.594x, since TimesFM's larger hidden/intermediate
dimensions of 1280 amortize the fixed 184-row overhead even further).

#### Case D — TimesFM 200M, a "typical longer" context — **assumption: context_len=2048 (L=64 tokens)**

The task's own architecture recap left context length "flexible"; TimesFM's paper and typical usage
support forecasting over multi-hundred-to-low-thousands-step histories, so **L=64 patches
(context_len=2048) is used here as a representative "longer, still realistic" case** — stated
explicitly as an assumption, not a hard spec.

Per token/layer at L=64: `Q·Kᵀ` rows (16 heads) = `16·64·264 = 270,336`; `scores·V` rows (16 heads)
= `16·80·248 = 317,440`; attention total = `587,776` rows, `163,840` mults (`2560·64`).

Per layer, L=64 tokens: 4-projection = `11,243,520·64 = 719,585,280` rows, `9,830,400·64 =
629,145,600` mults. Attention (L²) = `587,776·64 = 37,617,664` rows, `163,840·64 = 10,485,760`
mults. **Per layer: 757,202,944 rows, 639,631,360 mults.**

**All 20 layers: 15,144,058,880 rows (~15.14B), 12,792,627,200 scalar mults (~12.79B).** The
attention-score share of total mults grows from `0.42%` at L=16 to `1.64%` at L=64 — real, measured
growth with context length, though still small in absolute terms until context lengths get far
longer than either case tested here (the O(L²) term would only start to dominate at
several-hundred-token contexts, well beyond what's estimated here).

---

## Part 3: Actionable resource estimate

Time and memory are extrapolated from the Part 1 scaling law. Two figures are given per model:
a **central estimate** (pure-linear extrapolation from the k=20→22 measured rate, the most reliable
completed window) and a **wider band** (using the mildly super-linear exponents observed at
k=22→24, `p≈1.08` for time, `p≈0.93` for memory) to reflect the genuine uncertainty of extrapolating
2–3+ orders of magnitude past directly measured data (per the task's own guidance). **Setup and
keygen are one-time costs, reusable across every proof of the same circuit shape/weights; `prove`
is the marginal per-inference cost.**

| | **FinText 8M** (L=16, k=28) | **TimesFM 200M** (L=16, k=32) | **TimesFM 200M** (L=64, k=34) |
|---|---|---|---|
| Rows required | 147,789,824 | 3,701,473,280 | 15,144,058,880 |
| Required circuit `k` | 28 (2^28=268.4M, 55% full) | 32 (2^32=4.295B, 86% full) | 34 (2^34=17.18B, 88% full) |
| Setup (one-time) | ~1.06–1.4 h | ~26.7–37 h (1.1–1.5 d) | ~4.5–7 d |
| Keygen (one-time) | ~0.30–0.40 h | ~7.4–10 h | ~1.3–2 d |
| **One-time total (setup+keygen)** | **~1.4–1.8 h** | **~1.4–2.0 d** | **~5.8–9 d** |
| Prove (per-proof marginal) | ~34–48 min | ~14.2–20 h | ~2.4–3.7 d |
| **Grand total, first proof** | **~1.9–2.6 h** | **~2.0–3.5 d** | **~8.2–16 d** |
| Peak RAM | **~635–820 GB** | **~12.7–20.5 TB** | **~47–84 TB** |
| SRS on disk (compressed / raw) | 17.2 GB / 34.4 GB | 274.9 GB / 549.8 GB | 1.10 TB / 2.20 TB |
| Proving key (not separately measured) | roughly same order of magnitude as SRS above (1–3x), qualitative only | same | same |
| Recommended cores | as many as economical; ~8x measured speedup on 14 cores (~57% efficiency), diminishing returns expected beyond that, not tested past 14 | same guidance | same guidance |
| **Fits on 1 machine?** | **Yes** — within reach of a single 1–2TB memory-optimized cloud instance | **No** — exceeds the largest practical single-instance RAM SKUs (~24TB ceiling today) | **No** — same, by a wider margin |

**Given the extrapolation spans ~40x (FinText) to ~1,000–4,000x (TimesFM) beyond the largest
directly-measured row count (k=22, 3.77M rows), treat the TimesFM figures as order-of-magnitude,
not precise** — real hardware effects at genuinely enormous scale (memory-bandwidth/NUMA behavior
on a hypothetical 20–80TB machine, which behaves differently than the 48GB machine this benchmark
ran on) could plausibly push the real numbers meaningfully higher than even the wide band above;
they are very unlikely to be *lower*.

### Required code fixes before any of this is attemptable (blockers, not resource costs)

Confirmed directly in the current source this session (not just cited from the prior report):

1. **`ReduceSumChip`'s accumulator overflows for real hidden dimensions.**
   `crates/zkie-core/src/chips/reduce.rs` accumulates `partial_sums: Vec<i64>` directly in raw I18
   space (`partial_sums[i-1].checked_add(v.raw())` at line 134, and the same pattern again in the
   module's own test helpers) — meaning the *running sum itself*, not just the final mean, must
   stay within I18's representable magnitude (~9.22). Real per-channel reductions over any
   realistic hidden dimension (FinText's 264, TimesFM's 1280) will overflow this well before
   reaching the end of the sum. `DotProductChip` doesn't have this problem because its own
   accumulator (`crate::assembler::assign_dot_row`'s `raw_sum: i128`, and
   `DotProductChip::assign`'s identical `i128` accumulator in `dot_general.rs`) is already widened
   to `i128` — `ReduceSumChip`/`ReduceMeanChip` (and therefore `LayerNormChip`, which composes
   them) would need the same widening. Real, contained, but necessary fix.
2. **`Split`/multi-output-node support is missing from `zkie-compiler`.** Confirmed via
   `grep -rn "Split" crates/zkie-compiler/src/` returning zero hits. A fused QKV projection node
   (the real TimesFM/FinText export's actual graph shape) needs to split one projection's output
   into separate Q/K/V tensors — `zkie-compiler`'s current `Register`/`RegisterRef` model assumes
   one instruction produces exactly one output tensor, so this needs a real (not huge, but
   non-trivial) extension before a real exported graph's attention block could compile at all.

Both are correctness/capability blockers independent of the hardware numbers above — fixing them
doesn't change the row-count math in Part 2, but without them, neither full model can be compiled
and proven at all, regardless of how much RAM or time is available.

### Context: this is a cost of the *current circuit design*, not a fundamental ZKML limit

Every number above stems from a single design choice: `DotProductChip` proves one scalar
multiply-accumulate at a time, with no batched/vectorized/tiled matmul argument, and no GPU
acceleration for MSM/FFT. Production ZKML proving systems that use batched polynomial commitment
arguments (e.g., committing to whole matrices/tensors at once rather than one scalar product at a
time) and/or GPU-accelerated MSM/FFT routinely prove similarly-sized (and larger) models with
dramatically less time and memory than shown here — this is a well-known, hard, *engineering*
problem in ZKML generally, not evidence that proving an 8M or 200M-parameter model is inherently
this expensive. These figures describe what it costs *this project's specific, deliberately simple,
per-scalar-dot-product architecture*, built for sub-projects 1–3's ISA-coverage goals, not a
ceiling on ZKML as a field.

---

## Files

- `crates/zkie-core/examples/bench_rowcount.rs` — the throwaway benchmark tool (new, uncommitted,
  under `examples/` so it never affects `cargo test --workspace`). Usage:
  `cargo run --release --example bench_rowcount -- <k> [fill_fraction] [dump-srs]`.
- `.spike-test/bench/srs_k{10,12,14}_{raw,processed}.bin` — the real SRS files written to disk to
  measure exact serialized size (gitignored via the existing `.spike-test/` entry in `.gitignore`).
- This report: `docs/superpowers/specs/2026-07-26-zkie-full-model-resource-estimate.md`.
- No files under `crates/zkie-core/src` or `crates/zkie-compiler` were modified.

## Verification

`cargo test --workspace` was run once at the end of this session to confirm the baseline (164
tests, per the prior session's report) still passes and nothing was left broken: **164 passed, 0
failed, 0 ignored** (summed across all `test result:` lines in the run), confirming the new
`examples/bench_rowcount.rs` benchmark tool did not affect the normal test suite in any way.
