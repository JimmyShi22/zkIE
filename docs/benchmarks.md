# Benchmarks

Baseline measurements for the TimesFM 200M end-to-end proof (prologue + 20 layers + output head) through the unified op interface (prove_200m_ops).

Machine: 64-thread CPU, 3x NVIDIA L20 (46 GB each), 495 GB RAM.

| Backend | Wall time | Peak host memory |
|---|---|---|
| CPU (rayon, 64 threads) | ~404 s (6.7 min) | ~1.4 GB |
| GPU (CUDA, L20) | ~560 s (9.3 min) | ~1.65 GB |

Notes:

- GPU is currently slower than CPU (about 39 percent) because the model has about 29k small tensor commitments that are launch/transfer-bound; the GPU parallelism does not pay off at this granularity.
- The GPU run used one L20 and added about 0.9 GB of VRAM on top of the host memory reported above.
- These are pre-autotuning baselines. Autotuning targets coarser shards so commitments become fewer and larger, which is where the GPU is expected to win.

## IR batch-commit profiling (2026-09-28)

Per-op profiling of the IR two-phase executor (`prove_200m_ir`), release build,
1 layer, to locate the end-to-end bottleneck.

| Backend | commit (1 layer) | GKR sumcheck (k=2048) | WHIR open/verify (k=2048) |
|---|---|---|---|
| CPU (rayon, 64 threads) | 1.89 s | 24.0 ms | 345 ms |
| GPU (ZKIE_CUDA=1, L20) | 2.07 s | 23.3 ms | 397 ms |

Key finding:

- The dominant cost is **WHIR opening/verification** (the FRI opening proof),
  not the GKR sumcheck. Per 2048-dim matmul the open/verify is ~345 ms while
  the GKR sumcheck is ~24 ms (~14x smaller).
- The current CUDA backend (Sppark DFT + Poseidon2 Merkle) only accelerates
  the **commit** (building the Merkle tree); the opening/verification path is
  the upstream p3-merkle-tree CPU code, so the GPU does not accelerate the
  actual bottleneck.
- The GPU is slightly *slower* than the 64-thread CPU on both commit (2.07 s
  vs 1.89 s) and open/verify (397 ms vs 345 ms). The workload is launch-bound
  on the GPU, and the CPU AVX2 Poseidon2 is already competitive.

Implication: batching commitments (the earlier "29x batch commit" microbench)
targets the wrong stage. The real cost is the many small FRI opening proofs.
A meaningful speedup needs either (a) aggregating openings across ops, or
(b) a PCS with cheaper/fewer openings, rather than further GPU-izing the
commit.
