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
