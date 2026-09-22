# zkie-cuda

CUDA acceleration for the zkIE GKR/WHIR prover (Goldilocks), mirroring the
halo2-side CUDA MSM crate's build pattern.

## Contents

- `src/dft.rs` — `CudaDft`, a `p3_dft::TwoAdicSubgroupDft<Goldilocks>`.
  Only `dft_batch` is implemented; every other trait method derives from it.
  The kernel is a purpose-built radix-2 DIT over Sppark's `gl64_t`
  (canonical 64-bit arithmetic mod 2^64 − 2^32 + 1, byte-compatible with
  Plonky3's Goldilocks), with per-size twiddle tables generated on the host
  from p3's own `TWO_ADIC_GENERATORS` and uploaded once per size.
- `src/merkle.rs` — `CudaMerkleTreeMmcs`, a `p3_commit::Mmcs<Goldilocks>`
  with the 200M proof's exact configuration (Poseidon2-16,
  `PaddingFreeSponge` leaf hashing, `TruncatedPermutation` compression,
  arity 2, digest 8). `commit` builds all Merkle digest layers on the GPU
  and wraps them via the vendored `MerkleTree::from_parts`, so openings and
  verification remain the upstream p3-merkle-tree CPU code, bit-identical to
  a CPU-built tree. The Poseidon2 round constants are re-derived on the host
  from `SmallRng::seed_from_u64(1)` using p3's own public API — the same
  sequence the CPU permutation uses — and uploaded to device `__constant__`
  memory.
- `vendor/p3-merkle-tree` — unmodified crates.io 0.7.0 source plus one
  doc-hidden constructor (`MerkleTree::from_parts`), wired in through
  `[patch.crates-io]` in the workspace root.

Both backends fall back to their CPU counterparts when no GPU is available
or a kernel errors.

## Building

Tested with CUDA 12.4; native `sm_89` (Ada, e.g. the L20):

```bash
NVCC=off \
CUDA_NVCC=/usr/local/cuda-12.4/bin/nvcc \
CUDA_HOME=/usr/local/cuda-12.4 \
cargo build -p zkie-cuda --features cuda
```

`NVCC=off` stops the `sppark` crate's own build script from compiling a
second copy of `util/all_gpus.cpp`.

## Tests

`cargo test -p zkie-cuda --features cuda` runs bit-exactness suites (GPU vs
CPU, skipped when no device is present):

- `tests/dft_bit_exact.rs` — `dft_batch`/`idft_batch` over 2^8..2^16
  heights × widths {1,2,8,32,64}.
- `tests/merkle_bit_exact.rs` — Merkle caps over heights {1..1024} ×
  widths {1..100}, plus an open/verify roundtrip on a GPU-built tree.

At the zkie-gkr level, `cargo test -p zkie-gkr --features cuda --test
whir_cuda` pins full WHIR commitments (GPU vs an explicit CPU pipeline) and
a commit/open/verify roundtrip. `examples/bench_whir_cuda.rs` benchmarks
both backends at 2^15 and 2^22.

## Measured (L20, 64-core box)

| d   | commit CPU (1 thread) | commit GPU | speedup | commit CPU (64 rayon threads) |
|-----|-----------------------|------------|---------|-------------------------------|
| 15  | 0.022s                | 0.006s     | 3.5x    | 0.002s                        |
| 22  | 2.828s                | 0.109s     | 25.9x   | 0.112s                        |

The GPU path is launch/transfer-bound at these sizes: with 64 rayon threads
the CPU keeps pace, while the GPU wins decisively in the thread-constrained
regime and leaves CPU threads free for the sumcheck folding and opening
work.

## Known limitations / next steps

- The Sppark mixed-radix NTT kernels were evaluated first but their
  twiddle-table convention is incompatible with p3's generator (an impulse
  probe showed an effective generator of order 2), hence the purpose-built
  radix-2 kernel. A faster radix-32 variant is a straightforward follow-up.
- Per-commit H2D/D2H of the RS-expanded codewords and per-level digest
  downloads are the main transfer costs; a device-resident pipeline inside
  the WHIR round loop would remove them.
- Many small commitments (2^8..2^11 in the 200M proof) are launch-bound;
  batching commits per size or pipelining on streams would help.
