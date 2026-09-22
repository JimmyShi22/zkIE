//! Bit-exactness of `CudaDft::dft_batch` against the CPU
//! `Radix2DFTSmallBatch` (and the trait's derived `idft_batch`), over random
//! matrices of the sizes WHIR uses (heights 2^8..2^18, widths up to 64).
//!
//! Requires the `cuda` feature and a usable GPU; otherwise the test reports
//! itself as skipped.

#![cfg(feature = "cuda")]

use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::RngExt;
use rand::SeedableRng;
use zkie_cuda::dft::CudaDft;

fn random_matrix(lg: usize, w: usize, rng: &mut SmallRng) -> RowMajorMatrix<Goldilocks> {
    let h = 1 << lg;
    let values: Vec<Goldilocks> = rng.sample_iter(StandardUniform).take(h * w).collect();
    RowMajorMatrix::new(values, w)
}

#[test]
fn dft_batch_matches_cpu() {
    if !zkie_cuda::cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let gpu = CudaDft::new();
    assert!(gpu.is_cuda(), "expected a GPU-backed CudaDft");
    let cpu = Radix2DFTSmallBatch::new(1 << 18);

    let mut rng = SmallRng::seed_from_u64(0xabc123);
    for lg in [8usize, 10, 12, 14, 16] {
        for w in [1usize, 2, 8, 32, 64] {
            let mat = random_matrix(lg, w, &mut rng);
            let got = gpu.dft_batch(mat.clone());
            let want = cpu.dft_batch(mat);
            assert_eq!(
                got.values, want.values,
                "dft_batch mismatch at 2^{lg} x {w}"
            );
        }
    }
}

#[test]
fn idft_batch_matches_cpu() {
    if !zkie_cuda::cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let gpu = CudaDft::new();
    assert!(gpu.is_cuda(), "expected a GPU-backed CudaDft");
    let cpu = Radix2DFTSmallBatch::new(1 << 18);

    let mut rng = SmallRng::seed_from_u64(0xdef456);
    for lg in [8usize, 10, 12, 14] {
        for w in [1usize, 8, 32] {
            let mat = random_matrix(lg, w, &mut rng);
            let got = gpu.idft_batch(mat.clone());
            let want = cpu.idft_batch(mat);
            assert_eq!(
                got.values, want.values,
                "idft_batch mismatch at 2^{lg} x {w}"
            );
        }
    }
}
