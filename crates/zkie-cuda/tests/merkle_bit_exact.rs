//! Bit-exactness of `CudaMerkleTreeMmcs::commit` against the CPU
//! `MerkleTreeMmcs` with the 200M proof's exact configuration (Poseidon2-16,
//! PaddingFreeSponge leaf, TruncatedPermutation compression, arity 2, digest
//! 8): identical Merkle caps for random matrices of the shapes WHIR commits
//! (1-row witness padding aside, heights and widths spanning the SIMD batch
//! boundaries and the scalar tail paths).
//!
//! Requires the `cuda` feature and a usable GPU; otherwise skipped.

#![cfg(feature = "cuda")]

use p3_commit::Mmcs;
use p3_field::Field;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::RngExt;
use rand::SeedableRng;
use zkie_cuda::merkle::{
    CudaMerkleTreeMmcs, MerkleCompress, MerkleHash, Perm, DIGEST_ELEMS,
};

fn build_mmcs() -> (CudaMerkleTreeMmcs, zkie_cuda::merkle::CpuMerkleTreeMmcs) {
    use rand::RngExt;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MerkleHash::new(perm.clone());
    let compress = MerkleCompress::new(perm.clone());
    (
        CudaMerkleTreeMmcs::new(hash.clone(), compress.clone(), 0),
        zkie_cuda::merkle::CpuMerkleTreeMmcs::new(hash, compress, 0),
    )
}

#[test]
fn commit_root_matches_cpu() {
    if !zkie_cuda::cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let (gpu, cpu) = build_mmcs();
    assert!(gpu.is_cuda(), "expected a GPU-backed CudaMerkleTreeMmcs");

    let mut rng = SmallRng::seed_from_u64(0xfeedface);
    for h in [1usize, 2, 3, 5, 7, 8, 9, 16, 33, 100, 256, 1024] {
        for w in [1usize, 8, 16, 32, 64, 100] {
            let values: Vec<Goldilocks> = (&mut rng).sample_iter(StandardUniform).take(h * w).collect();
            let mat = RowMajorMatrix::new(values, w);

            let (gpu_cap, _) = gpu.commit_matrix(mat.clone());
            let (cpu_cap, _) = cpu.commit_matrix(mat);
            assert_eq!(gpu_cap, cpu_cap, "commit mismatch at {h} x {w}");
        }
    }
}

#[test]
fn open_verify_roundtrip_on_gpu_tree() {
    if !zkie_cuda::cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let (gpu, _cpu) = build_mmcs();
    assert!(gpu.is_cuda(), "expected a GPU-backed CudaMerkleTreeMmcs");

    let mut rng = SmallRng::seed_from_u64(0x77);
    let h = 512usize;
    let w = 32usize;
    let values: Vec<Goldilocks> = (&mut rng).sample_iter(StandardUniform).take(h * w).collect();
    let mat = RowMajorMatrix::new(values, w);
    let (cap, prover_data) = gpu.commit_matrix(mat);

    // Opening a row on the GPU-built tree must verify against the cap
    // through the upstream CPU verification code.
    let index = 42usize;
    let opening = gpu.open_batch(index, &prover_data);
    let dims = vec![p3_matrix::Dimensions {
        height: h,
        width: w,
    }];
    gpu.verify_batch(&cap, &dims, index, (&opening).into())
        .expect("GPU-built tree opening must verify");

    // And the opened row must equal the committed matrix's row.
    let matrices = gpu.get_matrices(&prover_data);
    let m = matrices[0];
    let expected: Vec<Goldilocks> = m.row(index).unwrap().into_iter().collect();
    assert_eq!(opening.opened_values[0], expected);
}
