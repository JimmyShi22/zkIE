//! CUDA-backed `Mmcs<Goldilocks>` with GPU Poseidon2 Merkle commitments.
//!
//! Mirrors the exact configuration of the 200M proof's CPU MMCS:
//! `MerkleTreeMmcs<PackedGoldilocksAVX2, PackedGoldilocksAVX2, PaddingFreeSponge<
//! Poseidon2Goldilocks<16>, 16, 8, 8>, TruncatedPermutation<..., 2, 8, 16>, 2, 8>`
//! over the base field. `commit` builds the tree's digest layers on the GPU
//! (leaf hashing in 8-row SIMD batches plus binary compression), then wraps
//! them in the vendored `MerkleTree::from_parts`, so openings and
//! verification are the upstream p3-merkle-tree CPU code, bit-identical to
//! the CPU-built tree.
//!
//! The round constants are the same ones the CPU permutation uses: derived
//! from `SmallRng::seed_from_u64(1)` with p3's own public API and uploaded
//! to device `__constant__` memory once at construction.
//!
//! Falls back to the CPU `MerkleTreeMmcs` whenever CUDA is unavailable, a
//! kernel errors, or a multi-matrix batch is committed (the GPU path handles
//! the single-matrix case the WHIR prover uses everywhere).

use std::sync::{Arc, Mutex};

use p3_commit::Mmcs;
use p3_field::{Field, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_merkle_tree::{MerkleCap, MerkleTree, MerkleTreeMmcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

#[cfg(feature = "cuda")]
use crate::buffer::SharedBuffer;
#[cfg(feature = "cuda")]
use crate::ffi::{
    cudaDeviceSynchronize, cudaMemcpy, cudaMemset, zkie_p2_compress_batch,
    zkie_p2_compress_scalar, zkie_p2_leaf_batch, zkie_p2_leaf_scalar,
    zkie_p2_upload_constants, CUDART_OK, MEMCPY_DEVICE_TO_HOST, MEMCPY_HOST_TO_DEVICE,
};

type F = Goldilocks;
pub type PackedF = <F as Field>::Packing;
pub type Perm = Poseidon2Goldilocks<16>;
pub type MerkleHash = PaddingFreeSponge<Perm, 16, 8, 8>;
pub type MerkleCompress = TruncatedPermutation<Perm, 2, 8, 16>;
pub const MERKLE_ARITY: usize = 2;
pub const DIGEST_ELEMS: usize = 8;

pub type CpuMerkleTreeMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, MERKLE_ARITY, DIGEST_ELEMS>;

/// Round constants derived exactly like `Perm::new_from_rng_128(&mut
/// SmallRng::seed_from_u64(1))` (8 external + 22 internal rounds), flattened
/// to u64 for the device.
#[cfg(feature = "cuda")]
pub fn derive_poseidon2_constants() -> ([u64; 4 * 16], [u64; 4 * 16], [u64; 22], u64) {
    use p3_goldilocks::MATRIX_DIAG_16_GOLDILOCKS;
    use p3_poseidon2::ExternalLayerConstants;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand::RngExt;

    let mut rng = SmallRng::seed_from_u64(1);
    let external = ExternalLayerConstants::<Goldilocks, 16>::new_from_rng(8, &mut rng);
    let internal: Vec<Goldilocks> = rng.sample_iter(StandardUniform).take(22).collect();

    let mut rc_init = [0u64; 4 * 16];
    for (r, round) in external.get_initial_constants().iter().enumerate() {
        for (i, v) in round.iter().enumerate() {
            rc_init[r * 16 + i] = v.as_canonical_u64();
        }
    }
    let mut rc_final = [0u64; 4 * 16];
    for (r, round) in external.get_terminal_constants().iter().enumerate() {
        for (i, v) in round.iter().enumerate() {
            rc_final[r * 16 + i] = v.as_canonical_u64();
        }
    }
    let mut rc_internal = [0u64; 22];
    for (i, v) in internal.iter().enumerate() {
        rc_internal[i] = v.as_canonical_u64();
    }
    let diag15 = MATRIX_DIAG_16_GOLDILOCKS[15].as_canonical_u64();

    (rc_init, rc_final, rc_internal, diag15)
}

#[cfg(feature = "cuda")]
struct GpuMerkle {
    arena: SharedBuffer,
}

#[cfg(feature = "cuda")]
impl GpuMerkle {
    fn try_init() -> Option<Self> {
        let (rc_init, rc_final, rc_internal, diag15) = derive_poseidon2_constants();
        let rc = unsafe {
            zkie_p2_upload_constants(
                rc_init.as_ptr(),
                rc_final.as_ptr(),
                rc_internal.as_ptr(),
                diag15,
            )
        };
        let rc = if rc == CUDART_OK { unsafe { cudaDeviceSynchronize() } } else { rc };
        if rc != CUDART_OK {
            return None;
        }
        Some(Self {
            arena: SharedBuffer::new(Mutex::new(crate::buffer::CudaBuffer::new(1 << 20)?)),
        })
    }
}

/// `padded_len` from p3-merkle-tree (leaf layer and every compression layer
/// are padded to an even length; 0/1 stay as-is).
fn padded_len(raw_len: usize, n: usize) -> usize {
    if raw_len <= 1 {
        raw_len
    } else if raw_len >= n {
        raw_len.div_ceil(n) * n
    } else {
        n
    }
}

/// CUDA-backed MMCS with the 200M proof's exact hash configuration.
#[derive(Clone)]
pub struct CudaMerkleTreeMmcs {
    fallback: CpuMerkleTreeMmcs,
    #[cfg(feature = "cuda")]
    gpu: Option<Arc<GpuMerkle>>,
}

impl CudaMerkleTreeMmcs {
    /// Same constructor shape as `MerkleTreeMmcs::new`.
    pub fn new(hash: MerkleHash, compress: MerkleCompress, cap_height: usize) -> Self {
        let fallback = CpuMerkleTreeMmcs::new(hash, compress, cap_height);
        #[cfg(feature = "cuda")]
        {
            let gpu = GpuMerkle::try_init().map(Arc::new);
            Self { fallback, gpu }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self { fallback }
        }
    }

    pub fn cap_height(&self) -> usize {
        self.fallback.cap_height()
    }

    /// Whether this instance commits on the GPU.
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaMerkleTreeMmcs {
    /// GPU path for a single matrix: hashes the leaf layer and every
    /// compression layer on the device, then assembles the digest layers
    /// into the vendored `MerkleTree` via `from_parts`.
    fn commit_gpu<M: Matrix<F>>(
        &self,
        gpu: &GpuMerkle,
        m: M,
    ) -> Result<(MerkleCap<F, [F; DIGEST_ELEMS]>, MerkleTree<F, F, M, MERKLE_ARITY, DIGEST_ELEMS>), M> {
        let w = m.width();
        if w == 0 {
            return Err(m);
        }
        // Row-major copy of the base-field values (one u64 per element),
        // obtained through the generic Matrix interface so any matrix type
        // p3-whir hands us works.
        let values: Vec<F> = m.rows().flatten().collect();
        let h = values.len() / w;
        if h == 0 {
            return Err(m);
        }
        let vals: Vec<u64> = values.iter().map(|f| f.as_canonical_u64()).collect();

        // Layer schedule for a single-matrix binary tree.
        let mut layer_lens = vec![padded_len(h, MERKLE_ARITY)];
        let mut arity_schedule = Vec::new();
        while *layer_lens.last().unwrap() > 1 {
            let raw_next = layer_lens.last().unwrap() / MERKLE_ARITY;
            layer_lens.push(padded_len(raw_next, MERKLE_ARITY));
            arity_schedule.push(MERKLE_ARITY);
        }

        let total_u64: usize = h * w + layer_lens.iter().sum::<usize>() * DIGEST_ELEMS;
        let mut guard = match gpu.arena.lock() {
            Ok(g) => g,
            Err(_) => return Err(m),
        };
        if guard.grow(total_u64).is_none() {
            return Err(m);
        }
        let d_mat = guard.ptr();
        let mut d_layers = Vec::with_capacity(layer_lens.len());
        let mut offset = h * w;
        for &len in &layer_lens {
            d_layers.push(unsafe { guard.ptr().add(offset) });
            offset += len * DIGEST_ELEMS;
        }

        let rc = unsafe { cudaMemcpy(d_mat, vals.as_ptr(), h * w * 8, MEMCPY_HOST_TO_DEVICE) };
        if rc != CUDART_OK {
            return Err(m);
        }
        // Padded layer tails must be zero digests (upstream initializes the
        // whole padded layer with the default digest).
        let rc = unsafe {
            cudaMemset(d_layers[0], 0, layer_lens[0] * DIGEST_ELEMS * 8)
        };
        if rc != CUDART_OK {
            return Err(m);
        }

        let batches = h / 8;
        let rc = unsafe { zkie_p2_leaf_batch(d_mat, d_layers[0], h as u32, w as u32) };
        if rc != CUDART_OK {
            return Err(m);
        }
        let rc = unsafe {
            zkie_p2_leaf_scalar(
                d_mat,
                d_layers[0],
                (batches * 8) as u32,
                h as u32,
                w as u32,
            )
        };
        if rc != CUDART_OK {
            return Err(m);
        }

        for i in 0..layer_lens.len() - 1 {
            let next_len_raw = layer_lens[i] / MERKLE_ARITY;
            let rc = unsafe {
                cudaMemset(d_layers[i + 1], 0, layer_lens[i + 1] * DIGEST_ELEMS * 8)
            };
            if rc != CUDART_OK {
                return Err(m);
            }
            let rc = unsafe {
                zkie_p2_compress_batch(
                    d_layers[i],
                    d_layers[i + 1],
                    next_len_raw as u32,
                    MERKLE_ARITY as u32,
                )
            };
            if rc != CUDART_OK {
                return Err(m);
            }
            let rc = unsafe {
                zkie_p2_compress_scalar(
                    d_layers[i],
                    d_layers[i + 1],
                    (next_len_raw / 8 * 8) as u32,
                    next_len_raw as u32,
                    MERKLE_ARITY as u32,
                )
            };
            if rc != CUDART_OK {
                return Err(m);
            }
        }

        let rc = unsafe { cudaDeviceSynchronize() };
        if rc != CUDART_OK {
            return Err(m);
        }

        let mut digest_layers: Vec<Vec<[F; DIGEST_ELEMS]>> = Vec::with_capacity(layer_lens.len());
        for (i, &len) in layer_lens.iter().enumerate() {
            let mut layer = vec![0u64; len * DIGEST_ELEMS];
            let rc = unsafe {
                cudaMemcpy(
                    layer.as_mut_ptr(),
                    d_layers[i],
                    len * DIGEST_ELEMS * 8,
                    MEMCPY_DEVICE_TO_HOST,
                )
            };
            if rc != CUDART_OK {
                return Err(m);
            }
            let digests: Vec<[F; DIGEST_ELEMS]> = layer
                .chunks_exact(DIGEST_ELEMS)
                .map(|c| {
                    std::array::from_fn(|j| F::new(c[j]))
                })
                .collect();
            digest_layers.push(digests);
        }
        drop(guard);

        let tree = MerkleTree::from_parts(vec![m], digest_layers, arity_schedule);
        let cap = tree.cap(self.cap_height().min(tree.num_layers().saturating_sub(1)));
        Ok((cap, tree))
    }
}

impl Mmcs<F> for CudaMerkleTreeMmcs {
    type ProverData<M> = MerkleTree<F, F, M, MERKLE_ARITY, DIGEST_ELEMS>;
    type Commitment = MerkleCap<F, [F; DIGEST_ELEMS]>;
    type Proof = Vec<[F; DIGEST_ELEMS]>;
    type MultiProof = p3_merkle_tree::PrunedMerklePaths<F, DIGEST_ELEMS>;
    type Error = p3_merkle_tree::MerkleTreeError;

    fn commit<M: Matrix<F>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        #[cfg(feature = "cuda")]
        {
            if inputs.len() == 1 {
                let m = inputs.into_iter().next().unwrap();
                if let Some(gpu) = &self.gpu {
                    match self.commit_gpu(gpu, m) {
                        Ok(out) => return out,
                        Err(m) => return self.fallback.commit(vec![m]),
                    }
                }
                return self.fallback.commit(vec![m]);
            }
            self.fallback.commit(inputs)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.fallback.commit(inputs)
        }
    }

    fn open_batch<M: Matrix<F>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> p3_commit::BatchOpening<F, Self> {
        // Delegate to the upstream CPU implementation (it only reads the
        // tree's digest layers and leaves, which are identical by
        // construction), then rewrap into our Mmcs type.
        let opening = self.fallback.open_batch(index, prover_data);
        p3_commit::BatchOpening::new(opening.opened_values, opening.opening_proof)
    }

    fn get_matrices<'a, M: Matrix<F>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        self.fallback.get_matrices(prover_data)
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: p3_commit::BatchOpeningRef<'_, F, Self>,
    ) -> Result<(), Self::Error> {
        let fallback_opening =
            p3_commit::BatchOpeningRef::new(batch_opening.opened_values, batch_opening.opening_proof);
        self.fallback
            .verify_batch(commit, dimensions, index, fallback_opening)
    }

    fn open_multi_batch<M: Matrix<F>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> (Vec<Vec<Vec<F>>>, Self::MultiProof) {
        self.fallback.open_multi_batch(indices, prover_data)
    }

    fn verify_multi_batch<R: AsRef<[F]> + PartialEq>(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        self.fallback
            .verify_multi_batch(commit, dimensions, indices, opened_values, proof)
    }
}
