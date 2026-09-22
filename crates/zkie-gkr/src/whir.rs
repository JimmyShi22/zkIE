//! Minimal WHIR multilinear PCS wrapper over Goldilocks.
//!
//! Exposes a clean commit -> prescribed-point open -> verify cycle for a single
//! flat multilinear extension (a `Vec<Goldilocks>` of length `2^d`). The opening
//! point is caller-chosen (base-field coordinates) and is embedded into the
//! degree-2 extension field internally, which is exactly the binding a GKR
//! matmul verifier needs for the two sum-check evaluations it would otherwise
//! recompute in `O(k)`.

use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Layout as _, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, PointSchedule, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, DEFAULT_MAX_POW,
};
use p3_whir::pcs::prover::WhirProver;
use rand::rngs::SmallRng;
use rand::SeedableRng;

type F = Goldilocks;
type EF = BinomialExtensionField<F, 2>;
type Perm = Poseidon2Goldilocks<16>;
type MerkleHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MerkleCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyLayout = SuffixProver<F, EF>;

type CpuMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type CpuDft = Radix2DFTSmallBatch<F>;

// The DFT and Merkle engines are swappable at the type level. Without the
// `cuda` feature these are the pure-CPU p3 implementations. With it, they
// are runtime-dispatched enums: measured on the 64-core dev box the GPU
// path is launch/transfer-bound and *slower* than 64-thread rayon for the
// 200M proof's many small commitments (548.7s vs 402.5s end-to-end), so the
// CPU engines stay the default and the GPU path is opt-in via
// `ZKIE_CUDA=1` (e.g. for thread-constrained deployments, where a 2^22
// commit measures 25.9x faster than single-threaded CPU).
#[cfg(feature = "cuda")]
mod backend {
    use super::*;
    use p3_commit::Mmcs as _;
    use p3_dft::TwoAdicSubgroupDft;
    use p3_matrix::Matrix as _;

    /// Whether the GPU engines should be used for this process.
    pub fn use_cuda() -> bool {
        matches!(std::env::var("ZKIE_CUDA"), Ok(v) if v == "1" || v == "true")
    }

    pub enum DftBackend {
        Cpu(CpuDft),
        #[cfg(feature = "cuda")]
        Cuda(zkie_cuda::dft::CudaDft),
    }

    impl Clone for DftBackend {
        fn clone(&self) -> Self {
            match self {
                Self::Cpu(d) => Self::Cpu(d.clone()),
                #[cfg(feature = "cuda")]
                Self::Cuda(d) => Self::Cuda(d.clone()),
            }
        }
    }

    impl Default for DftBackend {
        fn default() -> Self {
            Self::Cpu(CpuDft::default())
        }
    }

    impl TwoAdicSubgroupDft<F> for DftBackend {
        type Evaluations = RowMajorMatrix<F>;

        fn dft_batch(&self, mat: RowMajorMatrix<F>) -> Self::Evaluations {
            match self {
                Self::Cpu(d) => d.dft_batch(mat),
                #[cfg(feature = "cuda")]
                Self::Cuda(d) => d.dft_batch(mat),
            }
        }
    }

    pub enum MmcsBackend {
        Cpu(CpuMmcs),
        #[cfg(feature = "cuda")]
        Cuda(zkie_cuda::merkle::CudaMerkleTreeMmcs),
    }

    impl Clone for MmcsBackend {
        fn clone(&self) -> Self {
            match self {
                Self::Cpu(m) => Self::Cpu(m.clone()),
                #[cfg(feature = "cuda")]
                Self::Cuda(m) => Self::Cuda(m.clone()),
            }
        }
    }

    impl p3_commit::Mmcs<F> for MmcsBackend {
        type ProverData<M> = p3_merkle_tree::MerkleTree<F, F, M, 2, 8>;
        type Commitment = p3_symmetric::MerkleCap<F, [F; 8]>;
        type Proof = Vec<[F; 8]>;
        type MultiProof = p3_merkle_tree::PrunedMerklePaths<F, 8>;
        type Error = p3_merkle_tree::MerkleTreeError;

        fn commit<M: p3_matrix::Matrix<F>>(
            &self,
            inputs: Vec<M>,
        ) -> (Self::Commitment, Self::ProverData<M>) {
            match self {
                Self::Cpu(m) => m.commit(inputs),
                #[cfg(feature = "cuda")]
                Self::Cuda(m) => m.commit(inputs),
            }
        }

        fn open_batch<M: p3_matrix::Matrix<F>>(
            &self,
            index: usize,
            prover_data: &Self::ProverData<M>,
        ) -> p3_commit::BatchOpening<F, Self> {
            match self {
                Self::Cpu(m) => {
                    let o = m.open_batch(index, prover_data);
                    p3_commit::BatchOpening::new(o.opened_values, o.opening_proof)
                }
                #[cfg(feature = "cuda")]
                Self::Cuda(m) => {
                    let o = m.open_batch(index, prover_data);
                    p3_commit::BatchOpening::new(o.opened_values, o.opening_proof)
                }
            }
        }

        fn get_matrices<'a, M: p3_matrix::Matrix<F>>(
            &self,
            prover_data: &'a Self::ProverData<M>,
        ) -> Vec<&'a M> {
            match self {
                Self::Cpu(m) => m.get_matrices(prover_data),
                #[cfg(feature = "cuda")]
                Self::Cuda(m) => m.get_matrices(prover_data),
            }
        }

        fn verify_batch(
            &self,
            commit: &Self::Commitment,
            dimensions: &[p3_matrix::Dimensions],
            index: usize,
            batch_opening: p3_commit::BatchOpeningRef<'_, F, Self>,
        ) -> Result<(), Self::Error> {
            match self {
                Self::Cpu(m) => {
                    let o = p3_commit::BatchOpeningRef::<'_, F, CpuMmcs>::new(
                        batch_opening.opened_values,
                        batch_opening.opening_proof,
                    );
                    m.verify_batch(commit, dimensions, index, o)
                }
                #[cfg(feature = "cuda")]
                Self::Cuda(m) => {
                    let o = p3_commit::BatchOpeningRef::<
                        '_,
                        F,
                        zkie_cuda::merkle::CudaMerkleTreeMmcs,
                    >::new(batch_opening.opened_values, batch_opening.opening_proof);
                    m.verify_batch(commit, dimensions, index, o)
                }
            }
        }

        fn verify_multi_batch<R: AsRef<[F]> + PartialEq>(
            &self,
            commit: &Self::Commitment,
            dimensions: &[p3_matrix::Dimensions],
            indices: &[usize],
            opened_values: &[Vec<R>],
            proof: &Self::MultiProof,
        ) -> Result<(), Self::Error> {
            match self {
                Self::Cpu(m) => m.verify_multi_batch(
                    commit, dimensions, indices, opened_values, proof,
                ),
                #[cfg(feature = "cuda")]
                Self::Cuda(m) => m.verify_multi_batch(
                    commit, dimensions, indices, opened_values, proof,
                ),
            }
        }

        fn open_multi_batch<M: p3_matrix::Matrix<F>>(
            &self,
            indices: &[usize],
            prover_data: &Self::ProverData<M>,
        ) -> (Vec<Vec<Vec<F>>>, Self::MultiProof) {
            match self {
                Self::Cpu(m) => m.open_multi_batch(indices, prover_data),
                #[cfg(feature = "cuda")]
                Self::Cuda(m) => m.open_multi_batch(indices, prover_data),
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
type MyMmcs = CpuMmcs;
#[cfg(feature = "cuda")]
type MyMmcs = backend::MmcsBackend;
#[cfg(not(feature = "cuda"))]
type MyDft = CpuDft;
#[cfg(feature = "cuda")]
type MyDft = backend::DftBackend;
type MyPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, MyLayout>;

pub type Commitment = <MyPcs as MultilinearPcs<EF, MyChallenger>>::Commitment;
pub type ProverData = <MyPcs as MultilinearPcs<EF, MyChallenger>>::ProverData;
pub type Proof = <MyPcs as MultilinearPcs<EF, MyChallenger>>::Proof;
pub use p3_sumcheck::OpeningProtocol;

/// A WHIR PCS over Goldilocks configured for a single `2^num_variables` MLE.
pub struct Whir {
    pcs: MyPcs,
    perm: Perm,
    folding_factor: FoldingFactor,
    /// (commits, total seconds) spent inside `commit`, for performance telemetry.
    commit_stats: std::cell::Cell<(u64, f64)>,
}

impl Whir {
    /// Build a WHIR instance sized for a single `2^num_variables` multilinear
    /// polynomial. Security parameters are PoC-grade (90-bit, default PoW); the
    /// proof stays small enough that a `2^10` commitment runs in a couple seconds.
    pub fn new(num_variables: usize) -> Self {
        Self::with_params(num_variables, 90, DEFAULT_MAX_POW)
    }

    /// Fast, low-security instance for tests and local iteration. Do not use for
    /// anything that needs real soundness.
    pub fn new_testing(num_variables: usize) -> Self {
        Self::with_params(num_variables, 32, 10)
    }

    fn with_params(num_variables: usize, security_level: usize, pow_bits: usize) -> Self {
        let folding_factor = FoldingFactor::Constant(5);
        let (num_rounds, _) = folding_factor
            .compute_number_of_rounds(num_variables)
            .expect("valid folding schedule");
        let mut round_log_inv_rates = Vec::with_capacity(num_rounds);
        let mut rate = 1;
        for round in 0..num_rounds {
            rate += folding_factor.at_round(round) - 1;
            round_log_inv_rates.push(rate);
        }
        let params = ProtocolParameters {
            security_level,
            pow_bits,
            folding_factor: folding_factor.clone(),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            round_log_inv_rates,
        };

        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        #[cfg(not(feature = "cuda"))]
        let mmcs = MyMmcs::new(
            MerkleHash::new(perm.clone()),
            MerkleCompress::new(perm.clone()),
            0,
        );
        #[cfg(feature = "cuda")]
        let mmcs = {
            let hash = MerkleHash::new(perm.clone());
            let compress = MerkleCompress::new(perm.clone());
            if backend::use_cuda() {
                MyMmcs::Cuda(zkie_cuda::merkle::CudaMerkleTreeMmcs::new(hash, compress, 0))
            } else {
                MyMmcs::Cpu(CpuMmcs::new(hash, compress, 0))
            }
        };
        let config = WhirConfig::<EF, F, MyChallenger>::new(num_variables, params).unwrap();
        #[cfg(not(feature = "cuda"))]
        let dft = MyDft::new(1 << config.max_fft_size());
        #[cfg(feature = "cuda")]
        let dft = {
            if backend::use_cuda() {
                MyDft::Cuda(zkie_cuda::dft::CudaDft::new())
            } else {
                MyDft::Cpu(CpuDft::new(1 << config.max_fft_size()))
            }
        };
        let pcs = MyPcs::new(config, dft, mmcs);

        Whir {
            pcs,
            perm,
            folding_factor,
            commit_stats: std::cell::Cell::new((0, 0.0)),
        }
    }

    /// Number of `commit` calls and total wall-clock seconds spent inside them.
    pub fn commit_stats(&self) -> (u64, f64) {
        self.commit_stats.get()
    }

    fn fresh_challenger(&self) -> MyChallenger {
        let mut challenger = MyChallenger::new(self.perm.clone());
        let mut domain_separator = DomainSeparator::new(vec![]);
        self.pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut challenger);
        challenger
    }

    /// Commit to a flat MLE. Returns the commitment, prover data, and the public
    /// opening protocol (single table, single column, one point) used for open/verify.
    pub fn commit(&self, evals: &[Goldilocks]) -> (Commitment, ProverData, OpeningProtocol) {
        let t0 = std::time::Instant::now();
        let num_vars = evals.len().trailing_zeros() as usize;
        assert_eq!(evals.len(), 1 << num_vars, "MLE length must be a power of two");

        // One polynomial (one row) whose `2^num_vars` evaluations form the row.
        let table = Table::new(RowMajorMatrix::new(evals.to_vec(), 1 << num_vars));
        let folding = self.folding_factor.at_round(0);
        let witness = MyLayout::new_witness(vec![table], folding);

        let point_schedule: PointSchedule =
            std::iter::once(OpeningBatch::new(vec![0], Vec::new())).collect();
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_vars, 1),
            point_schedule,
        )])
        .pad_to_min_num_variables(folding);

        let (commitment, prover_data) =
            <MyPcs as MultilinearPcs<EF, MyChallenger>>::commit(
                &self.pcs,
                witness,
                &mut self.fresh_challenger(),
            );
        let (n, secs) = self.commit_stats.get();
        self.commit_stats
            .set((n + 1, secs + t0.elapsed().as_secs_f64()));
        (commitment, prover_data, protocol)
    }

    /// Open the committed MLE at `point` (base-field coordinates) and return the
    /// opening proof together with the claimed evaluation (in the extension field).
    pub fn open(
        &self,
        prover_data: ProverData,
        protocol: &OpeningProtocol,
        point: &[Goldilocks],
    ) -> (Proof, Goldilocks) {
        let ef_point = to_ef_point(point);
        let proof = self.pcs.open_at(
            prover_data,
            protocol,
            std::slice::from_ref(&ef_point),
            &mut self.fresh_challenger(),
        );
        let opened = proof.evals[0].current()[0];
        (proof, opened.as_base().expect("base-field MLE opens to a base element"))
    }

    /// Verify the opening proof at `point` and return the opened evaluation.
    pub fn verify(
        &self,
        commitment: &Commitment,
        proof: &Proof,
        protocol: &OpeningProtocol,
        point: &[Goldilocks],
    ) -> Result<Goldilocks, <MyPcs as MultilinearPcs<EF, MyChallenger>>::Error> {
        let ef_point = to_ef_point(point);
        let evals = self.pcs.verify_at(
            commitment,
            proof,
            protocol,
            std::slice::from_ref(&ef_point),
            &mut self.fresh_challenger(),
        )?;
        Ok(evals[0].current()[0].as_base().expect("base-field MLE opens to a base element"))
    }
}

/// Embed base-field coordinates into the degree-2 extension field.
///
/// The crate's hand-rolled `mle` uses "coordinate 0 = LSB of the flattened
/// index", while Plonky3's `Poly`/`Point` use "coordinate 0 = MSB" (big-endian).
/// Reversing here reconciles the two orderings at the commitment boundary.
fn to_ef_point(point: &[Goldilocks]) -> Point<EF> {
    Point::new(point.iter().rev().map(|&c| EF::from(c)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Goldilocks, XorShift64};
    use crate::mle;

    #[test]
    fn whir_prescribed_open_matches_mle_eval() {
        let mut rng = XorShift64::new(0x5eed);
        let d = 6;
        let evals: Vec<Goldilocks> = (0..(1 << d)).map(|_| rng.field()).collect();
        let whir = Whir::new(d);

        let point: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
        let (commitment, prover_data, protocol) = whir.commit(&evals);
        let (proof, opened) = whir.open(prover_data, &protocol, &point);
        let verified = whir.verify(&commitment, &proof, &protocol, &point).unwrap();

        assert_eq!(opened, verified);
        assert_eq!(verified, mle::eval(&evals, &point));
    }
}
