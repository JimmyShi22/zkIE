//! Minimal WHIR multilinear PCS wrapper over Goldilocks.
//!
//! Exposes a clean commit -> prescribed-point open -> verify cycle for a single
//! flat multilinear extension (a `Vec<Goldilocks>` of length `2^d`). The opening
//! point is caller-chosen (base-field coordinates) and is embedded into the
//! degree-2 extension field internally, which is exactly the binding a GKR
//! matmul verifier needs for the two sum-check evaluations it would otherwise
//! recompute in `O(k)`.

/// Global count of FRI opening proofs across all Whir instances (batch Whirs are
/// transient, so per-instance stats do not accumulate). Used for telemetry.
static GLOBAL_OPEN_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Total number of FRI opening proofs produced so far in this process.
pub fn global_open_count() -> u64 {
    GLOBAL_OPEN_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}


use p3_challenger::{CanObserve, DuplexChallenger};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
pub use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Layout as _, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, PointSchedule, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
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

    /// Resolve a per-engine override (`ZKIE_CUDA_DFT` / `ZKIE_CUDA_MMCS`),
    /// falling back to the global `ZKIE_CUDA` switch. This lets autotuning
    /// pin the DFT and Merkle engines independently of each other.
    fn pick_backend(name: &str, fallback: bool) -> bool {
        match std::env::var(name).ok().as_deref() {
            Some("1") | Some("true") | Some("cuda") => true,
            Some("0") | Some("false") | Some("cpu") => false,
            _ => fallback,
        }
    }

    pub fn use_cuda_dft() -> bool {
        pick_backend("ZKIE_CUDA_DFT", use_cuda())
    }

    pub fn use_cuda_mmcs() -> bool {
        pick_backend("ZKIE_CUDA_MMCS", use_cuda())
    }

    pub enum DftBackend {
        Cpu(CpuDft),
        #[cfg(feature = "cuda")]
        Cuda(crate::pcs::dft_cuda::CudaDft),
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
        Cuda(crate::pcs::merkle_cuda::CudaMerkleTreeMmcs),
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
                        crate::pcs::merkle_cuda::CudaMerkleTreeMmcs,
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

/// Verification failure for batch openings: either malformed public metadata
/// (rejected before any PCS verification — upstream `verify_at` *asserts* on
/// these) or an underlying WHIR PCS error.
#[derive(Debug)]
pub enum WhirVerifyError {
    /// Public metadata is malformed (protocol, table count, index, or point
    /// dimension); see the message.
    Malformed(&'static str),
    /// Underlying WHIR verification failure.
    Pcs(<MyPcs as MultilinearPcs<EF, MyChallenger>>::Error),
}

/// A WHIR PCS over Goldilocks configured for a single `2^num_variables` MLE.
pub struct Whir {
    pcs: MyPcs,
    perm: Perm,
    folding_factor: FoldingFactor,
    /// Security parameters this instance was configured with
    /// (`security_level` bits, `pow_bits` grinding budget). Batch commits must
    /// inherit these instead of silently falling back to test parameters.
    security_level: usize,
    pow_bits: usize,
    /// (commits, total seconds) spent inside `commit`, for performance telemetry.
    commit_stats: std::cell::Cell<(u64, f64)>,
    open_stats: std::cell::Cell<(u64, f64)>,
    verify_stats: std::cell::Cell<(u64, f64)>,
}

impl Whir {
    /// Build a WHIR instance sized for a single `2^num_variables` multilinear
    /// polynomial. Security parameters are PoC-grade (90-bit, default PoW); the
    /// proof stays small enough that a `2^10` commitment runs in a couple seconds.
    pub fn new(num_variables: usize) -> Self {
        Self::with_params(num_variables, 90, 32)
    }

    /// Fast, low-security instance for tests and local iteration. Do not use for
    /// anything that needs real soundness.
    pub fn new_testing(num_variables: usize) -> Self {
        Self::with_params(num_variables, 32, 10)
    }

    /// Explicit configuration with a caller-chosen security target and PoW
    /// budget (e.g. `new_target(n, 90, 0)` for the PCS-target-90 low-PoW
    /// prototype configuration). Genuinely fallible: returns `None` (never
    /// panics) when the folding schedule does not fit the variable count
    /// (tiny arity), the derived PoW schedule exceeds the budget, or the FFT
    /// domain would be absurdly large. No fallback or automatic downgrade.
    pub fn new_target(num_variables: usize, security_level: usize, pow_budget: usize) -> Option<Self> {
        Self::try_with_params(num_variables, security_level, pow_budget)
    }

    /// Whether the derived PoW schedule fits inside the configured budget
    /// (`WhirConfig::check_pow_bits`). False means the requested security
    /// target is not met by this instance.
    pub fn pow_bits_ok(&self) -> bool {
        self.pcs.config.check_pow_bits()
    }

    /// The derived maximum PoW bits across the schedule (capped by the budget
    /// when the construction succeeded).
    pub fn max_pow_bits(&self) -> usize {
        self.pcs.config.max_pow_bits()
    }

    /// Reject FFT domains above this many variables: `MyDft::new` would need
    /// `2^max_fft_size` elements, and anything beyond 2^32 is far outside
    /// anything this crate can run (and `1 << 33` no longer fits a 32-bit
    /// allocation graph). Kept well below `usize` shift limits.
    const MAX_FFT_VARIABLES: usize = 32;

    pub(crate) fn with_params(num_variables: usize, security_level: usize, pow_bits: usize) -> Self {
        // Legacy constructor (kept for `new` / `new_testing` and batch paths):
        // panics on configuration errors exactly as before. New code should
        // use the fallible `new_target`.
        Self::try_with_params(num_variables, security_level, pow_bits)
            .expect("invalid WHIR configuration")
    }

    fn try_with_params(num_variables: usize, security_level: usize, pow_bits: usize) -> Option<Self> {
        let folding_factor = FoldingFactor::Constant(5);
        let (num_rounds, _) = folding_factor
            .compute_number_of_rounds(num_variables)
            .ok()?;
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
            if backend::use_cuda_mmcs() {
                MyMmcs::Cuda(crate::pcs::merkle_cuda::CudaMerkleTreeMmcs::new(hash, compress, 0))
            } else {
                MyMmcs::Cpu(CpuMmcs::new(hash, compress, 0))
            }
        };
        let config = WhirConfig::<EF, F, MyChallenger>::new(num_variables, params).ok()?;
        if config.max_fft_size() > Self::MAX_FFT_VARIABLES {
            return None;
        }
        #[cfg(not(feature = "cuda"))]
        let dft = MyDft::new(1usize.checked_shl(config.max_fft_size() as u32)?);
        #[cfg(feature = "cuda")]
        let dft = {
            if backend::use_cuda_dft() {
                MyDft::Cuda(crate::pcs::dft_cuda::CudaDft::new())
            } else {
                MyDft::Cpu(CpuDft::new(1usize.checked_shl(config.max_fft_size() as u32)?))
            }
        };
        let pcs = MyPcs::new(config, dft, mmcs);

        Some(Whir {
            pcs,
            perm,
            folding_factor,
            security_level,
            pow_bits,
            commit_stats: std::cell::Cell::new((0, 0.0)),
            open_stats: std::cell::Cell::new((0, 0.0)),
            verify_stats: std::cell::Cell::new((0, 0.0)),
        })
    }

    /// The security parameters this instance was configured with:
    /// `(security_level_bits, pow_budget_bits)`. `pow_bits` is the grinding
    /// *budget*: p3-whir derives the actual per-round/final PoW from the gap
    /// between the security level and the query coverage, capped by this budget.
    pub fn security_params(&self) -> (usize, usize) {
        (self.security_level, self.pow_bits)
    }

    /// Number of variables of the committed MLE (log2 of its domain size).
    pub fn num_variables(&self) -> usize {
        self.pcs.num_vars()
    }

    /// The canonical opening protocol for `num_points` single-point openings of
    /// a `2^num_variables` MLE. Fully determined by public data: `commit` and
    /// `commit_with_points` build exactly this protocol, so a verifier can check
    /// a transported protocol against this reconstruction.
    pub fn opening_protocol(&self, num_variables: usize, num_points: usize) -> OpeningProtocol {
        let folding = self.folding_factor.at_round(0);
        let point_schedule: PointSchedule =
            (0..num_points).map(|_| OpeningBatch::new(vec![0], Vec::new())).collect();
        OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
            point_schedule,
        )])
        .pad_to_min_num_variables(folding)
    }

    /// The canonical opening protocol for a batch of `num_tables` MLEs of arity
    /// `arity` (one single-point opening per table). Fully determined by public
    /// data: `commit_batch` builds exactly this protocol, so a verifier can
    /// check a transported batch protocol against this reconstruction.
    pub fn batch_protocol(&self, arity: usize, num_tables: usize) -> OpeningProtocol {
        let folding = self.folding_factor.at_round(0);
        let point_schedule: PointSchedule =
            std::iter::once(OpeningBatch::new(vec![0], Vec::new())).collect();
        let specs: Vec<TableSpec> = (0..num_tables)
            .map(|_| TableSpec::new(TableShape::new(arity, 1), point_schedule.clone()))
            .collect();
        OpeningProtocol::new(specs).pad_to_min_num_variables(folding)
    }

    /// The derived WHIR configuration (per-round PoW/query schedule, final
    /// queries, folding schedule). Config-only: no commitments or openings.
    pub fn config(&self) -> &WhirConfig<EF, F, MyChallenger> {
        &self.pcs.config
    }

    /// Number of `commit` calls and total wall-clock seconds spent inside them.
    pub fn commit_stats(&self) -> (u64, f64) {
        self.commit_stats.get()
    }
    pub fn open_stats(&self) -> (u64, f64) {
        self.open_stats.get()
    }
    pub fn verify_stats(&self) -> (u64, f64) {
        self.verify_stats.get()
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
        self.commit_with_points(evals, 1)
    }

    /// Commit with a prescribed multi-point opening protocol (one opening per point).
    pub fn commit_with_points(&self, evals: &[Goldilocks], num_points: usize) -> (Commitment, ProverData, OpeningProtocol) {
        let t0 = std::time::Instant::now();
        let num_vars = evals.len().trailing_zeros() as usize;
        assert_eq!(evals.len(), 1 << num_vars, "MLE length must be a power of two");

        // One polynomial (one row) whose `2^num_vars` evaluations form the row.
        let table = Table::new(RowMajorMatrix::new(evals.to_vec(), 1 << num_vars));
        let folding = self.folding_factor.at_round(0);
        let witness = MyLayout::new_witness(vec![table], folding);

        let protocol = self.opening_protocol(num_vars, num_points);

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
    /// Commit to multiple flat MLEs of the same size in a single WHIR witness.
    /// This amortizes the per-commit launch/transfer overhead, which is the
    /// dominant cost on the GPU path for many small commitments.
    pub fn commit_batch(
        &self,
        evals_batch: &[&[Goldilocks]],
    ) -> (Commitment, ProverData, OpeningProtocol, Whir) {
        let t0 = std::time::Instant::now();
        assert!(!evals_batch.is_empty(), "batch must not be empty");
        let arity = evals_batch[0].len().trailing_zeros() as usize;
        for e in evals_batch {
            assert_eq!(e.len(), 1 << arity, "all MLEs in a batch must have the same size");
        }
        let total_slots = evals_batch.len() * (1usize << arity);
        let total_num_vars = total_slots.next_power_of_two().trailing_zeros() as usize;

        // The batch inherits the caller's security parameters. Previously this
        // silently built a `new_testing` instance, silently downgrading the
        // batch commitment to 32-bit/10-PoW-bits regardless of the caller.
        let whir = Whir::with_params(total_num_vars, self.security_level, self.pow_bits);
        let tables: Vec<Table<F>> = evals_batch
            .iter()
            .map(|e| Table::new(RowMajorMatrix::new(e.to_vec(), 1 << arity)))
            .collect();
        let folding = whir.folding_factor.at_round(0);
        let witness = MyLayout::new_witness(tables, folding);

        let protocol = whir.batch_protocol(arity, evals_batch.len());

        let (commitment, prover_data) = <MyPcs as MultilinearPcs<EF, MyChallenger>>::commit(
            &whir.pcs,
            witness,
            &mut whir.fresh_challenger(),
        );
        let (n, secs) = self.commit_stats.get();
        self.commit_stats
            .set((n + evals_batch.len() as u64, secs + t0.elapsed().as_secs_f64()));
        (commitment, prover_data, protocol, whir)
    }

    pub fn open(
        &self,
        prover_data: ProverData,
        protocol: &OpeningProtocol,
        point: &[Goldilocks],
    ) -> (Proof, Goldilocks) {
        let t0 = std::time::Instant::now();
        let ef_point = to_ef_point(point);
        let proof = self.pcs.open_at(
            prover_data,
            protocol,
            std::slice::from_ref(&ef_point),
            &mut self.fresh_challenger(),
        );
        let opened = proof.evals[0].current()[0];
        let (n, secs) = self.open_stats.get();
        self.open_stats.set((n + 1, secs + t0.elapsed().as_secs_f64()));
        GLOBAL_OPEN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        let t0 = std::time::Instant::now();
        let ef_point = to_ef_point(point);
        let evals = self.pcs.verify_at(
            commitment,
            proof,
            protocol,
            std::slice::from_ref(&ef_point),
            &mut self.fresh_challenger(),
        )?;
        let (n, secs) = self.verify_stats.get();
        self.verify_stats.set((n + 1, secs + t0.elapsed().as_secs_f64()));
        Ok(evals[0].current()[0].as_base().expect("base-field MLE opens to a base element"))
    }
    /// Open the committed MLE at a FULL extension-field point: the point and
    /// the returned evaluation are genuine `EF` elements (no base embedding,
    /// no downcast). The transcript is bound to the commitment before the
    /// opening, matching `commit`'s state.
    pub fn open_ef(
        &self,
        commitment: &Commitment,
        prover_data: ProverData,
        protocol: &OpeningProtocol,
        point: &Point<EF>,
    ) -> (Proof, EF) {
        let t0 = std::time::Instant::now();
        let mut challenger = self.fresh_challenger();
        challenger.observe(commitment);
        let proof = self.pcs.open_at(
            prover_data,
            protocol,
            std::slice::from_ref(point),
            &mut challenger,
        );
        let opened = proof.evals[0].current()[0];
        let (n, secs) = self.open_stats.get();
        self.open_stats.set((n + 1, secs + t0.elapsed().as_secs_f64()));
        GLOBAL_OPEN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        (proof, opened)
    }

    /// Verify a full-extension-field opening: returns the verified evaluation
    /// as an `EF` element. Verifier-only — never regenerates the opening.
    ///
    /// Pre-validates the public metadata BEFORE calling upstream `verify_at`
    /// (which asserts on mismatches): the point must have exactly
    /// `num_variables` coordinates and the protocol must be the canonical
    /// single-opening protocol for this instance.
    pub fn verify_ef(
        &self,
        commitment: &Commitment,
        proof: &Proof,
        protocol: &OpeningProtocol,
        point: &Point<EF>,
    ) -> Result<EF, WhirVerifyError> {
        if point.as_slice().len() != self.num_variables() {
            return Err(WhirVerifyError::Malformed("point arity does not match this Whir instance"));
        }
        if *protocol != self.opening_protocol(self.num_variables(), 1) {
            return Err(WhirVerifyError::Malformed(
                "protocol is not the canonical single-opening schedule",
            ));
        }
        let t0 = std::time::Instant::now();
        let mut challenger = self.fresh_challenger();
        challenger.observe(commitment);
        let evals = self.pcs
            .verify_at(
                commitment,
                proof,
                protocol,
                std::slice::from_ref(point),
                &mut challenger,
            )
            .map_err(WhirVerifyError::Pcs)?;
        let (n, secs) = self.verify_stats.get();
        self.verify_stats.set((n + 1, secs + t0.elapsed().as_secs_f64()));
        Ok(evals[0].current()[0])
    }

    /// Open the committed MLE at multiple points in one batched opening proof.
    /// Open the committed MLE at multiple points in one batched opening proof.
    pub fn open_multi(
        &self,
        prover_data: ProverData,
        protocol: &OpeningProtocol,
        points: &[Vec<Goldilocks>],
    ) -> (Proof, Vec<Goldilocks>) {
        let t0 = std::time::Instant::now();
        let ef_points: Vec<Point<EF>> = points.iter().map(|p| to_ef_point(p)).collect();
        let proof = self.pcs.open_at(
            prover_data,
            protocol,
            &ef_points,
            &mut self.fresh_challenger(),
        );
        let opened: Vec<Goldilocks> = proof
            .evals
            .iter()
            .map(|e| e.current()[0].as_base().expect("base-field MLE opens to a base element"))
            .collect();
        let (n, secs) = self.open_stats.get();
        self.open_stats.set((n + points.len() as u64, secs + t0.elapsed().as_secs_f64()));
        GLOBAL_OPEN_COUNT.fetch_add(points.len() as u64, std::sync::atomic::Ordering::Relaxed);
        (proof, opened)
    }

    /// Verify a multi-point opening proof and return every opened evaluation.
    pub fn verify_multi(
        &self,
        commitment: &Commitment,
        proof: &Proof,
        protocol: &OpeningProtocol,
        points: &[Vec<Goldilocks>],
    ) -> Result<Vec<Goldilocks>, <MyPcs as MultilinearPcs<EF, MyChallenger>>::Error> {
        let t0 = std::time::Instant::now();
        let ef_points: Vec<Point<EF>> = points.iter().map(|p| to_ef_point(p)).collect();
        let evals = self.pcs.verify_at(
            commitment,
            proof,
            protocol,
            &ef_points,
            &mut self.fresh_challenger(),
        )?;
        let (n, secs) = self.verify_stats.get();
        self.verify_stats.set((n + points.len() as u64, secs + t0.elapsed().as_secs_f64()));
        Ok(evals
            .iter()
            .map(|e| e.current()[0].as_base().expect("base-field MLE opens to a base element"))
            .collect())
    }

    /// Open the `table_index`-th MLE in a batch at `point`.
    pub fn open_batch(
        &self,
        prover_data: ProverData,
        protocol: &OpeningProtocol,
        table_index: usize,
        num_tables: usize,
        point: &[Goldilocks],
    ) -> (Proof, Goldilocks) {
        let t0 = std::time::Instant::now();
        let arity = point.len();
        let target = to_ef_point(point);
        let dummy = Point::new(vec![EF::from(Goldilocks::new(0)); arity]);
        let points: Vec<Point<EF>> = (0..num_tables)
            .map(|i| if i == table_index { target.clone() } else { dummy.clone() })
            .collect();
        let proof = self.pcs.open_at(
            prover_data,
            protocol,
            &points,
            &mut self.fresh_challenger(),
        );
        let opened = proof.evals[table_index].current()[0];
        let (n, secs) = self.open_stats.get();
        self.open_stats.set((n + 1, secs + t0.elapsed().as_secs_f64()));
        GLOBAL_OPEN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        (proof, opened.as_base().expect("base-field MLE opens to a base element"))
    }

    /// Verify the batch opening of the `table_index`-th MLE at `point`.
    ///
    /// Assumes well-formed public metadata: upstream `verify_at` *asserts*
    /// (does not return an error) when the protocol's opening count disagrees
    /// with the point count or the table index is out of range. Use
    /// [`Whir::verify_batch_checked`] to reject malformed metadata instead.
    pub fn verify_batch(
        &self,
        commitment: &Commitment,
        proof: &Proof,
        protocol: &OpeningProtocol,
        table_index: usize,
        num_tables: usize,
        point: &[Goldilocks],
    ) -> Result<Goldilocks, <MyPcs as MultilinearPcs<EF, MyChallenger>>::Error> {
        let arity = point.len();
        let target = to_ef_point(point);
        let dummy = Point::new(vec![EF::from(Goldilocks::new(0)); arity]);
        let points: Vec<Point<EF>> = (0..num_tables)
            .map(|i| if i == table_index { target.clone() } else { dummy.clone() })
            .collect();
        let evals = self.pcs.verify_at(
            commitment,
            proof,
            protocol,
            &points,
            &mut self.fresh_challenger(),
        )?;
        Ok(evals[table_index].current()[0].as_base().expect("base-field MLE opens to a base element"))
    }

    /// Verify the batch opening of the `table_index`-th MLE at `point`,
    /// rejecting malformed public metadata (non-canonical protocol, zero table
    /// count, out-of-range index, wrong point dimension) with an error BEFORE
    /// any PCS verification runs. Upstream `verify_at` asserts on such
    /// mismatches, so a verifier must pre-validate.
    pub fn verify_batch_checked(
        &self,
        commitment: &Commitment,
        proof: &Proof,
        protocol: &OpeningProtocol,
        table_index: usize,
        num_tables: usize,
        point: &[Goldilocks],
    ) -> Result<Goldilocks, WhirVerifyError> {
        self.check_batch_metadata(protocol, table_index, num_tables, point)
            .map_err(WhirVerifyError::Malformed)?;
        self.verify_batch(commitment, proof, protocol, table_index, num_tables, point)
            .map_err(WhirVerifyError::Pcs)
    }

    /// Shared batch-metadata validation: canonical protocol for
    /// `(point arity, num_tables)`, positive table count, in-range index,
    /// and point dimension matching this instance's configuration.
    fn check_batch_metadata(
        &self,
        protocol: &OpeningProtocol,
        table_index: usize,
        num_tables: usize,
        point: &[Goldilocks],
    ) -> Result<(), &'static str> {
        if num_tables == 0 {
            return Err("num_tables must be positive");
        }
        if table_index >= num_tables {
            return Err("table_index out of range");
        }
        let arity = point.len();
        let table_len = 1usize
            .checked_shl(arity as u32)
            .ok_or("point dimension too large")?;
        let total_slots = num_tables.checked_mul(table_len).ok_or("batch size overflow")?;
        // `next_power_of_two` overflows (panics) when total_slots > 2^63;
        // the checked form turns that into a rejection instead.
        let total_size = total_slots
            .checked_next_power_of_two()
            .ok_or("batch size overflow")?;
        let total_num_vars = total_size.trailing_zeros() as usize;
        if self.num_variables() != total_num_vars {
            return Err("batch size does not match this Whir instance");
        }
        if *protocol != self.batch_protocol(arity, num_tables) {
            return Err("protocol is not the canonical batch schedule");
        }
        Ok(())
    }

    /// Open every MLE in a batch at the same `point` in a single FRI proof,
    /// amortizing the O(N log N) folding across all tables instead of paying
    /// it once per table. Returns `(proof, evals)` where `evals[i]` is the
    /// opening of table `i`.
    pub fn open_batch_multi(
        &self,
        prover_data: ProverData,
        protocol: &OpeningProtocol,
        num_tables: usize,
        point: &[Goldilocks],
    ) -> (Proof, Vec<Goldilocks>) {
        let t0 = std::time::Instant::now();
        let target = to_ef_point(point);
        let points: Vec<Point<EF>> = (0..num_tables).map(|_| target.clone()).collect();
        let proof = self.pcs.open_at(
            prover_data,
            protocol,
            &points,
            &mut self.fresh_challenger(),
        );
        let opened: Vec<Goldilocks> = proof
            .evals
            .iter()
            .map(|e| e.current()[0].as_base().expect("base-field MLE opens to a base element"))
            .collect();
        let (n, secs) = self.open_stats.get();
        self.open_stats.set((n + 1, secs + t0.elapsed().as_secs_f64()));
        GLOBAL_OPEN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        (proof, opened)
    }

    /// Verify a batched opening of every MLE in the batch at the same `point`.
    pub fn verify_batch_multi(
        &self,
        commitment: &Commitment,
        proof: &Proof,
        protocol: &OpeningProtocol,
        num_tables: usize,
        point: &[Goldilocks],
    ) -> Result<Vec<Goldilocks>, <MyPcs as MultilinearPcs<EF, MyChallenger>>::Error> {
        let target = to_ef_point(point);
        let points: Vec<Point<EF>> = (0..num_tables).map(|_| target.clone()).collect();
        let evals = self.pcs.verify_at(
            commitment,
            proof,
            protocol,
            &points,
            &mut self.fresh_challenger(),
        )?;
        Ok(evals
            .iter()
            .map(|e| e.current()[0].as_base().expect("base-field MLE opens to a base element"))
            .collect())
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
    use crate::common::field::{Goldilocks, XorShift64};
    use crate::common::mle;

    #[test]
    fn whir_batch_open_matches_mle_eval() {
        let mut rng = XorShift64::new(0xbeef);
        let whir = Whir::new_testing(8);
        let n = 4usize;
        let arity = 8usize;
        let evals: Vec<Vec<Goldilocks>> = (0..n)
            .map(|_| (0..(1usize << arity)).map(|_| rng.field()).collect())
            .collect();
        let refs: Vec<&[Goldilocks]> = evals.iter().map(|e| e.as_slice()).collect();
        let (commitment, prover_data, protocol, batch_whir) = whir.commit_batch(&refs);
        for i in 0..n {
            let point: Vec<Goldilocks> = (0..arity).map(|_| rng.field()).collect();
            let (proof, opened) = batch_whir.open_batch(prover_data.clone(), &protocol, i, n, &point);
            let verified = batch_whir
                .verify_batch(&commitment, &proof, &protocol, i, n, &point)
                .unwrap();
            assert_eq!(opened, verified);
            assert_eq!(verified, mle::eval(&evals[i], &point));
        }
    }

    /// `new_target` is genuinely fallible: tiny arities (below the folding
    /// factor), PoW schedules exceeding the budget, and absurd FFT domains
    /// must yield `None`, never a panic.
    #[test]
    fn new_target_is_fallible() {
        assert!(Whir::new_target(4, 32, 10).is_none(), "arity below folding factor");
        // 90-bit with budget 0 IS valid at arity 6 (97 final queries cover
        // it); a 200-bit target with budget 0 needs >0 PoW bits and must be
        // rejected instead.
        assert!(Whir::new_target(6, 200, 0).is_none(), "PoW budget insufficient");
        assert!(Whir::new_target(40, 90, 32).is_none(), "FFT domain too large");
        let w = Whir::new_target(6, 90, 32).expect("valid configuration");
        assert!(w.pow_bits_ok());
    }

    /// `verify_ef` pre-validates point arity and the canonical single-opening
    /// protocol, returning `Err(Malformed)` instead of hitting upstream
    /// asserts.
    #[test]
    fn verify_ef_rejects_malformed_metadata() {
        let mut rng = XorShift64::new(0x7EE);
        let whir = Whir::new_testing(6);
        let evals: Vec<Goldilocks> = (0..(1 << 6)).map(|_| rng.field()).collect();
        let (root, pd, proto) = whir.commit(&evals);
        let point = Point::new(vec![EF::from(Goldilocks::new(7)); 6]);
        let (proof, _) = whir.open_ef(&root, pd, &proto, &point);

        let bad_pt = Point::new(vec![EF::from(Goldilocks::new(7)); 5]);
        assert!(matches!(
            whir.verify_ef(&root, &proof, &proto, &bad_pt),
            Err(WhirVerifyError::Malformed(_))
        ));
        let bad_proto = whir.opening_protocol(6, 2);
        assert!(matches!(
            whir.verify_ef(&root, &proof, &bad_proto, &point),
            Err(WhirVerifyError::Malformed(_))
        ));
        assert!(whir.verify_ef(&root, &proof, &proto, &point).is_ok());
    }

    /// Batch commits must inherit the caller's security parameters instead of
    /// silently downgrading to `new_testing` (32-bit / 10 PoW bits). Inspects
    /// parameters only: commit of a 2^6 tensor, no openings, no grinding.
    #[test]
    fn commit_batch_inherits_caller_security_params() {
        let mut rng = XorShift64::new(0x9e9e);
        let evals: Vec<Goldilocks> = (0..(1 << 6)).map(|_| rng.field()).collect();

        let whir = Whir::new(6); // 90-bit, 32 PoW-bit budget
        let (_, _, _, batch_whir) = whir.commit_batch(&[&evals[..]]);
        assert_eq!(batch_whir.security_params(), whir.security_params());
        assert_eq!(batch_whir.security_params(), (90, 32));

        let testing = Whir::new_testing(6); // 32-bit, 10 PoW-bit budget
        let (_, _, _, testing_batch) = testing.commit_batch(&[&evals[..]]);
        assert_eq!(testing_batch.security_params(), (32, 10));
    }

    #[test]
    fn whir_prescribed_open_matches_mle_eval() {
        let mut rng = XorShift64::new(0x5eed);
        let d = 6;
        let evals: Vec<Goldilocks> = (0..(1 << d)).map(|_| rng.field()).collect();
        // Functional roundtrip: use testing parameters. `Whir::new` (90-bit)
        // would grind real proof-of-work on every open, which is not what this
        // completeness test measures.
        let whir = Whir::new_testing(d);

        let point: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
        let (commitment, prover_data, protocol) = whir.commit(&evals);
        let (proof, opened) = whir.open(prover_data, &protocol, &point);
        let verified = whir.verify(&commitment, &proof, &protocol, &point).unwrap();

        assert_eq!(opened, verified);
        assert_eq!(verified, mle::eval(&evals, &point));
    }
}
