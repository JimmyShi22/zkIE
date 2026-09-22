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
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type MyDft = Radix2DFTSmallBatch<F>;
type MyLayout = SuffixProver<F, EF>;
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
        let mmcs = MyMmcs::new(
            MerkleHash::new(perm.clone()),
            MerkleCompress::new(perm.clone()),
            0,
        );
        let config = WhirConfig::<EF, F, MyChallenger>::new(num_variables, params).unwrap();
        let dft = MyDft::new(1 << config.max_fft_size());
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
