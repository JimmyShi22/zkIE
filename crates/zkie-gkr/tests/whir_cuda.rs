//! Integration test: the CUDA-backed `Whir` (feature `cuda`) produces
//! bit-identical commitments to an explicit CPU pipeline, and its
//! commit/open/verify roundtrip checks out.
//!
//! The CPU pipeline here mirrors `src/whir.rs` with the CPU engines
//! (`Radix2DFTSmallBatch` + `MerkleTreeMmcs`), so this test also pins the
//! GPU path's output against the CPU path at the full WHIR protocol level,
//! on top of the per-engine bit-exactness tests in zkie-cuda.

#![cfg(feature = "cuda")]

use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout as _, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PointSchedule, TableShape, TableSpec};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, DEFAULT_MAX_POW,
};
use p3_whir::pcs::prover::WhirProver;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use zkie_gkr::field::XorShift64;
use zkie_gkr::whir::Whir;

type F = Goldilocks;
type EF = BinomialExtensionField<F, 2>;
type Perm = Poseidon2Goldilocks<16>;
type MerkleHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MerkleCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type CpuChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type CpuMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type CpuDft = Radix2DFTSmallBatch<F>;
type CpuLayout = SuffixProver<F, EF>;
type CpuPcs = WhirProver<EF, F, CpuDft, CpuMmcs, CpuChallenger, CpuLayout>;

/// The CPU pipeline, configured exactly like `Whir::with_params`.
fn cpu_pcs(num_variables: usize, security_level: usize, pow_bits: usize) -> (CpuPcs, Perm) {
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
    let mmcs = CpuMmcs::new(
        MerkleHash::new(perm.clone()),
        MerkleCompress::new(perm.clone()),
        0,
    );
    let config = WhirConfig::<EF, F, CpuChallenger>::new(num_variables, params).unwrap();
    let dft = CpuDft::new(1 << config.max_fft_size());
    (CpuPcs::new(config, dft, mmcs), perm)
}

fn cpu_commit(
    pcs: &CpuPcs,
    perm: &Perm,
    evals: &[F],
) -> <CpuPcs as MultilinearPcs<EF, CpuChallenger>>::Commitment {
    let num_vars = evals.len().trailing_zeros() as usize;
    let table = Table::new(RowMajorMatrix::new(evals.to_vec(), 1 << num_vars));
    let witness = CpuLayout::new_witness(vec![table], 5);
    let mut challenger = CpuChallenger::new(perm.clone());
    let mut domain_separator = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut domain_separator);
    domain_separator.observe_domain_separator(&mut challenger);
    let (commitment, _) = pcs.commit(witness, &mut challenger);
    commitment
}

#[test]
fn whir_commitments_match_cpu() {
    if !zkie_cuda::cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    for &d in &[8usize, 12, 15] {
        let whir = Whir::new_testing(d); // CUDA-backed under this feature
        let (cpu_pcs, cpu_perm) = cpu_pcs(d, 32, 10);

        let mut rng = XorShift64::new(0x600);
        let evals: Vec<F> = (0..(1 << d)).map(|_| rng.field()).collect();

        let (gpu_commitment, _, _) = whir.commit(&evals);
        let cpu_commitment = cpu_commit(&cpu_pcs, &cpu_perm, &evals);
        assert_eq!(
            gpu_commitment, cpu_commitment,
            "WHIR commitment mismatch at 2^{d}"
        );
    }
}

#[test]
fn whir_cuda_roundtrip() {
    if !zkie_cuda::cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    for &d in &[8usize, 15] {
        let whir = Whir::new_testing(d);
        let mut rng = XorShift64::new(0x5eed);
        let evals: Vec<F> = (0..(1 << d)).map(|_| rng.field()).collect();
        let point: Vec<F> = (0..d).map(|_| rng.field()).collect();

        let (commitment, prover_data, protocol) = whir.commit(&evals);
        let (proof, opened) = whir.open(prover_data, &protocol, &point);
        let verified = whir
            .verify(&commitment, &proof, &protocol, &point)
            .expect("verify");
        assert_eq!(opened, verified);

        // Reference MLE evaluation (the `mle` helper's convention: LSB first).
        let mut expected = F::ZERO;
        for (i, &v) in evals.iter().enumerate() {
            let mut term = v;
            for (b, &p) in point.iter().enumerate() {
                if i & (1 << b) == 0 {
                    term *= F::ONE - p;
                } else {
                    term *= p;
                }
            }
            expected += term;
        }
        assert_eq!(opened, expected);
    }
}
