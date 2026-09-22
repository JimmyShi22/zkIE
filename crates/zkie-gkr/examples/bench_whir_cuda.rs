//! Benchmark CPU vs GPU WHIR commit/open at 2^15 and 2^22 (the dominant
//! commitment sizes of the 200M proof), with a bit-equality assert between
//! the two backends.
//!
//! Run with `--features cuda`. Falls back to comparing CPU against itself
//! when no CUDA device is available.

use std::time::Instant;

use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout as _, SuffixProver, Table};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
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

fn cpu_pcs(num_variables: usize, pow_bits: usize) -> (CpuPcs, Perm) {
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
        security_level: 32,
        pow_bits,
        folding_factor,
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

fn cpu_commit(pcs: &CpuPcs, perm: &Perm, evals: &[F]) -> <CpuPcs as MultilinearPcs<EF, CpuChallenger>>::Commitment {
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

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "auto".into());
    println!("rayon threads: {threads}, cuda available: {}", zkie_cuda::cuda_available());

    for &d in &[15usize, 22] {
        let whir = Whir::new_testing(d); // CUDA-backed under the cuda feature
        let (cpu_pcs, cpu_perm) = cpu_pcs(d, 10);
        let mut rng = XorShift64::new(0x600);
        let evals: Vec<F> = (0..(1 << d)).map(|_| rng.field()).collect();
        let point: Vec<F> = (0..d).map(|_| rng.field()).collect();

        // Warm-up both pipelines (twiddle tables, GPU buffers, constants).
        let cpu_commitment = {
            let table = Table::new(RowMajorMatrix::new(evals.clone(), 1 << d));
            let witness = CpuLayout::new_witness(vec![table], 5);
            let mut challenger = CpuChallenger::new(cpu_perm.clone());
            let mut ds = DomainSeparator::new(vec![]);
            cpu_pcs.add_domain_separator::<8>(&mut ds);
            ds.observe_domain_separator(&mut challenger);
            cpu_pcs.commit(witness, &mut challenger).0
        };
        let (gpu_commitment, gpu_pd, gpu_protocol) = whir.commit(&evals);
        assert_eq!(cpu_commitment, gpu_commitment, "backends disagree at 2^{d}");

        // Timed CPU commit.
        let t0 = Instant::now();
        let cpu_c = cpu_commit(&cpu_pcs, &cpu_perm, &evals);
        let cpu_commit_secs = t0.elapsed().as_secs_f64();
        assert_eq!(cpu_c, cpu_commitment);

        // Timed GPU commit.
        let t0 = Instant::now();
        let (gpu_c, _, _) = whir.commit(&evals);
        let gpu_commit_secs = t0.elapsed().as_secs_f64();
        assert_eq!(gpu_c, cpu_commitment);

        // Timed opens (both run the CPU sumcheck folding path; the GPU
        // backend's prover data is a normal Merkle tree, so open is shared).
        let t0 = Instant::now();
        let _ = whir.open(gpu_pd, &gpu_protocol, &point);
        let gpu_open_secs = t0.elapsed().as_secs_f64();

        println!(
            "d={d:2} ({:9} elems): commit cpu {:8.3}s | gpu {:8.3}s | speedup {:5.1}x | open {:.3}s",
            1 << d,
            cpu_commit_secs,
            gpu_commit_secs,
            cpu_commit_secs / gpu_commit_secs,
            gpu_open_secs,
        );
    }
}
