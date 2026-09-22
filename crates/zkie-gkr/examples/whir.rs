//! WHIR multilinear polynomial commitment scheme over Goldilocks.
//!
//! Demonstrates the full multilinear PCS lifecycle (commit -> open -> verify)
//! using Plonky3 0.7.0's `WhirProver`, which implements `MultilinearPcs`.
//! This is the polylog-opening replacement for the univariate `TwoAdicFriPcs`
//! used by the earlier `plonky3_fri` example.

use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
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

type F = Goldilocks;
type EF = BinomialExtensionField<F, 2>;
type Perm = Poseidon2Goldilocks<16>;

type MerkleHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MerkleCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type MyDft = Radix2DFTSmallBatch<F>;
type Layout = SuffixProver<F, EF>;
type MyPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, Layout>;

fn main() {
    // 2^num_variables evaluations committed under a single multilinear polynomial.
    let num_variables = 10;
    let num_evaluations = 1;
    let folding_factor = FoldingFactor::Constant(5);
    let starting_log_inv_rate = 1;

    let (num_rounds, _) = folding_factor
        .compute_number_of_rounds(num_variables)
        .expect("valid folding schedule");
    let mut round_log_inv_rates = Vec::with_capacity(num_rounds);
    let mut rate = starting_log_inv_rate;
    for round in 0..num_rounds {
        rate += folding_factor.at_round(round) - 1;
        round_log_inv_rates.push(rate);
    }

    let whir_params = ProtocolParameters {
        security_level: 90,
        pow_bits: DEFAULT_MAX_POW,
        folding_factor: folding_factor.clone(),
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate,
        round_log_inv_rates,
    };

    // Poseidon2-based Merkle hash + compression, and the Fiat-Shamir challenger.
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let merkle_hash = MerkleHash::new(perm.clone());
    let merkle_compress = MerkleCompress::new(perm.clone());
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    // One random single-column table over the hypercube.
    let mut rng = SmallRng::seed_from_u64(0);
    let table = Table::rand(&mut rng, 1, num_variables);
    let witness = Layout::new_witness(vec![table], folding_factor.at_round(0));

    let point_schedule: PointSchedule = (0..num_evaluations)
        .map(|_| OpeningBatch::new(vec![0], Vec::new()))
        .collect();
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        point_schedule,
    )])
    .pad_to_min_num_variables(folding_factor.at_round(0));
    assert_eq!(witness.table_shapes(), protocol.table_shapes());

    let config =
        WhirConfig::<EF, F, MyChallenger>::new(witness.num_variables(), whir_params).unwrap();
    let challenger = MyChallenger::new(perm.clone());
    let dft = Radix2DFTSmallBatch::<F>::new(1 << config.max_fft_size());
    let pcs = MyPcs::new(config, dft, mmcs);

    // Commit.
    let mut prover_challenger = challenger.clone();
    let mut domainsep = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut domainsep);
    domainsep.observe_domain_separator(&mut prover_challenger);
    let (commitment, prover_data) =
        <MyPcs as MultilinearPcs<EF, MyChallenger>>::commit(&pcs, witness, &mut prover_challenger);

    // Open.
    let proof = <MyPcs as MultilinearPcs<EF, MyChallenger>>::open(
        &pcs,
        prover_data,
        protocol.clone(),
        &mut prover_challenger,
    );

    // Verify with an independent transcript.
    let mut verifier_challenger = challenger;
    let mut domainsep = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut domainsep);
    domainsep.observe_domain_separator(&mut verifier_challenger);
    <MyPcs as MultilinearPcs<EF, MyChallenger>>::verify(
        &pcs,
        &commitment,
        &proof,
        &mut verifier_challenger,
        protocol,
    )
    .expect("WHIR verification failed");

    println!("WHIR multilinear commit + open + verify OK over Goldilocks (2^{num_variables} evals)");
}
