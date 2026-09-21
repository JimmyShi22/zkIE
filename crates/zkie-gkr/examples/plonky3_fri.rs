//! Minimal integration of Plonky3's audited FRI/MMCS over Goldilocks:
//! commit a matrix of polynomial evaluations, open it at a point, and verify.

use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_ceil_usize;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Val = Goldilocks;
type Challenge = Val;
type Perm = Poseidon<Val, MdsMatrixGoldilocks, 8, 7>;
type MyHash = PaddingFreeSponge<Perm, 8, 4, 4>;
type MyCompress = TruncatedPermutation<Perm, 2, 4, 8>;
type ValMmcs = FieldMerkleTreeMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    MyHash,
    MyCompress,
    4,
>;
type Dft = Radix2DitParallel;
type Challenger = DuplexChallenger<Val, Perm, 8, 4>;
type FriPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ValMmcs>;

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    let perm = Perm::new_from_rng(4, 22, MdsMatrixGoldilocks, &mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 40,
        proof_of_work_bits: 8,
        mmcs: val_mmcs.clone(),
    };
    let dft = Dft {};
    let pcs = FriPcs::new(log2_ceil_usize(1 << 10), dft, val_mmcs, fri_config);

    // One polynomial evaluated over the natural domain.
    let domain = <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << 10);
    let evals = RowMajorMatrix::<Val>::rand_nonzero(&mut rng, domain.size(), 1);

    let (commitment, prover_data) =
        <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, evals)]);

    // Sample the opening point (same value for prover and verifier).
    let mut prover_challenger = Challenger::new(perm.clone());
    let point: Challenge = prover_challenger.sample_ext_element();
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, vec![vec![point]])],
        &mut prover_challenger,
    );

    let mut verifier_challenger = Challenger::new(perm);
    let v_point: Challenge = verifier_challenger.sample_ext_element();
    assert_eq!(point, v_point);

    <FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        vec![(
            commitment,
            vec![(domain, vec![(point, opened[0][0][0].clone())])],
        )],
        &proof,
        &mut verifier_challenger,
    )
    .expect("FRI open should verify");

    println!("plonky3 FRI commit + open + verify OK over Goldilocks");
}
