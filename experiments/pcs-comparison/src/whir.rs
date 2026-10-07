//! Upstream p3-whir 0.7 over the Goldilocks BASE field with a quadratic
//! EXTENSION challenge field (`BinomialExtensionField<Goldilocks, 2>`),
//! benchmark-only. Mirrors `p3-whir/examples/whir.rs`; separate experimental
//! code — NOT a change to production GKR or the zkie-core `Whir` aliases.
//!
//! Target: 90-bit configured security with a PoW budget of 0. If the p3-whir
//! construction rejects those parameters (e.g. `PowBitsExceedBudget`), the
//! rejection reason is recorded as-is and security is NOT weakened. A
//! base-field testing-parameter run (32-bit, 10 PoW bits) is included as an
//! explicitly insecure functional baseline.
//!
//! Batching: the two tables are committed into ONE WHIR witness and opened at
//! two prescribed points with ONE FRI proof.
//!
//! Transcript contract (upstream requirement): `commit` leaves the commitment
//! absorbed in the prover's challenger, and `open_at` CONTINUES that
//! challenger. `verify_at` does NOT absorb the commitment itself — the caller
//! absorbs it once into a fresh domain-separated challenger before calling.

use std::time::{Duration, Instant};

use p3_challenger::{CanObserve, DuplexChallenger};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Layout as _, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PointSchedule, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
};
use p3_whir::pcs::prover::WhirProver;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = Goldilocks;
/// Quadratic extension challenge field (Goldilocks base).
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

pub struct WhirReport {
    pub per_table_vars: usize,
    pub security_level: usize,
    pub pow_budget: usize,
    /// `Some(err)` if `WhirConfig::new` rejected these parameters; no proof
    /// was produced and security was not weakened.
    pub config_error: Option<String>,
    /// Config derivation + Poseidon2 permutation + Merkle MMCS construction.
    pub setup: Option<Duration>,
    /// DFT twiddle-table allocation (`MyDft::new`).
    pub dft_init: Option<Duration>,
    pub commit: Option<Duration>,
    pub open: Option<Duration>,
    pub verify: Option<Duration>,
    /// ONE batched proof covers BOTH tables (single witness, two points).
    pub proof_bytes: Option<usize>,
    /// One batched commitment covers both tables.
    pub commitment_bytes: Option<usize>,
    pub final_queries: Option<usize>,
    pub max_pow_bits: Option<usize>,
    pub wrong_point_rejected: Option<bool>,
    pub wrong_commitment_rejected: Option<bool>,
    /// Opening the MUTATED witness against the honest commitment is rejected.
    pub mutated_value_rejected: Option<bool>,
}

/// Raw integer values of the shared generator for table `table_index`
/// (2^n entries in 0..99). Used by tests to prove both schemes share inputs.
pub fn table_values(n: usize, table_index: usize) -> Vec<u64> {
    crate::tables::values(n, table_index)
}

/// Integer-valued 2^n table (shared generator, embedded into the Goldilocks
/// base field). Same integer pattern as the KZG side, but embedded into a
/// DIFFERENT field; no cross-field equality is claimed.
fn int_table(n: usize, table_index: usize) -> Vec<F> {
    table_values(n, table_index).into_iter().map(F::new).collect()
}

struct WhirParts {
    config: WhirConfig<EF, F, MyChallenger>,
    perm: Perm,
    mmcs: MyMmcs,
}

/// Derive the round-by-round config and build the Poseidon2/Merkle parts.
/// Returns `(Err(reason), None, None)` if the parameters are rejected.
fn build_parts(
    total_num_vars: usize,
    security_level: usize,
    pow_bits: usize,
) -> (Result<WhirParts, String>, Option<usize>, Option<usize>) {
    let folding_factor = FoldingFactor::Constant(5);
    let (num_rounds, _) = folding_factor
        .compute_number_of_rounds(total_num_vars)
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
    let config = match WhirConfig::<EF, F, MyChallenger>::new(total_num_vars, params) {
        Ok(c) => c,
        // Record the rejection reason; do NOT weaken security.
        Err(e) => return (Err(format!("{e:?}")), None, None),
    };
    let final_queries = config.final_queries;
    let max_pow_bits = config.max_pow_bits();

    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let mmcs = MyMmcs::new(
        MerkleHash::new(perm.clone()),
        MerkleCompress::new(perm.clone()),
        0,
    );
    (Ok(WhirParts { config, perm, mmcs }), Some(final_queries), Some(max_pow_bits))
}

/// Fresh challenger with only the domain separator observed.
fn fresh_challenger(pcs: &MyPcs, perm: &Perm) -> MyChallenger {
    let mut challenger = MyChallenger::new(perm.clone());
    let mut domain_separator = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut domain_separator);
    domain_separator.observe_domain_separator(&mut challenger);
    challenger
}

/// Fresh challenger at the transcript state `[domain_separator, commitment]`:
/// the state both `open_at` (prover) and `verify_at` (verifier) must start
/// from. The prover's `commit` leaves exactly this state via `commit_base`
/// (`challenger.observe(root)`), so re-absorbing the commitment into a fresh
/// domain-separated challenger reproduces it for every open/verify pair —
/// binding each opening transcript to the commitment.
fn commitment_challenger(
    pcs: &MyPcs,
    perm: &Perm,
    commitment: &<MyPcs as MultilinearPcs<EF, MyChallenger>>::Commitment,
) -> MyChallenger {
    let mut challenger = fresh_challenger(pcs, perm);
    challenger.observe(commitment);
    challenger
}

/// Embed base-field coordinates into the extension field (LSB-first
/// convention of the local `mle` utilities, mirrored here for the experiment).
fn to_ef_point(point: &[F]) -> Point<EF> {
    Point::new(point.iter().rev().map(|&c| EF::from(c)).collect())
}

/// Independent MLE evaluation of the base-field table at an extension point,
/// OUTSIDE any timed region. The flat table is LSB-first while p3 points are
/// MSB-first, so the fold iterates the reversed coordinate order (matching
/// `to_ef_point`).
fn ef_eval(base_evals: &[F], point: &Point<EF>) -> EF {
    let mut buf: Vec<EF> = base_evals.iter().map(|&v| EF::from(v)).collect();
    let mut size = buf.len();
    for &p in point.iter().rev() {
        let half = size / 2;
        for i in 0..half {
            buf[i] = buf[2 * i] + p * (buf[2 * i + 1] - buf[2 * i]);
        }
        size = half;
    }
    buf[0]
}

/// Run the WHIR cycle for `num_tables` integer tables of `2^per_table_vars`
/// entries over `rounds` rounds of fresh random challenge points.
pub fn run(per_table_vars: usize, security_level: usize, pow_budget: usize, rounds: usize) -> WhirReport {
    let num_tables = 2usize;
    let total_slots = num_tables * (1usize << per_table_vars);
    let total_num_vars = total_slots.next_power_of_two().trailing_zeros() as usize;

    let t_setup = Instant::now();
    let (parts_res, final_queries, max_pow_bits) =
        build_parts(total_num_vars, security_level, pow_budget);
    let setup = t_setup.elapsed();
    let parts = match parts_res {
        Ok(p) => p,
        Err(e) => {
            return WhirReport {
                per_table_vars,
                security_level,
                pow_budget,
                config_error: Some(e),
                setup: Some(setup),
                dft_init: None,
                commit: None,
                open: None,
                verify: None,
                proof_bytes: None,
                commitment_bytes: None,
                final_queries,
                max_pow_bits,
                wrong_point_rejected: None,
                wrong_commitment_rejected: None,
                mutated_value_rejected: None,
            }
        }
    };

    let t_dft = Instant::now();
    let dft = MyDft::new(1 << parts.config.max_fft_size());
    let dft_init = t_dft.elapsed();
    let pcs = MyPcs::new(parts.config, dft, parts.mmcs);

    let base_tables: Vec<Vec<F>> = (0..num_tables).map(|i| int_table(per_table_vars, i)).collect();
    let tables: Vec<Table<F>> = base_tables
        .iter()
        .map(|evals| Table::new(RowMajorMatrix::new(evals.clone(), 1 << per_table_vars)))
        .collect();
    let folding = FoldingFactor::Constant(5).at_round(0);
    let witness = MyLayout::new_witness(tables, folding);

    // One single-point opening batch per table (the schedule itself has ONE
    // entry and is cloned per table; a per-table multi-entry schedule would
    // inflate the opening count).
    let point_schedule: PointSchedule =
        std::iter::once(OpeningBatch::new(vec![0], Vec::new())).collect();
    let specs: Vec<TableSpec> = (0..num_tables)
        .map(|_| TableSpec::new(TableShape::new(per_table_vars, 1), point_schedule.clone()))
        .collect();
    let protocol = OpeningProtocol::new(specs).pad_to_min_num_variables(folding);

    // Prover transcript: commit absorbs the commitment into the challenger.
    let mut prover_challenger = fresh_challenger(&pcs, &parts.perm);
    let t0 = Instant::now();
    let (commitment, prover_data) =
        <MyPcs as MultilinearPcs<EF, MyChallenger>>::commit(&pcs, witness, &mut prover_challenger);
    let commit = t0.elapsed();
    let commitment_bytes = postcard::to_allocvec(&commitment).expect("commitment serialization").len();

    // Mutated witness (one value doubled) for the mutated-value rejection.
    let mut bad_base_tables = base_tables.clone();
    bad_base_tables[0][0] = bad_base_tables[0][0] + bad_base_tables[0][0];
    let bad_tables: Vec<Table<F>> = bad_base_tables
        .iter()
        .map(|evals| Table::new(RowMajorMatrix::new(evals.clone(), 1 << per_table_vars)))
        .collect();
    let bad_witness = MyLayout::new_witness(bad_tables, folding);
    let mut bad_challenger = fresh_challenger(&pcs, &parts.perm);
    let (bad_commitment, _bad_prover_data) =
        <MyPcs as MultilinearPcs<EF, MyChallenger>>::commit(&pcs, bad_witness, &mut bad_challenger);

    let mut rng = SmallRng::seed_from_u64(0x77);
    let mut open_elapsed = Duration::ZERO;
    let mut verify_elapsed = Duration::ZERO;
    let mut proof_bytes = 0usize;
    let mut last_points: Vec<Vec<F>> = Vec::new();
    let mut last_proof = None;
    for _ in 0..rounds {
        // One fresh challenge vector per table, per round.
        let points: Vec<Vec<F>> = (0..num_tables)
            .map(|_| (0..per_table_vars).map(|_| rng.random::<F>()).collect())
            .collect();
        let ef_points: Vec<Point<EF>> = points.iter().map(|p| to_ef_point(p)).collect();

        // Each open/verify pair starts a FRESH transcript at
        // [domain_separator, commitment] on BOTH sides (matching the state
        // `commit` leaves), so repeated rounds do not accumulate transcript
        // state and each proof is bound to the commitment.
        let mut open_challenger = commitment_challenger(&pcs, &parts.perm, &commitment);
        let o0 = Instant::now();
        let proof = pcs.open_at(
            prover_data.clone(),
            &protocol,
            &ef_points,
            &mut open_challenger,
        );
        open_elapsed += o0.elapsed();
        proof_bytes = postcard::to_allocvec(&proof).expect("proof serialization").len();

        let mut verifier_challenger = commitment_challenger(&pcs, &parts.perm, &commitment);
        let v0 = Instant::now();
        let evals = pcs
            .verify_at(
                &commitment,
                &proof,
                &protocol,
                &ef_points,
                &mut verifier_challenger,
            )
            .expect("honest opening must verify");
        verify_elapsed += v0.elapsed();
        assert_eq!(evals.len(), num_tables);

        // Independent check OUTSIDE the timers: the returned opening values
        // must equal the MLE evaluation of the intended tables in the
        // extension field.
        for (i, p) in ef_points.iter().enumerate() {
            let expected = ef_eval(&base_tables[i], p);
            let got = evals[i].current()[0];
            assert_eq!(expected, got, "opened value must match independent MLE eval");
        }
        last_points = points;
        last_proof = Some(proof);
    }
    let open = open_elapsed.div_f64(rounds as f64);
    let verify = verify_elapsed.div_f64(rounds as f64);

    // ---- rejection tests on the last proof ----
    let ef_points: Vec<Point<EF>> = last_points.iter().map(|p| to_ef_point(p)).collect();
    let proof = last_proof.unwrap();

    // wrong point: verify the last proof at fresh (different) points
    let other_points: Vec<Vec<F>> = (0..num_tables)
        .map(|_| (0..per_table_vars).map(|_| rng.random::<F>()).collect())
        .collect();
    let ef_other: Vec<Point<EF>> = other_points.iter().map(|p| to_ef_point(p)).collect();
    let mut wc = commitment_challenger(&pcs, &parts.perm, &commitment);
    let wrong_point_rejected = pcs
        .verify_at(&commitment, &proof, &protocol, &ef_other, &mut wc)
        .is_err();

    // wrong commitment: honest proof checked against the flipped-table commit
    let mut wc2 = commitment_challenger(&pcs, &parts.perm, &bad_commitment);
    let wrong_commitment_rejected = pcs
        .verify_at(&bad_commitment, &proof, &protocol, &ef_points, &mut wc2)
        .is_err();

    // mutated claimed value: CLONE the honest proof and change the opening
    // evaluation claimed for table 0 (`proof.evals[0]`), preserving the honest
    // root, points, and the rest of the proof. The OOD binding must reject it.
    let mut mutated_proof = proof.clone();
    let cur = mutated_proof.evals[0].current().to_vec();
    let mut bad_cur = cur;
    bad_cur[0] = bad_cur[0] + bad_cur[0]; // one claimed opening evaluation doubled
    mutated_proof.evals[0] = OpeningBatch::new(bad_cur, mutated_proof.evals[0].next().to_vec());
    let mut mvc = commitment_challenger(&pcs, &parts.perm, &commitment);
    let mutated_value_rejected = pcs
        .verify_at(&commitment, &mutated_proof, &protocol, &ef_points, &mut mvc)
        .is_err();

    WhirReport {
        per_table_vars,
        security_level,
        pow_budget,
        config_error: None,
        setup: Some(setup),
        dft_init: Some(dft_init),
        commit: Some(commit),
        open: Some(open),
        verify: Some(verify),
        proof_bytes: Some(proof_bytes),
        commitment_bytes: Some(commitment_bytes),
        final_queries,
        max_pow_bits,
        wrong_point_rejected: Some(wrong_point_rejected),
        wrong_commitment_rejected: Some(wrong_commitment_rejected),
        mutated_value_rejected: Some(mutated_value_rejected),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Functional baseline (explicitly insecure testing parameters).
    #[test]
    fn whir_positive_and_rejection_testing_params() {
        let r = run(6, 32, 10, 2);
        assert!(r.config_error.is_none(), "testing params must construct: {:?}", r.config_error);
        assert!(r.commit.unwrap().as_secs_f64() >= 0.0);
        assert!(r.proof_bytes.unwrap() > 0);
        assert!(r.wrong_point_rejected.unwrap(), "wrong point must be rejected");
        assert!(r.wrong_commitment_rejected.unwrap(), "wrong commitment must be rejected");
        assert!(r.mutated_value_rejected.unwrap(), "mutated value must be rejected");
    }

    /// 90-bit with zero PoW budget must either construct (with more queries)
    /// or be rejected with a recorded reason — never silently weakened.
    #[test]
    fn whir_90bit_zero_pow_constructs_or_records_rejection() {
        let r = run(6, 90, 0, 1);
        if let Some(err) = &r.config_error {
            // Rejection is acceptable: record and continue; do not weaken.
            assert!(!err.is_empty());
        } else {
            assert!(r.wrong_point_rejected.unwrap());
            assert!(r.wrong_commitment_rejected.unwrap());
            assert!(r.mutated_value_rejected.unwrap());
        }
    }
}
