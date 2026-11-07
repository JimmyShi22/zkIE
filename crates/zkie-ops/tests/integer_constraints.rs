//! Integer constraints: sumcheck, quotient/remainder and limb/carry checks.

use zkie_core::error::Error;
use zkie_core::types::Digest;

use zkie_engine::pcs::request::field::Fp3;

use zkie_ops::proof::binding::{commit_rows_with_openings, OracleOpening, OracleRegistry};
use zkie_ops::proof::integer::{
    prove_integer_relations, verify_integer_relations, IntegerProof, IntegerVectors,
};
use zkie_ops::proof::sumcheck::{accounting, prove_sumcheck, verify_sumcheck};

fn field(value: u64) -> Fp3 {
    Fp3::from_u64(value)
}

#[test]
fn sumcheck_checks_every_round_and_reports_its_budget() {
    let values = vec![field(1), field(2), field(3), field(4)];
    let challenges = vec![field(2), field(3)];
    let proof = prove_sumcheck(&values, &challenges, field(10)).unwrap();
    verify_sumcheck(&proof, &challenges, field(10), proof.final_value).unwrap();
    assert!(prove_sumcheck(&values, &challenges, field(11)).is_err());

    let mut tampered = proof.clone();
    tampered.rounds[0][0] = tampered.rounds[0][0].add(&Fp3::ONE);
    assert!(verify_sumcheck(&tampered, &challenges, field(10), proof.final_value).is_err());

    let small = accounting(2);
    let large = accounting(10);
    assert!(small.error_log2_floor < large.error_log2_floor);
    assert_eq!(small.field_bits, 192);
    assert!(small.error_log2_floor < 0);
}

fn commit(values: &[i64], registry: &mut OracleRegistry, label: u8) -> Vec<OracleOpening> {
    let rows: Vec<Vec<u64>> = values.iter().map(|value| vec![*value as u64]).collect();
    let (_, openings) =
        commit_rows_with_openings(registry, &Digest([label; 32]), &rows).unwrap();
    openings
}

#[test]
fn quotient_remainder_and_limbs_are_exact() {
    let vectors = IntegerVectors {
        numerators: vec![7, -7],
        denominators: vec![3, 3],
        quotients: vec![2, -3],
        remainders: vec![1, 2],
    };
    let mut registry = OracleRegistry::new();
    let numerators = commit(&vectors.numerators, &mut registry, 1);
    let denominators = commit(&vectors.denominators, &mut registry, 2);
    let quotients = commit(&vectors.quotients, &mut registry, 3);
    let remainders = commit(&vectors.remainders, &mut registry, 4);
    let draft = IntegerProof {
        numerators,
        denominators,
        quotients,
        remainders,
        limbs: Vec::new(),
        carries: vec![0, 0],
        base_bits: 16,
    };
    let proof = prove_integer_relations(&vectors, &draft, &registry).unwrap();
    verify_integer_relations(&proof, &registry).unwrap();

    let mut tampered = draft.clone();
    tampered.quotients[0].values[0] += 1;
    assert!(prove_integer_relations(&vectors, &tampered, &registry).is_err());

    let false_relation = IntegerVectors {
        quotients: vec![3, -3],
        ..vectors.clone()
    };
    assert!(verify_integer_relations(&proof, &registry).is_ok());
    assert!(prove_integer_relations(&false_relation, &proof, &registry).is_err());
}

#[test]
fn zero_and_negative_divisors_are_rejected() {
    let vectors = IntegerVectors {
        numerators: vec![7],
        denominators: vec![0],
        quotients: vec![0],
        remainders: vec![7],
    };
    let mut registry = OracleRegistry::new();
    let draft = IntegerProof {
        numerators: commit(&vectors.numerators, &mut registry, 5),
        denominators: commit(&vectors.denominators, &mut registry, 6),
        quotients: commit(&vectors.quotients, &mut registry, 7),
        remainders: commit(&vectors.remainders, &mut registry, 8),
        limbs: Vec::new(),
        carries: vec![0],
        base_bits: 16,
    };
    assert!(matches!(
        prove_integer_relations(&vectors, &draft, &registry),
        Err(Error::DivisionByZero)
    ));

    let mut negative = vectors.clone();
    negative.denominators = vec![-3];
    let mut registry = OracleRegistry::new();
    let draft = IntegerProof {
        numerators: commit(&negative.numerators, &mut registry, 9),
        denominators: commit(&negative.denominators, &mut registry, 10),
        quotients: commit(&negative.quotients, &mut registry, 11),
        remainders: commit(&negative.remainders, &mut registry, 12),
        limbs: Vec::new(),
        carries: vec![0],
        base_bits: 16,
    };
    assert!(prove_integer_relations(&negative, &draft, &registry).is_err());
}

#[test]
fn multiplicity_planes_recombine_counts_and_mask_padding() {
    use zkie_ops::proof::lookup::{commit_count_planes, verify_count_planes};

    let counts = vec![5u64, 0, 3, 0];
    let mut registry = OracleRegistry::new();
    let (oracle, count_openings) = commit_rows_with_openings(
        &mut registry,
        &Digest([0x31; 32]),
        &counts.iter().map(|count| vec![*count]).collect::<Vec<_>>(),
    )
    .unwrap();
    let planes = commit_count_planes(&mut registry, &Digest([0x32; 32]), &counts, 3).unwrap();
    verify_count_planes(&registry, &count_openings, &planes).unwrap();

    // A non-boolean plane bit is rejected.
    let mut bad_bit = planes.clone();
    bad_bit.plane_openings[0][0].values[0] = 2;
    assert!(verify_count_planes(&registry, &count_openings, &bad_bit).is_err());

    // A padding row cannot carry a nonzero count.
    let mut bad_mask = planes.clone();
    bad_mask.mask_openings[3].values[0] = 1;
    assert!(verify_count_planes(&registry, &count_openings, &bad_mask).is_err());

    // Counts must match the plane recombination.
    let mut changed = count_openings.clone();
    changed[0].values[0] = 6;
    assert!(verify_count_planes(&registry, &changed, &planes).is_err());

    // The commitment of a plane is authenticated.
    let mut moved = planes.clone();
    moved.plane_openings[0][0].oracle = oracle;
    assert!(verify_count_planes(&registry, &count_openings, &moved).is_err());
}

#[test]
fn lookup_planes_and_mask_share_one_entry_with_multiplicity_ood() {
    use zkie_engine::pcs::request::chain::Chain;
    use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;
    use zkie_ops::proof::lookup::{
        commit_multiplicities, prove_lookup_with_planes, verify_lookup_with_planes,
    };
    use zkie_ops::proof::request::lkp::LookupTable;

    let table = LookupTable::new(vec![vec![1], vec![2]]).unwrap();
    let mut registry = OracleRegistry::new();
    let label = Digest([0x41; 32]);
    let (_, queries) = commit_rows_with_openings(
        &mut registry,
        &label,
        &[vec![1], vec![1], vec![2], vec![2]],
    )
    .unwrap();
    let (oracle, openings) =
        commit_multiplicities(&mut registry, &label, &table, &queries, &[]).unwrap();
    let mut chain =
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest([4u8; 32]), b"lookup", &[], None).unwrap();
    let proof = prove_lookup_with_planes(
        &mut chain,
        &table,
        &queries,
        &[],
        &mut registry,
        oracle,
        &openings,
        &Digest([0x42; 32]),
    )
    .unwrap();
    let mut verify_chain =
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest([4u8; 32]), b"lookup", &[], None).unwrap();
    verify_lookup_with_planes(&mut verify_chain, &table, &queries, &[], &registry, &proof).unwrap();

    let mut bad_mask = proof.clone();
    bad_mask.planes.mask_openings[0].values[0] = 0;
    let mut mask_chain =
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest([4u8; 32]), b"lookup", &[], None).unwrap();
    assert!(verify_lookup_with_planes(&mut mask_chain, &table, &queries, &[], &registry, &bad_mask).is_err());

    let mut bad_bit = proof.clone();
    bad_bit.planes.plane_openings[0][0].values[0] = 2;
    let mut bit_chain =
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest([4u8; 32]), b"lookup", &[], None).unwrap();
    assert!(verify_lookup_with_planes(&mut bit_chain, &table, &queries, &[], &registry, &bad_bit).is_err());

    let mut bad_binding = proof.clone();
    bad_binding.multiplicity.query_digest = Digest([0x77; 32]);
    let mut binding_chain =
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest([4u8; 32]), b"lookup", &[], None).unwrap();
    assert!(verify_lookup_with_planes(
        &mut binding_chain,
        &table,
        &queries,
        &[],
        &registry,
        &bad_binding
    )
    .is_err());

    let mut swapped = queries.clone();
    swapped.swap(0, 3);
    let mut swap_chain =
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest([4u8; 32]), b"lookup", &[], None).unwrap();
    assert!(verify_lookup_with_planes(
        &mut swap_chain,
        &table,
        &swapped,
        &[],
        &registry,
        &proof
    )
    .is_err());

    let mut bad_fold = proof.clone();
    bad_fold.fold.rounds[0].polynomial[0] =
        bad_fold.fold.rounds[0].polynomial[0].add(&Fp3::ONE);
    let mut fold_chain =
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest([4u8; 32]), b"lookup", &[], None).unwrap();
    assert!(verify_lookup_with_planes(
        &mut fold_chain,
        &table,
        &queries,
        &[],
        &registry,
        &bad_fold
    )
    .is_err());

    let mut bad_target = proof.clone();
    bad_target.fold_target = bad_target.fold_target.add(&Fp3::ONE);
    let mut target_chain =
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest([4u8; 32]), b"lookup", &[], None).unwrap();
    assert!(verify_lookup_with_planes(
        &mut target_chain,
        &table,
        &queries,
        &[],
        &registry,
        &bad_target
    )
    .is_err());

    let mut bad_ood = proof.clone();
    bad_ood.multiplicity.answer = bad_ood.multiplicity.answer.add(&Fp3::ONE);
    let mut ood_chain =
        Chain::begin(REQUEST_CHAIN_LABEL, &Digest([4u8; 32]), b"lookup", &[], None).unwrap();
    assert!(verify_lookup_with_planes(&mut ood_chain, &table, &queries, &[], &registry, &bad_ood).is_err());
}
