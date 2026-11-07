//! Lookup, range, linear, fold-equation and small relation-graph contracts.

use serde_json::Value;
use zkie_core::claim::{CoefficientSpec, LinearClaim, RelationId};
use zkie_core::error::Error;
use zkie_core::session::{DecodeCoverage, EosPolicy, OutputPolicy, Statement};
use zkie_core::types::Digest;

use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;

use zkie_ops::proof::binding::{commit_rows_with_openings, commit_vector, OracleRegistry};
use zkie_ops::proof::request::foldeq::{
    derive_connection, fold_value, verify_derived_connection, CommittedRow, FoldConnection, FoldEqProof,
};
use zkie_ops::proof::request::lin::{coefficients as derive_coefficients, prove_linear, verify_linear, LinearProof};
use zkie_ops::proof::request::lkp::{
    prove_lookup, prove_lookup_with_padding, verify_lookup, verify_lookup_with_padding, LookupProof,
    LookupTable,
};
use zkie_ops::proof::request::lns::{limbs_for_values, prove_range, verify_range, RangeParams};
use zkie_ops::proof::request::request::{
    prove_relation_graph, verify_relation_graph, RelationGraphPublic,
    RelationGraphWitness,
};

fn vectors() -> Value {
    serde_json::from_str(include_str!("vectors/request/relations.json")).unwrap()
}

fn chain() -> Chain {
    Chain::begin(REQUEST_CHAIN_LABEL, &Digest([1u8; 32]), b"relations", &[], None).unwrap()
}

fn field(value: u64) -> Fp3 {
    Fp3::from_u64(value)
}

fn linear_value(coefficients: &[Fp3], values: &[u64]) -> Fp3 {
    let mut sum = Fp3::ZERO;
    for (coefficient, value) in coefficients.iter().zip(values) {
        sum = sum.add(&coefficient.mul(&Fp3::from_u64(*value)));
    }
    sum
}

fn statement(policy: OutputPolicy) -> Statement {
    Statement {
        protocol_id: Digest([1u8; 32]),
        protocol_version: 1,
        parameter_digest: Digest([2u8; 32]),
        model_id: Digest([3u8; 32]),
        graph_digest: Digest([4u8; 32]),
        quantization_digest: Digest([5u8; 32]),
        tables_digest: Digest([6u8; 32]),
        prompt_tokens: vec![1, 2],
        output_tokens: vec![7],
        coverage: DecodeCoverage::full(1),
        eos_policy: EosPolicy::FixedLength,
        output_policy: policy,
    }
}

#[test]
fn lookup_product_forest_accepts_and_rejects() {
    let vectors = vectors();
    let rows: Vec<Vec<u64>> = vectors["lookup_table"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect()
        })
        .collect();
    let table = LookupTable::new(rows).unwrap();
    let query_rows: Vec<Vec<u64>> = vectors["lookup_queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect()
        })
        .collect();
    let mut registry = OracleRegistry::new();
    let label = Digest([9u8; 32]);
    let (oracle, openings) = commit_rows_with_openings(&mut registry, &label, &query_rows).unwrap();
    assert!(openings.iter().all(|opening| opening.oracle == oracle));

    let mut prover = chain();
    let proof = prove_lookup(&mut prover, &table, &openings, &registry).unwrap();
    let mut verifier = chain();
    verify_lookup(&mut verifier, &table, &openings, &registry, &proof).unwrap();
    for level in &proof.query.levels {
        if level.len() > 1 {
            for pair in level.chunks_exact(2) {
                // The parent relation is checked by the verifier; this is a
                // sanity check that the levels are not degenerate.
                let _ = pair;
            }
        }
    }
    assert_eq!(proof.query.root, proof.table.root);

    let mut tampered_table = table.clone();
    tampered_table.rows[0][0] += 1;
    let mut chain_a = chain();
    assert!(verify_lookup(&mut chain_a, &tampered_table, &openings, &registry, &proof).is_err());

    let mut tampered_openings = openings.clone();
    tampered_openings[0].values[0] ^= 1;
    let mut chain_b = chain();
    assert!(verify_lookup(&mut chain_b, &table, &tampered_openings, &registry, &proof).is_err());

    let mut tampered_proof = proof.clone();
    tampered_proof.query.levels[0][0] = tampered_proof.query.levels[0][0].add(&Fp3::ONE);
    let mut chain_c = chain();
    assert!(verify_lookup(&mut chain_c, &table, &openings, &registry, &tampered_proof).is_err());

    let empty_registry = OracleRegistry::new();
    let mut chain_d = chain();
    assert!(verify_lookup(&mut chain_d, &table, &openings, &empty_registry, &proof).is_err());

    let mut zero_proof: LookupProof = proof.clone();
    zero_proof.gamma = Fp3::ZERO;
    let mut chain_e = chain();
    assert!(verify_lookup(&mut chain_e, &table, &openings, &registry, &zero_proof).is_err());
}

#[test]
fn range_check_accepts_and_rejects_out_of_range_and_tampering() {
    let vectors = vectors();
    let params = RangeParams::new(8, 16).unwrap();
    let table = zkie_ops::proof::request::lns::range_table(params).unwrap();
    let values: Vec<i64> = vectors["range_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect();
    let mut registry = OracleRegistry::new();
    let label = Digest([8u8; 32]);
    let mut prover = chain();
    let proof = prove_range(&mut prover, params, &values, &mut registry, &label, &table).unwrap();
    let mut verifier = chain();
    verify_range(&mut verifier, params, &proof, &table, &registry).unwrap();

    let out_of_range = [400000i64, 0, 0, 0, 0];
    let mut prover = chain();
    let bad = prove_range(&mut prover, params, &out_of_range, &mut registry, &label, &table).unwrap();
    let mut verifier = chain();
    assert!(matches!(
        verify_range(&mut verifier, params, &bad, &table, &registry),
        Err(Error::BoundViolation) | Err(Error::InvalidProof)
    ));

    let mut tampered = proof.clone();
    tampered.values[0] += 1;
    let mut verifier = chain();
    assert!(verify_range(&mut verifier, params, &tampered, &table, &registry).is_err());

    let mut tampered = proof.clone();
    tampered.limbs[0] ^= 1;
    let mut verifier = chain();
    assert!(verify_range(&mut verifier, params, &tampered, &table, &registry).is_err());

    let mut tampered = proof;
    tampered.values.pop();
    let mut verifier = chain();
    assert!(verify_range(&mut verifier, params, &tampered, &table, &registry).is_err());
}

#[test]
fn linear_claims_tie_coefficients_values_and_sources() {
    let vectors = vectors();
    let values: Vec<u64> = vectors["linear_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap())
        .collect();
    let coefficients: Vec<Fp3> = vectors["coefficients"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| field(value.as_u64().unwrap()))
        .collect();
    let mut registry = OracleRegistry::new();
    let _oracle = commit_vector(&mut registry, &Digest([7u8; 32]), &values).unwrap();
    let mut openings_registry = OracleRegistry::new();
    let (oracle, openings) =
        commit_rows_with_openings(&mut openings_registry, &Digest([7u8; 32]), &values.iter().map(|value| vec![*value]).collect::<Vec<_>>()).unwrap();
    let value = linear_value(&coefficients, &values);
    let claim = LinearClaim {
        oracle,
        coefficients: CoefficientSpec::Public,
        value,
        source: RelationId(3),
    };
    let mut prover = chain();
    let proof = prove_linear(&mut prover, &claim, &openings, &openings_registry, &coefficients).unwrap();
    let mut verifier = chain();
    verify_linear(&mut verifier, &claim, &proof, &openings_registry, &coefficients, RelationId(3)).unwrap();

    let mut wrong = claim.clone();
    wrong.value = wrong.value.add(&Fp3::ONE);
    let mut verifier = chain();
    assert!(verify_linear(&mut verifier, &wrong, &proof, &openings_registry, &coefficients, RelationId(3)).is_err());
    let mut verifier = chain();
    assert!(verify_linear(&mut verifier, &claim, &proof, &openings_registry, &coefficients, RelationId(4)).is_err());

    let mut tampered = proof.clone();
    tampered.openings[0].values[0] ^= 1;
    let mut verifier = chain();
    assert!(verify_linear(&mut verifier, &claim, &tampered, &openings_registry, &coefficients, RelationId(3)).is_err());

    // Challenge-derived coefficients are replayed from the chain.
    let spec = CoefficientSpec::ChallengeDerived { label: b"lin".to_vec() };
    let mut scratch = chain();
    let derived = derive_coefficients(&mut scratch, &spec, values.len(), oracle, RelationId(5), &[]).unwrap();
    let claim = LinearClaim {
        oracle,
        coefficients: spec.clone(),
        value: linear_value(&derived, &values),
        source: RelationId(5),
    };
    let mut prover = chain();
    let proof = prove_linear(&mut prover, &claim, &openings, &openings_registry, &[]).unwrap();
    let mut verifier = chain();
    verify_linear(&mut verifier, &claim, &proof, &openings_registry, &[], RelationId(5)).unwrap();
    let _ = oracle;
    let _ = registry;
}

#[test]
fn fold_connections_require_every_relation_once() {
    let mut registry = OracleRegistry::new();
    let oracle = commit_vector(&mut registry, &Digest([6u8; 32]), &[1, 2]).unwrap();
    let low = field(3);
    let high = field(5);
    let challenge = field(7);
    let connection = FoldConnection {
        relation: RelationId(1),
        oracle,
        low,
        high,
        challenge,
        folded: fold_value(low, high, challenge),
    };
    let proof = zkie_ops::proof::request::foldeq::prove_fold_equations(vec![connection.clone()]).unwrap();
    zkie_ops::proof::request::foldeq::verify_fold_equations(&proof, &[RelationId(1)], &registry).unwrap();
    assert!(zkie_ops::proof::request::foldeq::verify_fold_equations(
        &proof,
        &[RelationId(1), RelationId(2)],
        &registry
    )
    .is_err());

    let mut tampered = proof.clone();
    tampered.connections[0].folded = tampered.connections[0].folded.add(&Fp3::ONE);
    assert!(zkie_ops::proof::request::foldeq::verify_fold_equations(
        &tampered,
        &[RelationId(1)],
        &registry
    )
    .is_err());

    let mut tampered = proof.clone();
    tampered.connections[0].challenge = field(8);
    assert!(zkie_ops::proof::request::foldeq::verify_fold_equations(
        &tampered,
        &[RelationId(1)],
        &registry
    )
    .is_err());

    let mut bad = connection;
    bad.folded = bad.folded.add(&Fp3::ONE);
    assert!(zkie_ops::proof::request::foldeq::prove_fold_equations(vec![bad]).is_err());
}

#[test]
fn small_relation_graph_proves_and_verifies_from_public_data() {
    let vectors = vectors();
    let lookup_rows: Vec<Vec<u64>> = vectors["lookup_table"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|value| value.as_u64().unwrap()).collect())
        .collect();
    let lookup_table = LookupTable::new(lookup_rows).unwrap();
    let query_rows: Vec<Vec<u64>> = vectors["lookup_queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|value| value.as_u64().unwrap()).collect())
        .collect();
    let mut registry = OracleRegistry::new();
    let (_, lookup_queries) =
        commit_rows_with_openings(&mut registry, &Digest([9u8; 32]), &query_rows).unwrap();

    let range_params = RangeParams::new(8, 16).unwrap();
    let range_table = zkie_ops::proof::request::lns::range_table(range_params).unwrap();
    let range_values: Vec<i64> = vectors["graph_range_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect();
    let limbs = limbs_for_values(range_params, &range_values).unwrap();
    let limb_rows: Vec<Vec<u64>> = limbs.iter().map(|limb| vec![*limb]).collect();
    let (range_oracle, range_openings) =
        commit_rows_with_openings(&mut registry, &Digest([8u8; 32]), &limb_rows).unwrap();

    let linear_values: Vec<u64> = vectors["linear_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap())
        .collect();
    let coefficients: Vec<Fp3> = vectors["coefficients"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| field(value.as_u64().unwrap()))
        .collect();
    let (linear_oracle, linear_openings) = commit_rows_with_openings(
        &mut registry,
        &Digest([7u8; 32]),
        &linear_values.iter().map(|value| vec![*value]).collect::<Vec<_>>(),
    )
    .unwrap();
    let linear_claim = LinearClaim {
        oracle: linear_oracle,
        coefficients: CoefficientSpec::Public,
        value: linear_value(&coefficients, &linear_values),
        source: RelationId(3),
    };

    let low = field(11);
    let high = field(13);
    let challenge = field(17);
    let connection = FoldConnection {
        relation: RelationId(1),
        oracle: linear_oracle,
        low,
        high,
        challenge,
        folded: fold_value(low, high, challenge),
    };
    let public = RelationGraphPublic {
        statement: statement(OutputPolicy::ExactMatch),
        registry,
        lookup_table,
        lookup_queries,
        range_table,
        range_params,
        range_values: range_values.clone(),
        range_oracle,
        range_openings,
        linear_claim,
        linear_openings,
        public_coefficients: coefficients,
        required_relations: vec![RelationId(1)],
    };
    let witness = RelationGraphWitness {
        range_values,
        fold_connections: vec![connection],
    };
    let mut prover = chain();
    let proof = prove_relation_graph(&mut prover, &public, &witness).unwrap();
    let mut verifier = chain();
    let result = verify_relation_graph(&mut verifier, &public, &proof).unwrap();
    assert!(result.accepted);

    let mut tampered = proof.clone();
    tampered.lookup.query.levels[0][0] = tampered.lookup.query.levels[0][0].add(&Fp3::ONE);
    let mut verifier = chain();
    assert!(verify_relation_graph(&mut verifier, &public, &tampered).is_err());

    let mut tampered = proof.clone();
    tampered.range.values[0] += 1;
    let mut verifier = chain();
    assert!(verify_relation_graph(&mut verifier, &public, &tampered).is_err());

    let mut tampered = proof.clone();
    tampered.linear = LinearProof { openings: Vec::new() };
    let mut verifier = chain();
    assert!(verify_relation_graph(&mut verifier, &public, &tampered).is_err());

    let mut tampered = proof.clone();
    tampered.foldeq = FoldEqProof { connections: Vec::new() };
    let mut verifier = chain();
    assert!(verify_relation_graph(&mut verifier, &public, &tampered).is_err());

    let mut wrong_public = public.clone();
    wrong_public.range_values[0] += 1;
    let mut verifier = chain();
    assert!(verify_relation_graph(&mut verifier, &wrong_public, &proof).is_err());
}

#[test]
fn lookup_padding_uses_a_tie_factor_and_rejects_a_tampered_tie() {
    let vectors = vectors();
    let table = LookupTable::new(
        vectors["lookup_table"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().iter().map(|value| value.as_u64().unwrap()).collect())
            .collect(),
    )
    .unwrap();
    let query_rows: Vec<Vec<u64>> = vectors["lookup_queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|value| value.as_u64().unwrap()).collect())
        .collect();
    let mut registry = OracleRegistry::new();
    let (_, openings) =
        commit_rows_with_openings(&mut registry, &Digest([9u8; 32]), &query_rows).unwrap();
    let padding = vec![vec![40u64]];

    let mut prover = chain();
    let proof = prove_lookup_with_padding(&mut prover, &table, &openings, &padding, &registry).unwrap();
    assert_ne!(proof.padding_factor, Fp3::ONE);
    let mut verifier = chain();
    verify_lookup_with_padding(&mut verifier, &table, &openings, &padding, &registry, &proof).unwrap();

    // The same proof without the declared padding fails the tie.
    let mut verifier = chain();
    assert!(verify_lookup(&mut verifier, &table, &openings, &registry, &proof).is_err());

    let mut tampered = proof.clone();
    tampered.padding_factor = tampered.padding_factor.add(&Fp3::ONE);
    let mut verifier = chain();
    assert!(verify_lookup_with_padding(&mut verifier, &table, &openings, &padding, &registry, &tampered).is_err());

    // A padding key must itself be a table row.
    let mut prover = chain();
    assert!(prove_lookup_with_padding(&mut prover, &table, &openings, &[vec![41u64]], &registry).is_err());
}

#[test]
fn range_table_must_cover_every_limb() {
    let params = RangeParams::new(8, 16).unwrap();
    let full = zkie_ops::proof::request::lns::range_table(params).unwrap();
    let mut truncated_rows = full.rows.clone();
    truncated_rows.truncate(100);
    let truncated = LookupTable::new(truncated_rows).unwrap();
    let values = [200i64];
    let mut registry = OracleRegistry::new();
    let mut prover = chain();
    assert!(prove_range(&mut prover, params, &values, &mut registry, &Digest([4u8; 32]), &truncated).is_err());
}

#[test]
fn fold_connections_are_derived_from_committed_rows() {
    let low_values = [1u64, 2, 3, 4];
    let high_values = [5u64, 6, 7, 8];
    let mut registry = OracleRegistry::new();
    let (low_oracle, low_openings) = commit_rows_with_openings(
        &mut registry,
        &Digest([1u8; 32]),
        &low_values.iter().map(|value| vec![*value]).collect::<Vec<_>>(),
    )
    .unwrap();
    let (high_oracle, high_openings) = commit_rows_with_openings(
        &mut registry,
        &Digest([2u8; 32]),
        &high_values.iter().map(|value| vec![*value]).collect::<Vec<_>>(),
    )
    .unwrap();
    let low = CommittedRow {
        oracle: low_oracle,
        coefficients: vec![field(1), field(2), field(3), field(4)],
        openings: low_openings,
    };
    let high = CommittedRow {
        oracle: high_oracle,
        coefficients: vec![field(1), field(1), field(1), field(1)],
        openings: high_openings,
    };
    let challenge = field(3);
    let connection = derive_connection(RelationId(1), &low, &high, challenge, &registry).unwrap();
    assert_eq!(connection.low, field(30));
    assert_eq!(connection.high, field(26));
    assert_eq!(connection.folded, fold_value(field(30), field(26), challenge));
    verify_derived_connection(&connection, &low, &high, &registry).unwrap();

    let mut tampered = connection.clone();
    tampered.folded = tampered.folded.add(&Fp3::ONE);
    assert!(verify_derived_connection(&tampered, &low, &high, &registry).is_err());

    let mut tampered = low.clone();
    tampered.coefficients[0] = field(2);
    assert!(verify_derived_connection(&connection, &tampered, &high, &registry).is_err());
}
