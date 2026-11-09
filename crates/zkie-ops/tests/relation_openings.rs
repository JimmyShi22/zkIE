use zkie_core::error::Error;
use zkie_core::types::Digest;
use zkie_ops::affine_scale::AffineScaleOp;
use zkie_ops::matmul::MatMulOp;
use zkie_ops::proof::binding::commit_rows_with_openings;
use zkie_ops::proof::relations::{
    proof_input_values, prove_op_relation_shared, verify_op_relation_with_external,
    OpRelationProof, OpRelationPublic, RelationConfig,
};

fn matmul_fixture() -> (OpRelationPublic, OpRelationProof, Vec<i64>) {
    let op = MatMulOp {
        m: 3,
        n: 2,
        k: 2,
        shift: 0,
    };
    let lhs = (1..=6).collect::<Vec<i64>>();
    let rhs = (1..=4).map(|value| value * 2).collect::<Vec<i64>>();
    let inputs = vec![lhs, rhs];
    let config = op.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation_shared(&op, &inputs, &config, &[]).unwrap();
    (public, proof, config)
}

fn affine_fixture() -> (OpRelationPublic, OpRelationProof, Vec<i64>) {
    let op = AffineScaleOp {
        multiplier: 3,
        addend: 5,
        shift: 2,
        clamp: None,
    };
    let inputs = vec![vec![1, 2, 3]];
    let config = op.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation_shared(&op, &inputs, &config, &[]).unwrap();
    (public, proof, config)
}

fn verify_code(public: &OpRelationPublic, proof: &OpRelationProof, config: &[i64]) -> String {
    verify_op_relation_with_external(public, proof, config, &[])
        .unwrap_err()
        .code()
        .to_owned()
}

#[test]
fn positive_group_openings_verify() {
    let (public, proof, config) = matmul_fixture();
    verify_op_relation_with_external(&public, &proof, &config, &[]).unwrap();
    let inputs = proof_input_values(&proof).unwrap();
    assert_eq!(inputs[0].len(), 6);
    assert_eq!(inputs[1].len(), 4);

    let (public, proof, config) = affine_fixture();
    verify_op_relation_with_external(&public, &proof, &config, &[]).unwrap();
}

#[test]
fn reordered_and_duplicated_openings_are_rejected() {
    let (public, proof, config) = matmul_fixture();

    let mut duplicate = proof.clone();
    duplicate.inputs[0][1] = duplicate.inputs[0][0].clone();
    assert_eq!(verify_code(&public, &duplicate, &config), "invalid_index");
    assert!(proof_input_values(&duplicate).is_err());

    let mut swapped = proof.clone();
    swapped.inputs[0].swap(0, 1);
    assert_eq!(verify_code(&public, &swapped, &config), "invalid_index");

    let mut missing = proof.clone();
    missing.inputs[0][2] = missing.inputs[0][0].clone();
    assert_eq!(verify_code(&public, &missing, &config), "invalid_index");

    let mut output_swap = proof.clone();
    output_swap.outputs[0].swap(0, 2);
    assert_eq!(verify_code(&public, &output_swap, &config), "invalid_index");
}

#[test]
fn width_padding_depth_length_and_coverage_are_rejected() {
    let (public, proof, config) = matmul_fixture();

    let mut width = proof.clone();
    width.inputs[0][0].values.pop();
    assert_eq!(verify_code(&public, &width, &config), "shape_mismatch");

    let mut padding = proof.clone();
    padding.inputs[0][3].values[0] = 1;
    assert_eq!(verify_code(&public, &padding, &config), "statement_mismatch");

    let mut depth = proof.clone();
    depth.inputs[0][0].path.push(Digest([0; 32]));
    assert_eq!(verify_code(&public, &depth, &config), "invalid_proof");

    let mut coverage = proof.clone();
    coverage.inputs[0].pop();
    assert_eq!(verify_code(&public, &coverage, &config), "statement_mismatch");

    let mut length = proof.clone();
    length.lengths.inputs[0] = 8;
    assert_eq!(verify_code(&public, &length, &config), "shape_mismatch");

    let mut odd_length = proof.clone();
    odd_length.lengths.inputs[0] = 5;
    assert_eq!(verify_code(&public, &odd_length, &config), "shape_mismatch");
}

#[test]
fn wide_group_structure_is_checked() {
    let (public, proof, config) = affine_fixture();

    let mut width = proof.clone();
    width.raw[0][0].values.pop();
    assert_eq!(verify_code(&public, &width, &config), "shape_mismatch");

    let mut index = proof.clone();
    index.raw[0][0].index = 1;
    assert_eq!(verify_code(&public, &index, &config), "invalid_index");

    let mut depth = proof.clone();
    depth.raw[0][0].path.push(Digest([0; 32]));
    assert_eq!(verify_code(&public, &depth, &config), "invalid_proof");

    let mut coverage = proof.clone();
    coverage.raw.push(coverage.raw[0].clone());
    assert_eq!(verify_code(&public, &coverage, &config), "shape_mismatch");
}

#[test]
fn bounds_do_not_panic() {
    let (public, mut proof, config) = matmul_fixture();
    proof.lengths.inputs[0] = usize::MAX;
    let result = verify_op_relation_with_external(&public, &proof, &config, &[]);
    assert!(matches!(result, Err(Error::ShapeMismatch)));
}

#[test]
fn empty_and_missing_ports_are_rejected() {
    let (public, proof, config) = matmul_fixture();

    let mut empty_outputs = proof.clone();
    empty_outputs.outputs.clear();
    empty_outputs.lengths.outputs.clear();
    let mut public_empty_outputs = public.clone();
    public_empty_outputs.output_oracles.clear();
    assert_eq!(
        verify_code(&public_empty_outputs, &empty_outputs, &config),
        "shape_mismatch"
    );

    let mut empty_inputs = proof.clone();
    empty_inputs.inputs.clear();
    empty_inputs.lengths.inputs.clear();
    let mut public_empty_inputs = public.clone();
    public_empty_inputs.input_oracles.clear();
    assert_eq!(
        verify_code(&public_empty_inputs, &empty_inputs, &config),
        "shape_mismatch"
    );

    let mut empty_raw = proof.clone();
    empty_raw.raw.clear();
    empty_raw.lengths.raw.clear();
    assert_eq!(verify_code(&public, &empty_raw, &config), "shape_mismatch");
}

#[test]
fn extra_zero_leaves_are_rejected() {
    let (mut public, proof, config) = matmul_fixture();
    let mut rows: Vec<Vec<u64>> = proof.inputs[0]
        .iter()
        .map(|opening| opening.values.clone())
        .collect();
    let width = rows[0].len();
    while rows.len() < 8 {
        rows.push(vec![0; width]);
    }
    let (oracle, openings) =
        commit_rows_with_openings(&mut public.registry, &Digest([9; 32]), &rows).unwrap();
    let mut tampered = proof.clone();
    tampered.inputs[0] = openings;
    public.input_oracles[0] = oracle;
    assert_eq!(
        verify_code(&public, &tampered, &config),
        "statement_mismatch"
    );
}
