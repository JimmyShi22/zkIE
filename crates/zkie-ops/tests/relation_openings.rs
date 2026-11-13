use zkie_core::error::Error;
use zkie_core::types::Digest;
use zkie_ops::affine_scale::AffineScaleOp;
use zkie_ops::matmul::MatMulOp;
use zkie_ops::op::OpCode;
use zkie_ops::proof::binding::{
    commit_rows_with_openings, OracleId, OracleOpening, OracleRegistry,
};
use zkie_ops::proof::relations::{
    proof_input_values, prove_op_relation_shared, prove_op_relation_with_inputs,
    verify_op_relation_with_external, OpRelationProof, OpRelationPublic, RelationConfig,
    RelationLengths,
};
use zkie_ops::softmax::SoftmaxOp;

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
    let (public, proof) = prove_op_relation_with_inputs(&op, &inputs, &config, &[]).unwrap();
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
    let (public, proof) = prove_op_relation_with_inputs(&op, &inputs, &config, &[]).unwrap();
    (public, proof, config)
}

fn shared_fixture() -> (OpRelationPublic, OpRelationProof, Vec<i64>, Vec<Vec<i64>>) {
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
    (public, proof, config, inputs)
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
    // The prover writes the full-open form: every leaf is present and no
    // Merkle path is carried; the verifier rebuilds each root from the leaves.
    for group in proof
        .inputs
        .iter()
        .chain(proof.outputs.iter())
        .chain(proof.quotients.iter())
        .chain(proof.helpers.iter())
        .chain(proof.aux.iter())
    {
        assert!(group.iter().all(|opening| opening.path.is_empty()));
    }
    assert!(proof.raw.iter().all(|group| group.iter().all(|opening| opening.path.is_empty())));

    let (public, proof, config) = affine_fixture();
    verify_op_relation_with_external(&public, &proof, &config, &[]).unwrap();
}

#[test]
fn pathless_leaf_tamper_is_rejected() {
    let (public, proof, config) = matmul_fixture();
    let mut changed = proof.clone();
    changed.inputs[0][0].values[0] ^= 1;
    assert_eq!(verify_code(&public, &changed, &config), "invalid_proof");

    let mut changed = proof.clone();
    changed.outputs[0][1].values[1] ^= 1;
    assert_eq!(verify_code(&public, &changed, &config), "invalid_proof");
}

#[test]
fn pathless_partial_coverage_is_rejected() {
    let (public, proof, config) = matmul_fixture();
    let mut missing = proof.clone();
    missing.outputs[0].pop();
    assert_eq!(verify_code(&public, &missing, &config), "statement_mismatch");

    let mut reordered = proof.clone();
    reordered.outputs[0].swap(1, 2);
    assert_eq!(verify_code(&public, &reordered, &config), "invalid_index");
}

#[test]
fn shared_form_carries_no_input_openings() {
    let (public, proof, config, inputs) = shared_fixture();
    assert!(proof.inputs.iter().all(|group| group.is_empty()));
    assert!(public
        .input_oracles
        .iter()
        .all(|oracle| oracle.0 == u64::MAX));
    let values: Vec<Option<Vec<i64>>> = inputs.iter().cloned().map(Some).collect();
    verify_op_relation_with_external(&public, &proof, &config, &values).unwrap();

    // A materialized value that differs from the prover's input is rejected by
    // the relation equation.
    let mut changed = inputs.clone();
    changed[0][0] ^= 1;
    let wrong: Vec<Option<Vec<i64>>> = changed.into_iter().map(Some).collect();
    assert_eq!(
        verify_op_relation_with_external(&public, &proof, &config, &wrong)
            .unwrap_err()
            .code(),
        "invalid_proof"
    );

    // Missing materialized values are missing authentication.
    assert_eq!(
        verify_op_relation_with_external(&public, &proof, &config, &[])
            .unwrap_err()
            .code(),
        "statement_mismatch"
    );
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

fn layout_fixture(cells: usize) -> (OpRelationPublic, OpRelationProof, Vec<i64>) {
    let config = vec![0, 2, 2, 4, 1];
    let values: Vec<i64> = (1..=cells as i64).collect();
    let mut rows: Vec<Vec<u64>> = values.iter().map(|value| vec![*value as u64]).collect();
    rows.resize(rows.len().max(1).next_power_of_two(), vec![0]);
    let mut registry = OracleRegistry::new();
    let (input_oracle, inputs) =
        commit_rows_with_openings(&mut registry, &Digest([11; 32]), &rows).unwrap();
    let (output_oracle, outputs) =
        commit_rows_with_openings(&mut registry, &Digest([12; 32]), &rows).unwrap();
    let proof = OpRelationProof {
        lengths: RelationLengths {
            inputs: vec![cells],
            outputs: vec![cells],
            raw: Vec::new(),
            quotients: Vec::new(),
            helpers: Vec::new(),
            aux: Vec::new(),
        },
        inputs: vec![inputs],
        outputs: vec![outputs],
        raw: Vec::new(),
        quotients: Vec::new(),
        helpers: Vec::new(),
        aux: Vec::new(),
    };
    let public = OpRelationPublic {
        opcode: OpCode::Layout,
        config: config.clone(),
        registry,
        input_oracles: vec![input_oracle],
        output_oracles: vec![output_oracle],
    };
    (public, proof, config)
}

#[test]
fn layout_lengths_must_match_the_grid() {
    let (public, proof, config) = layout_fixture(4);
    verify_op_relation_with_external(&public, &proof, &config, &[]).unwrap();

    let (short_public, short_proof, short_config) = layout_fixture(3);
    assert_eq!(
        verify_code(&short_public, &short_proof, &short_config),
        "shape_mismatch"
    );
}

#[test]
fn softmax_raw_length_must_match_the_input() {
    let op = SoftmaxOp { shift: 4 };
    let inputs = vec![vec![1i64, 2, 3, 4], vec![1i64, 2, 3, 4]];
    let config = op.relation_config(&inputs).unwrap();
    let (mut public, proof) =
        prove_op_relation_with_inputs(&op, &inputs, &config, &[]).unwrap();
    verify_op_relation_with_external(&public, &proof, &config, &[]).unwrap();

    let words = proof.raw[0][0].values.clone();
    let shortened = words[..words.len() - 2].to_vec();
    let (oracle, openings) =
        commit_rows_with_openings(&mut public.registry, &Digest([13; 32]), &[shortened]).unwrap();
    let mut tampered = proof.clone();
    tampered.raw[0] = openings;
    tampered.lengths.raw[0] -= 1;
    assert_eq!(verify_code(&public, &tampered, &config), "shape_mismatch");
}

fn width_group(
    registry: &mut OracleRegistry,
    label: u8,
    values: &[i64],
    width: usize,
) -> (OracleId, Vec<OracleOpening>) {
    assert!(width > 0 && values.len() % width == 0);
    let mut rows: Vec<Vec<u64>> = values
        .chunks_exact(width)
        .map(|chunk| chunk.iter().map(|value| *value as u64).collect())
        .collect();
    assert!(!rows.is_empty());
    rows.resize(rows.len().max(1).next_power_of_two(), vec![0; width]);
    commit_rows_with_openings(registry, &Digest([label; 32]), &rows).unwrap()
}

fn wide_group(
    registry: &mut OracleRegistry,
    label: u8,
    values: &[i64],
) -> (OracleId, Vec<OracleOpening>) {
    let mut words = Vec::with_capacity(values.len() * 2);
    for value in values {
        let wide = *value as i128;
        words.push(wide as u64);
        words.push((wide >> 64) as u64);
    }
    commit_rows_with_openings(registry, &Digest([label; 32]), &[words]).unwrap()
}

fn manual_relation(
    opcode: OpCode,
    config: Vec<i64>,
    inputs: Vec<(Vec<i64>, usize)>,
    outputs: Vec<(Vec<i64>, usize)>,
    raw: Option<Vec<i64>>,
    aux: Vec<Vec<i64>>,
) -> (OpRelationPublic, OpRelationProof) {
    let mut registry = OracleRegistry::new();
    let mut label = 40u8;
    let mut input_oracles = Vec::new();
    let mut input_openings = Vec::new();
    let mut input_lengths = Vec::new();
    for (values, width) in inputs {
        let (oracle, openings) = width_group(&mut registry, label, &values, width);
        label += 1;
        input_oracles.push(oracle);
        input_openings.push(openings);
        input_lengths.push(values.len());
    }
    let mut output_oracles = Vec::new();
    let mut output_openings = Vec::new();
    let mut output_lengths = Vec::new();
    for (values, width) in outputs {
        let (oracle, openings) = width_group(&mut registry, label, &values, width);
        label += 1;
        output_oracles.push(oracle);
        output_openings.push(openings);
        output_lengths.push(values.len());
    }
    let mut raw_lengths = Vec::new();
    let mut raw_openings = Vec::new();
    if let Some(values) = raw {
        let (_, openings) = wide_group(&mut registry, label, &values);
        label += 1;
        raw_lengths.push(values.len());
        raw_openings.push(openings);
    }
    let mut aux_lengths = Vec::new();
    let mut aux_openings = Vec::new();
    for values in aux {
        let (_, openings) = width_group(&mut registry, label, &values, 1);
        label += 1;
        aux_lengths.push(values.len());
        aux_openings.push(openings);
    }
    let proof = OpRelationProof {
        lengths: RelationLengths {
            inputs: input_lengths,
            outputs: output_lengths,
            raw: raw_lengths,
            quotients: Vec::new(),
            helpers: Vec::new(),
            aux: aux_lengths,
        },
        inputs: input_openings,
        outputs: output_openings,
        raw: raw_openings,
        quotients: Vec::new(),
        helpers: Vec::new(),
        aux: aux_openings,
    };
    let public = OpRelationPublic {
        opcode,
        config: config.clone(),
        registry,
        input_oracles,
        output_oracles,
    };
    (public, proof)
}

#[test]
fn profile_rms_norm_square_overflow_is_rejected() {
    let min = i64::MIN;
    let (public, proof) = manual_relation(
        OpCode::ProfileRmsNorm,
        vec![1, 2, 2, 0],
        vec![
            (vec![min, min], 1),
            (vec![1, 1], 2),
            (vec![1], 1),
        ],
        vec![(vec![0, 0], 2)],
        Some(vec![0, 0]),
        vec![vec![0], vec![0]],
    );
    assert_eq!(verify_code(&public, &proof, &[1, 2, 2, 0]), "bound_violation");
}

#[test]
fn profile_layer_norm_variance_overflow_is_rejected() {
    let (public, proof) = manual_relation(
        OpCode::ProfileLayerNorm,
        vec![1, 2, 2, 0],
        vec![
            (vec![i64::MIN, i64::MAX], 1),
            (vec![1, 1], 2),
            (vec![0, 0], 2),
            (vec![1], 1),
        ],
        vec![(vec![0, 0], 2)],
        Some(vec![0, 0]),
        vec![vec![0], vec![0], vec![0]],
    );
    assert_eq!(verify_code(&public, &proof, &[1, 2, 2, 0]), "bound_violation");
}

#[test]
fn rope_projection_short_raw_is_rejected() {
    let (public, proof) = manual_relation(
        OpCode::RopeProjection,
        vec![0],
        vec![
            (vec![1, 2, 3, 4], 1),
            (vec![1, 1], 1),
            (vec![0, 0], 1),
        ],
        vec![(vec![1, 2, 3, 4], 1)],
        Some(vec![1, 2]),
        Vec::new(),
    );
    assert_eq!(verify_code(&public, &proof, &[0]), "shape_mismatch");
}

#[test]
fn unchecked_accumulation_branches_fail_closed() {
    let min = i64::MIN;

    let (public, proof) = manual_relation(
        OpCode::BatchedMatMul,
        vec![1, 1, 1, 2, 0],
        vec![(vec![min, min], 2), (vec![min, min], 1)],
        vec![(vec![0], 1)],
        Some(vec![0]),
        Vec::new(),
    );
    assert_eq!(
        verify_code(&public, &proof, &[1, 1, 1, 2, 0]),
        "bound_violation"
    );

    let (public, proof) = manual_relation(
        OpCode::GatherMatMul,
        vec![2, 1, 0],
        vec![
            (vec![min, min], 1),
            (vec![0], 1),
            (vec![min, min], 1),
        ],
        vec![(vec![0], 1)],
        Some(vec![0]),
        Vec::new(),
    );
    assert_eq!(verify_code(&public, &proof, &[2, 1, 0]), "bound_violation");

    let (public, proof) = manual_relation(
        OpCode::CausalAttention,
        vec![1, 2, 0],
        vec![
            (vec![min, min], 1),
            (vec![min, min], 1),
            (vec![0, 0], 1),
            (vec![1], 1),
        ],
        vec![(vec![0, 0], 1)],
        Some(vec![0, 0]),
        vec![vec![0], vec![0], vec![0]],
    );
    assert_eq!(verify_code(&public, &proof, &[1, 2, 0]), "bound_violation");
}
