//! Directed protocol mutation matrix.
//!
//! Every test builds a valid object, mutates exactly one protocol-relevant
//! field, and requires the verifier to reject the result. The categories are
//! the ones fixed by the comprehensive-acceptance card: identities, weights,
//! inputs/outputs, LUTs, quantization, shape/layout/padding, graph edges,
//! lookup keys, product trees, quotient/limb/carry/range, KV state, RoPE,
//! causal masking, seals, batch order, shared results, OOD answers, fold
//! connections, terminal equations, PoW, versions, parameters and output
//! policy.

use std::fs;

use zkie_core::claim::RelationId;
use zkie_core::error::Result;
use zkie_core::oracle::OracleId;
use zkie_core::session::{
    ClippedGapPolicy, DecodeCoverage, EosPolicy, OutputPolicy, Statement,
};
use zkie_core::types::{DType, Digest, LayoutSpec, TensorId, TensorSpec};

use zkie_engine::pcs::request::chain::{Chain, KIND_SEAL};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::fold::{prove_fold, verify_fold, FoldClaim, FoldShape};
use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;
use zkie_engine::pcs::request::pow::{search_nonce, verify_nonce, work_test};

use zkie_ops::causal_attention::CausalAttentionOp;
use zkie_ops::div_round::DivRoundOp;
use zkie_ops::index_gen::TableIndexOp;
use zkie_ops::kv_cache::KvCacheOp;
use zkie_ops::matmul::MatMulOp;
use zkie_ops::profile_norm::ProfileLayerNormOp;
use zkie_ops::proof::binding::{
    bind_edge, commit_rows_with_openings, verify_edge, OracleRegistry,
};
use zkie_ops::proof::integer::{
    prove_integer_relations, verify_integer_relations, IntegerProof, IntegerVectors,
};
use zkie_ops::proof::lookup::{
    commit_multiplicities, prove_multiplicity_ood, verify_multiplicity_ood,
};
use zkie_ops::proof::model_commitment::{
    commit_assets, open_asset, ModelAssetSpec, ModelCommitmentPublic,
};
use zkie_ops::proof::relations::{
    prove_op_relation, verify_op_relation, OpRelationProof, RelationConfig,
};
use zkie_ops::proof::request::batt::{
    build_binding_tree, freeze_batch, open_slot, prove_shared, verify_shared, verify_slot,
    SealedMember,
};
use zkie_ops::proof::request::foldeq::{
    derive_connection, fold_value, prove_fold_equations, verify_derived_connection,
    verify_fold_equations, CommittedRow, FoldConnection,
};
use zkie_ops::proof::request::lkp::{prove_lookup, verify_lookup, LookupTable};
use zkie_ops::proof::request::output_policy::{
    policy_digest, validate_statement_policy, PolicyParams,
};
use zkie_ops::proof::request::request::{
    check_sampling_preconditions, check_weight_pass_preconditions, ColumnQuery, SealGate,
    SessionLifecycle, WeightPass, WeightPassRequest,
};
use zkie_ops::range::RangeOp;
use zkie_ops::registry::{PureForward, TempScratch};
use zkie_ops::rope_projection::RopeProjectionOp;
use zkie_ops::table_lookup::TableLookupOp;

fn chain() -> Chain {
    Chain::begin(
        REQUEST_CHAIN_LABEL,
        &Digest([1u8; 32]),
        b"protocol-mutations",
        &[],
        None,
    )
    .unwrap()
}

fn reject<T>(what: &str, result: Result<T>) {
    assert!(result.is_err(), "{what} must be rejected");
}

fn prove<T: PureForward + RelationConfig>(op: &T, inputs: &[Vec<i64>]) -> (Vec<i64>, zkie_ops::proof::relations::OpRelationPublic, OpRelationProof) {
    let config = op.relation_config(inputs).unwrap();
    let (public, proof) = prove_op_relation(op, inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();
    (config, public, proof)
}

#[test]
fn model_identity_weight_and_opening_are_bound() {
    let temp = TempScratch::new("protocol-mutations-model").unwrap();
    let path = temp.path.join("w.bin");
    let bytes: Vec<u8> = [5i32, 6, 7, 8]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fs::write(&path, bytes).unwrap();
    let spec = ModelAssetSpec {
        asset_id: 3,
        path: "w.bin".into(),
        element_count: 4,
        row_width: 2,
        encoding: "i32-le".into(),
    };
    let commitment = commit_assets(
        &temp.path,
        Digest([0x44; 32]),
        Digest([0x55; 32]),
        &[spec],
    )
    .unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let opening = open_asset(&commitment, 3, 0, 2).unwrap();
    public.verify_opening(&opening).unwrap();

    let mut bad = public.clone();
    bad.anchor.model_commitment = Digest([0x99; 32]);
    reject("model id", bad.verify_opening(&opening));

    let mut bad = public.clone();
    bad.assets[0].root = Digest([0x98; 32]);
    reject("weight digest", bad.verify_opening(&opening));

    let mut bad = opening.clone();
    bad.rows[0].values[0] += 1;
    reject("weight row", public.verify_opening(&bad));
}

#[test]
fn input_output_and_lut_mutations_are_rejected() {
    let op = TableLookupOp;
    let inputs = vec![vec![10, 20, 30], vec![2, 0, 1]];
    let (_config, public, proof) = prove(&op, &inputs);

    let mut bad = proof.clone();
    bad.inputs[0][0].values[0] += 1;
    reject("lut content", verify_op_relation(&public, &bad));

    let mut bad = proof.clone();
    bad.inputs[1][0].values[0] = 0;
    reject("lookup key", verify_op_relation(&public, &bad));

    let mut bad = proof.clone();
    bad.outputs[0][0].values[0] += 1;
    reject("output", verify_op_relation(&public, &bad));

    let op = MatMulOp { m: 2, n: 2, k: 2, shift: 0 };
    let inputs = vec![vec![1, 2, 3, 4], vec![1, 0, 0, 1]];
    let (_config, public, proof) = prove(&op, &inputs);

    let mut bad = proof.clone();
    bad.inputs[0][0].values[0] += 1;
    reject("input", verify_op_relation(&public, &bad));

    let mut bad = proof.clone();
    bad.inputs[0][0].values[0] = 9;
    bad.inputs[1][0].values[0] = 9;
    reject("matmul operands", verify_op_relation(&public, &bad));
}

#[test]
fn quantization_shape_and_padding_are_bound() {
    let op = MatMulOp { m: 2, n: 2, k: 2, shift: 0 };
    let inputs = vec![vec![1, 2, 3, 4], vec![1, 0, 0, 1]];
    let (_config, public, proof) = prove(&op, &inputs);
    let mut bad = public.clone();
    bad.config[3] = 1;
    reject("quantization shift", verify_op_relation(&bad, &proof));

    let table = vec![2i64, 2, 2, 2];
    let op = ProfileLayerNormOp {
        rows: 1,
        cols: 4,
        valid_cols: 2,
        output_shift: 32,
        clamp: None,
    };
    let inputs = vec![vec![1, 2, 3, 4], vec![1, 1, 1, 1], vec![0, 0, 0, 0], table];
    let (_config, public, proof) = prove(&op, &inputs);
    let mut bad = public.clone();
    bad.config[1] = 5;
    reject("shape columns", verify_op_relation(&bad, &proof));
    let mut bad = public.clone();
    bad.config[2] = 3;
    reject("padding valid columns", verify_op_relation(&bad, &proof));
}

#[test]
fn graph_edge_layout_and_padding_are_bound() {
    let from = TensorSpec::new(DType::I64, vec![2, 2]);
    let to = TensorSpec::new(DType::I64, vec![4]);
    let binding = bind_edge(7, TensorId(1), TensorId(2), &from, &to).unwrap();
    verify_edge(&binding, 7, TensorId(1), TensorId(2), &from, &to).unwrap();

    reject(
        "graph edge endpoint",
        verify_edge(&binding, 7, TensorId(1), TensorId(3), &from, &to),
    );
    reject(
        "graph edge id",
        verify_edge(&binding, 8, TensorId(1), TensorId(2), &from, &to),
    );

    let mut column_major = to.clone();
    column_major.layout = LayoutSpec::ColumnMajor;
    reject(
        "layout",
        verify_edge(&binding, 7, TensorId(1), TensorId(2), &from, &column_major),
    );

    let mut padded = to.clone();
    padded.padding.valid_len = Some(2);
    reject(
        "padding",
        verify_edge(&binding, 7, TensorId(1), TensorId(2), &from, &padded),
    );
}

#[test]
fn lookup_key_product_tree_and_ood_answer_are_bound() {
    let table = LookupTable::new(vec![vec![1], vec![2], vec![3]]).unwrap();
    let mut registry = OracleRegistry::new();
    let (_oracle, queries) =
        commit_rows_with_openings(&mut registry, &Digest([7; 32]), &[vec![1], vec![2]]).unwrap();

    let mut prove_chain = chain();
    let proof = prove_lookup(&mut prove_chain, &table, &queries, &registry).unwrap();
    let mut verify_chain = chain();
    verify_lookup(&mut verify_chain, &table, &queries, &registry, &proof).unwrap();

    let mut bad = proof.clone();
    bad.gamma = bad.gamma.add(&Fp3::ONE);
    reject(
        "lookup challenge",
        verify_lookup(&mut chain(), &table, &queries, &registry, &bad),
    );

    let mut bad = proof.clone();
    bad.query.levels[0][0] = bad.query.levels[0][0].add(&Fp3::ONE);
    reject(
        "product tree node",
        verify_lookup(&mut chain(), &table, &queries, &registry, &bad),
    );

    let mut bad_queries = queries.clone();
    bad_queries[0].values[0] = 3;
    reject(
        "lookup key",
        verify_lookup(
            &mut chain(),
            &table,
            &bad_queries,
            &registry,
            &proof,
        ),
    );

    let mut bad_table = table.clone();
    bad_table.rows[0][0] = 9;
    reject(
        "lut table",
        verify_lookup(
            &mut chain(),
            &bad_table,
            &queries,
            &registry,
            &proof,
        ),
    );

    let mut ood_registry = OracleRegistry::new();
    let (_queries_oracle, ood_queries) =
        commit_rows_with_openings(&mut ood_registry, &Digest([7; 32]), &[vec![1], vec![2]])
            .unwrap();
    let (oracle, openings) = commit_multiplicities(
        &mut ood_registry,
        &Digest([8; 32]),
        &table,
        &ood_queries,
        &[],
    )
    .unwrap();
    let mut prove_chain = chain();
    let ood = prove_multiplicity_ood(
        &mut prove_chain,
        &table,
        &ood_queries,
        &[],
        &ood_registry,
        oracle,
        &openings,
    )
    .unwrap();
    let mut verify_chain = chain();
    verify_multiplicity_ood(&mut verify_chain, &table, &ood_queries, &[], &ood_registry, &ood).unwrap();

    let mut bad = ood.clone();
    bad.answer = bad.answer.add(&Fp3::ONE);
    reject(
        "ood answer",
        verify_multiplicity_ood(
            &mut chain(),
            &table,
            &ood_queries,
            &[],
            &ood_registry,
            &bad,
        ),
    );
}

#[test]
fn quotient_remainder_limb_carry_and_range_are_bound() {
    let op = DivRoundOp;
    let inputs = vec![vec![7, -5], vec![2, 2]];
    let (_config, public, proof) = prove(&op, &inputs);

    let mut bad = proof.clone();
    bad.quotients[0][0].values[0] += 1;
    reject("quotient", verify_op_relation(&public, &bad));

    let mut bad = proof.clone();
    bad.outputs[0][2].values[0] += 1;
    reject("remainder", verify_op_relation(&public, &bad));

    let op = RangeOp { low: 0, high: 9 };
    let inputs = vec![vec![5, -1, 10]];
    let (_config, public, proof) = prove(&op, &inputs);

    let mut bad = proof.clone();
    bad.helpers[0][0].values[0] = 0;
    reject("range helper", verify_op_relation(&public, &bad));

    let mut bad = proof.clone();
    bad.outputs[0][1].values[0] = 1;
    reject("range output", verify_op_relation(&public, &bad));

    let mut registry = OracleRegistry::new();
    let label = Digest([9; 32]);
    let (_, numerators) =
        commit_rows_with_openings(&mut registry, &label, &[vec![7]]).unwrap();
    let (_, denominators) =
        commit_rows_with_openings(&mut registry, &label, &[vec![2]]).unwrap();
    let (_, quotients) =
        commit_rows_with_openings(&mut registry, &label, &[vec![3]]).unwrap();
    let (_, remainders) =
        commit_rows_with_openings(&mut registry, &label, &[vec![1]]).unwrap();
    let vectors = IntegerVectors {
        numerators: vec![7],
        denominators: vec![2],
        quotients: vec![3],
        remainders: vec![1],
    };
    let openings = IntegerProof {
        numerators,
        denominators,
        quotients,
        remainders,
        limbs: Vec::new(),
        carries: Vec::new(),
        base_bits: 8,
    };
    let integer = prove_integer_relations(&vectors, &openings, &registry).unwrap();
    verify_integer_relations(&integer, &registry).unwrap();

    let mut bad = integer.clone();
    bad.carries[0] = 1;
    reject("carry", verify_integer_relations(&bad, &registry));

    let mut bad = integer.clone();
    bad.limbs[0] += 1;
    reject("limb", verify_integer_relations(&bad, &registry));
}

#[test]
fn kv_address_and_history_are_bound() {
    let op = KvCacheOp {
        capacity: 2,
        width: 2,
        position: 1,
    };
    let inputs = vec![
        vec![1, 2, 3, 4],
        vec![5, 6, 7, 8],
        vec![9, 10],
        vec![11, 12],
    ];
    let (_config, public, proof) = prove(&op, &inputs);

    let mut bad = public.clone();
    bad.config[2] = 0;
    reject("kv address", verify_op_relation(&bad, &proof));

    let mut bad = proof.clone();
    bad.inputs[0][0].values[0] += 1;
    reject("kv history", verify_op_relation(&public, &bad));

    let mut bad = proof.clone();
    bad.outputs[0][0].values[0] += 1;
    reject("kv write", verify_op_relation(&public, &bad));
}

#[test]
fn rope_offset_and_causal_mask_are_bound() {
    let op = RopeProjectionOp { shift: 16 };
    let inputs = vec![vec![65536, 0], vec![1], vec![0]];
    let (_config, public, proof) = prove(&op, &inputs);

    let mut bad = proof.clone();
    bad.inputs[1][0].values[0] = 0;
    reject("rope offset", verify_op_relation(&public, &bad));

    let mut bad = public.clone();
    bad.config[0] = 15;
    reject("rope shift", verify_op_relation(&bad, &proof));

    let op = CausalAttentionOp {
        seq: 2,
        head_dim: 2,
        shift: 0,
    };
    let inputs = vec![
        vec![1, 0, 0, 1],
        vec![1, 0, 0, 1],
        vec![1, 2, 3, 4],
        vec![1, 2, 3, 4],
    ];
    let (_config, public, proof) = prove(&op, &inputs);

    let mut bad = proof.clone();
    bad.outputs[0][2].values[0] += 1;
    reject("causal mask output", verify_op_relation(&public, &bad));

    let mut bad = proof.clone();
    bad.aux[1][0].values[0] += 1;
    reject("causal denominator terms", verify_op_relation(&public, &bad));
}

#[test]
fn request_seal_and_batch_order_are_bound() {
    let mut lifecycle = SessionLifecycle::new();
    reject("seal before start", lifecycle.seal_request());
    lifecycle.start_request().unwrap();
    reject("freeze before seal", lifecycle.freeze_batch());
    lifecycle.seal_request().unwrap();
    lifecycle.freeze_batch().unwrap();
    lifecycle.begin_shared().unwrap();
    reject("finalize before shared seal", lifecycle.begin_final());
    lifecycle.seal_shared().unwrap();
    lifecycle.begin_final().unwrap();
    reject("finalize with open claims", lifecycle.finalize());
    lifecycle.close_claims().unwrap();
    lifecycle.finalize().unwrap();

    struct Gate(bool);
    impl SealGate for Gate {
        fn is_sealed(&self, _seal: Digest) -> bool {
            self.0
        }
    }
    let query = ColumnQuery {
        oracle: OracleId(1),
        depends_on_seals: vec![Digest([1; 32])],
        count: 1,
    };
    reject(
        "sampling with unsealed dependency",
        check_sampling_preconditions(&query, &Gate(false), true),
    );
    reject(
        "sampling before commitment",
        check_sampling_preconditions(&query, &Gate(true), false),
    );
    check_sampling_preconditions(&query, &Gate(true), true).unwrap();

    let request = WeightPassRequest {
        pass: WeightPass::Second,
        layer_start: 0,
        layer_end: 1,
        tensor_ids: Vec::new(),
        depends_on_seals: vec![Digest([2; 32])],
    };
    reject(
        "second pass before seal",
        check_weight_pass_preconditions(&request, &Gate(false)),
    );
    check_weight_pass_preconditions(&request, &Gate(true)).unwrap();
}

#[test]
fn shared_result_and_binding_slot_are_bound() {
    let members = vec![
        SealedMember {
            request_id: Digest([1; 32]),
            seal: Digest([2; 32]),
            claims: vec![(RelationId(1), Fp3::from_u64(5))],
        },
        SealedMember {
            request_id: Digest([3; 32]),
            seal: Digest([4; 32]),
            claims: vec![(RelationId(1), Fp3::from_u64(7))],
        },
    ];
    reject(
        "duplicate batch member",
        freeze_batch(vec![members[0].clone(), members[0].clone()]),
    );
    let batch = freeze_batch(members.clone()).unwrap();
    let mut prove_chain = chain();
    let receipt = prove_shared(&mut prove_chain, &batch).unwrap();
    let mut verify_chain = chain();
    verify_shared(
        &mut verify_chain,
        &batch.root,
        &batch,
        &receipt,
        &[RelationId(1)],
    )
    .unwrap();

    let mut bad = receipt.clone();
    bad.combined[0].1 = bad.combined[0].1.add(&Fp3::ONE);
    reject(
        "shared result",
        verify_shared(
            &mut chain(),
            &batch.root,
            &batch,
            &bad,
            &[RelationId(1)],
        ),
    );

    let mut bad = receipt.clone();
    bad.root = Digest([9; 32]);
    reject(
        "shared root",
        verify_shared(
            &mut chain(),
            &batch.root,
            &batch,
            &bad,
            &[RelationId(1)],
        ),
    );

    let mut reversed = members.clone();
    reversed.reverse();
    let reordered = freeze_batch(reversed).unwrap();
    reject(
        "batch order",
        verify_shared(
            &mut chain(),
            &batch.root,
            &reordered,
            &receipt,
            &[RelationId(1)],
        ),
    );

    let configuration = Digest([5; 32]);
    let padding: Vec<Digest> = (0..4).map(|index| Digest([index as u8; 32])).collect();
    let seal = Digest([6; 32]);
    let salt = vec![7u8; 40];
    let tree =
        build_binding_tree(2, configuration, &[(0, seal, salt.clone())], &padding).unwrap();
    let proof = open_slot(&tree, 0, &seal, salt).unwrap();
    verify_slot(&tree.root, &configuration, &proof).unwrap();

    let mut bad = proof.clone();
    bad.seal = Digest([8; 32]);
    reject(
        "binding seal",
        verify_slot(&tree.root, &configuration, &bad),
    );

    let mut bad = proof.clone();
    bad.path[0] = Digest([9; 32]);
    reject(
        "binding path",
        verify_slot(&tree.root, &configuration, &bad),
    );
}

#[test]
fn fold_connection_and_terminal_relation_are_bound() {
    let mut registry = OracleRegistry::new();
    let (low_oracle, low_openings) =
        commit_rows_with_openings(&mut registry, &Digest([10; 32]), &[vec![1]]).unwrap();
    let (high_oracle, high_openings) =
        commit_rows_with_openings(&mut registry, &Digest([11; 32]), &[vec![2]]).unwrap();
    let oracle = low_oracle;
    let connection = FoldConnection {
        relation: RelationId(1),
        oracle,
        low: Fp3::from_u64(1),
        high: Fp3::from_u64(2),
        challenge: Fp3::from_u64(3),
        folded: fold_value(Fp3::from_u64(1), Fp3::from_u64(2), Fp3::from_u64(3)),
    };
    let proof = prove_fold_equations(vec![connection]).unwrap();
    verify_fold_equations(&proof, &[RelationId(1)], &registry).unwrap();

    let mut bad = proof.clone();
    bad.connections[0].folded = bad.connections[0].folded.add(&Fp3::ONE);
    reject(
        "fold connection",
        verify_fold_equations(&bad, &[RelationId(1)], &registry),
    );

    let mut bad = proof.clone();
    bad.connections.clear();
    reject(
        "fold connection count",
        verify_fold_equations(&bad, &[RelationId(1)], &registry),
    );

    reject(
        "fold relation order",
        verify_fold_equations(&proof, &[RelationId(2)], &registry),
    );

    let low = CommittedRow {
        oracle: low_oracle,
        coefficients: vec![Fp3::ONE],
        openings: low_openings,
    };
    let high = CommittedRow {
        oracle: high_oracle,
        coefficients: vec![Fp3::ONE],
        openings: high_openings,
    };
    let derived =
        derive_connection(RelationId(1), &low, &high, Fp3::from_u64(4), &registry).unwrap();
    verify_derived_connection(&derived, &low, &high, &registry).unwrap();
    let mut bad = derived.clone();
    bad.high = bad.high.add(&Fp3::ONE);
    reject(
        "derived fold connection",
        verify_derived_connection(&bad, &low, &high, &registry),
    );

    let shape = FoldShape::new(2, 1, 4).unwrap();
    let inner = |left: &[Fp3], right: &[Fp3]| {
        let mut sum = Fp3::ZERO;
        for (a, b) in left.iter().zip(right) {
            sum = sum.add(&a.mul(b));
        }
        sum
    };
    let main: Vec<Fp3> = (1..=8u64).map(Fp3::from_u64).collect();
    let coefficients_main: Vec<Fp3> = (1..=8u64).map(|value| Fp3::from_u64(value * 2 + 1)).collect();
    let mask: Vec<Fp3> = (1..=8u64).map(|value| Fp3::from_u64(value + 20)).collect();
    let coefficients_mask: Vec<Fp3> =
        (1..=8u64).map(|value| Fp3::from_u64(value + 40)).collect();
    let kept = Fp3::from_u64(3);
    let target = inner(&coefficients_main, &main)
        .add(&inner(&coefficients_mask, &mask))
        .add(&kept);
    let claim = FoldClaim {
        main_rows: 4,
        main_cols: 2,
        main,
        coefficients_main,
        mask,
        coefficients_mask,
        kept,
        target,
    };
    let coins: Vec<Fp3> = (1..=5u64).map(Fp3::from_u64).collect();
    let mut prove_chain = chain();
    let receipt = prove_fold(&shape, &claim, &coins, &mut prove_chain).unwrap();
    let mut verify_chain = chain();
    verify_fold(&shape, target, &receipt, &mut verify_chain).unwrap();

    let mut bad = receipt.clone();
    bad.output.target = bad.output.target.add(&Fp3::ONE);
    reject(
        "terminal target",
        verify_fold(&shape, target, &bad, &mut chain()),
    );

    let mut bad = receipt.clone();
    bad.output.coefficients_main[0] = bad.output.coefficients_main[0].add(&Fp3::ONE);
    reject(
        "terminal equation",
        verify_fold(&shape, target, &bad, &mut chain()),
    );
}

#[test]
fn pow_version_parameters_and_output_policy_are_bound() {
    let mut chain_state = chain();
    let prefix = chain_state.message().prefix(KIND_SEAL, 8).unwrap();
    let nonce = search_nonce(&prefix, 1 << 20).unwrap();
    assert!(verify_nonce(&prefix, nonce, 8).unwrap());
    let mut rejected = 0;
    for offset in 1..=16u64 {
        if !verify_nonce(&prefix, nonce.wrapping_add(offset), 8).unwrap_or(false) {
            rejected += 1;
        }
    }
    assert!(rejected > 0, "wrong nonce was accepted for every candidate");
    reject("pow work bits", work_test(0, 65));

    let mut statement = sample_statement();
    assert!(statement.validate().is_ok());
    statement.protocol_version = 0;
    reject("protocol version", statement.validate());
    statement.protocol_version = 1;
    statement.model_id = Digest::ZERO;
    reject("statement model id", statement.validate());
    statement.model_id = Digest([3; 32]);
    statement.parameter_digest = Digest::ZERO;
    reject("statement parameters", statement.validate());

    let params = PolicyParams::exact_match(7).unwrap();
    assert!(params.validate().is_ok());
    let mut bad_params = params;
    bad_params.theta = bad_params.c.saturating_add(1);
    reject("policy parameters", bad_params.validate());
    reject("policy length", PolicyParams::decode(&[0u8; 7]));

    let clipped_params = PolicyParams::clipped(64, 32, 0, 0, 7, 3).unwrap();
    let clipped = ClippedGapPolicy {
        parameters: clipped_params.encode().to_vec(),
        parameters_digest: policy_digest(&OutputPolicy::ClippedGap(ClippedGapPolicy {
            parameters: clipped_params.encode().to_vec(),
            parameters_digest: Digest::ZERO,
        }))
        .unwrap(),
    };
    let mut statement = sample_statement();
    statement.output_policy = OutputPolicy::ClippedGap(clipped);
    validate_statement_policy(&statement).unwrap();
    if let OutputPolicy::ClippedGap(ref mut policy) = statement.output_policy {
        policy.parameters_digest = Digest([0x77; 32]);
    }
    reject(
        "output policy digest",
        validate_statement_policy(&statement),
    );
}

fn sample_statement() -> Statement {
    Statement {
        protocol_id: Digest([1; 32]),
        protocol_version: 1,
        parameter_digest: Digest([2; 32]),
        model_id: Digest([3; 32]),
        graph_digest: Digest([4; 32]),
        quantization_digest: Digest([5; 32]),
        tables_digest: Digest([6; 32]),
        prompt_tokens: vec![0, 1],
        output_tokens: vec![2, 3],
        coverage: DecodeCoverage::full(2),
        eos_policy: EosPolicy::FixedLength,
        output_policy: OutputPolicy::ExactMatch,
    }
}


#[test]
fn seq512_relations_reject_row_weight_and_parameter_tampering() {
    use zkie_ops::matmul::MatMulOp;
    use zkie_ops::proof::relations::{prove_op_relation, verify_op_relation, RelationConfig};
    use zkie_ops::softmax_rows::SoftmaxRowsOp;

    let rows = 512usize;
    let cols = 512usize;
    let mut x = vec![0i64; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            x[row * cols + col] = ((row * 37 + col * 11) % 17) as i64;
        }
    }
    let table: Vec<i64> = (0..1024).map(|index| (index as i64 % 100) + 1).collect();
    let op = SoftmaxRowsOp { rows, cols, shift: 16 };
    let inputs = vec![x, table];
    let config = op.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&op, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();

    let mut bad = proof.clone();
    bad.outputs[0][511].values[0] += 1;
    assert!(verify_op_relation(&public, &bad).is_err());

    let mut bad = proof.clone();
    bad.inputs[1][0].values[0] += 1;
    assert!(verify_op_relation(&public, &bad).is_err());

    let mut bad = public.clone();
    bad.config[1] = 511;
    assert!(verify_op_relation(&bad, &proof).is_err());

    let op = MatMulOp {
        m: 512,
        n: 2,
        k: 2,
        shift: 0,
    };
    let inputs = vec![vec![1i64; 512 * 2], vec![1, 0, 0, 1]];
    let config = op.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&op, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();
    let mut bad = proof.clone();
    bad.inputs[1][0].values[0] += 1;
    assert!(verify_op_relation(&public, &bad).is_err());
}
