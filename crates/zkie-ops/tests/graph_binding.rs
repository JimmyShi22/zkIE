//! Graph edges are bound by exact tensor ids, layout, padding and spec.

use std::collections::BTreeMap;

use zkie_core::types::{DType, Digest, LayoutSpec, PaddingSpec, TensorId, TensorSpec};

const TEST_PREFIX: Digest = Digest([0x5C; 32]);

use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::storage::EngineStorage;

use zkie_ops::execute::{GraphExecutor, OperationRegistry, SequentialExecutor};
use zkie_ops::graph::{GraphNode, GraphPlan};
use zkie_ops::op::{OpCode, OpId};
use zkie_ops::proof::binding::{bind_edge, graph_binding_digest, verify_edge};
use zkie_ops::registry::{
    read_i64_view, spec_1d, write_i64_scratch, NullProgress, StorageOnlyPort, TempScratch,
};
use zkie_ops::relu::ReluOp;

#[test]
fn edges_bind_ids_layout_and_padding() {
    let input = TensorId(1);
    let output = TensorId(2);
    let other = TensorId(3);
    let from_spec = TensorSpec::new(DType::I64, vec![2, 2]);
    let to_spec = TensorSpec::new(DType::I64, vec![4]);
    let binding = bind_edge(7, input, output, &from_spec, &to_spec).unwrap();
    verify_edge(&binding, 7, input, output, &from_spec, &to_spec).unwrap();

    // Same content, different id: not the same edge.
    assert!(verify_edge(&binding, 7, input, other, &from_spec, &to_spec).is_err());
    assert!(verify_edge(&binding, 8, input, output, &from_spec, &to_spec).is_err());

    // Layout changes the binding.
    let mut column_major = to_spec.clone();
    column_major.layout = LayoutSpec::ColumnMajor;
    assert!(verify_edge(&binding, 7, input, output, &from_spec, &column_major).is_err());

    // Padding changes the binding.
    let mut padded = to_spec.clone();
    padded.padding = PaddingSpec {
        value: 5,
        valid_len: Some(3),
        mask_digest: None,
    };
    assert!(verify_edge(&binding, 7, input, output, &from_spec, &padded).is_err());

    let changed = bind_edge(7, input, output, &from_spec, &padded).unwrap();
    assert_ne!(graph_binding_digest(&[binding]), graph_binding_digest(&[changed]));
}

struct Registry {
    relu: ReluOp,
}

impl OperationRegistry<Fp3> for Registry {
    fn operation(&self, opcode: OpCode) -> Option<&dyn zkie_ops::op::Operation<Fp3>> {
        if opcode == OpCode::Relu {
            Some(&self.relu)
        } else {
            None
        }
    }
}

#[test]
fn sequential_executor_uses_exact_graph_edges() {
    let temp = TempScratch::new("graph").unwrap();
    let storage = EngineStorage::new(&temp.path).unwrap();
    let governor = BudgetedMemoryGovernor::new(1 << 20);
    let input_spec = spec_1d(3);
    let output_spec = spec_1d(3);
    let input = TensorId(1);
    let output = TensorId(2);

    let mut graph = GraphPlan::new();
    graph.add_tensor(input, input_spec.clone()).unwrap();
    graph.add_tensor(output, output_spec.clone()).unwrap();
    let mut node = GraphNode::new(OpId(1), OpCode::Relu);
    node.inputs = vec![input];
    node.outputs = vec![output];
    graph.add_node(node).unwrap();

    let input_view =
        write_i64_scratch(&storage, &governor, &input_spec, &[-3, 0, 4]).unwrap();
    let mut inputs = BTreeMap::new();
    inputs.insert(input, input_view);
    let registry = Registry { relu: ReluOp };
    let progress = NullProgress;
    let mut port = StorageOnlyPort::<Fp3>::new(&storage, &governor, &progress);
    let mut executor = SequentialExecutor::default();
    let outcome = executor
        .execute(&graph, &registry, &mut port, &inputs)
        .unwrap();
    assert_eq!(executor.trace.len(), 1);
    let values = read_i64_view(&storage, outcome.outputs.get(&output).unwrap()).unwrap();
    assert_eq!(values, vec![0, 0, 4]);
}

#[test]
fn op_relations_are_checked_without_forward() {
    use zkie_ops::add::AddOp;
    use zkie_ops::matmul::MatMulOp;
    use zkie_ops::projection::ProjectionOp;
    use zkie_ops::proof::relations::{prove_op_relation, verify_op_relation};
    use zkie_ops::relu::ReluOp;

    let matmul = MatMulOp { m: 2, n: 2, k: 2, shift: 0 };
    let (public, proof) = prove_op_relation(
        &matmul,
        &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
        &[2, 2, 2, 0, 0],
    )
    .unwrap();
    verify_op_relation(&public, &proof).unwrap();

    let mut bad_output = proof.clone();
    bad_output.outputs[0][0].values[0] += 1;
    assert!(verify_op_relation(&public, &bad_output).is_err());

    let mut bad_input = proof.clone();
    bad_input.inputs[0][0].values[0] += 1;
    assert!(verify_op_relation(&public, &bad_input).is_err());

    let mut bad_raw = proof.clone();
    bad_raw.raw[0][0].values[0] += 1;
    assert!(verify_op_relation(&public, &bad_raw).is_err());

    let mut bad_config = public.clone();
    bad_config.config[3] = 1;
    assert!(verify_op_relation(&bad_config, &proof).is_err());

    let projection = ProjectionOp {
        m: 2,
        n: 2,
        k: 2,
        shift: 0,
        bias: Some(vec![1, 2, 3, 4]),
    };
    let (public, proof) = prove_op_relation(
        &projection,
        &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
        &[2, 2, 2, 0, 4, 1, 2, 3, 4],
    )
    .unwrap();
    verify_op_relation(&public, &proof).unwrap();
    let mut bad_bias = public.clone();
    bad_bias.config[4] += 1;
    assert!(verify_op_relation(&bad_bias, &proof).is_err());

    let relu = ReluOp;
    let (public, proof) = prove_op_relation(&relu, &[vec![-3, 0, 4]], &[]).unwrap();
    verify_op_relation(&public, &proof).unwrap();
    let mut bad_relu = proof.clone();
    bad_relu.outputs[0][0].values[0] = 1;
    assert!(verify_op_relation(&public, &bad_relu).is_err());

    let add = AddOp;
    let (public, proof) =
        prove_op_relation(&add, &[vec![1, 2, 3, 4], vec![4, 3, 2, 1]], &[]).unwrap();
    verify_op_relation(&public, &proof).unwrap();
}

#[test]
fn every_operator_relation_closes_and_rejects_output_tampering() {
    use zkie_ops::affine_scale::AffineScaleOp;
    use zkie_ops::argmax::ArgmaxOp;
    use zkie_ops::batched_matmul::BatchedMatMulOp;
    use zkie_ops::causal_attention::CausalAttentionOp;
    use zkie_ops::div_round::DivRoundOp;
    use zkie_ops::gated_ffn::GatedFfnOp;
    use zkie_ops::gather::GatherOp;
    use zkie_ops::gather_matmul::GatherMatMulOp;
    use zkie_ops::hadamard::HadamardOp;
    use zkie_ops::index_compose::IndexComposeOp;
    use zkie_ops::int_sqrt::IntSqrtOp;
    use zkie_ops::kv_cache::KvCacheOp;
    use zkie_ops::layer_norm::LayerNormOp;
    use zkie_ops::layout::{LayoutKind, LayoutOp};
    use zkie_ops::moe_down::MoeDownOp;
    use zkie_ops::pad::PadOp;
    use zkie_ops::patchify::PatchifyOp;
    use zkie_ops::proof::relations::{
        prove_op_relation, verify_op_relation, RelationConfig,
    };
    use zkie_ops::range::RangeOp;
    use zkie_ops::regression_head::RegressionHeadOp;
    use zkie_ops::requantize::RequantizeOp;
    use zkie_ops::rms_norm::RmsNormOp;
    use zkie_ops::rope_projection::RopeProjectionOp;
    use zkie_ops::route_gate::RouteGateOp;
    use zkie_ops::scatter::ScatterOp;
    use zkie_ops::softmax::SoftmaxOp;
    use zkie_ops::table_lookup::TableLookupOp;
    use zkie_ops::topk::TopKOp;
    use zkie_ops::registry::PureForward;

    fn close<T: PureForward + RelationConfig>(op: &T, inputs: &[Vec<i64>]) {
        let config = op.relation_config(inputs).unwrap();
        let (public, proof) = prove_op_relation(op, inputs, &config).unwrap();
        verify_op_relation(&public, &proof).unwrap_or_else(|error| panic!("op {:?}: {:?}", op.opcode(), error));
        let mut tampered = proof.clone();
        tampered.outputs[0][0].values[0] = tampered.outputs[0][0].values[0].wrapping_add(1);
        assert!(
            verify_op_relation(&public, &tampered).is_err(),
            "op {:?} must reject a tampered output row",
            op.opcode()
        );
    }

    close(
        &LayoutOp {
            kind: LayoutKind::Transpose2d,
            from: vec![2, 2],
            to: vec![2, 2],
        },
        &[vec![1, 2, 3, 4]],
    );
    close(
        &LayoutOp {
            kind: LayoutKind::Reshape,
            from: vec![2, 2],
            to: vec![4],
        },
        &[vec![1, 2, 3, 4]],
    );
    close(&RequantizeOp { shift: 1 }, &[vec![3, -3, 5]]);
    close(
        &AffineScaleOp {
            multiplier: 2,
            addend: 1,
            shift: 1,
            clamp: Some((-2, 10)),
        },
        &[vec![1, 2, -3]],
    );
    close(&zkie_ops::relu::ReluOp, &[vec![-3, 0, 4]]);
    close(
        &zkie_ops::matmul::MatMulOp { m: 2, n: 2, k: 2, shift: 0 },
        &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
    );
    close(
        &zkie_ops::projection::ProjectionOp {
            m: 2,
            n: 2,
            k: 2,
            shift: 0,
            bias: Some(vec![1, 2, 3, 4]),
        },
        &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
    );
    close(&zkie_ops::add::AddOp, &[vec![1, 2, 3, 4], vec![4, 3, 2, 1]]);
    close(
        &TableLookupOp,
        &[vec![10, 20, 30], vec![2, 0, 1]],
    );
    close(
        &SoftmaxOp { shift: 4 },
        &[vec![1, 2, 3], vec![1, 1, 1]],
    );
    close(
        &LayerNormOp {
            rows: 1,
            cols: 2,
            valid_cols: 2,
            shift: 1,
            clamp: None,
        },
        &[vec![4, 6], vec![1, 1]],
    );
    close(
        &RmsNormOp {
            rows: 1,
            cols: 2,
            valid_cols: 2,
            shift: 1,
            clamp: None,
        },
        &[vec![2, 2], vec![1, 1]],
    );
    close(
        &RopeProjectionOp { shift: 0 },
        &[vec![1, 2], vec![1], vec![0]],
    );
    close(&TopKOp { k: 2 }, &[vec![3, 1, 2, 4]]);
    close(&ArgmaxOp, &[vec![1, 5, 3]]);
    close(
        &RouteGateOp { top_k: 2, threshold: 2 },
        &[vec![5, 1, 4, 3]],
    );
    close(
        &ScatterOp { size: 4 },
        &[vec![1, 3], vec![7, 9]],
    );
    close(
        &GatherOp { cols: 2 },
        &[vec![1, 2, 3, 4, 5, 6], vec![2, 0]],
    );
    close(
        &BatchedMatMulOp { batch: 2, m: 1, n: 1, k: 2, shift: 0 },
        &[vec![1, 2, 3, 4], vec![1, 1, 2, 2]],
    );
    close(
        &GatherMatMulOp { cols: 2, n: 2, shift: 0 },
        &[vec![1, 2, 3, 4], vec![1], vec![1, 0, 0, 1]],
    );
    close(&IndexComposeOp, &[vec![2, 0], vec![7, 8, 9]]);
    close(&PadOp { valid_len: 2, value: 5 }, &[vec![1, 2, 3, 4]]);
    close(
        &HadamardOp { shift: 1 },
        &[vec![2, 3], vec![4, 5]],
    );
    close(&RangeOp { low: 0, high: 2 }, &[vec![-1, 0, 3]]);
    close(
        &DivRoundOp,
        &[vec![7, -7], vec![3, 3]],
    );
    close(&IntSqrtOp, &[vec![0, 1, 8, 9]]);
    close(
        &GatedFfnOp { shift: 1 },
        &[vec![2, 3], vec![3, 4]],
    );
    close(
        &MoeDownOp {
            shared_weights: vec![1, 2],
            routed_weights: vec![3, 4],
            width: 2,
            out_cols: 1,
            shift: 0,
        },
        &[vec![1, 1], vec![1, 1]],
    );
    close(
        &KvCacheOp { capacity: 2, width: 2, position: 1 },
        &[vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![30, 40], vec![50, 60]],
    );
    close(&PatchifyOp { patch: 2 }, &[vec![1, 2, 3, 4]]);
    close(
        &RegressionHeadOp {
            m: 1,
            n: 2,
            k: 2,
            shift: 0,
            bias: Some(vec![1, 2]),
        },
        &[vec![1, 2], vec![3, 4, 5, 6]],
    );
    close(
        &CausalAttentionOp { seq: 2, head_dim: 2, shift: 0 },
        &[
            vec![1, 2, 3, 4],
            vec![1, 0, 0, 1],
            vec![10, 20, 30, 40],
            vec![1, 1, 1],
        ],
    );
}

#[test]
fn matrix_weights_use_vector_leaves_not_element_openings() {
    use zkie_ops::matmul::MatMulOp;
    use zkie_ops::proof::relations::{prove_op_relation, verify_op_relation};

    let op = MatMulOp { m: 4, n: 4, k: 4, shift: 0 };
    let a = vec![1i64; 16];
    let b = vec![2i64; 16];
    let (public, proof) = prove_op_relation(&op, &[a, b], &[4, 4, 4, 0, 0]).unwrap();
    assert_eq!(proof.inputs[0].len(), 4);
    assert_eq!(proof.inputs[1].len(), 4);
    assert!(proof.inputs[1].iter().all(|opening| opening.values.len() == 4));
    assert_eq!(proof.outputs[0].len(), 4);
    verify_op_relation(&public, &proof).unwrap();

    let mut tampered = proof.clone();
    tampered.inputs[1][0].values[0] += 1;
    assert!(verify_op_relation(&public, &tampered).is_err());
}

#[test]
fn profile_norm_and_index_generation_relations_close() {
    use zkie_ops::index_gen::{EmbeddingIndexOp, TableIndexOp};
    use zkie_ops::profile_norm::ProfileLayerNormOp;
    use zkie_ops::proof::relations::{prove_op_relation, verify_op_relation, RelationConfig};

    let mut table = vec![0i64; 8];
    table[0] = 2;
    let norm = ProfileLayerNormOp {
        rows: 1,
        cols: 2,
        valid_cols: 2,
        output_shift: 0,
        clamp: None,
    };
    let inputs = vec![
        vec![4, 6],
        vec![1, 1],
        vec![0, 0],
        table,
    ];
    let config = norm.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&norm, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();
    assert_eq!(public.output_oracles.len(), 1);

    let index = TableIndexOp {
        offset: 1 << 3,
        table_len: 1 << 4,
    };
    let inputs = vec![vec![-10, 0, 4, 100]];
    let config = index.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&index, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();

    let embedding = EmbeddingIndexOp {
        hidden: 2,
        vocab: 3,
    };
    let inputs = vec![vec![0, 2]];
    let config = embedding.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&embedding, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();

    let mut tampered = proof.clone();
    tampered.outputs[0][0].values[0] += 1;
    assert!(verify_op_relation(&public, &tampered).is_err());
}

#[test]
fn model_commitment_shares_asset_openings_across_intervals() {
    use zkie_ops::proof::model_commitment::{
        commit_assets, open_asset, verify_asset_opening, ModelAssetSpec,
    };

    let temp = TempScratch::new("model-commitment").unwrap();
    let root = temp.path.clone();
    let write = |name: &str, values: &[i32]| {
        let path = root.join(name);
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    };
    write("a.bin", &[1, 2, 3, 4, 5, 6, 7, 8]);
    write("b.bin", &[9, 10, 11, 12]);
    let specs = vec![
        ModelAssetSpec {
            asset_id: 0,
            path: "a.bin".into(),
            element_count: 8,
            row_width: 4,
            encoding: "i32-le".into(),
        },
        ModelAssetSpec {
            asset_id: 1,
            path: "b.bin".into(),
            element_count: 4,
            row_width: 2,
            encoding: "i32-le".into(),
        },
    ];
    let commitment =
        commit_assets(&root, Digest([0x11; 32]), Digest([0x22; 32]), &specs).unwrap();
    let opening = open_asset(&commitment, 0, 0, 2).unwrap();
    verify_asset_opening(&commitment, &opening).unwrap();

    let mut tampered = opening.clone();
    tampered.rows[0].values[0] = 99;
    assert!(verify_asset_opening(&commitment, &tampered).is_err());

    let mut moved = opening.clone();
    moved.root = Digest([0x33; 32]);
    assert!(verify_asset_opening(&commitment, &moved).is_err());

    let second = open_asset(&commitment, 1, 1, 1).unwrap();
    verify_asset_opening(&commitment, &second).unwrap();
    let mut link = second.clone();
    link.link.registry_index = 0;
    assert!(verify_asset_opening(&commitment, &link).is_err());
}

#[test]
fn asset_store_reads_ranges_and_rejects_changed_backing_file() {
    use zkie_core::types::Digest;
    use zkie_ops::proof::model_commitment::{
        commit_assets, verify_asset_opening, AssetStore, ModelAssetSpec,
    };

    let temp = TempScratch::new("asset-store-ranges").unwrap();
    let path = temp.path.join("a.bin");
    let bytes: Vec<u8> = [1i32, 2, 3, 4, 5, 6, 7, 8]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(&path, &bytes).unwrap();
    let spec = ModelAssetSpec {
        asset_id: 0,
        path: "a.bin".into(),
        element_count: 8,
        row_width: 4,
        encoding: "i32-le".into(),
    };
    let commitment = commit_assets(
        &temp.path,
        Digest([0x66; 32]),
        Digest([0x77; 32]),
        &[spec],
    )
    .unwrap();

    {
        let store = AssetStore::open(&commitment).unwrap();
        assert_eq!(store.value_range("a.bin", 1, 3).unwrap(), vec![2, 3, 4]);
        assert_eq!(
            store.rows("a.bin", 4, 0, 2).unwrap(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(store.columns("a.bin", 4, 1, 3).unwrap(), vec![2, 3, 6, 7]);
        assert_eq!(store.value("a.bin").unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let opening = store.open_asset(0).unwrap();
        verify_asset_opening(&commitment, &opening).unwrap();
    }

    let mut changed = bytes.clone();
    changed[7 * 4] = 99;
    std::fs::write(&path, &changed).unwrap();
    let store = AssetStore::open(&commitment).unwrap();
    assert_eq!(store.value_range("a.bin", 7, 1).unwrap(), vec![99]);
    let opening = store.open_asset(0).unwrap();
    assert!(verify_asset_opening(&commitment, &opening).is_err());
}

#[test]
fn shared_external_inputs_avoid_local_weight_openings() {
    use zkie_ops::matmul::MatMulOp;
    use zkie_ops::proof::relations::{
        prove_op_relation_shared, verify_op_relation_with_external,
    };

    let op = MatMulOp { m: 2, n: 2, k: 2, shift: 0 };
    let a = vec![1i64, 2, 3, 4];
    let b = vec![5i64, 6, 7, 8];
    let (public, proof) =
        prove_op_relation_shared(&op, &[a.clone(), b.clone()], &[2, 2, 2, 0, 0], &[1], &TEST_PREFIX).unwrap();
    assert!(proof.inputs[1].is_empty());
    let external = vec![None, Some(b.clone())];
    verify_op_relation_with_external(&public, &proof, &public.config, &external).unwrap();

    let mut wrong = external.clone();
    wrong[1] = Some(vec![5, 6, 7, 9]);
    assert!(verify_op_relation_with_external(&public, &proof, &public.config, &wrong).is_err());
    assert!(verify_op_relation_with_external(&public, &proof, &public.config, &[]).is_err());
    assert_eq!(a.len(), 4);
}

#[test]
fn shared_trace_references_one_model_asset_opening() {
    use zkie_core::types::Digest;
    use zkie_ops::matmul::MatMulOp;
    use zkie_ops::proof::model_commitment::{
        commit_assets, open_asset, verify_shared_trace,
        ModelAssetSpec, ModelCommitmentPublic, SharedTrace, SharedTraceStep,
    };
    use zkie_ops::proof::relations::ExternalAssetRef;
    use zkie_ops::proof::relations::{prove_op_relation_shared, RelationReceiptMessage};
    use zkie_ops::registry::PureForward;

    let temp = TempScratch::new("shared-trace").unwrap();
    let path = temp.path.join("b.bin");
    let bytes: Vec<u8> = [5i32, 6, 7, 8]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(&path, bytes).unwrap();
    let spec = ModelAssetSpec {
        asset_id: 3,
        path: "b.bin".into(),
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

    let op = MatMulOp { m: 2, n: 2, k: 2, shift: 0 };
    let a = vec![1i64, 2, 3, 4];
    let (relation_public, proof) = prove_op_relation_shared(
        &op,
        &[a, vec![5, 6, 7, 8]],
        &[2, 2, 2, 0, 0],
        &[1],
        &TEST_PREFIX,
    )
    .unwrap();
    zkie_ops::proof::relations::verify_op_relation_with_external(
        &relation_public,
        &proof,
        &[2, 2, 2, 0, 0],
        &[None, Some(vec![5, 6, 7, 8])],
    )
    .unwrap();
    let mut roots = Vec::new();
    for id in relation_public.registry.ids() {
        let root = relation_public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    let step = SharedTraceStep {
        message: RelationReceiptMessage {
            opcode: op.opcode() as u16,
            config: vec![2, 2, 2, 0, 0],
            roots,
            external_inputs: vec![1],
            proof,
        },
        external: vec![
            None,
            Some(ExternalAssetRef::rows(3, 0, 2)),
        ],
        public_luts: vec![None, None],
        sources: vec![None, None],
    };
    let mut trace = SharedTrace {
        tokens: vec![0],
        assets: vec![opening.clone()],
        steps: vec![step],
        argmax: vec![0],
        output: Vec::new(),
    };
    public.verify_opening(&opening).unwrap();
    verify_shared_trace(&public, &trace, &[0], &TEST_PREFIX).unwrap();

    let mut tampered = trace.clone();
    tampered.assets[0].rows[0].values[0] = 99;
    assert!(verify_shared_trace(&public, &tampered, &[0], &TEST_PREFIX).is_err());

    let mut tampered = trace.clone();
    tampered.steps[0].message.config[0] = 3;
    assert!(verify_shared_trace(&public, &tampered, &[0], &TEST_PREFIX).is_err());

    let mut tampered = trace.clone();
    tampered.steps[0].message.proof.outputs[0][0].values[0] += 1;
    assert!(verify_shared_trace(&public, &tampered, &[0], &TEST_PREFIX).is_err());

    let mut tampered = trace.clone();
    tampered.steps[0].message.proof.inputs[0][0].values[0] += 1;
    assert!(verify_shared_trace(&public, &tampered, &[0], &TEST_PREFIX).is_err());

    let mut tampered = trace.clone();
    tampered.steps[0].external[1] = Some(ExternalAssetRef::rows(3, 0, 1));
    assert!(verify_shared_trace(&public, &tampered, &[0], &TEST_PREFIX).is_err());

    let mut tampered = trace.clone();
    tampered.steps[0].external[1] = Some(ExternalAssetRef::rows(99, 0, 2));
    assert!(verify_shared_trace(&public, &tampered, &[0], &TEST_PREFIX).is_err());
}

#[test]
fn gemma_profile_norm_rope_and_row_embedding_relations_close() {
    use zkie_ops::index_gen::RowEmbeddingIndexOp;
    use zkie_ops::profile_rms::ProfileRmsNormOp;
    use zkie_ops::profile_rope::ProfileRopeOp;
    use zkie_ops::proof::relations::{prove_op_relation, verify_op_relation, RelationConfig};

    let mut table = vec![0i64; 8];
    table[0] = 2;
    let norm = ProfileRmsNormOp {
        rows: 1,
        cols: 2,
        valid_cols: 2,
        output_shift: 0,
        clamp: None,
    };
    let inputs = vec![vec![4, 6], vec![1, 1], table];
    let config = norm.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&norm, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();

    let rope = ProfileRopeOp { head_dim: 2, shift: 0 };
    let inputs = vec![vec![1, 2], vec![1], vec![0]];
    let config = rope.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&rope, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();

    let mut base = vec![0i64; 8];
    base[0] = 2;
    let mut extension = vec![0i64; 9];
    extension[8] = 1;
    let norm = ProfileRmsNormOp {
        rows: 1,
        cols: 1,
        valid_cols: 1,
        output_shift: 0,
        clamp: None,
    };
    let inputs = vec![vec![2048], vec![1], base, extension];
    let config = norm.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&norm, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();
    assert_eq!(proof.outputs[0][0].values, vec![2048]);

    let embedding = RowEmbeddingIndexOp { hidden: 2, vocab: 3 };
    let inputs = vec![vec![1, 0]];
    let config = embedding.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&embedding, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();
    assert_eq!(proof.outputs[0][0].values, vec![2, 3, 0, 1]);
}

#[test]
fn timesfm_profile_standardize_relation_closes() {
    use zkie_ops::profile_standardize::ProfileStandardizeOp;
    use zkie_ops::proof::relations::{prove_op_relation, verify_op_relation, RelationConfig};

    let mut table = vec![0i64; 8];
    table[0] = 2;
    let op = ProfileStandardizeOp {
        rows: 2,
        cols: 2,
        output_cols: 4,
        valid_cols: 2,
        source_row: 0,
        output_shift: 0,
        clamp: None,
    };
    let inputs = vec![vec![4, 6, 10, 14], table];
    let config = op.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&op, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();
    let values: Vec<i64> = proof.outputs[0][0].values.iter().map(|value| *value as i64).collect();
    assert_eq!(values, vec![-2, 2, 0, 0, 10, 18, 0, 0]);
}

#[test]
fn profile_relations_reject_tampered_outputs() {
    use zkie_ops::index_gen::RowEmbeddingIndexOp;
    use zkie_ops::profile_rms::ProfileRmsNormOp;
    use zkie_ops::profile_rope::ProfileRopeOp;
    use zkie_ops::profile_standardize::ProfileStandardizeOp;
    use zkie_ops::proof::relations::{prove_op_relation, verify_op_relation, RelationConfig};

    let mut table = vec![0i64; 8];
    table[0] = 2;
    let norm = ProfileRmsNormOp { rows: 1, cols: 2, valid_cols: 2, output_shift: 0, clamp: None };
    let inputs = vec![vec![4, 6], vec![1, 1], table.clone()];
    let config = norm.relation_config(&inputs).unwrap();
    let (public, mut proof) = prove_op_relation(&norm, &inputs, &config).unwrap();
    proof.outputs[0][0].values[1] += 1;
    assert!(verify_op_relation(&public, &proof).is_err());

    let rope = ProfileRopeOp { head_dim: 2, shift: 0 };
    let inputs = vec![vec![1, 2], vec![1], vec![0]];
    let config = rope.relation_config(&inputs).unwrap();
    let (public, mut proof) = prove_op_relation(&rope, &inputs, &config).unwrap();
    proof.outputs[0][0].values[0] += 1;
    assert!(verify_op_relation(&public, &proof).is_err());

    let embedding = RowEmbeddingIndexOp { hidden: 2, vocab: 3 };
    let inputs = vec![vec![1, 0]];
    let config = embedding.relation_config(&inputs).unwrap();
    let (public, mut proof) = prove_op_relation(&embedding, &inputs, &config).unwrap();
    proof.outputs[0][0].values[0] += 1;
    assert!(verify_op_relation(&public, &proof).is_err());

    let standardize = ProfileStandardizeOp {
        rows: 2,
        cols: 2,
        output_cols: 4,
        valid_cols: 2,
        source_row: 0,
        output_shift: 0,
        clamp: None,
    };
    let inputs = vec![vec![4, 6, 10, 14], table];
    let config = standardize.relation_config(&inputs).unwrap();
    let (public, mut proof) = prove_op_relation(&standardize, &inputs, &config).unwrap();
    proof.outputs[0][0].values[5] += 1;
    assert!(verify_op_relation(&public, &proof).is_err());
}


#[test]
fn softmax_rows_matches_per_row_and_rejects_tampering() {
    use zkie_ops::proof::relations::{prove_op_relation, verify_op_relation, RelationConfig};
    use zkie_ops::registry::PureForward;
    use zkie_ops::softmax::SoftmaxOp;
    use zkie_ops::softmax_rows::SoftmaxRowsOp;

    let table = vec![1i64, 2, 3, 4];
    let x = vec![1i64, 0, 0, 1];
    let op = SoftmaxRowsOp {
        rows: 2,
        cols: 2,
        shift: 16,
    };
    let inputs = vec![x.clone(), table.clone()];
    let config = op.relation_config(&inputs).unwrap();
    let (public, proof) = prove_op_relation(&op, &inputs, &config).unwrap();
    verify_op_relation(&public, &proof).unwrap();

    let batched = op.forward_values(&inputs).unwrap().outputs[0].clone();
    let mut per_row = Vec::new();
    for row in 0..2 {
        let row_values = x[row * 2..(row + 1) * 2].to_vec();
        let record = SoftmaxOp { shift: 16 }
            .forward_values(&[row_values, table.clone()])
            .unwrap();
        per_row.extend_from_slice(&record.outputs[0]);
    }
    assert_eq!(batched, per_row);

    let mut bad = proof.clone();
    bad.outputs[0][0].values[0] += 1;
    assert!(verify_op_relation(&public, &bad).is_err());

    let mut bad = proof.clone();
    bad.raw[0][0].values[0] += 1;
    assert!(verify_op_relation(&public, &bad).is_err());

    let mut bad = proof.clone();
    bad.aux[0][0].values[0] += 1;
    assert!(verify_op_relation(&public, &bad).is_err());

    let mut bad = public.clone();
    bad.config[1] = 3;
    assert!(verify_op_relation(&bad, &proof).is_err());
}
