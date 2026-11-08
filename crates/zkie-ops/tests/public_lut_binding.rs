//! Negative tests for the public-LUT channel and explicit asset intervals.
//!
//! Every case first serializes the valid trace, verifies it, then mutates the
//! serialized proof via a decode/modify/encode roundtrip and runs the full
//! shared-trace verifier again. No test skips the verifier.

use std::path::PathBuf;
use std::sync::OnceLock;

use zkie_core::types::Digest;
use zkie_ops::proof::model_commitment::{
    build_claim_set, commit_assets, open_asset, verify_shared_trace, AssetOpening, ModelAssetSpec,
    ModelCommitment, ModelCommitmentPublic, SharedTrace, SharedTraceStep,
};
use zkie_ops::proof::relations::{
    prove_op_relation_shared, ExternalAssetRef, ExternalLayout, RelationReceiptMessage,
};
use zkie_ops::registry::PureForward;
use zkie_ops::public_lut::{self, PublicLutKind, PublicLutRef};
use zkie_ops::add::AddOp;
use zkie_ops::softmax::SoftmaxOp;
use zkie_ops::table_lookup::TableLookupOp;

struct Fixture {
    public: ModelCommitmentPublic,
    exp: PublicLutRef,
    rsqrt: PublicLutRef,
    rsqrt_opening: AssetOpening,
    weights: ExternalAssetRef,
    weights_opening: AssetOpening,
}

fn temp_root() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "zkie-lut-neg-{}-{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn write_table(path: &std::path::Path, values: &[i64]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&(*value as i32).to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let root = temp_root();
        let exp_values = public_lut::values(&PublicLutRef {
            asset_id: 0,
            kind: PublicLutKind::Exp,
            valid_len: public_lut::EXP_LEN,
            padded_len: public_lut::padded_len(public_lut::EXP_LEN),
        })
        .unwrap();
        let rsqrt_values = public_lut::values(&PublicLutRef {
            asset_id: 1,
            kind: PublicLutKind::Rsqrt,
            valid_len: public_lut::RSQRT_LEN,
            padded_len: public_lut::padded_len(public_lut::RSQRT_LEN),
        })
        .unwrap();
        write_table(&root.join("exp_table_i32.bin"), exp_values.as_ref());
        write_table(&root.join("rsqrt_table_i32.bin"), rsqrt_values.as_ref());
        write_table(&root.join("weights.bin"), &[1, 2, 3, 4]);
        let specs = vec![
            ModelAssetSpec {
                asset_id: 0,
                path: "exp_table_i32.bin".into(),
                element_count: public_lut::EXP_LEN,
                row_width: public_lut::EXP_LEN as usize,
                encoding: "i32-le".into(),
            },
            ModelAssetSpec {
                asset_id: 1,
                path: "rsqrt_table_i32.bin".into(),
                element_count: public_lut::RSQRT_LEN,
                row_width: public_lut::RSQRT_LEN as usize,
                encoding: "i32-le".into(),
            },
            ModelAssetSpec {
                asset_id: 2,
                path: "weights.bin".into(),
                element_count: 4,
                row_width: 2,
                encoding: "i32-le".into(),
            },
        ];
        let commitment = commit_assets(&root, Digest([0x31; 32]), Digest([0x32; 32]), &specs)
            .expect("fixture commitment");
        let public = ModelCommitmentPublic::from_commitment(&commitment);
        let exp = reference_for(&public, 0, PublicLutKind::Exp);
        let rsqrt = reference_for(&public, 1, PublicLutKind::Rsqrt);
        let rsqrt_opening = opening(&commitment, 1);
        let weights_opening = opening(&commitment, 2);
        let weights = ExternalAssetRef::rows(2, 0, 2);
        Fixture {
            public,
            exp,
            rsqrt,
            rsqrt_opening,
            weights,
            weights_opening,
        }
    })
}

fn reference_for(
    public: &ModelCommitmentPublic,
    asset_id: u64,
    kind: PublicLutKind,
) -> PublicLutRef {
    let meta = public
        .assets
        .iter()
        .find(|asset| asset.asset_id == asset_id)
        .unwrap();
    PublicLutRef {
        asset_id,
        kind,
        valid_len: meta.element_count,
        padded_len: public_lut::padded_len(meta.element_count),
    }
}

fn opening(commitment: &ModelCommitment, asset_id: u64) -> AssetOpening {
    let asset = commitment
        .assets
        .iter()
        .find(|asset| asset.asset_id == asset_id)
        .unwrap();
    open_asset(commitment, asset_id, 0, asset.rows).unwrap()
}

fn root_of(public: &ModelCommitmentPublic, reference: &PublicLutRef) -> Digest {
    public
        .assets
        .iter()
        .find(|asset| asset.asset_id == reference.asset_id)
        .unwrap()
        .root
}

fn config_with_lut(base: &[i64], public: &ModelCommitmentPublic, reference: &PublicLutRef) -> Vec<i64> {
    let mut config = base.to_vec();
    config.extend(public_lut::identity_words(reference, &root_of(public, reference)));
    config
}

fn step_with(
    opcode: u16,
    config: Vec<i64>,
    roots: Vec<(Digest, Digest, u64)>,
    external_inputs: Vec<usize>,
    proof: zkie_ops::proof::relations::OpRelationProof,
    external: Vec<Option<ExternalAssetRef>>,
    public_luts: Vec<Option<PublicLutRef>>,
) -> SharedTraceStep {
    let sources = vec![None; external.len()];
    SharedTraceStep {
        message: RelationReceiptMessage {
            opcode,
            config,
            roots,
            external_inputs,
            proof,
        },
        external,
        public_luts,
        sources,
    }
}

fn softmax_trace() -> SharedTrace {
    let fixture = fixture();
    let values = public_lut::values(&fixture.exp).unwrap();
    let config = config_with_lut(&[16], &fixture.public, &fixture.exp);
    let op = SoftmaxOp { shift: 16 };
    let (relation_public, proof) = prove_op_relation_shared(
        &op,
        &[vec![0, 0], (*values).clone()],
        &config,
        &[1],
    )
    .unwrap();
    let mut roots = Vec::new();
    for id in relation_public.registry.ids() {
        let root = relation_public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    seal(SharedTrace {
        tokens: vec![0],
        assets: Vec::new(),
        steps: vec![step_with(
            op.opcode() as u16,
            config,
            roots,
            vec![1],
            proof,
            vec![None, None],
            vec![None, Some(fixture.exp)],
        )],
        argmax: vec![0],
        output: Vec::new(),
        claims_root: Digest::ZERO,
    })
}

fn rsqrt_lookup_trace() -> SharedTrace {
    let fixture = fixture();
    let values = public_lut::values(&fixture.rsqrt).unwrap();
    let config = config_with_lut(&[], &fixture.public, &fixture.rsqrt);
    let op = TableLookupOp;
    let (relation_public, proof) = prove_op_relation_shared(
        &op,
        &[(*values).clone(), vec![0]],
        &config,
        &[0],
    )
    .unwrap();
    let mut roots = Vec::new();
    for id in relation_public.registry.ids() {
        let root = relation_public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    seal(SharedTrace {
        tokens: vec![0],
        assets: Vec::new(),
        steps: vec![step_with(
            op.opcode() as u16,
            config,
            roots,
            vec![0],
            proof,
            vec![None, None],
            vec![Some(fixture.rsqrt), None],
        )],
        argmax: vec![0],
        output: Vec::new(),
        claims_root: Digest::ZERO,
    })
}

fn asset_add_trace() -> SharedTrace {
    let fixture = fixture();
    let op = AddOp;
    let config = Vec::new();
    let (relation_public, proof) = prove_op_relation_shared(
        &op,
        &[vec![1, 1, 1, 1], vec![1, 2, 3, 4]],
        &config,
        &[1],
    )
    .unwrap();
    let mut roots = Vec::new();
    for id in relation_public.registry.ids() {
        let root = relation_public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    seal(SharedTrace {
        tokens: vec![0],
        assets: vec![fixture.weights_opening.clone()],
        steps: vec![step_with(
            op.opcode() as u16,
            config,
            roots,
            vec![1],
            proof,
            vec![None, Some(fixture.weights)],
            vec![None, None],
        )],
        argmax: vec![0],
        output: Vec::new(),
        claims_root: Digest::ZERO,
    })
}

fn layout_trace(values: Vec<i64>, reference: ExternalAssetRef) -> SharedTrace {
    let fixture = fixture();
    let op = AddOp;
    let config = Vec::new();
    let left: Vec<i64> = (0..values.len()).map(|index| 5 + index as i64).collect();
    let (relation_public, proof) =
        prove_op_relation_shared(&op, &[left, values], &config, &[1]).unwrap();
    let mut roots = Vec::new();
    for id in relation_public.registry.ids() {
        let root = relation_public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    seal(SharedTrace {
        tokens: vec![0],
        assets: vec![fixture.weights_opening.clone()],
        steps: vec![step_with(
            op.opcode() as u16,
            config,
            roots,
            vec![1],
            proof,
            vec![None, Some(reference)],
            vec![None, None],
        )],
        argmax: vec![0],
        output: Vec::new(),
        claims_root: Digest::ZERO,
    })
}

fn broadcast_trace() -> SharedTrace {
    layout_trace(
        vec![1, 2, 1, 2],
        ExternalAssetRef {
            asset_id: 2,
            row_start: 0,
            row_count: 1,
            layout: ExternalLayout::Broadcast { rows: 2 },
        },
    )
}

fn columns_trace() -> SharedTrace {
    layout_trace(
        vec![1, 3],
        ExternalAssetRef {
            asset_id: 2,
            row_start: 0,
            row_count: 2,
            layout: ExternalLayout::Columns {
                source_row_width: 2,
                col_start: 0,
                col_count: 1,
            },
        },
    )
}

fn broadcast_columns_trace() -> SharedTrace {
    layout_trace(
        vec![1, 1],
        ExternalAssetRef {
            asset_id: 2,
            row_start: 0,
            row_count: 1,
            layout: ExternalLayout::BroadcastColumns {
                source_row_width: 2,
                col_start: 0,
                col_count: 1,
                rows: 2,
            },
        },
    )
}

fn seal(mut trace: SharedTrace) -> SharedTrace {
    trace.claims_root =
        build_claim_set(&trace, &fixture().public.anchor.model_commitment)
            .unwrap()
            .root;
    trace
}

fn roundtrip(trace: &SharedTrace) -> SharedTrace {
    let bytes = bincode::serialize(trace).unwrap();
    bincode::deserialize(&bytes).unwrap()
}

fn assert_accepts(trace: &SharedTrace) {
    let bytes = bincode::serialize(trace).unwrap();
    let decoded: SharedTrace = bincode::deserialize(&bytes).unwrap();
    verify_shared_trace(&fixture().public, &decoded, &[0]).expect("valid trace must pass");
}

fn assert_rejects(trace: &SharedTrace) {
    let bytes = bincode::serialize(trace).unwrap();
    let decoded: SharedTrace = bincode::deserialize(&bytes).unwrap();
    assert!(verify_shared_trace(&fixture().public, &decoded, &[0]).is_err());
}

#[test]
fn table_kind_replacement_rejected() {
    let base = softmax_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].public_luts[1].as_mut().unwrap().kind = PublicLutKind::Gelu;
    assert_rejects(&tampered);
}

#[test]
fn table_valid_len_replacement_rejected() {
    let base = softmax_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].public_luts[1].as_mut().unwrap().valid_len -= 1;
    assert_rejects(&tampered);
}

#[test]
fn table_padded_len_replacement_rejected() {
    let base = softmax_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].public_luts[1].as_mut().unwrap().padded_len += 1;
    assert_rejects(&tampered);
}

#[test]
fn table_root_replacement_rejected() {
    let base = softmax_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let last = tampered.steps[0].message.config.len() - 1;
    tampered.steps[0].message.config[last] ^= 1;
    assert_rejects(&tampered);
}

#[test]
fn index_offset_rebinding_rejected() {
    let base = softmax_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].message.config[0] = 15;
    assert_rejects(&tampered);
}

#[test]
fn padding_region_index_rejected() {
    let base = rsqrt_lookup_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].message.proof.inputs[1][0].values[0] =
        fixture().rsqrt.valid_len;
    assert_rejects(&tampered);
}

#[test]
fn out_of_range_index_rejected() {
    let base = rsqrt_lookup_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].message.proof.inputs[1][0].values[0] =
        fixture().rsqrt.valid_len + 100;
    assert_rejects(&tampered);
}

#[test]
fn public_lut_and_external_collision_rejected() {
    let base = rsqrt_lookup_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].external[0] = Some(ExternalAssetRef::rows(
        fixture().rsqrt.asset_id,
        0,
        fixture().rsqrt.valid_len,
    ));
    assert_rejects(&tampered);
}

#[test]
fn table_asset_reopened_rejected() {
    let base = rsqrt_lookup_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.assets.push(fixture().rsqrt_opening.clone());
    assert_rejects(&tampered);
}

#[test]
fn duplicate_asset_opening_rejected() {
    let base = rsqrt_lookup_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered
        .assets
        .push(fixture().rsqrt_opening.clone());
    tampered
        .assets
        .push(fixture().rsqrt_opening.clone());
    assert_rejects(&tampered);
}

#[test]
fn duplicate_reference_inconsistent_rejected() {
    let base = softmax_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let mut second = tampered.steps[0].clone();
    second.public_luts[1].as_mut().unwrap().padded_len += 1;
    tampered.steps.push(second);
    assert_rejects(&tampered);
}

#[test]
fn asset_value_tamper_rejected() {
    let base = asset_add_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.assets[0].rows[0].values[0] ^= 1;
    assert_rejects(&tampered);
}

#[test]
fn asset_interval_tamper_rejected() {
    let base = asset_add_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    reference.row_start = 1;
    reference.row_count = 1;
    assert_rejects(&tampered);
}

#[test]
fn duplicate_asset_reference_rejected() {
    let base = asset_add_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.assets.push(fixture().weights_opening.clone());
    assert_rejects(&tampered);
}

#[test]
fn step_list_missing_or_forward_reference_rejected() {
    let base = softmax_trace();
    assert_accepts(&base);

    let mut tampered = roundtrip(&base);
    tampered.steps.clear();
    assert_rejects(&tampered);

    let mut tampered = roundtrip(&base);
    tampered.steps[0].sources[0] = Some((0, 0));
    assert_rejects(&tampered);
}

#[test]
fn broadcast_layout_repeats_committed_row() {
    let base = broadcast_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::Broadcast { rows } = &mut reference.layout {
        *rows = 1;
    }
    assert_rejects(&tampered);
}

#[test]
fn broadcast_requires_single_source_row_rejected() {
    let base = broadcast_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].external[1].as_mut().unwrap().row_count = 2;
    assert_rejects(&tampered);
}

#[test]
fn broadcast_zero_repeat_rejected() {
    let base = broadcast_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::Broadcast { rows } = &mut reference.layout {
        *rows = 0;
    }
    assert_rejects(&tampered);
}

#[test]
fn columns_layout_selects_committed_window() {
    let base = columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::Columns { col_start, .. } = &mut reference.layout {
        *col_start = 1;
    }
    assert_rejects(&tampered);
}

#[test]
fn columns_source_width_mismatch_rejected() {
    let base = columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::Columns {
        source_row_width, ..
    } = &mut reference.layout
    {
        *source_row_width = 3;
    }
    assert_rejects(&tampered);
}

#[test]
fn columns_window_overflow_rejected() {
    let base = columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::Columns {
        col_start,
        col_count,
        ..
    } = &mut reference.layout
    {
        *col_start = 1;
        *col_count = 2;
    }
    assert_rejects(&tampered);
}

#[test]
fn columns_zero_width_rejected() {
    let base = columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::Columns { col_count, .. } = &mut reference.layout {
        *col_count = 0;
    }
    assert_rejects(&tampered);
}

#[test]
fn broadcast_columns_layout_selects_window() {
    let base = broadcast_columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::BroadcastColumns { col_start, .. } = &mut reference.layout {
        *col_start = 1;
    }
    assert_rejects(&tampered);
}

#[test]
fn broadcast_columns_requires_single_source_row_rejected() {
    let base = broadcast_columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].external[1].as_mut().unwrap().row_count = 2;
    assert_rejects(&tampered);
}

#[test]
fn broadcast_columns_zero_repeat_rejected() {
    let base = broadcast_columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::BroadcastColumns { rows, .. } = &mut reference.layout {
        *rows = 0;
    }
    assert_rejects(&tampered);
}

#[test]
fn broadcast_columns_source_width_mismatch_rejected() {
    let base = broadcast_columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::BroadcastColumns {
        source_row_width, ..
    } = &mut reference.layout
    {
        *source_row_width = 3;
    }
    assert_rejects(&tampered);
}

#[test]
fn broadcast_columns_zero_width_rejected() {
    let base = broadcast_columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::BroadcastColumns { col_count, .. } = &mut reference.layout {
        *col_count = 0;
    }
    assert_rejects(&tampered);
}

#[test]
fn broadcast_columns_window_overflow_rejected() {
    let base = broadcast_columns_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    let reference = tampered.steps[0].external[1].as_mut().unwrap();
    if let ExternalLayout::BroadcastColumns {
        col_start,
        col_count,
        ..
    } = &mut reference.layout
    {
        *col_start = 1;
        *col_count = 2;
    }
    assert_rejects(&tampered);
}

fn local_add_step(
    left: Vec<i64>,
    right: Vec<i64>,
    sources: Vec<Option<(usize, usize)>>,
) -> SharedTraceStep {
    let op = AddOp;
    let config = Vec::new();
    let (relation_public, proof) =
        prove_op_relation_shared(&op, &[left, right], &config, &[]).unwrap();
    let mut roots = Vec::new();
    for id in relation_public.registry.ids() {
        let root = relation_public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    let mut step = step_with(
        op.opcode() as u16,
        config,
        roots,
        Vec::new(),
        proof,
        vec![None, None],
        vec![None, None],
    );
    step.sources = sources;
    step
}

fn producer_chain_trace() -> SharedTrace {
    let first = local_add_step(vec![1, 1], vec![1, 1], vec![None, None]);
    let second = local_add_step(vec![1, 1], vec![1, 1], vec![None, None]);
    let consumer = local_add_step(vec![2, 2], vec![1, 1], vec![Some((0, 0)), None]);
    seal(SharedTrace {
        tokens: vec![0],
        assets: Vec::new(),
        steps: vec![first, second, consumer],
        argmax: vec![0],
        output: Vec::new(),
        claims_root: Digest::ZERO,
    })
}

fn local_pair_trace() -> SharedTrace {
    let step = local_add_step(vec![1, 2], vec![3, 4], vec![None, None]);
    seal(SharedTrace {
        tokens: vec![0],
        assets: Vec::new(),
        steps: vec![step],
        argmax: vec![0],
        output: Vec::new(),
        claims_root: Digest::ZERO,
    })
}

fn shared_asset_steps_trace() -> SharedTrace {
    let fixture = fixture();
    let weights = fixture.weights;
    let mut opening = fixture.weights_opening.clone();
    opening.last_use_step = 1;
    let op = AddOp;
    let mut steps = Vec::new();
    for _ in 0..2 {
        let config = Vec::new();
        let (relation_public, proof) = prove_op_relation_shared(
            &op,
            &[vec![5, 6, 7, 8], vec![1, 2, 3, 4]],
            &config,
            &[1],
        )
        .unwrap();
        let mut roots = Vec::new();
        for id in relation_public.registry.ids() {
            let root = relation_public.registry.get(id).unwrap();
            roots.push((root.root, root.start, root.count));
        }
        steps.push(step_with(
            op.opcode() as u16,
            config,
            roots,
            vec![1],
            proof,
            vec![None, Some(weights)],
            vec![None, None],
        ));
    }
    seal(SharedTrace {
        tokens: vec![0],
        assets: vec![opening],
        steps,
        argmax: vec![0],
        output: Vec::new(),
        claims_root: Digest::ZERO,
    })
}

#[test]
fn source_rebind_with_equal_values_rejected() {
    let base = producer_chain_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[2].sources[0] = Some((1, 0));
    assert_rejects(&tampered);
}

#[test]
fn claim_port_content_swap_rejected() {
    let base = local_pair_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.steps[0].message.proof.inputs.swap(0, 1);
    assert_rejects(&tampered);
}

#[test]
fn early_last_use_rejected() {
    let base = shared_asset_steps_trace();
    assert_accepts(&base);
    let mut tampered = roundtrip(&base);
    tampered.assets[0].last_use_step = 0;
    assert_rejects(&tampered);
}

#[test]
fn different_claim_ids_share_one_slice_accepted() {
    let trace = shared_asset_steps_trace();
    assert_accepts(&trace);
}
