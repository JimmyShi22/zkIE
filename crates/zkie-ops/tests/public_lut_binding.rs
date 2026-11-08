//! Negative tests for the public-LUT channel and explicit asset intervals.
//!
//! Every case first serializes the valid trace, verifies it, then mutates the
//! serialized proof via a decode/modify/encode roundtrip and runs the full
//! shared-trace verifier again. No test skips the verifier.

use std::path::PathBuf;
use std::sync::OnceLock;

use zkie_core::types::Digest;
use zkie_ops::proof::model_commitment::{
    commit_assets, open_asset, verify_shared_trace, AssetOpening, ModelAssetSpec,
    ModelCommitment, ModelCommitmentPublic, SharedTrace, SharedTraceStep,
};
use zkie_ops::proof::relations::{
    prove_op_relation_shared, ExternalAssetRef, RelationReceiptMessage,
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
        let weights = ExternalAssetRef {
            asset_id: 2,
            row_start: 0,
            row_count: 2,
        };
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
    SharedTrace {
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
    }
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
    SharedTrace {
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
    }
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
    SharedTrace {
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
    }
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
    tampered.steps[0].external[0] = Some(ExternalAssetRef {
        asset_id: fixture().rsqrt.asset_id,
        row_start: 0,
        row_count: fixture().rsqrt.valid_len,
    });
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
