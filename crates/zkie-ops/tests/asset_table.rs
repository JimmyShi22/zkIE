//! Encoded asset table identity and commit binding.

use std::fs;
use std::path::{Path, PathBuf};

use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::CodeParams;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::table::{IndexShape, TableDistances};
use zkie_ops::proof::asset_table::{
    validate_asset_table_params, AssetTableInput, AssetTableLayout, AssetTableMeta,
    AssetTableParams, AssetTableRandomness,
};
use zkie_ops::proof::model_commitment::{
    commit_assets_with_tables, verify_asset_opening, verify_model_anchor,
    verify_model_anchor_with_tables, AssetStore, ModelAssetSpec, ModelCommitmentPublic,
};

const MODEL: Digest = Digest([0x11; 32]);
const PARAMS_DIGEST: Digest = Digest([0x22; 32]);
const ASSET_ID: u64 = 7;

fn temp_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-asset-table-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_asset(path: &Path, values: &[i32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn shape() -> IndexShape {
    IndexShape {
        code: CodeParams {
            log_n: 5,
            ell: 11,
            r: 0,
        },
        v: 2,
        answers: 1,
        salt_bytes: 40,
        distances: TableDistances {
            radius: 0,
            later_radius: 0,
            m_once: 0,
            m_later: 0,
        },
    }
}

fn spec() -> ModelAssetSpec {
    ModelAssetSpec {
        asset_id: ASSET_ID,
        path: "asset.bin".into(),
        element_count: 24,
        row_width: 8,
        encoding: "i32-le".into(),
    }
}

fn params() -> AssetTableParams {
    AssetTableParams {
        asset_id: ASSET_ID,
        path: "asset.bin".into(),
        element_count: 24,
        row_width: 8,
        encoding: "i32-le".into(),
        shape: shape(),
        valid_min: -1000,
        valid_max: 1000,
        layout: AssetTableLayout {
            row_width: 8,
            message_len: 11,
            tail_len: 0,
            pads: 3,
            valid_elements: 24,
            padding_zero: true,
        },
    }
}

fn randomness() -> AssetTableRandomness {
    AssetTableRandomness {
        salts: (0..32)
            .map(|position| vec![(position as u8) ^ 0x5a; 40])
            .collect(),
        tails: vec![Vec::new(); 4],
    }
}

fn values() -> Vec<i32> {
    (0..24).map(|index| (index as i32 - 12) * 3).collect()
}

fn input() -> AssetTableInput {
    AssetTableInput {
        params: params(),
        randomness: randomness(),
    }
}

fn commit(name: &str, values: &[i32]) -> (ModelCommitmentPublic, Vec<AssetTableMeta>) {
    let root = temp_dir(name);
    write_asset(&root.join("asset.bin"), values);
    let commitment =
        commit_assets_with_tables(&root, MODEL, PARAMS_DIGEST, &[spec()], &[input()]).unwrap();
    assert_eq!(commitment.tables.len(), 1);
    assert_eq!(commitment.anchor.record.leaf_count, 2);
    (
        ModelCommitmentPublic::from_commitment(&commitment),
        commitment.tables,
    )
}

/// The commit creates the table's column ledger, empty and initialized; a
/// repeated commit of the same table never resets a consumed count.
#[test]
fn the_table_commit_creates_its_column_ledger_and_a_recommit_keeps_it() {
    let root = temp_dir("ledger-init");
    write_asset(&root.join("asset.bin"), &values());
    let commitment =
        commit_assets_with_tables(&root, MODEL, PARAMS_DIGEST, &[spec()], &[input()]).unwrap();
    let meta = commitment.tables.first().unwrap();
    let ledger = zkie_ops::proof::column_session::column_ledger_dir(&root, meta.root);
    assert!(ledger.is_dir());
    let budget = zkie_engine::pcs::request::read::DurableColumnBudget::open(
        ledger.clone(),
        ledger.join("columns.count"),
        2,
        4,
    )
    .unwrap();
    assert_eq!(budget.count().unwrap(), 0);
    budget.reserve().unwrap();
    assert_eq!(budget.count().unwrap(), 2);

    let again =
        commit_assets_with_tables(&root, MODEL, PARAMS_DIGEST, &[spec()], &[input()]).unwrap();
    assert_eq!(again.tables.first().unwrap().root, meta.root);
    let reopened = zkie_engine::pcs::request::read::DurableColumnBudget::open(
        ledger.clone(),
        ledger.join("columns.count"),
        2,
        4,
    )
    .unwrap();
    assert_eq!(reopened.count().unwrap(), 2);
}

#[test]
fn table_commit_binds_the_anchor_and_verifies() {
    let (public, metas) = commit("positive", &values());
    let anchor = public.anchor.anchor_id;
    verify_model_anchor_with_tables(
        &public,
        anchor,
        MODEL,
        PARAMS_DIGEST,
        &[spec()],
        &metas,
        &[params()],
    )
    .unwrap();
    let bytes = bincode::serialize(&public).unwrap();
    let decoded: ModelCommitmentPublic = bincode::deserialize(&bytes).unwrap();
    verify_model_anchor_with_tables(
        &decoded,
        anchor,
        MODEL,
        PARAMS_DIGEST,
        &[spec()],
        &metas,
        &[params()],
    )
    .unwrap();
    assert!(verify_model_anchor(&decoded, anchor, MODEL, PARAMS_DIGEST, &[spec()]).is_err());
}

#[test]
fn changed_committed_table_is_rejected() {
    let (base, metas) = commit("changed", &values());
    let anchor = base.anchor.anchor_id;
    let check = |public: &ModelCommitmentPublic, metas: &[AssetTableMeta]| {
        verify_model_anchor_with_tables(
            public,
            anchor,
            MODEL,
            PARAMS_DIGEST,
            &[spec()],
            metas,
            &[params()],
        )
    };
    let mut cases: Vec<(&str, Box<dyn Fn(&mut Vec<AssetTableMeta>)>)> = Vec::new();
    cases.push((
        "root",
        Box::new(|metas| metas[0].root.0[0] ^= 1),
    ));
    cases.push((
        "configuration",
        Box::new(|metas| metas[0].configuration.0[0] ^= 1),
    ));
    cases.push((
        "identity",
        Box::new(|metas| metas[0].identity.0[0] ^= 1),
    ));
    cases.push((
        "shape-v",
        Box::new(|metas| metas[0].shape.v += 1),
    ));
    cases.push((
        "shape-code",
        Box::new(|metas| metas[0].shape.code.log_n += 1),
    ));
    cases.push((
        "record",
        Box::new(|metas| metas[0].record.y.push(Fp3::ZERO)),
    ));
    cases.push((
        "valid-range",
        Box::new(|metas| metas[0].valid_min -= 1),
    ));
    cases.push((
        "removed",
        Box::new(|metas| metas.clear()),
    ));
    cases.push((
        "duplicated",
        Box::new(|metas| {
            let table = metas[0].clone();
            metas.push(table);
        }),
    ));
    cases.push((
        "asset-id",
        Box::new(|metas| metas[0].asset_id += 1),
    ));
    cases.push((
        "path",
        Box::new(|metas| metas[0].path.push('x')),
    ));
    for (name, mutate) in cases {
        let mut metas = metas.clone();
        mutate(&mut metas);
        assert!(
            check(&base, &metas).is_err(),
            "a changed table field must be rejected: {name}"
        );
    }
}

#[test]
fn asset_openings_prove_against_the_table_anchor() {
    let root = temp_dir("opening");
    write_asset(&root.join("asset.bin"), &values());
    let commitment =
        commit_assets_with_tables(&root, MODEL, PARAMS_DIGEST, &[spec()], &[input()]).unwrap();
    let store = AssetStore::open(&commitment).unwrap();
    let opening = store.open_asset(0).unwrap();
    verify_asset_opening(&commitment, &opening).unwrap();
    let mut changed = opening.clone();
    changed.rows[0].values[0] ^= 1;
    assert!(verify_asset_opening(&commitment, &changed).is_err());
}

#[test]
fn out_of_range_rows_are_rejected_at_commit() {
    let root = temp_dir("range");
    let mut out = values();
    out[3] = 5000;
    write_asset(&root.join("asset.bin"), &out);
    assert!(commit_assets_with_tables(&root, MODEL, PARAMS_DIGEST, &[spec()], &[input()]).is_err());
}

#[test]
fn wrong_random_material_is_rejected_at_commit() {
    let root = temp_dir("random");
    write_asset(&root.join("asset.bin"), &values());
    let mut bad_salts = input();
    bad_salts.randomness.salts.pop();
    assert!(
        commit_assets_with_tables(&root, MODEL, PARAMS_DIGEST, &[spec()], &[bad_salts]).is_err()
    );
    let mut bad_tails = input();
    bad_tails.randomness.tails.pop();
    assert!(
        commit_assets_with_tables(&root, MODEL, PARAMS_DIGEST, &[spec()], &[bad_tails]).is_err()
    );
}

#[test]
fn params_must_describe_the_asset() {
    let mut changed = params();
    changed.element_count += 1;
    assert!(validate_asset_table_params(&spec(), &changed).is_err());
    let mut changed = params();
    changed.layout.padding_zero = false;
    assert!(validate_asset_table_params(&spec(), &changed).is_err());
    let mut changed = params();
    changed.valid_min = 1;
    changed.valid_max = -1;
    assert!(validate_asset_table_params(&spec(), &changed).is_err());
    let mut changed = params();
    changed.shape.v += 1;
    assert!(validate_asset_table_params(&spec(), &changed).is_err());
}
