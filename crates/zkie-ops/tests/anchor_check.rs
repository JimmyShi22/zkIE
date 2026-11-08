use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::types::Digest;
use zkie_ops::proof::model_commitment::{
    commit_assets, verify_model_anchor, ModelAssetSpec, ModelCommitmentPublic,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-anchor-check-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_i32(path: &PathBuf, values: &[i32]) {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fs::write(path, bytes).unwrap();
}

fn fixture() -> (ModelCommitmentPublic, Vec<ModelAssetSpec>) {
    let root = temp_dir();
    write_i32(&root.join("a_i32.bin"), &[1, 2, 3, 4, 5, 6]);
    write_i32(&root.join("b_i32.bin"), &[7, 8, 9, 10]);
    let specs = vec![
        ModelAssetSpec {
            asset_id: 0,
            path: "a_i32.bin".into(),
            element_count: 6,
            row_width: 3,
            encoding: "i32-le".into(),
        },
        ModelAssetSpec {
            asset_id: 1,
            path: "b_i32.bin".into(),
            element_count: 4,
            row_width: 2,
            encoding: "i32-le".into(),
        },
    ];
    let commitment =
        commit_assets(&root, Digest([1; 32]), Digest([2; 32]), &specs).unwrap();
    (ModelCommitmentPublic::from_commitment(&commitment), specs)
}

fn verify(public: &ModelCommitmentPublic, specs: &[ModelAssetSpec]) -> zkie_core::error::Result<()> {
    verify_model_anchor(
        public,
        public.anchor.anchor_id,
        public.anchor.model_commitment,
        public.anchor.parameters_digest,
        specs,
    )
}

#[test]
fn positive_anchor_and_negative_mutations() {
    let (public, specs) = fixture();
    verify(&public, &specs).unwrap();

    let mut wrong = public.clone();
    wrong.anchor.anchor_id = Digest([9; 32]);
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.anchor.model_commitment = Digest([8; 32]);
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.anchor.parameters_digest = Digest([7; 32]);
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets[0].asset_id = 9;
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets[0].rows += 1;
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets[0].start = Digest([6; 32]);
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets[0].root = Digest([5; 32]);
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets[0].leaf_record.leaf_root = Digest([4; 32]);
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets[0].leaf_record.table_identity = Digest([3; 32]);
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets[0].leaf_record.leaf_id = 9;
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets[0].path = "other_i32.bin".into();
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets.pop();
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong = public.clone();
    wrong.assets.push(wrong.assets[0].clone());
    assert!(verify(&wrong, &specs).is_err());

    let mut wrong_specs = specs.clone();
    wrong_specs[0].path = "other_i32.bin".into();
    assert!(verify(&public, &wrong_specs).is_err());

    let mut wrong_specs = specs.clone();
    wrong_specs[0].encoding = "i64-le".into();
    assert!(verify(&public, &wrong_specs).is_err());

    let mut wrong_specs = specs.clone();
    wrong_specs[0].element_count = 5;
    assert!(verify(&public, &wrong_specs).is_err());

    let mut wrong_specs = specs.clone();
    wrong_specs[0].row_width = 2;
    assert!(verify(&public, &wrong_specs).is_err());

    let mut wrong_specs = specs.clone();
    wrong_specs[1] = wrong_specs[0].clone();
    assert!(verify(&public, &wrong_specs).is_err());
}
