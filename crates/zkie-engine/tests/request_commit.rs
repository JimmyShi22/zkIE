//! Commit-side contracts: column commitments, anchor/leaf/link, WF, hiding
//! budgets and artifact persistence.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;
use zkie_core::artifact::digest_from_hex;
use zkie_core::common::random::RandomSource;
use zkie_core::session::{
    BudgetItem, BudgetReservation, ExposureRecord, LeaseOutcome, ProverStateStore,
};
use zkie_core::error::Error;
use zkie_core::types::Digest;

use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp, symbol_at_fp};
use zkie_engine::pcs::request::field::{fp_from_u64, fp_to_u64};
use zkie_engine::pcs::request::format::{load_commit, publish_commit, PublicCommit};
use zkie_engine::pcs::request::hiding::{
    allocate_random_tail, derive_mask_stream, validate_hiding_state, ExposureKind, HidingLimits,
    HidingState, LeafLifecycle,
};
use zkie_engine::pcs::request::link::{
    make_anchor, prove_link, verify_link, AnchorRecord, LeafRecord, LinkProof,
};
use zkie_engine::pcs::request::merkle::{
    build_column_tree, open_columns, verify_columns, ColumnOpening, ColumnTreeSpec,
};
use zkie_engine::pcs::request::short_code::{fold_mask_columns, short_position};
use zkie_engine::pcs::state::BudgetedProverStateStore;
use zkie_engine::pcs::request::wf::{
    commit_weights, prove_wf, verify_wf, WfParams,
};

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zkie-commit-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct TestRng {
    seed: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl RandomSource for TestRng {
    fn fill(
        &mut self,
        domain: &[u8],
        logical_object: u64,
        offset: u64,
        out: &mut [u8],
    ) -> zkie_core::error::Result<()> {
        let mut state = self.seed ^ 0x9e37_79b9_7f4a_7c15;
        for byte in domain {
            state = state.wrapping_mul(1099511628211).wrapping_add(*byte as u64);
        }
        state = state.wrapping_add(logical_object.rotate_left(17));
        state = state.wrapping_add(offset.rotate_left(33));
        for byte in out {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        Ok(())
    }
}

fn vectors() -> Value {
    serde_json::from_str(include_str!("vectors/request/commit.json")).unwrap()
}

fn hex_bytes(value: &str) -> Vec<u8> {
    (0..value.len() / 2)
        .map(|index| u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap())
        .collect()
}

fn digest_from_value(value: &Value) -> Digest {
    digest_from_hex(value.as_str().unwrap()).unwrap()
}

fn u64s(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry.as_u64().unwrap())
        .collect()
}

fn build_tree(vectors: &Value) -> (ColumnTreeSpec, zkie_engine::pcs::request::merkle::ColumnTree) {
    let spec = ColumnTreeSpec::new(
        digest_from_value(&vectors["table_identity_hex"]),
        digest_from_value(&vectors["parameters_hex"]),
        0,
        4,
        40,
    )
    .unwrap();
    let salts = vectors["salts_hex"].as_array().unwrap();
    let symbols = vectors["symbols"].as_array().unwrap();
    let columns: Vec<(Vec<u64>, Vec<u8>)> = symbols
        .iter()
        .zip(salts)
        .map(|(row, salt)| (u64s(row), hex_bytes(salt.as_str().unwrap())))
        .collect();
    let tree = build_column_tree(&spec, &columns).unwrap();
    (spec, tree)
}

#[test]
fn column_tree_matches_vectors() {
    let vectors = vectors();
    let (_spec, tree) = build_tree(&vectors);
    assert_eq!(tree.root, digest_from_value(&vectors["column_root_hex"]));
    for (index, expected) in vectors["leaf_digests_hex"].as_array().unwrap().iter().enumerate() {
        assert_eq!(tree.leaves[index].digest, digest_from_value(expected));
    }
}

#[test]
fn column_openings_verify_and_reject_tampering() {
    let vectors = vectors();
    let (spec, tree) = build_tree(&vectors);
    let indices: Vec<u64> = (0..4).collect();
    let opening = open_columns(&tree, &indices).unwrap();
    verify_columns(&opening, &tree.root, spec.symbol_count, spec.salt_bytes).unwrap();

    let mut wrong_root = opening.clone();
    wrong_root.root = Digest([9u8; 32]);
    assert!(matches!(
        verify_columns(&wrong_root, &wrong_root.root, spec.symbol_count, spec.salt_bytes),
        Err(Error::InvalidProof) | Err(Error::ArtifactIntegrityMismatch)
    ));
    assert!(matches!(
        verify_columns(&opening, &wrong_root.root, spec.symbol_count, spec.salt_bytes),
        Err(Error::ArtifactIntegrityMismatch)
    ));

    let mut tampered = opening.clone();
    tampered.leaves[0].symbols[0] ^= 1;
    assert!(matches!(
        verify_columns(&tampered, &tree.root, spec.symbol_count, spec.salt_bytes),
        Err(Error::ArtifactIntegrityMismatch)
    ));

    let mut bad_path = opening.clone();
    bad_path.leaves[2].path[0] = Digest([5u8; 32]);
    assert!(matches!(
        verify_columns(&bad_path, &tree.root, spec.symbol_count, spec.salt_bytes),
        Err(Error::InvalidProof)
    ));

    assert!(matches!(
        verify_columns(&opening, &tree.root, 5, spec.salt_bytes),
        Err(Error::ShapeMismatch)
    ));
    assert!(open_columns(&tree, &[0, 0]).is_err());
    assert!(open_columns(&tree, &[4]).is_err());

    let encoded = opening.encode().unwrap();
    assert_eq!(ColumnOpening::decode(&encoded).unwrap().root, opening.root);
    let mut corrupt = encoded;
    corrupt[0] ^= 1;
    assert!(ColumnOpening::decode(&corrupt).is_err());
}

#[test]
fn anchor_and_link_match_vectors_and_reject_tampering() {
    let vectors = vectors();
    let (_spec, tree) = build_tree(&vectors);
    let records: Vec<LeafRecord> = vectors["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| LeafRecord {
            leaf_id: record["leaf_id"].as_u64().unwrap(),
            generation: record["generation"].as_u64().unwrap(),
            table_identity: digest_from_value(&record["table_identity_hex"]),
            leaf_root: digest_from_value(&record["leaf_root_hex"]),
        })
        .collect();
    let anchor = make_anchor(
        digest_from_value(&vectors["model_commitment_hex"]),
        digest_from_value(&vectors["parameters_hex"]),
        &records,
    )
    .unwrap();
    assert_eq!(anchor.record.registry_root, digest_from_value(&vectors["registry_root_hex"]));
    assert_eq!(anchor.record.anchor_id, digest_from_value(&vectors["anchor_id_hex"]));

    let link = prove_link(&anchor, &records, 0).unwrap();
    verify_link(
        &anchor.record,
        &link,
        &records[0].table_identity,
        &records[0].leaf_root,
    )
    .unwrap();
    assert_eq!(link.record.leaf_root, tree.root);

    assert!(matches!(
        verify_link(
            &anchor.record,
            &link,
            &records[1].table_identity,
            &records[0].leaf_root,
        ),
        Err(Error::StatementMismatch)
    ));
    let mut wrong_path = link.clone();
    wrong_path.path[0] = Digest([7u8; 32]);
    assert!(matches!(
        verify_link(
            &anchor.record,
            &wrong_path,
            &records[0].table_identity,
            &records[0].leaf_root,
        ),
        Err(Error::InvalidProof)
    ));
    let mut wrong_binding = link.clone();
    wrong_binding.record.generation += 1;
    assert!(matches!(
        verify_link(
            &anchor.record,
            &wrong_binding,
            &records[0].table_identity,
            &records[0].leaf_root,
        ),
        Err(Error::ArtifactIntegrityMismatch)
    ));
    let mut wrong_anchor = anchor.record.clone();
    wrong_anchor.anchor_id = Digest([3u8; 32]);
    assert!(verify_link(
        &wrong_anchor,
        &link,
        &records[0].table_identity,
        &records[0].leaf_root,
    )
    .is_err());

    let malformed = AnchorRecord::decode(&anchor.record.encode()).unwrap();
    assert_eq!(malformed, anchor.record);
    let roundtrip = LinkProof::decode(&link.encode().unwrap()).unwrap();
    assert_eq!(roundtrip.binding, link.binding);
}

#[test]
fn weight_range_proof_accepts_in_range_and_rejects_out_of_range() {
    let vectors = vectors();
    let case = &vectors["wf"];
    let params = WfParams {
        bound: case["bound"].as_u64().unwrap(),
        base_bits: case["base_bits"].as_u64().unwrap() as u32,
    };
    let identity = digest_from_value(&case["table_identity_hex"]);
    let parameters = digest_from_value(&vectors["parameters_hex"]);
    let weights: Vec<i64> = case["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect();
    let salts: Vec<Vec<u8>> = case["salts_hex"]
        .as_array()
        .unwrap()
        .iter()
        .map(|salt| hex_bytes(salt.as_str().unwrap()))
        .collect();
    let (commitment, tree) =
        commit_weights(params, identity, parameters, &weights, &salts).unwrap();
    assert_eq!(commitment.root, digest_from_value(&case["root_hex"]));
    let proof = prove_wf(&tree, &commitment).unwrap();
    verify_wf(&commitment, &proof).unwrap();

    let mut wrong_root = proof.clone();
    wrong_root.commitment = Digest([1u8; 32]);
    assert!(matches!(
        verify_wf(&commitment, &wrong_root),
        Err(Error::StatementMismatch)
    ));

    let mut short = proof.clone();
    short.leaves.leaves.pop();
    assert!(matches!(
        verify_wf(&commitment, &short),
        Err(Error::IncompleteArtifact)
    ));

    let out_of_range = [1001i64, 0, 0, 0];
    let (bad_commitment, bad_tree) = commit_weights(
        params,
        identity,
        parameters,
        &out_of_range,
        &salts,
    )
    .unwrap();
    let bad_proof = prove_wf(&bad_tree, &bad_commitment).unwrap();
    assert!(matches!(
        verify_wf(&bad_commitment, &bad_proof),
        Err(Error::BoundViolation)
    ));

    let mut tampered = proof.clone();
    tampered.leaves.leaves[1].symbols[1] += 1;
    assert!(verify_wf(&commitment, &tampered).is_err());
}

#[test]
fn hiding_budget_is_per_exposed_object_and_never_refunded() {
    let limits = HidingLimits::new()
        .with(ExposureKind::Query, 3)
        .with(ExposureKind::Ood, 1)
        .with(ExposureKind::Column, 10);
    let mut state = HidingState::new(limits);
    let query = state.reserve(ExposureKind::Query, 2).unwrap();
    assert_eq!(state.reserved(ExposureKind::Query), 2);
    state.record_exposure(query, 1).unwrap();
    assert_eq!(state.exposed(ExposureKind::Query), 1);
    state.finish(query);
    assert_eq!(state.reserved(ExposureKind::Query), 2);

    let second = state.reserve(ExposureKind::Query, 1).unwrap();
    assert_eq!(state.reserved(ExposureKind::Query), 3);
    assert!(matches!(
        state.reserve(ExposureKind::Query, 1),
        Err(Error::LeafBudgetExhausted)
    ));
    state.record_exposure(second, 1).unwrap();
    assert!(matches!(
        state.record_exposure(second, 2),
        Err(Error::LeafBudgetExhausted)
    ));
    let ood = state.reserve(ExposureKind::Ood, 1).unwrap();
    state.record_exposure(ood, 1).unwrap();
    assert!(matches!(
        state.reserve(ExposureKind::Ood, 1),
        Err(Error::LeafBudgetExhausted)
    ));
    validate_hiding_state(&state).unwrap();
    assert!(state.reserve(ExposureKind::Query, 0).is_err());
}

#[test]
fn leaf_rotation_needs_authorization_and_never_resets_budgets() {
    let mut lifecycle = LeafLifecycle::new(0, 0, 2, true).unwrap();
    assert!(lifecycle.rotate_leaf().is_err());
    lifecycle.record_use().unwrap();
    lifecycle.record_use().unwrap();
    assert!(lifecycle.record_use().is_err());
    let next = lifecycle.rotate_leaf().unwrap();
    assert_eq!(next, 1);
    assert_eq!(lifecycle.generation, 1);

    let mut denied = LeafLifecycle::new(0, 0, 1, false).unwrap();
    denied.record_use().unwrap();
    assert!(matches!(denied.rotate_leaf(), Err(Error::InvalidSessionState)));

    let limits = HidingLimits::new().with(ExposureKind::Query, 1);
    let mut state = HidingState::new(limits);
    let _ = state.reserve(ExposureKind::Query, 1).unwrap();
    let _ = lifecycle;
    assert_eq!(state.reserved(ExposureKind::Query), 1);
}

#[test]
fn mask_streams_and_tails_are_domain_separated() {
    let mut rng = TestRng::new(11);
    let first = derive_mask_stream(&mut rng, b"request/fold", 7, 3, 4).unwrap();
    let second = derive_mask_stream(&mut rng, b"request/fold", 7, 3, 4).unwrap();
    assert_eq!(first, second);
    let other_domain = derive_mask_stream(&mut rng, b"request/ood", 7, 3, 4).unwrap();
    let other_leaf = derive_mask_stream(&mut rng, b"request/fold", 8, 3, 4).unwrap();
    let other_round = derive_mask_stream(&mut rng, b"request/fold", 7, 4, 4).unwrap();
    assert_ne!(first, other_domain);
    assert_ne!(first, other_leaf);
    assert_ne!(first, other_round);

    let tail = allocate_random_tail(&mut rng, 7, 0, 8).unwrap();
    let tail_again = allocate_random_tail(&mut rng, 7, 0, 8).unwrap();
    assert_eq!(tail, tail_again);
    let tail_other = allocate_random_tail(&mut rng, 8, 0, 8).unwrap();
    assert_ne!(tail, tail_other);
}

#[test]
fn short_positions_match_vectors() {
    let vectors = vectors();
    let case = &vectors["short_positions"];
    let log_n = case["log_n"].as_u64().unwrap() as u32;
    let log_short = case["log_short"].as_u64().unwrap() as u32;
    let positions = u64s(&case["positions"]);
    let expected = u64s(&case["short"]);
    for (position, short) in positions.iter().zip(expected) {
        assert_eq!(short_position(log_n, log_short, *position).unwrap(), short);
    }
    assert_eq!(fold_mask_columns(2, 1).unwrap(), 4);
    assert!(short_position(4, 5, 0).is_err());
}

#[test]
fn public_commit_is_loadable_in_a_new_process() {
    let temp = TempDir::new();
    let vectors = vectors();
    let (spec, tree) = build_tree(&vectors);
    let records: Vec<LeafRecord> = vectors["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| LeafRecord {
            leaf_id: record["leaf_id"].as_u64().unwrap(),
            generation: record["generation"].as_u64().unwrap(),
            table_identity: digest_from_value(&record["table_identity_hex"]),
            leaf_root: digest_from_value(&record["leaf_root_hex"]),
        })
        .collect();
    let anchor = make_anchor(
        digest_from_value(&vectors["model_commitment_hex"]),
        digest_from_value(&vectors["parameters_hex"]),
        &records,
    )
    .unwrap();
    let link = prove_link(&anchor, &records, 0).unwrap();
    let params = WfParams {
        bound: vectors["wf"]["bound"].as_u64().unwrap(),
        base_bits: vectors["wf"]["base_bits"].as_u64().unwrap() as u32,
    };
    let weights: Vec<i64> = vectors["wf"]["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect();
    let salts: Vec<Vec<u8>> = vectors["wf"]["salts_hex"]
        .as_array()
        .unwrap()
        .iter()
        .map(|salt| hex_bytes(salt.as_str().unwrap()))
        .collect();
    let (wf, _wf_tree) = commit_weights(
        params,
        digest_from_value(&vectors["wf"]["table_identity_hex"]),
        digest_from_value(&vectors["parameters_hex"]),
        &weights,
        &salts,
    )
    .unwrap();
    let public = PublicCommit {
        anchor: anchor.record.clone(),
        leaf: records[0].clone(),
        link,
        column_spec: spec.clone(),
        column_root: tree.root,
        wf,
    };
    let commit_dir = temp.path().join("commit");
    publish_commit(&commit_dir, &public).unwrap();
    let loaded = load_commit(&commit_dir).unwrap();
    assert_eq!(loaded.column_root, tree.root);
    assert_eq!(loaded.anchor.anchor_id, public.anchor.anchor_id);

    let opening = open_columns(&tree, &[0, 1, 2, 3]).unwrap();
    let opening_path = temp.path().join("opening.bin");
    fs::write(&opening_path, opening.encode().unwrap()).unwrap();

    let exe = std::env::current_exe().unwrap();
    let output = std::process::Command::new(exe)
        .arg("child_loads_commit_in_new_process")
        .arg("--exact")
        .arg("--nocapture")
        .env("ZKIE_CHILD_COMMIT_DIR", &commit_dir)
        .env("ZKIE_CHILD_OPENING", &opening_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("child-commit-ok"), "{stdout}");
}

#[test]
fn child_loads_commit_in_new_process() {
    let Ok(commit_dir) = std::env::var("ZKIE_CHILD_COMMIT_DIR") else {
        return;
    };
    let opening_path = std::env::var("ZKIE_CHILD_OPENING").unwrap();
    let commit = load_commit(Path::new(&commit_dir)).unwrap();
    let opening = ColumnOpening::decode(&fs::read(opening_path).unwrap()).unwrap();
    verify_columns(
        &opening,
        &commit.column_root,
        commit.column_spec.symbol_count,
        commit.column_spec.salt_bytes,
    )
    .unwrap();
    verify_link(
        &commit.anchor,
        &commit.link,
        &commit.leaf.table_identity,
        &commit.leaf.leaf_root,
    )
    .unwrap();
    println!("child-commit-ok");
}

#[test]
fn hiding_budget_leases_use_the_private_ledger() {
    let kind = ExposureKind::Query;
    let mut store = BudgetedProverStateStore::new(vec![BudgetItem {
        kind: kind.budget_kind().to_owned(),
        amount: 2,
    }]);
    let reservation = BudgetReservation {
        commitment: Digest([1u8; 32]),
        leaf: 0,
        generation: 0,
        run_id: "run-1".into(),
        items: vec![BudgetItem {
            kind: kind.budget_kind().to_owned(),
            amount: 2,
        }],
    };
    let lease = store.reserve(&reservation).unwrap();
    store
        .mark_exposed(
            &lease,
            &ExposureRecord {
                items: vec![BudgetItem {
                    kind: kind.budget_kind().to_owned(),
                    amount: 1,
                }],
            },
        )
        .unwrap();
    store.finish(lease, LeaseOutcome::Completed).unwrap();
    let mut again = reservation.clone();
    again.items[0].amount = 1;
    assert!(matches!(
        store.reserve(&again),
        Err(Error::LeafBudgetExhausted)
    ));
}

#[test]
fn rs_product_form_transform_matches_direct_evaluation() {
    let vectors = vectors();
    let case = &vectors["code_transform"];
    let plan = build_code_plan(
        case["log_n"].as_u64().unwrap() as u32,
        case["ell"].as_u64().unwrap(),
        case["r"].as_u64().unwrap(),
    )
    .unwrap();
    let message: Vec<_> = case["message"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| fp_from_u64(value.as_u64().unwrap()))
        .collect();
    let tail: Vec<_> = case["tail"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| fp_from_u64(value.as_u64().unwrap()))
        .collect();
    let expected: Vec<u64> = case["word"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap())
        .collect();
    let word = encode_row_fp(&plan, &message, &tail).unwrap();
    assert_eq!(
        word.iter().map(|value| fp_to_u64(*value)).collect::<Vec<_>>(),
        expected
    );
    for position in 0..plan.n() {
        assert_eq!(
            fp_to_u64(symbol_at_fp(&plan, position, &message, &tail).unwrap()),
            expected[position as usize]
        );
    }
}

#[test]
fn leaf_record_encoding_is_locked() {
    let vectors = vectors();
    let case = &vectors["leaf_record"];
    let record = LeafRecord {
        leaf_id: case["leaf_id"].as_u64().unwrap(),
        generation: case["generation"].as_u64().unwrap(),
        table_identity: digest_from_value(&case["table_identity_hex"]),
        leaf_root: digest_from_value(&case["leaf_root_hex"]),
    };
    assert_eq!(
        record.encode(),
        hex_bytes(vectors["leaf_record_hex"].as_str().unwrap())
    );
    assert_eq!(LeafRecord::decode(&record.encode()).unwrap(), record);
    let mut corrupt = record.encode();
    corrupt[0] ^= 1;
    assert!(LeafRecord::decode(&corrupt).is_err());
}

#[test]
fn column_tree_requires_a_power_of_two_leaf_count() {
    let vectors = vectors();
    let spec = ColumnTreeSpec::new(
        digest_from_value(&vectors["table_identity_hex"]),
        digest_from_value(&vectors["parameters_hex"]),
        0,
        4,
        40,
    )
    .unwrap();
    let salt = hex_bytes(vectors["salts_hex"][0].as_str().unwrap());
    let column = (vec![1u64, 2, 3, 4], salt.clone());
    let one = build_column_tree(&spec, &[column.clone()]).unwrap();
    assert_eq!(one.leaves.len(), 1);
    assert_eq!(one.structure.padded_count, 1);
    for count in [3usize, 5] {
        let columns: Vec<(Vec<u64>, Vec<u8>)> =
            (0..count).map(|_| column.clone()).collect();
        assert!(matches!(
            build_column_tree(&spec, &columns),
            Err(Error::InvalidConfig)
        ));
    }
}
