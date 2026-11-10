use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::artifact::digest_to_hex;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::link::AnchorRecord;
use zkie_ops::proof::model_commitment::{ModelCommitmentPublic, SharedTrace};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-policy-cli-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn write<T: serde::Serialize>(path: &Path, value: &T) {
    fs::write(path, bincode::serialize(value).unwrap()).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn setup(
    root: &Path,
    model: &str,
    case_id: &str,
    tokens: &[u32],
    statement_commitment: Digest,
    anchor_id: Digest,
    anchor_commitment: Digest,
    policy: Option<&str>,
) -> (PathBuf, PathBuf) {
    let public_dir = root.join("public");
    fs::create_dir_all(&public_dir).unwrap();
    let public = ModelCommitmentPublic {
        anchor: AnchorRecord {
            anchor_id,
            model_commitment: anchor_commitment,
            parameters_digest: Digest([0x33; 32]),
            registry_root: Digest([0x44; 32]),
            leaf_count: 0,
        },
        assets: Vec::new(),
    };
    write(&public_dir.join("model-commitment.bin"), &public);
    let statement = zkie::provider::StatementData {
        case_id: case_id.into(),
        model: model.into(),
        model_commitment: statement_commitment,
        tokens: tokens.to_vec(),
        expected_argmax: vec![0; tokens.len()],
        expected_output: Vec::new(),
    };
    write(&root.join("statement.bin"), &statement);
    let trace = SharedTrace {
        tokens: tokens.to_vec(),
        assets: Vec::new(),
        steps: Vec::new(),
        argmax: vec![0; tokens.len()],
        output: Vec::new(),
    };
    write(&root.join("proof.bin"), &trace);
    write(&root.join("public-inputs.bin"), &vec![0u32; tokens.len()]);
    let policy_path = root.join("case.policy");
    if let Some(text) = policy {
        fs::write(&policy_path, text).unwrap();
    }
    (public_dir, policy_path)
}

fn verify_code(root: &Path, public_dir: &Path, policy: &Path, model_id: &str) -> i32 {
    zkie::run_cli(vec![
        "verify".into(),
        "--public".into(),
        public_dir.to_string_lossy().into_owned(),
        "--statement".into(),
        root.join("statement.bin").to_string_lossy().into_owned(),
        "--proof".into(),
        root.join("proof.bin").to_string_lossy().into_owned(),
        "--public-inputs".into(),
        root.join("public-inputs.bin").to_string_lossy().into_owned(),
        "--model-id".into(),
        model_id.to_owned(),
        "--policy".into(),
        policy.to_string_lossy().into_owned(),
        "--run-dir".into(),
        root.join("run").to_string_lossy().into_owned(),
    ])
}

fn identity_block(case_id: &str, anchor: &Digest, model_id: &Digest) -> String {
    format!(
        "\n[case.limited_table_identity]\nversion = 1\nmodel = \"gpt2\"\nmodel_version = 1\ncase_id = \"{case_id}\"\nanchor_id = \"{}\"\nmodel_id = \"{}\"\nasset_id = 7\nasset_table_identity = \"{a}\"\ntable_root = \"{r}\"\ntable_record_digest = \"{d}\"\ntable_configuration = \"{c}\"\ncode_log_n = 3\ncode_ell = 5\ncode_r = 3\ntable_v = 1\nanswers = 1\nsalt_bytes = 40\ndistance_radius = 1\ndistance_later_radius = 1\ndistance_m_once = 0\ndistance_m_later = 0\nmatrix_rows_log = 1\nk_lo = 1\nk_col = 2\nprotocol_identity = \"{p}\"\n",
        digest_to_hex(anchor),
        digest_to_hex(model_id),
        a = "33".repeat(32),
        r = "44".repeat(32),
        d = "55".repeat(32),
        c = "66".repeat(32),
        p = "77".repeat(32),
    )
}

fn token_policy(sequence: usize, anchor: &Digest, model_id: &Digest) -> String {
    format!(
        "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"gpt2-seq16\"\nsequence_length = {sequence}\nanchor_id = \"{}\"\nmodel_id = \"{}\"\noutput_mode = \"none\"\n{}",
        digest_to_hex(anchor),
        digest_to_hex(model_id),
        identity_block("gpt2-seq16", anchor, model_id)
    )
}

#[test]
fn missing_policy_exits_nonzero() {
    let root = temp_dir();
    let (public_dir, policy) = setup(
        &root,
        "gpt2",
        "case",
        &[1],
        Digest([0x11; 32]),
        Digest([0x22; 32]),
        Digest([0x11; 32]),
        None,
    );
    assert_ne!(verify_code(&root, &public_dir, &policy, &"11".repeat(32)), 0);
}

#[test]
fn malformed_policy_exits_nonzero() {
    let root = temp_dir();
    let (public_dir, policy) = setup(
        &root,
        "gpt2",
        "case",
        &[1],
        Digest([0x11; 32]),
        Digest([0x22; 32]),
        Digest([0x11; 32]),
        Some("policy_version = \"bad\"\n"),
    );
    assert_ne!(verify_code(&root, &public_dir, &policy, &"11".repeat(32)), 0);
}

#[test]
fn unknown_model_exits_nonzero() {
    let root = temp_dir();
    let policy = format!(
        "policy_version = 1\nmodel = \"unknown\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"case\"\nsequence_length = 1\nanchor_id = \"{}\"\nmodel_id = \"{}\"\noutput_mode = \"none\"\n",
        "22".repeat(32),
        "11".repeat(32)
    );
    let (public_dir, policy) = setup(
        &root,
        "unknown",
        "case",
        &[1],
        Digest([0x11; 32]),
        Digest([0x22; 32]),
        Digest([0x11; 32]),
        Some(&policy),
    );
    assert_ne!(verify_code(&root, &public_dir, &policy, &"11".repeat(32)), 0);
}

#[test]
fn ambiguous_case_exits_nonzero() {
    let root = temp_dir();
    let anchor = Digest([0x22; 32]);
    let model_id = Digest([0x11; 32]);
    let identity = identity_block("case", &anchor, &model_id);
    let policy = format!(
        "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"case\"\nsequence_length = 1\nanchor_id = \"{}\"\nmodel_id = \"{}\"\noutput_mode = \"none\"\n{identity}\n\n[[case]]\ncase_id = \"case\"\nsequence_length = 1\nanchor_id = \"{}\"\nmodel_id = \"{}\"\noutput_mode = \"none\"\n{identity}\n",
        "22".repeat(32),
        "11".repeat(32),
        "22".repeat(32),
        "11".repeat(32),
    );
    let (public_dir, policy) = setup(
        &root,
        "gpt2",
        "case",
        &[1],
        Digest([0x11; 32]),
        Digest([0x22; 32]),
        Digest([0x11; 32]),
        Some(&policy),
    );
    assert_ne!(verify_code(&root, &public_dir, &policy, &"11".repeat(32)), 0);
}

#[test]
fn input_replacement_exits_nonzero() {
    let root = temp_dir();
    let committed = vec![1u32; 16];
    let replaced = vec![2u32; 16];
    let model_id = zkie_models_gpt2::plan::model_commitment(&committed).unwrap();
    let anchor = Digest([0x22; 32]);
    let policy = token_policy(committed.len(), &anchor, &model_id);
    let (public_dir, policy_path) = setup(
        &root,
        "gpt2",
        "gpt2-seq16",
        &replaced,
        model_id,
        anchor,
        model_id,
        Some(&policy),
    );
    assert_ne!(
        verify_code(&root, &public_dir, &policy_path, &digest_to_hex(&model_id)),
        0
    );
}

#[test]
fn model_version_mismatch_exits_nonzero() {
    let root = temp_dir();
    let tokens = vec![1u32; 16];
    let model_id = zkie_models_gpt2::plan::model_commitment(&tokens).unwrap();
    let anchor = Digest([0x22; 32]);
    let policy = token_policy(tokens.len(), &anchor, &model_id)
        .replace("model_version = 1", "model_version = 2");
    let (public_dir, policy_path) = setup(
        &root,
        "gpt2",
        "gpt2-seq16",
        &tokens,
        model_id,
        anchor,
        model_id,
        Some(&policy),
    );
    assert_ne!(
        verify_code(&root, &public_dir, &policy_path, &digest_to_hex(&model_id)),
        0
    );
}

#[test]
fn anchor_mismatch_exits_nonzero() {
    let root = temp_dir();
    let tokens = vec![1u32; 16];
    let model_id = zkie_models_gpt2::plan::model_commitment(&tokens).unwrap();
    let trusted_anchor = Digest([0x22; 32]);
    let policy = token_policy(tokens.len(), &trusted_anchor, &model_id);
    let (public_dir, policy_path) = setup(
        &root,
        "gpt2",
        "gpt2-seq16",
        &tokens,
        model_id,
        Digest([0x99; 32]),
        model_id,
        Some(&policy),
    );
    assert_ne!(
        verify_code(&root, &public_dir, &policy_path, &digest_to_hex(&model_id)),
        0
    );
}
