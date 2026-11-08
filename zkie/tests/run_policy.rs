use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-run-policy-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_case(root: &PathBuf) -> PathBuf {
    let assets = root.join("assets");
    fs::create_dir_all(&assets).unwrap();
    fs::write(assets.join("input_i32.bin"), vec![0u8; 64]).unwrap();
    let case = root.join("case.toml");
    fs::write(
        &case,
        format!(
            "root = \"{}\"\n\n[[case]]\nid = \"gpt2-seq16\"\ninput = {{ path = \"input_i32.bin\" }}\nseq_tokens = 16\n",
            assets.to_string_lossy()
        ),
    )
    .unwrap();
    case
}

#[test]
fn run_rejects_missing_policy() {
    let root = temp_dir();
    let case = write_case(&root);
    let code = zkie::run_cli(vec![
        "run".into(),
        "--commit-policy".into(),
        "create".into(),
        "--case".into(),
        case.to_string_lossy().into_owned(),
        "--assets-manifest".into(),
        case.to_string_lossy().into_owned(),
        "--parameters".into(),
        root.join("parameters.toml").to_string_lossy().into_owned(),
        "--input".into(),
        root.join("assets/input_i32.bin").to_string_lossy().into_owned(),
        "--model-id".into(),
        "11".repeat(32),
        "--policy".into(),
        root.join("missing.policy").to_string_lossy().into_owned(),
        "--run-dir".into(),
        root.join("run").to_string_lossy().into_owned(),
    ]);
    assert_ne!(code, 0);
}

#[test]
fn matrix_rejects_missing_case_policy() {
    let root = temp_dir();
    let case = write_case(&root);
    let code = zkie::run_cli(vec![
        "matrix".into(),
        "--cases".into(),
        case.to_string_lossy().into_owned(),
        "--out".into(),
        root.join("matrix").to_string_lossy().into_owned(),
    ]);
    assert_ne!(code, 0);
    assert!(!root.join("matrix/gpt2-seq16/cold-0/commit").exists());
}

#[test]
fn run_records_policy_source() {
    let root = temp_dir();
    let case = write_case(&root);
    let policy_bytes = b"policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n";
    let policy = root.join("case.policy");
    fs::write(&policy, policy_bytes).unwrap();
    let commit = root.join("commit");
    fs::create_dir_all(&commit).unwrap();
    fs::write(commit.join("COMPLETE"), b"complete
").unwrap();
    let run_dir = root.join("run");
    let code = zkie::run_cli(vec![
        "run".into(),
        "--commit-policy".into(),
        "reuse".into(),
        "--commit-dir".into(),
        commit.to_string_lossy().into_owned(),
        "--case".into(),
        case.to_string_lossy().into_owned(),
        "--assets-manifest".into(),
        case.to_string_lossy().into_owned(),
        "--parameters".into(),
        root.join("parameters.toml").to_string_lossy().into_owned(),
        "--input".into(),
        root.join("assets/input_i32.bin").to_string_lossy().into_owned(),
        "--model-id".into(),
        "11".repeat(32),
        "--policy".into(),
        policy.to_string_lossy().into_owned(),
        "--run-dir".into(),
        run_dir.to_string_lossy().into_owned(),
    ]);
    assert_ne!(code, 0);
    assert_eq!(fs::read(run_dir.join("policy.toml")).unwrap(), policy_bytes);
    let source: serde_json::Value =
        serde_json::from_slice(&fs::read(run_dir.join("policy-source.json")).unwrap()).unwrap();
    assert_eq!(source["path"], "case.policy");
    assert_eq!(source["used_path"], "policy.toml");
    assert_eq!(
        source["sha256"],
        zkie_core::artifact::digest_to_hex(&zkie_core::artifact::digest_bytes(policy_bytes))
    );
    assert!(source["registered_at_unix"].is_u64());
}
