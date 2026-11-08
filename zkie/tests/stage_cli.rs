//! Stage-boundary tests: prove never commits, verify accepts only public flags,
//! and run plans execute commit exactly once.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie::{prove, run, verify};
use zkie_core::artifact::{digest_to_hex, COMPLETE_FILE};
use zkie_core::error::Error;
use zkie_core::types::Digest;
use zkie_ops::pipeline::UnimplementedProvider;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zkie-stage-cli-{}-{}",
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

fn write(path: &Path, body: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, body).unwrap();
}

#[test]
fn prove_without_commit_is_rejected_and_does_not_create_commit() {
    let temp = TempDir::new();
    let input = temp.path().join("input.bin");
    write(&input, b"input");
    let commit = temp.path().join("commit");
    let out = temp.path().join("proof");
    let run_dir = temp.path().join("run-prove");
    let args = vec![
        "--commit".into(),
        commit.to_string_lossy().into_owned(),
        "--input".into(),
        input.to_string_lossy().into_owned(),
        "--model-id".into(),
        digest_to_hex(&Digest([7u8; 32])),
        "--out".into(),
        out.to_string_lossy().into_owned(),
        "--run-dir".into(),
        run_dir.to_string_lossy().into_owned(),
    ];
    let error = prove::run(&args, &UnimplementedProvider).unwrap_err();
    assert!(matches!(error, Error::MissingCommitArtifact));
    assert!(!commit.exists());
    assert!(!out.exists());
    let report = fs::read_to_string(run_dir.join("result.json")).unwrap();
    assert!(report.contains("BLOCKED") || report.contains("FAIL"));
}

#[test]
fn verify_parser_rejects_private_and_weight_flags() {
    let temp = TempDir::new();
    let base = vec![
        "--public".into(),
        temp.path().join("public").to_string_lossy().into_owned(),
        "--statement".into(),
        temp.path().join("statement.bin").to_string_lossy().into_owned(),
        "--proof".into(),
        temp.path().join("proof.bin").to_string_lossy().into_owned(),
        "--public-inputs".into(),
        temp.path().join("public-inputs.bin").to_string_lossy().into_owned(),
        "--model-id".into(),
        digest_to_hex(&Digest([8u8; 32])),
        "--policy".into(),
        temp.path().join("policy.toml").to_string_lossy().into_owned(),
        "--run-dir".into(),
        temp.path().join("run").to_string_lossy().into_owned(),
    ];
    assert!(verify::VerifyArgs::parse(&base).is_ok());
    let mut private = base.clone();
    private.push("--private".into());
    private.push(temp.path().join("private").to_string_lossy().into_owned());
    assert!(verify::VerifyArgs::parse(&private).is_err());
    let mut weights = base;
    weights.push("--weights".into());
    weights.push(temp.path().join("weights").to_string_lossy().into_owned());
    assert!(verify::VerifyArgs::parse(&weights).is_err());
}

#[test]
fn run_plan_commits_once_and_reuse_skips_commit() {
    let temp = TempDir::new();
    let case = temp.path().join("case.toml");
    let assets = temp.path().join("assets.json");
    let parameters = temp.path().join("parameters.toml");
    let input = temp.path().join("input.bin");
    let policy = temp.path().join("policy.toml");
    for path in [&case, &assets, &parameters, &input, &policy] {
        write(path, b"x");
    }
    let model_id = digest_to_hex(&Digest([9u8; 32]));
    let base = vec![
        "--case".into(),
        case.to_string_lossy().into_owned(),
        "--assets-manifest".into(),
        assets.to_string_lossy().into_owned(),
        "--parameters".into(),
        parameters.to_string_lossy().into_owned(),
        "--input".into(),
        input.to_string_lossy().into_owned(),
        "--model-id".into(),
        model_id,
        "--policy".into(),
        policy.to_string_lossy().into_owned(),
    ];
    let mut create = vec!["--commit-policy".into(), "create".into()];
    create.extend(base.clone());
    create.extend([
        "--run-dir".into(),
        temp.path().join("run-create").to_string_lossy().into_owned(),
    ]);
    let plan = run::RunArgs::parse(&create).unwrap().plan().unwrap();
    assert_eq!(plan.len(), 3);
    assert_eq!(plan[0][0], "commit");
    assert_eq!(plan[1][0], "prove");
    assert_eq!(plan[2][0], "verify");

    let commit_dir = temp.path().join("existing-commit");
    write(&commit_dir.join(COMPLETE_FILE), b"complete");
    let mut reuse = vec![
        "--commit-policy".into(),
        "reuse".into(),
        "--commit-dir".into(),
        commit_dir.to_string_lossy().into_owned(),
    ];
    reuse.extend(base);
    reuse.extend([
        "--run-dir".into(),
        temp.path().join("run-reuse").to_string_lossy().into_owned(),
    ]);
    let plan = run::RunArgs::parse(&reuse).unwrap().plan().unwrap();
    assert_eq!(plan.len(), 2);
    assert_eq!(plan[0][0], "prove");
    assert_eq!(plan[1][0], "verify");
}
