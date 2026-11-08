//! CLI and artifact lifecycle contract tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use zkie::progress::{HeartbeatGuard, JsonlProgressSink};
use zkie::report::RunReport;
use zkie::{commit, prove, verify};
use zkie_core::artifact::{
    digest_bytes, digest_to_hex, ArtifactFile, ArtifactKind, ArtifactManifest, PrivacyMode,
    COMPLETE_FILE,
};
use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink, RunOutcome};
use zkie_core::types::Digest;
use zkie_engine::artifact::{verify_artifact, ArtifactPublisher};
use zkie_ops::pipeline::{
    CommitReceipt, PipelineContext, PipelineProvider, ProofReceipt, UnimplementedProvider,
    VerifyReceipt,
};

const TEST_BACKEND: u32 = 0xFFFF_FFFF;

struct TestProvider;

fn digest(seed: u8) -> Digest {
    Digest([seed; 32])
}

fn commit_manifest() -> (ArtifactManifest, Vec<u8>, Vec<u8>) {
    let public = b"public-protocol".to_vec();
    let private = b"private-state".to_vec();
    let manifest = ArtifactManifest {
        artifact_schema_version: 1,
        artifact_kind: ArtifactKind::Commit,
        protocol_id: digest(1),
        protocol_version: 1,
        backend_id: TEST_BACKEND,
        backend_format_version: 1,
        model_commitment_id: digest(2),
        parameters_digest: digest(3),
        model_config_digest: digest(4),
        quantization_digest: digest(5),
        tables_digest: digest(6),
        privacy_mode: PrivacyMode::Hidden,
        files: vec![
            ArtifactFile::new(
                "public/protocol.bin",
                public.len() as u64,
                digest_bytes(&public),
            ),
            ArtifactFile::new(
                "private/state.bin",
                private.len() as u64,
                digest_bytes(&private),
            ),
        ],
    };
    (manifest, public, private)
}

fn proof_manifest() -> (ArtifactManifest, Vec<u8>, Vec<u8>, Vec<u8>) {
    let statement = b"statement".to_vec();
    let proof = b"proof".to_vec();
    let inputs = b"inputs".to_vec();
    let manifest = ArtifactManifest {
        artifact_schema_version: 1,
        artifact_kind: ArtifactKind::Proof,
        protocol_id: digest(1),
        protocol_version: 1,
        backend_id: TEST_BACKEND,
        backend_format_version: 1,
        model_commitment_id: digest(2),
        parameters_digest: digest(3),
        model_config_digest: digest(4),
        quantization_digest: digest(5),
        tables_digest: digest(6),
        privacy_mode: PrivacyMode::Public,
        files: vec![
            ArtifactFile::new(
                "statement.bin",
                statement.len() as u64,
                digest_bytes(&statement),
            ),
            ArtifactFile::new("proof.bin", proof.len() as u64, digest_bytes(&proof)),
            ArtifactFile::new(
                "public-inputs.bin",
                inputs.len() as u64,
                digest_bytes(&inputs),
            ),
        ],
    };
    (manifest, statement, proof, inputs)
}

impl PipelineProvider for TestProvider {
    fn commit_model(
        &self,
        request: &zkie_ops::pipeline::CommitRequest,
        ctx: &mut PipelineContext<'_>,
    ) -> Result<CommitReceipt> {
        let (manifest, public, private) = commit_manifest();
        let files: Vec<(&str, &[u8])> = vec![
            ("public/protocol.bin", public.as_slice()),
            ("private/state.bin", private.as_slice()),
        ];
        ArtifactPublisher::publish(&request.output_dir, &manifest, &files)?;
        ctx.metrics.increment("test_commit");
        Ok(CommitReceipt {
            artifact_dir: request.output_dir.clone(),
            manifest_digest: manifest.manifest_digest()?,
        })
    }

    fn prove_request(
        &self,
        request: &zkie_ops::pipeline::ProveRequest,
        ctx: &mut PipelineContext<'_>,
    ) -> Result<ProofReceipt> {
        if !request.commit_dir.join(COMPLETE_FILE).exists() {
            return Err(Error::MissingCommitArtifact);
        }
        let (manifest, statement, proof, inputs) = proof_manifest();
        let files: Vec<(&str, &[u8])> = vec![
            ("statement.bin", statement.as_slice()),
            ("proof.bin", proof.as_slice()),
            ("public-inputs.bin", inputs.as_slice()),
        ];
        ArtifactPublisher::publish(&request.output_dir, &manifest, &files)?;
        ctx.metrics.increment("test_prove");
        Ok(ProofReceipt {
            proof_dir: request.output_dir.clone(),
            statement_digest: digest(9),
            test_only: true,
        })
    }

    fn verify_request(
        &self,
        request: &zkie_ops::pipeline::VerifyRequest,
        _ctx: &mut PipelineContext<'_>,
    ) -> Result<VerifyReceipt> {
        if !request.public_bundle_dir.join("protocol.bin").exists() {
            return Err(Error::IncompleteArtifact);
        }
        Ok(VerifyReceipt {
            accepted: true,
            outcome: RunOutcome::Passed,
            output_binding: None,
            authenticated_output: None,
            authenticated_output_digest: None,
        })
    }
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zkie-cli-contract-{}-{}",
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

fn write_file(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, bytes).unwrap();
}

fn texts(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}

fn model_id() -> String {
    digest_to_hex(&digest(2))
}

#[test]
fn commit_prove_verify_lifecycle_uses_public_artifacts() {
    let temp = TempDir::new();
    let case = temp.path().join("case.toml");
    let assets = temp.path().join("assets.json");
    let parameters = temp.path().join("parameters.toml");
    let input = temp.path().join("input.bin");
    let policy = temp.path().join("policy.toml");
    write_file(&case, b"case");
    write_file(&assets, b"assets");
    write_file(&parameters, b"parameters");
    write_file(&input, b"input");
    write_file(&policy, b"policy");

    let commit_dir = temp.path().join("commit");
    let commit_run = temp.path().join("run-commit");
    let commit_args = vec![
        "--case".into(),
        case.to_string_lossy().into_owned(),
        "--assets-manifest".into(),
        assets.to_string_lossy().into_owned(),
        "--parameters".into(),
        parameters.to_string_lossy().into_owned(),
        "--out".into(),
        commit_dir.to_string_lossy().into_owned(),
        "--run-dir".into(),
        commit_run.to_string_lossy().into_owned(),
    ];
    let provider = TestProvider;
    let commit_receipt = commit::run(&commit_args, &provider).unwrap();
    assert_eq!(commit_receipt.artifact_dir, commit_dir);
    assert!(commit_dir.join(COMPLETE_FILE).exists());
    verify_artifact(&commit_dir).unwrap();
    assert!(texts(&commit_run.join("result.json")).contains("PASS"));

    let proof_dir = temp.path().join("proof");
    let proof_run = temp.path().join("run-prove");
    let prove_args = vec![
        "--commit".into(),
        commit_dir.to_string_lossy().into_owned(),
        "--input".into(),
        input.to_string_lossy().into_owned(),
        "--model-id".into(),
        model_id(),
        "--out".into(),
        proof_dir.to_string_lossy().into_owned(),
        "--run-dir".into(),
        proof_run.to_string_lossy().into_owned(),
    ];
    let proof_receipt = prove::run(&prove_args, &provider).unwrap();
    assert!(proof_receipt.test_only);
    verify_artifact(&proof_dir).unwrap();
    assert!(texts(&proof_run.join("result.json")).contains("PASS"));

    let verify_run = temp.path().join("run-verify");
    let verify_args = vec![
        "--public".into(),
        commit_dir.join("public").to_string_lossy().into_owned(),
        "--statement".into(),
        proof_dir.join("statement.bin").to_string_lossy().into_owned(),
        "--proof".into(),
        proof_dir.join("proof.bin").to_string_lossy().into_owned(),
        "--public-inputs".into(),
        proof_dir
            .join("public-inputs.bin")
            .to_string_lossy()
            .into_owned(),
        "--model-id".into(),
        model_id(),
        "--policy".into(),
        policy.to_string_lossy().into_owned(),
        "--run-dir".into(),
        verify_run.to_string_lossy().into_owned(),
    ];
    let verify_receipt = verify::run(&verify_args, &provider).unwrap();
    assert!(verify_receipt.accepted);
    assert!(texts(&verify_run.join("result.json")).contains("PASS"));
}

#[test]
fn prove_without_commit_fails_without_creating_artifacts() {
    let temp = TempDir::new();
    let input = temp.path().join("input.bin");
    write_file(&input, b"input");
    let output = temp.path().join("proof");
    let run_dir = temp.path().join("run");
    let args = vec![
        "--commit".into(),
        temp.path().join("missing-commit").to_string_lossy().into_owned(),
        "--input".into(),
        input.to_string_lossy().into_owned(),
        "--model-id".into(),
        model_id(),
        "--out".into(),
        output.to_string_lossy().into_owned(),
        "--run-dir".into(),
        run_dir.to_string_lossy().into_owned(),
    ];
    assert!(matches!(
        prove::run(&args, &TestProvider),
        Err(Error::MissingCommitArtifact)
    ));
    assert!(!output.exists());
    assert!(texts(&run_dir.join("progress.jsonl")).contains("stage_failed"));
    assert!(texts(&run_dir.join("result.json")).contains("BLOCKED"));
}

#[test]
fn verify_parser_rejects_private_and_weight_flags() {
    let base = vec![
        "--public".to_string(),
        "/tmp/public".into(),
        "--statement".into(),
        "/tmp/statement".into(),
        "--proof".into(),
        "/tmp/proof".into(),
        "--public-inputs".into(),
        "/tmp/inputs".into(),
        "--model-id".into(),
        model_id(),
        "--policy".into(),
        "/tmp/policy".into(),
        "--run-dir".into(),
        "/tmp/run".into(),
    ];
    for forbidden in ["--private", "--weights", "--prover-state"] {
        let mut args = base.clone();
        args.push(forbidden.into());
        args.push("/tmp/secret".into());
        assert!(matches!(
            verify::VerifyArgs::parse(&args),
            Err(Error::InvalidConfig)
        ));
    }
}

#[test]
fn publisher_is_atomic_and_rejects_overwrite() {
    let temp = TempDir::new();
    let dir = temp.path().join("artifact");
    let (manifest, public, private) = commit_manifest();
    let files: Vec<(&str, &[u8])> = vec![
        ("public/protocol.bin", public.as_slice()),
        ("private/state.bin", private.as_slice()),
    ];
    ArtifactPublisher::publish(&dir, &manifest, &files).unwrap();
    verify_artifact(&dir).unwrap();
    assert!(matches!(
        ArtifactPublisher::publish(&dir, &manifest, &files),
        Err(Error::ArtifactAlreadyExists)
    ));

    let mut bad_version = manifest.clone();
    bad_version.artifact_schema_version = 0;
    assert!(matches!(
        bad_version.validate(),
        Err(Error::UnsupportedVersion)
    ));

    let mut traversal = manifest.clone();
    traversal.files[0].relative_path = "../escape.bin".into();
    assert!(traversal.validate().is_err());
}

#[test]
fn progress_and_report_records_are_structured() {
    let temp = TempDir::new();
    let sink = Arc::new(JsonlProgressSink::create(temp.path(), "run-1").unwrap());
    sink.emit(ProgressEvent::new(
        "run-1",
        "commit",
        ProgressKind::StageStarted,
    ))
    .unwrap();
    sink.emit(ProgressEvent::new(
        "run-1",
        "optional",
        ProgressKind::StageSkipped,
    ))
    .unwrap();
    let heartbeat = HeartbeatGuard::start_with_interval(
        Arc::clone(&sink),
        Duration::from_millis(10),
    );
    std::thread::sleep(Duration::from_millis(35));
    drop(heartbeat);
    sink.flush().unwrap();

    let progress = texts(sink.path());
    assert!(progress.contains("\"kind\":\"stage_started\""));
    assert!(progress.contains("\"kind\":\"stage_skipped\""));
    assert!(progress.contains("\"kind\":\"heartbeat\""));
    assert!(zkie::report::vm_hwm_kib().is_some());

    let skipped = RunReport::new("verify", RunOutcome::Skipped);
    let skip_path = skipped.write(&temp.path().join("skip-run")).unwrap();
    assert!(texts(&skip_path).contains("\"SKIP\""));

    let mut report = RunReport::new("commit", RunOutcome::Passed);
    report.progress_path = Some(sink.path().to_path_buf());
    report.last_event_seq = sink.last_event_seq();
    report.last_completed_stage = Some("commit".into());
    let path = report.write(temp.path()).unwrap();
    let text = texts(&path);
    assert!(text.contains("\"PASS\""));
    assert!(text.contains("progress_path"));
    assert!(text.contains("last_event_seq"));
}

#[test]
fn run_plan_requires_explicit_create_or_reuse() {
    let temp = TempDir::new();
    let case = temp.path().join("case.toml");
    let assets = temp.path().join("assets.json");
    let parameters = temp.path().join("parameters.toml");
    let input = temp.path().join("input.bin");
    let policy = temp.path().join("policy.toml");
    for path in [&case, &assets, &parameters, &input, &policy] {
        write_file(path, b"x");
    }

    let create = zkie::run::RunArgs::parse(&[
        "--commit-policy".into(),
        "create".into(),
        "--case".into(),
        case.to_string_lossy().into_owned(),
        "--assets-manifest".into(),
        assets.to_string_lossy().into_owned(),
        "--parameters".into(),
        parameters.to_string_lossy().into_owned(),
        "--input".into(),
        input.to_string_lossy().into_owned(),
        "--model-id".into(),
        model_id(),
        "--policy".into(),
        policy.to_string_lossy().into_owned(),
        "--run-dir".into(),
        temp.path().join("run").to_string_lossy().into_owned(),
    ])
    .unwrap();
    let plan = create.plan().unwrap();
    assert_eq!(plan.len(), 3);
    assert_eq!(plan[0][0], "commit");
    assert_eq!(plan[1][0], "prove");
    assert_eq!(plan[2][0], "verify");

    let commit_dir = temp.path().join("commit");
    fs::create_dir_all(&commit_dir).unwrap();
    write_file(&commit_dir.join(COMPLETE_FILE), b"x");
    let reuse = zkie::run::RunArgs::parse(&[
        "--commit-policy".into(),
        "reuse".into(),
        "--commit-dir".into(),
        commit_dir.to_string_lossy().into_owned(),
        "--case".into(),
        case.to_string_lossy().into_owned(),
        "--assets-manifest".into(),
        assets.to_string_lossy().into_owned(),
        "--parameters".into(),
        parameters.to_string_lossy().into_owned(),
        "--input".into(),
        input.to_string_lossy().into_owned(),
        "--model-id".into(),
        model_id(),
        "--policy".into(),
        policy.to_string_lossy().into_owned(),
        "--run-dir".into(),
        temp.path().join("run2").to_string_lossy().into_owned(),
    ])
    .unwrap();
    let plan = reuse.plan().unwrap();
    assert_eq!(plan.len(), 2);
    assert_eq!(plan[0][0], "prove");
}

#[test]
fn production_provider_refuses_to_fabricate_a_proof() {
    let temp = TempDir::new();
    let case = temp.path().join("case.toml");
    let assets = temp.path().join("assets.json");
    let parameters = temp.path().join("parameters.toml");
    for path in [&case, &assets, &parameters] {
        write_file(path, b"x");
    }
    let run_dir = temp.path().join("run");
    let args = vec![
        "--case".into(),
        case.to_string_lossy().into_owned(),
        "--assets-manifest".into(),
        assets.to_string_lossy().into_owned(),
        "--parameters".into(),
        parameters.to_string_lossy().into_owned(),
        "--out".into(),
        temp.path().join("commit").to_string_lossy().into_owned(),
        "--run-dir".into(),
        run_dir.to_string_lossy().into_owned(),
    ];
    assert!(matches!(
        commit::run(&args, &UnimplementedProvider),
        Err(Error::UnsupportedPrimitive)
    ));
    let result = texts(&run_dir.join("result.json"));
    assert!(result.contains("BLOCKED"));
    assert!(!temp.path().join("commit").exists());
}
