//! Real-verifier negative matrix for the timesfm norm-only segment carrier.
//!
//! Ignored by default. It consumes one completed acceptance round (a commit
//! bundle, a proof artifact and the round policy) and injects every tamper
//! through the full public verify path. Proof-artifact tampers rewrite the
//! artifact manifest entry and complete marker, so the artifact-integrity
//! layer accepts the file and the segment checks must reject it. The gain
//! reader-root and record-digest cases tamper the commit bundle's asset-table
//! metadata (the trusted source of the L0 gain identity) and rewrite the
//! commit manifest the same way.
//!
//! Environment:
//! - ZKIE_NEG_ROUND: the round directory (commit/, proof/, policy.toml)
//! - ZKIE_NEG_BIN: the zkie binary
//! - ZKIE_NEG_OUT: scratch output directory
//! - ZKIE_NEG_SKIP_TRACE: any value skips the large source-row case

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use zkie_core::artifact::{
    digest_bytes, digest_to_hex, ArtifactManifest, COMPLETE_FILE, MANIFEST_FILE, PROOF_FILE,
};
use zkie_engine::artifact::load_manifest;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_ops::proof::asset_table::AssetTableMeta;
use zkie_ops::proof::model_commitment::SharedTrace;

use zkie::provider::StatementData;
use zkie::timesfm_norm_segment_provider::{TimesfmNormCarrier, TIMESFM_NORM_CARRIER_FILE};

const GAIN_PATH: &str = "models/timesfm/weights/L0_lnw_i32.bin";

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}

fn stage_outcome(run_dir: &Path) -> String {
    let text = fs::read_to_string(run_dir.join("result.json")).unwrap_or_default();
    let value: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    value["outcome"].as_str().unwrap_or("").to_owned()
}

fn link_or_copy(from: &Path, to: &Path) {
    if fs::hard_link(from, to).is_err() {
        fs::copy(from, to).unwrap();
    }
}

/// Rewrites one artifact with the tampered file bytes and a manifest that
/// binds the new length and digest, so the artifact layer accepts it.
fn publish_tampered(source_root: &Path, dir: &Path, tampered: &str, bytes: &[u8]) -> ArtifactManifest {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let mut manifest = load_manifest(source_root).unwrap();
    for file in &manifest.files {
        let name = &file.relative_path;
        let target = dir.join(name);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        if name == tampered {
            fs::write(&target, bytes).unwrap();
        } else {
            link_or_copy(&source_root.join(name), &target);
        }
    }
    let entry = manifest
        .files
        .iter_mut()
        .find(|file| file.relative_path == tampered)
        .expect("the manifest lists the tampered file");
    entry.byte_length = bytes.len() as u64;
    entry.content_digest = digest_bytes(bytes);
    let digest = manifest.manifest_digest().unwrap();
    fs::write(dir.join(MANIFEST_FILE), manifest.to_json().unwrap()).unwrap();
    fs::write(
        dir.join(COMPLETE_FILE),
        format!("{}\n", digest_to_hex(&digest)),
    )
    .unwrap();
    manifest
}

/// Rewrites the artifact without one file and with a tampered statement, so
/// neither the manifest nor the complete marker mention the omitted file.
fn publish_omitted_with_statement(
    source_root: &Path,
    dir: &Path,
    omitted: &str,
    statement: &[u8],
) -> ArtifactManifest {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let mut manifest = load_manifest(source_root).unwrap();
    manifest
        .files
        .retain(|file| file.relative_path != omitted);
    for file in &manifest.files {
        let name = &file.relative_path;
        let target = dir.join(name);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        if name == "statement.bin" {
            fs::write(&target, statement).unwrap();
        } else {
            link_or_copy(&source_root.join(name), &target);
        }
    }
    let entry = manifest
        .files
        .iter_mut()
        .find(|file| file.relative_path == "statement.bin")
        .expect("the manifest lists the statement");
    entry.byte_length = statement.len() as u64;
    entry.content_digest = digest_bytes(statement);
    let digest = manifest.manifest_digest().unwrap();
    fs::write(dir.join(MANIFEST_FILE), manifest.to_json().unwrap()).unwrap();
    fs::write(
        dir.join(COMPLETE_FILE),
        format!("{}\n", digest_to_hex(&digest)),
    )
    .unwrap();
    manifest
}

fn run_verify(
    bin: &Path,
    commit_root: &Path,
    policy: &Path,
    proof_dir: &Path,
    run_dir: &Path,
    model_id: &str,
) -> (bool, String) {
    let _ = fs::remove_dir_all(run_dir);
    let status = Command::new(bin)
        .args([
            "verify",
            "--public",
            commit_root.join("public").to_str().unwrap(),
            "--statement",
            proof_dir.join("statement.bin").to_str().unwrap(),
            "--proof",
            proof_dir.join("proof.bin").to_str().unwrap(),
            "--public-inputs",
            proof_dir.join("public-inputs.bin").to_str().unwrap(),
            "--model-id",
            model_id,
            "--policy",
            policy.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    (status.success(), stage_outcome(run_dir))
}

fn carrier_version_two(carrier: &TimesfmNormCarrier) -> Vec<u8> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(2);
    writer.put_bytes(&carrier.first);
    writer.put_bytes(&carrier.second);
    writer.into_bytes()
}

#[test]
#[ignore = "real-verifier negatives"]
fn timesfm_norm_only_negatives() {
    let round = PathBuf::from(required("ZKIE_NEG_ROUND"));
    let bin = PathBuf::from(required("ZKIE_NEG_BIN"));
    let out = PathBuf::from(required("ZKIE_NEG_OUT"));
    fs::create_dir_all(&out).unwrap();
    let commit = round.join("commit");
    let proof = round.join("proof");

    let statement: StatementData =
        bincode::deserialize(&fs::read(proof.join("statement.bin")).unwrap()).unwrap();
    let model_id = digest_to_hex(&statement.model_commitment);
    let original = fs::read(proof.join(TIMESFM_NORM_CARRIER_FILE)).unwrap();
    let carrier = TimesfmNormCarrier::decode(&original).expect("the round carried a segment carrier");

    let mut summary = Vec::new();

    // Control: the untouched artifact must pass through the same harness.
    let (ok, outcome) = run_verify(&bin, &commit, &round.join("policy.toml"), &proof, &out.join("control-run"), &model_id);
    assert!(ok && outcome == "PASS", "control: {ok} {outcome}");
    summary.push(serde_json::json!({"case": "control", "outcome": outcome}));

    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();

    cases.push(("version", carrier_version_two(&carrier)));

    // The carrier has no reader field; the reader root and record digest are
    // the gain table identity in the commit bundle and are covered below.

    let mut truncated = original.clone();
    truncated.pop();
    cases.push(("messages_truncated", truncated));

    for (name, bytes) in cases {
        let dir = out.join(format!("case-{name}"));
        publish_tampered(&proof, &dir, TIMESFM_NORM_CARRIER_FILE, &bytes);
        let (ok, outcome) = run_verify(
            &bin,
            &commit,
            &round.join("policy.toml"),
            &dir,
            &out.join(format!("run-{name}")),
            &model_id,
        );
        assert!(!ok || outcome != "PASS", "case {name} was accepted");
        summary.push(serde_json::json!({"case": name, "exit_ok": ok, "outcome": outcome}));
    }

    // The segment declaration rides the statement; a carrier without the
    // declaration must not verify.
    let mut tampered_statement: StatementData =
        bincode::deserialize(&fs::read(proof.join("statement.bin")).unwrap()).unwrap();
    tampered_statement.norm_only = false;
    let bytes = bincode::serialize(&tampered_statement).unwrap();
    let dir = out.join("case-statement_identity");
    publish_tampered(&proof, &dir, "statement.bin", &bytes);
    let (ok, outcome) = run_verify(
        &bin,
        &commit,
        &round.join("policy.toml"),
        &dir,
        &out.join("run-statement_identity"),
        &model_id,
    );
    assert!(!ok || outcome != "PASS", "the statement identity tamper was accepted");
    summary.push(serde_json::json!({
        "case": "statement_identity",
        "exit_ok": ok,
        "outcome": outcome,
    }));

    // Omission: the statement turns the declaration off and the artifact no
    // longer contains the carrier at all. The trusted case still requires the
    // segment, so the verify must reject rather than skip it.
    let mut omitted_statement: StatementData =
        bincode::deserialize(&fs::read(proof.join("statement.bin")).unwrap()).unwrap();
    omitted_statement.norm_only = false;
    let bytes = bincode::serialize(&omitted_statement).unwrap();
    let dir = out.join("case-declaration_omitted");
    publish_omitted_with_statement(&proof, &dir, TIMESFM_NORM_CARRIER_FILE, &bytes);
    let (ok, outcome) = run_verify(
        &bin,
        &commit,
        &round.join("policy.toml"),
        &dir,
        &out.join("run-declaration_omitted"),
        &model_id,
    );
    assert!(
        !ok || outcome != "PASS",
        "the declaration omission was accepted"
    );
    summary.push(serde_json::json!({
        "case": "declaration_omitted",
        "exit_ok": ok,
        "outcome": outcome,
    }));

    // Reader root and record digest: the trusted gain identity lives in the
    // commit bundle's asset-table metadata; tamper it and rewrite the commit
    // manifest so the artifact layer accepts the bundle.
    let table_bytes = fs::read(commit.join("public/asset-tables.bin")).unwrap();
    let tables: Vec<AssetTableMeta> = bincode::deserialize(&table_bytes).unwrap();
    for (case, mutate) in [
        ("reader_root", 0u8),
        ("record_digest", 1u8),
    ] {
        let mut tampered_tables = tables.clone();
        let meta = tampered_tables
            .iter_mut()
            .find(|meta| meta.path == GAIN_PATH)
            .expect("the commit lists the gain table");
        if mutate == 0 {
            meta.root = zkie_core::types::Digest([0xAB; 32]);
        } else {
            meta.record.y[0] = Fp3::from_u64(2);
        }
        let bytes = bincode::serialize(&tampered_tables).unwrap();
        let dir = out.join(format!("case-{case}"));
        publish_tampered(&commit, &dir, "public/asset-tables.bin", &bytes);
        let (ok, outcome) = run_verify(
            &bin,
            &dir,
            &round.join("policy.toml"),
            &proof,
            &out.join(format!("run-{case}")),
            &model_id,
        );
        assert!(!ok || outcome != "PASS", "case {case} was accepted");
        summary.push(serde_json::json!({"case": case, "exit_ok": ok, "outcome": outcome}));
    }

    if std::env::var_os("ZKIE_NEG_SKIP_TRACE").is_none() {
        // The materialized activation rows: change one L0 RMS-norm output
        // value in the authenticated trace; the segment's activation identity
        // must not match.
        let trace_bytes = fs::read(proof.join(PROOF_FILE)).unwrap();
        let mut trace: SharedTrace = bincode::deserialize(&trace_bytes).unwrap();
        let target = zkie_ops::op::OpCode::ProfileRmsNorm as u16;
        let step = trace
            .steps
            .iter_mut()
            .find(|step| step.message.opcode == target)
            .expect("the trace carries the L0 RMS norm step");
        step.message.proof.outputs[0][0].values[0] =
            step.message.proof.outputs[0][0].values[0].wrapping_add(1);
        let tampered = bincode::serialize(&trace).unwrap();
        let dir = out.join("case-source_rows");
        publish_tampered(&proof, &dir, PROOF_FILE, &tampered);
        let (ok, outcome) = run_verify(
            &bin,
            &commit,
            &round.join("policy.toml"),
            &dir,
            &out.join("run-source_rows"),
            &model_id,
        );
        assert!(!ok || outcome != "PASS", "the source-row tamper was accepted");
        summary.push(serde_json::json!({
            "case": "source_rows",
            "exit_ok": ok,
            "outcome": outcome,
        }));
    }

    fs::write(
        out.join("negatives.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();
}
