//! Real-verifier negative matrix for the final norm segment carrier.
//!
//! Ignored by default. It consumes one completed acceptance round (a commit
//! public bundle, a proof artifact and the round policy) and injects every
//! tamper through the full public verify path: the tampered file is written
//! into a self-consistent proof artifact (its manifest entry and the complete
//! marker are recomputed), so the artifact-integrity layer accepts it and the
//! segment checks must reject it.
//!
//! Environment:
//! - ZKIE_NEG_ROUND: the round directory (commit/, proof/, policy.toml)
//! - ZKIE_NEG_BIN: the zkie binary
//! - ZKIE_NEG_OUT: scratch output directory
//! - ZKIE_NEG_SKIP_TRACE: any value skips the large source-row case

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use bincode::Options;
use serde_json::Value;
use zkie_core::artifact::{
    digest_bytes, digest_to_hex, ArtifactManifest, COMPLETE_FILE, MANIFEST_FILE,
};
use zkie_engine::artifact::load_manifest;
use zkie_engine::pcs::request::field::Fp3;
use zkie_ops::proof::model_commitment::SharedTrace;

use zkie::norm_segment_provider::{NormSegmentCarrier, NORM_SEGMENT_CARRIER_FILE};

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

/// Rewrites one proof artifact with the tampered file bytes and a manifest
/// that binds the new length and digest, so the artifact layer accepts it.
fn publish_tampered(
    round: &Path,
    dir: &Path,
    tampered: &str,
    bytes: &[u8],
) -> ArtifactManifest {
    let proof = round.join("proof");
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    let mut manifest = load_manifest(&proof).unwrap();
    for file in &manifest.files {
        let name = &file.relative_path;
        let target = dir.join(name);
        if name == tampered {
            fs::write(&target, bytes).unwrap();
        } else {
            link_or_copy(&proof.join(name), &target);
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

fn run_verify(
    bin: &Path,
    round: &Path,
    proof_dir: &Path,
    run_dir: &Path,
    model_id: &str,
) -> (bool, String) {
    let _ = fs::remove_dir_all(run_dir);
    let status = Command::new(bin)
        .args([
            "verify",
            "--public",
            round.join("commit/public").to_str().unwrap(),
            "--statement",
            proof_dir.join("statement.bin").to_str().unwrap(),
            "--proof",
            proof_dir.join("proof.bin").to_str().unwrap(),
            "--public-inputs",
            proof_dir.join("public-inputs.bin").to_str().unwrap(),
            "--model-id",
            model_id,
            "--policy",
            round.join("policy.toml").to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    (status.success(), stage_outcome(run_dir))
}

fn bincode_fixint<T: serde::Serialize>(value: &T) -> Vec<u8> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(value)
        .unwrap()
}

#[test]
#[ignore = "real-verifier negatives"]
fn norm_segment_carrier_negatives() {
    let round = PathBuf::from(required("ZKIE_NEG_ROUND"));
    let bin = PathBuf::from(required("ZKIE_NEG_BIN"));
    let out = PathBuf::from(required("ZKIE_NEG_OUT"));
    fs::create_dir_all(&out).unwrap();
    let proof = round.join("proof");

    let statement: zkie::provider::StatementData =
        bincode::deserialize(&fs::read(proof.join("statement.bin")).unwrap()).unwrap();
    let model_id = digest_to_hex(&statement.model_commitment);
    let original = fs::read(proof.join(NORM_SEGMENT_CARRIER_FILE)).unwrap();
    let carrier = NormSegmentCarrier::decode(&original).expect("the round carried a segment carrier");

    let mut summary = Vec::new();

    // Control: the untouched artifact must pass through the same harness.
    let (ok, outcome) = run_verify(&bin, &round, &proof, &out.join("control-run"), &model_id);
    assert!(ok && outcome == "PASS", "control: {ok} {outcome}");
    summary.push(serde_json::json!({"case": "control", "outcome": outcome}));

    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();

    let mut bumped = carrier.clone();
    bumped.version += 1;
    cases.push(("version", bincode_fixint(&bumped)));

    let mut wrong_root = carrier.clone();
    wrong_root.reader.root = zkie_core::types::Digest([0xAA; 32]);
    cases.push(("reader_root", wrong_root.encode().unwrap()));

    // The segment identity rides the statement; a changed identity must not
    // verify against the same carrier.
    let mut tampered_statement: zkie::provider::StatementData =
        bincode::deserialize(&fs::read(proof.join("statement.bin")).unwrap()).unwrap();
    if let Some(identity) = tampered_statement.segment_identity.as_mut() {
        identity.table_root = zkie_core::types::Digest([0xAB; 32]);
        let bytes = bincode::serialize(&tampered_statement).unwrap();
        let dir = out.join("case-statement_identity");
        publish_tampered(&round, &dir, "statement.bin", &bytes);
        let (ok, outcome) = run_verify(
            &bin,
            &round,
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
    }

    let mut wrong_record = carrier.clone();
    wrong_record.reader.record.y[0] = Fp3::from_u64(2);
    cases.push(("record_digest", wrong_record.encode().unwrap()));

    // The statement names the bounded table source; a statement that claims a
    // different source must not verify against the same carrier.
    let mut tampered_statement: zkie::provider::StatementData =
        bincode::deserialize(&fs::read(proof.join("statement.bin")).unwrap()).unwrap();
    if let Some(identity) = tampered_statement.segment_identity.as_mut() {
        identity.source_identity = zkie_core::types::Digest([0xAC; 32]);
        let bytes = bincode::serialize(&tampered_statement).unwrap();
        let dir = out.join("case-statement_source_identity");
        publish_tampered(&round, &dir, "statement.bin", &bytes);
        let (ok, outcome) = run_verify(
            &bin,
            &round,
            &dir,
            &out.join("run-statement_source_identity"),
            &model_id,
        );
        assert!(
            !ok || outcome != "PASS",
            "the statement source identity tamper was accepted"
        );
        summary.push(serde_json::json!({
            "case": "statement_source_identity",
            "exit_ok": ok,
            "outcome": outcome,
        }));
    }

    let mut truncated = carrier.clone();
    truncated
        .messages
        .truncate(truncated.messages.len().saturating_sub(1));
    cases.push(("messages_truncated", truncated.encode().unwrap()));

    for (name, bytes) in cases {
        let dir = out.join(format!("case-{name}"));
        publish_tampered(&round, &dir, NORM_SEGMENT_CARRIER_FILE, &bytes);
        let (ok, outcome) = run_verify(
            &bin,
            &round,
            &dir,
            &out.join(format!("run-{name}")),
            &model_id,
        );
        assert!(!ok || outcome != "PASS", "case {name} was accepted");
        summary.push(serde_json::json!({"case": name, "exit_ok": ok, "outcome": outcome}));
    }

    if std::env::var_os("ZKIE_NEG_SKIP_TRACE").is_none() {
        // The materialized source rows: change one final-norm output value in
        // the authenticated trace; the activation identity must not match.
        let trace_bytes = fs::read(proof.join("proof.bin")).unwrap();
        let mut trace: SharedTrace = bincode::deserialize(&trace_bytes).unwrap();
        let target = zkie_ops::op::OpCode::ProfileLayerNorm as u16;
        let step = trace
            .steps
            .iter_mut()
            .rev()
            .find(|step| step.message.opcode == target)
            .expect("the trace carries the final norm step");
        step.message.proof.outputs[0][0].values[0] =
            step.message.proof.outputs[0][0].values[0].wrapping_add(1);
        let tampered = bincode::serialize(&trace).unwrap();
        let dir = out.join("case-source_rows");
        publish_tampered(&round, &dir, zkie_core::artifact::PROOF_FILE, &tampered);
        let (ok, outcome) = run_verify(
            &bin,
            &round,
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
