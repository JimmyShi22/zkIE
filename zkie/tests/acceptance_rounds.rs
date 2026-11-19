//! Acceptance rounds for one case through the production three-stage CLI.
//!
//! Ignored by default. One invocation runs `rounds` independent rounds; every
//! round creates its own commit (a fresh table instance, as the durable column
//! budget permits exactly one request per instance) and provisions the
//! verifier policy from that commit's public bundle. The three stages run as
//! separate CLI processes, so the stage wall time and VmHWM are per-stage
//! measurements. Oversized proof binaries are deleted after their measurements
//! are recorded; every textual artifact stays.
//!
//! Environment:
//! - ZKIE_ACCEPT_CASE_FILE, ZKIE_ACCEPT_CASE_ID, ZKIE_ACCEPT_MODEL
//! - ZKIE_ACCEPT_POLICY_TEMPLATE
//! - ZKIE_ACCEPT_ROUNDS, ZKIE_ACCEPT_OUT
//! - ZKIE_ACCEPT_BIN (default `target/release/zkie`)
//! - ZKIE_ACCEPT_KEEP (any value keeps proof binaries)

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use zkie_core::artifact::digest_to_hex;
use zkie_ops::proof::asset_table::AssetTableMeta;
use zkie_ops::proof::model_commitment::ModelCommitmentPublic;

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}

fn stage_result(run_dir: &Path) -> Value {
    let text = fs::read_to_string(run_dir.join("result.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn run_stage(bin: &Path, args: &[String], run_dir: &Path, case_id: &str) -> Value {
    let status = Command::new(bin)
        .args(args)
        .env("ZKIE_CASE_ID", case_id)
        .status()
        .unwrap();
    assert!(status.success(), "stage failed: {args:?}");
    let result = stage_result(run_dir);
    assert_eq!(result["outcome"], "PASS", "stage outcome: {result}");
    result
}

/// The single-case policy: the template's trusted fields with the anchor and
/// model identity of this commit. The gpt2 case also gets the committed table
/// identity derived from the same public bundle.
fn write_policy(
    template: &Path,
    case_id: &str,
    model: &str,
    anchor_id: &str,
    model_id: &str,
    metas: &[AssetTableMeta],
    keep_output_head: bool,
) -> String {
    let text = fs::read_to_string(template).unwrap();
    let mut header = String::new();
    let mut blocks: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_case = false;
    for line in text.lines() {
        if line.trim() == "[[case]]" {
            if in_case {
                blocks.push(current);
                current = String::new();
            }
            in_case = true;
            current.push_str(line);
            current.push('\n');
        } else if in_case {
            current.push_str(line);
            current.push('\n');
        } else {
            header.push_str(line);
            header.push('\n');
        }
    }
    if in_case {
        blocks.push(current);
    }
    let mut block = blocks
        .into_iter()
        .find(|block| block.contains(&format!("case_id = \"{case_id}\"")))
        .expect("the template lists the case");
    let mut kept_output_head: Option<String> = None;
    if let Some(cut) = block.find("[case.limited_table_identity]") {
        if keep_output_head && model == "gpt2" {
            let mut fixed = String::new();
            for line in block[cut..].lines() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("anchor_id = ") {
                    fixed.push_str(&format!("anchor_id = \"{anchor_id}\"\n"));
                } else if trimmed.starts_with("model_id = ") {
                    fixed.push_str(&format!("model_id = \"{model_id}\"\n"));
                } else {
                    fixed.push_str(line);
                    fixed.push('\n');
                }
            }
            kept_output_head = Some(fixed);
        }
        block.truncate(cut);
    }
    let model_version = header
        .lines()
        .find_map(|line| line.trim().strip_prefix("model_version = "))
        .unwrap_or("1")
        .to_owned();
    let mut out = header;
    for line in block.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("anchor_id = ") {
            out.push_str(&format!("anchor_id = \"{anchor_id}\"\n"));
        } else if trimmed.starts_with("model_id = ") {
            out.push_str(&format!("model_id = \"{model_id}\"\n"));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    let identity_meta = if model == "gpt2" {
        if let Some(kept) = kept_output_head {
            out.push_str(&kept);
            None
        } else {
            metas.first()
        }
    } else if !metas.is_empty() {
        metas.first()
    } else {
        None
    };
    if let Some(meta) = identity_meta {
        out.push_str(&format!(
            "\n[case.limited_table_identity]\nversion = 1\nmodel = \"{model}\"\nmodel_version = {model_version}\ncase_id = \"{case_id}\"\nanchor_id = \"{anchor_id}\"\nmodel_id = \"{model_id}\"\nasset_id = {}\nasset_table_identity = \"{}\"\ntable_root = \"{}\"\ntable_record_digest = \"{}\"\ntable_configuration = \"{}\"\ncode_log_n = {}\ncode_ell = {}\ncode_r = {}\ntable_v = {}\nanswers = {}\nsalt_bytes = {}\ndistance_radius = {}\ndistance_later_radius = {}\ndistance_m_once = {}\ndistance_m_later = {}\nmatrix_rows_log = {}\nk_lo = 0\nk_col = {}\nprotocol_identity = \"{}\"\n",
            meta.asset_id,
            digest_to_hex(&meta.identity),
            digest_to_hex(&meta.root),
            digest_to_hex(&zkie_core::artifact::digest_bytes(&meta.record.encode())),
            digest_to_hex(&meta.configuration),
            meta.shape.code.log_n,
            meta.shape.code.ell,
            meta.shape.code.r,
            meta.shape.v,
            meta.shape.answers,
            meta.shape.salt_bytes,
            meta.shape.distances.radius,
            meta.shape.distances.later_radius,
            meta.shape.distances.m_once,
            meta.shape.distances.m_later,
            meta.shape.v,
            meta.shape.v,
            digest_to_hex(&zkie_core::types::Digest([0x5A; 32])),
        ));
    }
    out
}

#[test]
#[ignore = "acceptance rounds"]
fn acceptance_rounds() {
    let case_file = PathBuf::from(required("ZKIE_ACCEPT_CASE_FILE"));
    let case_id = required("ZKIE_ACCEPT_CASE_ID");
    let model = required("ZKIE_ACCEPT_MODEL");
    let template = PathBuf::from(required("ZKIE_ACCEPT_POLICY_TEMPLATE"));
    let rounds: u32 = required("ZKIE_ACCEPT_ROUNDS").parse().unwrap();
    let out = PathBuf::from(required("ZKIE_ACCEPT_OUT"));
    let bin = std::env::var_os("ZKIE_ACCEPT_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/release/zkie"));
    let keep = std::env::var_os("ZKIE_ACCEPT_KEEP").is_some();
    let bin = if bin.is_absolute() {
        bin
    } else {
        std::env::current_dir().unwrap().join(bin)
    };
    fs::create_dir_all(&out).unwrap();

    let mut summary = Vec::new();
    for round in 1..=rounds {
        let dir = out.join(format!("round-{round}"));
        let _ = fs::remove_dir_all(&dir);
        let commit = dir.join("commit");
        let proof = dir.join("proof");
        let commit_run = dir.join("commit-run");
        let prove_run = dir.join("prove-run");
        let verify_run = dir.join("verify-run");
        fs::create_dir_all(&dir).unwrap();
        let parameters = dir.join("parameters.toml");
        fs::write(&parameters, "profile = \"case\"\n").unwrap();

        let commit_result = run_stage(
            &bin,
            &[
                "commit".into(),
                "--case".into(),
                case_file.to_string_lossy().into_owned(),
                "--assets-manifest".into(),
                case_file.to_string_lossy().into_owned(),
                "--parameters".into(),
                parameters.to_string_lossy().into_owned(),
                "--out".into(),
                commit.to_string_lossy().into_owned(),
                "--run-dir".into(),
                commit_run.to_string_lossy().into_owned(),
            ],
            &commit_run,
            &case_id,
        );

        let public: ModelCommitmentPublic =
            bincode::deserialize(&fs::read(commit.join("public/model-commitment.bin")).unwrap())
                .unwrap();
        let anchor_id = digest_to_hex(&public.anchor.anchor_id);
        let model_id = digest_to_hex(&public.anchor.model_commitment);
        let metas: Vec<AssetTableMeta> = fs::read(commit.join("public/asset-tables.bin"))
            .ok()
            .and_then(|bytes| bincode::deserialize(&bytes).ok())
            .unwrap_or_default();
        let policy_text = write_policy(
            &template,
            &case_id,
            &model,
            &anchor_id,
            &model_id,
            &metas,
            std::env::var("ZKIE_ACCEPT_OUTPUT_HEAD_IDENTITY")
                .map(|value| value == "template")
                .unwrap_or(false),
        );
        let policy = dir.join("policy.toml");
        fs::write(&policy, policy_text).unwrap();

        let prove_result = run_stage(
            &bin,
            &[
                "prove".into(),
                "--commit".into(),
                commit.to_string_lossy().into_owned(),
                "--input".into(),
                case_file.to_string_lossy().into_owned(),
                "--model-id".into(),
                model_id.clone(),
                "--out".into(),
                proof.to_string_lossy().into_owned(),
                "--run-dir".into(),
                prove_run.to_string_lossy().into_owned(),
            ],
            &prove_run,
            &case_id,
        );
        let proof_bytes = prove_result["proof_bytes"].clone();

        let verify_result = run_stage(
            &bin,
            &[
                "verify".into(),
                "--public".into(),
                commit.join("public").to_string_lossy().into_owned(),
                "--statement".into(),
                proof.join("statement.bin").to_string_lossy().into_owned(),
                "--proof".into(),
                proof.join("proof.bin").to_string_lossy().into_owned(),
                "--public-inputs".into(),
                proof.join("public-inputs.bin").to_string_lossy().into_owned(),
                "--model-id".into(),
                model_id.clone(),
                "--policy".into(),
                policy.to_string_lossy().into_owned(),
                "--run-dir".into(),
                verify_run.to_string_lossy().into_owned(),
            ],
            &verify_run,
            &case_id,
        );

        let deleted = if keep {
            None
        } else {
            let path = proof.join("proof.bin");
            let bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            fs::remove_file(&path).unwrap();
            Some(bytes)
        };
        summary.push(serde_json::json!({
            "round": round,
            "anchor_id": anchor_id,
            "model_id": model_id,
            "commit": commit_result,
            "prove": prove_result,
            "verify": verify_result,
            "proof_bytes": proof_bytes,
            "proof_bin_deleted_bytes": deleted,
        }));
        eprintln!(
            "ROUND {} commit={} prove={} verify={} proof_bytes={}",
            round,
            summary.last().unwrap()["commit"]["stage_wall_ms"],
            summary.last().unwrap()["prove"]["stage_wall_ms"],
            summary.last().unwrap()["verify"]["stage_wall_ms"],
            proof_bytes
        );
    }
    let body = serde_json::to_string_pretty(&serde_json::json!({
        "case_id": case_id,
        "model": model,
        "rounds": summary,
    }))
    .unwrap();
    fs::write(out.join("summary.json"), body).unwrap();
}
