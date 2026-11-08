//! `run` subcommand: serial orchestration of three independent processes.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use zkie_core::artifact::COMPLETE_FILE;
use zkie_core::error::{Error, Result};

use crate::report::{parse_flags, path_flag, require_flag};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitPolicy {
    Create,
    Reuse(PathBuf),
}

#[derive(Clone, Debug)]
pub struct RunArgs {
    pub policy: CommitPolicy,
    pub case_config: PathBuf,
    pub assets_manifest: PathBuf,
    pub parameters: PathBuf,
    pub input_request: PathBuf,
    pub model_id: String,
    pub verifier_policy: PathBuf,
    pub run_dir: PathBuf,
}

impl RunArgs {
    pub fn parse(args: &[String]) -> Result<Self> {
        let flags = parse_flags(
            args,
            &[
                "commit-policy",
                "commit-dir",
                "case",
                "assets-manifest",
                "parameters",
                "input",
                "model-id",
                "policy",
                "run-dir",
            ],
        )?;
        let policy = match require_flag(&flags, "commit-policy")? {
            "create" => CommitPolicy::Create,
            "reuse" => CommitPolicy::Reuse(path_flag(&flags, "commit-dir")?),
            _ => return Err(Error::InvalidConfig),
        };
        Ok(Self {
            policy,
            case_config: path_flag(&flags, "case")?,
            assets_manifest: path_flag(&flags, "assets-manifest")?,
            parameters: path_flag(&flags, "parameters")?,
            input_request: path_flag(&flags, "input")?,
            model_id: require_flag(&flags, "model-id")?.to_owned(),
            verifier_policy: path_flag(&flags, "policy")?,
            run_dir: path_flag(&flags, "run-dir")?,
        })
    }

    pub fn plan(&self) -> Result<Vec<Vec<String>>> {
        let commit_dir = match &self.policy {
            CommitPolicy::Create => self.run_dir.join("commit"),
            CommitPolicy::Reuse(dir) => {
                if !dir.join(COMPLETE_FILE).exists() {
                    return Err(Error::MissingCommitArtifact);
                }
                dir.clone()
            }
        };
        let proof_dir = self.run_dir.join("proof");
        let run_dir = self.run_dir.to_string_lossy();
        let commit_dir_text = commit_dir.to_string_lossy();
        let proof_dir_text = proof_dir.to_string_lossy();
        let public_dir = commit_dir.join("public");

        let mut steps = Vec::new();
        if matches!(self.policy, CommitPolicy::Create) {
            steps.push(vec![
                "commit".into(),
                "--case".into(),
                self.case_config.to_string_lossy().into_owned(),
                "--assets-manifest".into(),
                self.assets_manifest.to_string_lossy().into_owned(),
                "--parameters".into(),
                self.parameters.to_string_lossy().into_owned(),
                "--out".into(),
                commit_dir_text.clone().into_owned(),
                "--run-dir".into(),
                format!("{run_dir}/commit-run"),
            ]);
        }
        steps.push(vec![
            "prove".into(),
            "--commit".into(),
            commit_dir_text.clone().into_owned(),
            "--input".into(),
            self.input_request.to_string_lossy().into_owned(),
            "--model-id".into(),
            self.model_id.clone(),
            "--out".into(),
            proof_dir_text.clone().into_owned(),
            "--run-dir".into(),
            format!("{run_dir}/prove"),
        ]);
        steps.push(vec![
            "verify".into(),
            "--public".into(),
            public_dir.to_string_lossy().into_owned(),
            "--statement".into(),
            format!("{proof_dir_text}/statement.bin"),
            "--proof".into(),
            format!("{proof_dir_text}/proof.bin"),
            "--public-inputs".into(),
            format!("{proof_dir_text}/public-inputs.bin"),
            "--model-id".into(),
            self.model_id.clone(),
            "--policy".into(),
            self.verifier_policy.to_string_lossy().into_owned(),
            "--run-dir".into(),
            format!("{run_dir}/verify"),
        ]);
        Ok(steps)
    }
}

fn manifest_model_hex(commit_dir: &Path) -> Result<String> {
    let text = fs::read_to_string(commit_dir.join("manifest.json"))
        .map_err(|_| Error::MissingCommitArtifact)?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|_| Error::InvalidEncoding)?;
    let bytes = value
        .get("model_commitment_id")
        .and_then(|value| value.as_array())
        .ok_or(Error::InvalidEncoding)?;
    let mut hex = String::with_capacity(64);
    for byte in bytes {
        let byte = byte.as_u64().ok_or(Error::InvalidEncoding)? as u8;
        hex.push_str(&format!("{byte:02x}"));
    }
    Ok(hex)
}

pub fn run(args: &[String]) -> Result<Vec<i32>> {
    let parsed = RunArgs::parse(args)?;
    let plan = parsed.plan()?;
    let executable = std::env::current_exe().map_err(|_| Error::Io)?;
    let mut codes = Vec::with_capacity(plan.len());
    let commit_dir = match &parsed.policy {
        CommitPolicy::Create => parsed.run_dir.join("commit"),
        CommitPolicy::Reuse(dir) => dir.clone(),
    };
    let mut index = 0usize;
    if matches!(parsed.policy, CommitPolicy::Create) {
        let status = Command::new(&executable)
            .args(&plan[0])
            .env("ZKIE_CASE_ID", std::env::var("ZKIE_CASE_ID").unwrap_or_default())
            .status()
            .map_err(|_| Error::Io)?;
        codes.push(status.code().unwrap_or(1));
        index = 1;
    }
    let model_id = manifest_model_hex(&commit_dir).unwrap_or_else(|_| parsed.model_id.clone());
    for step in &plan[index..] {
        let mut step = step.clone();
        if let Some(position) = step.iter().position(|arg| arg == "--model-id") {
            if let Some(value) = step.get_mut(position + 1) {
                *value = model_id.clone();
            }
        }
        let status = Command::new(&executable)
            .args(&step)
            .env("ZKIE_CASE_ID", std::env::var("ZKIE_CASE_ID").unwrap_or_default())
            .status()
            .map_err(|_| Error::Io)?;
        codes.push(status.code().unwrap_or(1));
    }
    Ok(codes)
}

// ---------------------------------------------------------------------------
// matrix / decode-matrix orchestration.
//
// Cases are executed serially; each run is a separate `run` subprocess, which
// in turn starts commit/prove/verify as separate processes. Reports are
// collected without rewriting stage outcomes.


use std::path::Path;

use zkie_core::artifact::{digest_bytes, digest_to_hex};


#[derive(Clone, Debug, Default)]
pub(crate) struct CaseEntry {
    pub(crate) id: String,
    pub(crate) input: String,
    pub(crate) seq_tokens: usize,
}

pub(crate) fn parse_cases(path: &Path) -> Result<(std::path::PathBuf, Vec<CaseEntry>)> {
    let text = fs::read_to_string(path).map_err(|_| Error::Io)?;
    let mut root = std::path::PathBuf::new();
    let mut entries = Vec::new();
    let mut current: Option<CaseEntry> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("root = ") {
            let value = trimmed.trim_start_matches("root = ").trim_matches('"');
            root = std::path::PathBuf::from(value);
        }
        if trimmed == "[[case]]" {
            if let Some(entry) = current.take() {
                entries.push(entry);
            }
            current = Some(CaseEntry::default());
            continue;
        }
        if let Some(entry) = current.as_mut() {
            if let Some(rest) = trimmed.strip_prefix("id = ") {
                entry.id = rest.trim().trim_matches('"').to_owned();
            } else if let Some(rest) = trimmed.strip_prefix("input = {") {
                if let Some(position) = rest.find("path = \"") {
                    let after = &rest[position + 8..];
                    if let Some(end) = after.find('"') {
                        entry.input = after[..end].to_owned();
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("seq_tokens = ") {
                entry.seq_tokens = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = trimmed.strip_prefix("patch_tokens = ") {
                entry.seq_tokens = rest.trim().parse().unwrap_or(0);
            }
        }
    }
    if let Some(entry) = current.take() {
        entries.push(entry);
    }
    Ok((root, entries))
}

fn write_placeholder(path: &Path, body: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|_| Error::Io)?;
    }
    fs::write(path, body).map_err(|_| Error::Io)
}

fn run_one(
    executable: &Path,
    case: &Path,
    case_id: &str,
    input: &Path,
    run_dir: &Path,
    commit_policy: &str,
    commit_dir: Option<&Path>,
) -> Result<serde_json::Value> {
    let _ = fs::remove_dir_all(run_dir);
    fs::create_dir_all(run_dir).map_err(|_| Error::Io)?;
    let parameters = run_dir.join("parameters.toml");
    let policy = run_dir.join("policy.toml");
    write_placeholder(&parameters, b"profile = \"case\"\n")?;
    write_placeholder(&policy, b"policy = \"exact_match\"\n")?;
    let model_id = digest_to_hex(&digest_bytes(
        case.to_string_lossy().as_bytes(),
    ));
    let mut args: Vec<String> = vec![
        "run".into(),
        "--commit-policy".into(),
        commit_policy.into(),
    ];
    if let Some(dir) = commit_dir {
        args.push("--commit-dir".into());
        args.push(dir.to_string_lossy().into_owned());
    }
    args.extend([
        "--case".into(),
        case.to_string_lossy().into_owned(),
        "--assets-manifest".into(),
        case.to_string_lossy().into_owned(),
        "--parameters".into(),
        parameters.to_string_lossy().into_owned(),
        "--input".into(),
        input.to_string_lossy().into_owned(),
        "--model-id".into(),
        model_id,
        "--policy".into(),
        policy.to_string_lossy().into_owned(),
        "--run-dir".into(),
        run_dir.to_string_lossy().into_owned(),
    ]);
    let status = std::process::Command::new(executable)
        .args(&args)
        .env("ZKIE_CASE_ID", case_id)
        .status()
        .map_err(|_| Error::Io)?;
    let mut outcome = serde_json::json!({"exit_code": status.code().unwrap_or(1)});
    for (stage, subdir) in [("commit", "commit-run"), ("prove", "prove"), ("verify", "verify")] {
        let path = run_dir.join(subdir).join("result.json");
        if let Ok(text) = fs::read_to_string(&path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                outcome[stage] = value;
            }
        }
    }
    Ok(outcome)
}

fn collect(args: &[String], decode: bool) -> Result<serde_json::Value> {
    let mut cases: Vec<String> = Vec::new();
    let mut out: Option<std::path::PathBuf> = None;
    let mut cold_runs = 1u32;
    let mut warm_runs = 3u32;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--cases" => {
                index += 1;
                while index < args.len() && !args[index].starts_with("--") {
                    cases.push(args[index].clone());
                    index += 1;
                }
                continue;
            }
            "--out" => {
                index += 1;
                out = Some(std::path::PathBuf::from(
                    args.get(index).ok_or(Error::InvalidConfig)?,
                ));
            }
            "--cold-runs" => {
                index += 1;
                cold_runs = args
                    .get(index)
                    .and_then(|value| value.parse().ok())
                    .ok_or(Error::InvalidConfig)?;
            }
            "--warm-runs" => {
                index += 1;
                warm_runs = args
                    .get(index)
                    .and_then(|value| value.parse().ok())
                    .ok_or(Error::InvalidConfig)?;
            }
            "--prompt-len" | "--decode-lengths" | "--compute" | "--asset-root" => {
                index += 1;
                while index < args.len() && !args[index].starts_with("--") {
                    index += 1;
                }
                continue;
            }
            other => return Err(if other.is_empty() { Error::InvalidConfig } else { Error::InvalidConfig }),
        }
        index += 1;
    }
    let out = out.ok_or(Error::InvalidConfig)?;
    fs::create_dir_all(&out).map_err(|_| Error::Io)?;
    let executable = std::env::current_exe().map_err(|_| Error::Io)?;
    let mut report = Vec::new();
    for case in &cases {
        let case_path = std::path::PathBuf::from(case);
        let (root, entries) = parse_cases(&case_path)?;
        for entry in entries {
            if decode && !case.contains("gpt2") && !case.contains("gemma3") {
                continue;
            }
            let input = root.join(&entry.input);
            if !input.exists() {
                report.push(serde_json::json!({
                    "case_id": entry.id,
                    "outcome": "BLOCKED",
                    "error": "input missing",
                    "input": input,
                }));
                continue;
            }
            let case_dir = out.join(&entry.id);
            fs::create_dir_all(&case_dir).map_err(|_| Error::Io)?;
            let commit_dir = case_dir.join("cold-0").join("commit");
            let mut runs = Vec::new();
            for run in 0..cold_runs {
                let run_dir = case_dir.join(format!("cold-{run}"));
                runs.push(run_one(
                    &executable,
                    &case_path,
                    &entry.id,
                    &input,
                    &run_dir,
                    "create",
                    None,
                )?);
            }
            for run in 0..warm_runs {
                let run_dir = case_dir.join(format!("warm-{run}"));
                runs.push(run_one(
                    &executable,
                    &case_path,
                    &entry.id,
                    &input,
                    &run_dir,
                    "reuse",
                    Some(&commit_dir),
                )?);
            }
            report.push(serde_json::json!({
                "case_id": entry.id,
                "seq_tokens": entry.seq_tokens,
                "input": input,
                "runs": runs,
            }));
        }
    }
    let body = serde_json::to_string_pretty(&serde_json::json!({
        "schema": 1,
        "kind": if decode { "decode-matrix" } else { "matrix" },
        "cases": report,
    }))
    .map_err(|_| Error::InvalidConfig)?;
    fs::write(out.join(if decode { "decode-matrix.json" } else { "matrix.json" }), body)
        .map_err(|_| Error::Io)?;
    Ok(serde_json::json!({"cases": report.len()}))
}

pub fn matrix(args: &[String]) -> Result<serde_json::Value> {
    collect(args, false)
}

pub fn decode_matrix(args: &[String]) -> Result<serde_json::Value> {
    collect(args, true)
}
