//! Profile matrix: repeat cases across thread counts and record every sample.
//!
//! Each run is a separate `run` subprocess, which spawns commit/prove/verify as
//! independent processes. The first run of a group creates the commitment; the
//! remaining runs reuse it. All stage samples and operator/kernel counters are
//! written; large proof artifacts are removed after their measurements are
//! recorded unless `--keep-artifacts` is given.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde_json::json;
use zkie_core::artifact::{digest_bytes, digest_to_hex};
use zkie_core::error::{Error, Result};

use crate::report::{summarize_profile, ProfileSample};
use crate::run::parse_cases;

#[derive(Clone, Debug)]
pub struct ProfileArgs {
    pub cases: Vec<PathBuf>,
    pub threads: Vec<usize>,
    pub runs: u32,
    pub out: PathBuf,
    pub case_ids: Vec<String>,
    pub keep_artifacts: bool,
}

impl ProfileArgs {
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut cases = Vec::new();
        let mut threads = Vec::new();
        let mut runs = 3u32;
        let mut out: Option<PathBuf> = None;
        let mut case_ids = Vec::new();
        let mut keep_artifacts = false;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--cases" => {
                    index += 1;
                    while index < args.len() && !args[index].starts_with("--") {
                        cases.push(PathBuf::from(&args[index]));
                        index += 1;
                    }
                    continue;
                }
                "--threads" => {
                    index += 1;
                    while index < args.len() && !args[index].starts_with("--") {
                        threads.push(args[index].parse().map_err(|_| Error::InvalidConfig)?);
                        index += 1;
                    }
                    continue;
                }
                "--case-ids" => {
                    index += 1;
                    while index < args.len() && !args[index].starts_with("--") {
                        case_ids.push(args[index].clone());
                        index += 1;
                    }
                    continue;
                }
                "--runs" => {
                    index += 1;
                    runs = args
                        .get(index)
                        .and_then(|value| value.parse().ok())
                        .ok_or(Error::InvalidConfig)?;
                }
                "--out" => {
                    index += 1;
                    out = Some(PathBuf::from(
                        args.get(index).ok_or(Error::InvalidConfig)?,
                    ));
                }
                "--keep-artifacts" => {
                    keep_artifacts = true;
                }
                "--compute" | "--asset-root" => {
                    index += 1;
                    while index < args.len() && !args[index].starts_with("--") {
                        index += 1;
                    }
                    continue;
                }
                _ => return Err(Error::InvalidConfig),
            }
            index += 1;
        }
        if cases.is_empty() || threads.is_empty() || runs == 0 {
            return Err(Error::InvalidConfig);
        }
        if threads.iter().any(|threads| *threads == 0) {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            cases,
            threads,
            runs,
            out: out.ok_or(Error::InvalidConfig)?,
            case_ids,
            keep_artifacts,
        })
    }
}

fn read_json(path: &std::path::Path) -> Option<serde_json::Value> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn add_profile_entry(
    operators: &mut BTreeMap<(String, usize, String, String), (u64, u64)>,
    stage_wall: &mut BTreeMap<(String, usize, String), u64>,
    case_id: &str,
    threads: usize,
    stage: &str,
    stage_wall_ms: u64,
    profile: &serde_json::Value,
) {
    stage_wall.insert((case_id.to_owned(), threads, stage.to_owned()), stage_wall_ms);
    if let Some(entries) = profile.get("operators").and_then(|value| value.as_array()) {
        for entry in entries {
            let name = entry
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown")
                .to_owned();
            let wall_ns = entry.get("wall_ns").and_then(|value| value.as_u64()).unwrap_or(0);
            let calls = entry.get("calls").and_then(|value| value.as_u64()).unwrap_or(0);
            let key = (case_id.to_owned(), threads, stage.to_owned(), name);
            let slot = operators.entry(key).or_insert((0, 0));
            slot.0 = slot.0.saturating_add(wall_ns);
            slot.1 = slot.1.saturating_add(calls);
        }
    }
}

fn add_kernel_entry(
    kernels: &mut BTreeMap<(String, usize, String, String), (u64, u64)>,
    case_id: &str,
    threads: usize,
    stage: &str,
    profile: &serde_json::Value,
) {
    if let Some(entries) = profile.get("kernels").and_then(|value| value.as_array()) {
        for entry in entries {
            let name = entry
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown")
                .to_owned();
            let wall_ns = entry.get("wall_ns").and_then(|value| value.as_u64()).unwrap_or(0);
            let calls = entry.get("calls").and_then(|value| value.as_u64()).unwrap_or(0);
            let key = (case_id.to_owned(), threads, stage.to_owned(), name);
            let slot = kernels.entry(key).or_insert((0, 0));
            slot.0 = slot.0.saturating_add(wall_ns);
            slot.1 = slot.1.saturating_add(calls);
        }
    }
}

fn summary_json(
    entries: &BTreeMap<(String, usize, String, String), (u64, u64)>,
    stage_wall: &BTreeMap<(String, usize, String), u64>,
    kind: &str,
) -> serde_json::Value {
    let mut category_totals: BTreeMap<(String, usize, String), u64> = BTreeMap::new();
    for ((case_id, threads, stage, _), (wall_ns, _)) in entries {
        let slot = category_totals
            .entry((case_id.clone(), *threads, stage.clone()))
            .or_insert(0);
        *slot = slot.saturating_add(*wall_ns);
    }
    let mut rows = Vec::new();
    for ((case_id, threads, stage, name), (wall_ns, calls)) in entries {
        let wall_ms = *wall_ns as f64 / 1_000_000.0;
        let stage_wall_ms = stage_wall
            .get(&(case_id.clone(), *threads, stage.clone()))
            .copied()
            .unwrap_or(0);
        let category_total_ns = category_totals
            .get(&(case_id.clone(), *threads, stage.clone()))
            .copied()
            .unwrap_or(0);
        let share = if category_total_ns == 0 {
            0.0
        } else {
            *wall_ns as f64 / category_total_ns as f64
        };
        let cumulative_over_stage = if stage_wall_ms == 0 {
            0.0
        } else {
            wall_ms / stage_wall_ms as f64
        };
        rows.push(json!({
            "case_id": case_id,
            "threads": threads,
            "stage": stage,
            "name": name,
            "wall_ms": wall_ms,
            "calls": calls,
            "stage_wall_ms": stage_wall_ms,
            "category_total_ms": category_total_ns as f64 / 1_000_000.0,
            "share_of_category": share,
            "cumulative_over_stage_wall": cumulative_over_stage,
            "kind": kind,
        }));
    }
    rows.sort_by(|left, right| {
        left["case_id"]
            .as_str()
            .cmp(&right["case_id"].as_str())
            .then(left["threads"].as_u64().cmp(&right["threads"].as_u64()))
            .then(left["stage"].as_str().cmp(&right["stage"].as_str()))
            .then_with(|| {
                right["wall_ms"]
                    .as_f64()
                    .partial_cmp(&left["wall_ms"].as_f64())
                    .unwrap_or(Ordering::Equal)
            })
    });
    serde_json::Value::Array(rows)
}

fn write_human_summary(path: &std::path::Path, rows: &serde_json::Value) -> Result<()> {
    let mut text = String::new();
    text.push_str(
        "profile summary: cumulative wall over all calls/threads; share is of the same-kind\n         category total; cumulative/stage can exceed 1 under parallelism.\n",
    );
    let empty = Vec::new();
    for row in rows.as_array().unwrap_or(&empty) {
        let case_id = row["case_id"].as_str().unwrap_or("");
        let threads = row["threads"].as_u64().unwrap_or(0);
        let stage = row["stage"].as_str().unwrap_or("");
        let name = row["name"].as_str().unwrap_or("");
        let kind = row["kind"].as_str().unwrap_or("");
        let wall_ms = row["wall_ms"].as_f64().unwrap_or(0.0);
        let calls = row["calls"].as_u64().unwrap_or(0);
        let share = row["share_of_category"].as_f64().unwrap_or(0.0) * 100.0;
        let over = row["cumulative_over_stage_wall"].as_f64().unwrap_or(0.0);
        text.push_str(&format!(
            "{case_id}\tthreads={threads}\t{stage}\t{kind}\t{name}\t{wall_ms:.1} ms\t{calls} calls\tshare={share:.1}%\tcum/stage={over:.2}\n"
        ));
    }
    fs::write(path, text).map_err(|_| Error::Io)
}

pub fn profile_matrix(args: &[String]) -> Result<serde_json::Value> {
    let parsed = ProfileArgs::parse(args)?;
    fs::create_dir_all(&parsed.out).map_err(|_| Error::Io)?;
    let executable = std::env::current_exe().map_err(|_| Error::Io)?;
    let mut cases_json = Vec::new();
    let mut samples: Vec<ProfileSample> = Vec::new();
    let mut profiles: Vec<serde_json::Value> = Vec::new();
    let mut operator_entries: BTreeMap<(String, usize, String, String), (u64, u64)> =
        BTreeMap::new();
    let mut kernel_entries: BTreeMap<(String, usize, String, String), (u64, u64)> = BTreeMap::new();
    let mut stage_wall: BTreeMap<(String, usize, String), u64> = BTreeMap::new();
    for case in &parsed.cases {
        let (root, entries) = parse_cases(case)?;
        for entry in entries {
            if !parsed.case_ids.is_empty() && !parsed.case_ids.iter().any(|id| *id == entry.id) {
                continue;
            }
            let input = root.join(&entry.input);
            if !input.exists() {
                cases_json.push(json!({
                    "case_id": entry.id,
                    "outcome": "BLOCKED",
                    "error": "input missing",
                    "input": input,
                }));
                continue;
            }
            for &threads in &parsed.threads {
                let group = parsed
                    .out
                    .join(&entry.id)
                    .join(format!("threads-{threads}"));
                let commit_dir = group.join("run-0").join("commit");
                let mut runs_json = Vec::new();
                for run in 0..parsed.runs {
                    let run_dir = group.join(format!("run-{run}"));
                    let _ = fs::remove_dir_all(&run_dir);
                    fs::create_dir_all(&run_dir).map_err(|_| Error::Io)?;
                    let parameters = run_dir.join("parameters.toml");
                    let policy = run_dir.join("policy.toml");
                    fs::write(&parameters, b"profile = \"case\"\n").map_err(|_| Error::Io)?;
                    fs::write(&policy, b"policy = \"exact_match\"\n").map_err(|_| Error::Io)?;
                    let model_id = digest_to_hex(&digest_bytes(
                        case.to_string_lossy().as_bytes(),
                    ));
                    let commit_policy = if run == 0 { "create" } else { "reuse" };
                    let mut run_args: Vec<String> = vec![
                        "run".into(),
                        "--commit-policy".into(),
                        commit_policy.into(),
                    ];
                    if run > 0 {
                        run_args.push("--commit-dir".into());
                        run_args.push(commit_dir.to_string_lossy().into_owned());
                    }
                    run_args.extend([
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
                    std::env::set_var("ZKIE_CASE_ID", &entry.id);
                    std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
                    std::env::set_var("ZKIE_PROFILE", "1");
                    let status = std::process::Command::new(&executable)
                        .args(&run_args)
                        .status()
                        .map_err(|_| Error::Io)?;
                    let mut stages = serde_json::Map::new();
                    for (stage, subdir) in [
                        ("commit", "commit-run"),
                        ("prove", "prove"),
                        ("verify", "verify"),
                    ] {
                        let stage_dir = run_dir.join(subdir);
                        let result_path = stage_dir.join("result.json");
                        let Ok(text) = fs::read_to_string(&result_path) else {
                            continue;
                        };
                        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
                            continue;
                        };
                        let stage_wall_ms = value
                            .get("stage_wall_ms")
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0);
                        let sample = ProfileSample {
                            case_id: entry.id.clone(),
                            threads,
                            run_index: run,
                            stage: stage.into(),
                            outcome: value
                                .get("outcome")
                                .and_then(|value| value.as_str())
                                .unwrap_or("INCOMPLETE")
                                .to_owned(),
                            stage_wall_ms,
                            protocol_wall_ms: value
                                .get("protocol_wall_ms")
                                .and_then(|value| value.as_u64())
                                .unwrap_or(0),
                            vmhwm_kib: value.get("vmhwm_kib").and_then(|value| value.as_u64()),
                            proof_bytes: value.get("proof_bytes").and_then(|value| value.as_u64()),
                        };
                        stages.insert(
                            stage.into(),
                            serde_json::to_value(&sample)
                                .map_err(|_| Error::InvalidEncoding)?,
                        );
                        samples.push(sample);
                        let profile = read_json(&stage_dir.join("profile.json"));
                        if let Some(profile) = &profile {
                            profiles.push(json!({
                                "case_id": entry.id,
                                "threads": threads,
                                "run": run,
                                "stage": stage,
                                "profile": profile,
                            }));
                            add_profile_entry(
                                &mut operator_entries,
                                &mut stage_wall,
                                &entry.id,
                                threads,
                                stage,
                                stage_wall_ms,
                                profile,
                            );
                            add_kernel_entry(
                                &mut kernel_entries,
                                &entry.id,
                                threads,
                                stage,
                                profile,
                            );
                        }
                    }
                    runs_json.push(json!({
                        "run": run,
                        "commit_policy": commit_policy,
                        "exit_code": status.code().unwrap_or(1),
                        "stages": stages,
                    }));
                    if !parsed.keep_artifacts {
                        let _ = fs::remove_dir_all(run_dir.join("proof"));
                    }
                }
                if !parsed.keep_artifacts {
                    let _ = fs::remove_dir_all(&commit_dir);
                }
                cases_json.push(json!({
                    "case_id": entry.id,
                    "threads": threads,
                    "runs": runs_json,
                }));
            }
        }
    }
    std::env::remove_var("RAYON_NUM_THREADS");
    std::env::remove_var("ZKIE_CASE_ID");
    std::env::remove_var("ZKIE_PROFILE");
    let summary = summarize_profile(&samples)?;
    let operators = summary_json(&operator_entries, &stage_wall, "operator");
    let kernels = summary_json(&kernel_entries, &stage_wall, "kernel");
    let mut human_rows = operators.as_array().cloned().unwrap_or_default();
    human_rows.extend(kernels.as_array().cloned().unwrap_or_default());
    let human = serde_json::Value::Array(human_rows);
    write_human_summary(&parsed.out.join("profile-summary.txt"), &human)?;
    let body = serde_json::to_string_pretty(&json!({
        "schema": 1,
        "kind": "profile-matrix",
        "artifacts_retained": parsed.keep_artifacts,
        "cases": cases_json,
        "samples": samples,
        "summary": summary,
        "profiles": profiles,
        "operator_summary": operators,
        "kernel_summary": kernels,
    }))
    .map_err(|_| Error::InvalidEncoding)?;
    fs::write(parsed.out.join("profile-matrix.json"), body).map_err(|_| Error::Io)?;
    Ok(json!({
        "case_groups": cases_json.len(),
        "samples": samples.len(),
    }))
}
