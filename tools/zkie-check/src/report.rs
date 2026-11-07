//! Run-report check.
//!
//! Reads a matrix run directory (the `matrix.json` produced by the bench
//! runner, or the per-case directories) and writes an acceptance report. A
//! case is `PASS` only when every run completed every stage with `PASS`.
//! Mixed evidence, missing stages and budget-exhaustion errors are never
//! promoted to `PASS`; they are reported as `STATE_CONFLICT`, `INCOMPLETE` or
//! `BUDGET_EXHAUSTED`. Cases whose id names a GPU or Llama track are reported
//! as `DEFERRED` and excluded from the required denominator.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::parse_flags;

const REPORT_FILE: &str = "acceptance-report.json";
const PROFILE: &str = "cpu-original-cases";
const REQUIRED_CASES: [&str; 3] = ["gpt2-seq16", "gemma3-seq16", "timesfm-16patch"];
const BUDGET_ERRORS: [&str; 5] = [
    "resource_limit",
    "budget_exhausted",
    "out_of_memory",
    "oom",
    "memory_limit",
];
const DEFERRED_MARKERS: [&str; 2] = ["gpu", "llama"];

pub fn main(args: &[String]) -> i32 {
    match check(args) {
        Ok(path) => {
            println!("{{\"ok\":true,\"result\":\"{}\"}}", path.display());
            0
        }
        Err(error) => {
            eprintln!("report check failed: {error}");
            1
        }
    }
}

pub fn check(args: &[String]) -> Result<PathBuf, String> {
    let flags = parse_flags(args, &["input", "require-profile"])?;
    let input = PathBuf::from(flags.get("input").ok_or("missing --input")?);
    if !input.is_dir() {
        return Err(format!("not a directory: {}", input.display()));
    }
    let profile = flags.get("require-profile").cloned();
    if let Some(profile) = &profile {
        if profile.is_empty() || profile != PROFILE {
            return Err(format!("unsupported profile: {profile}"));
        }
    }

    let cases = collect_cases(&input)?;
    if let Some(profile) = &profile {
        for case in &cases {
            if is_deferred(&case.id) {
                continue;
            }
            if !original_family(&case.id) {
                return Err(format!(
                    "case {} is outside the {profile} profile",
                    case.id
                ));
            }
        }
        for required in REQUIRED_CASES {
            if !cases.iter().any(|case| case.id == required) {
                return Err(format!("profile {profile} is missing case {required}"));
            }
        }
    }

    let entries: Vec<Value> = cases.iter().map(analyze_case).collect();
    let status = |name: &str| {
        entries
            .iter()
            .filter(|entry| entry.get("status").and_then(Value::as_str) == Some(name))
            .count()
    };
    let deferred = status("DEFERRED");
    let passed = status("PASS");
    let required = entries.len().saturating_sub(deferred);
    let (blocked, incomplete, conflicts, budget) = (
        status("BLOCKED"),
        status("INCOMPLETE"),
        status("STATE_CONFLICT"),
        status("BUDGET_EXHAUSTED"),
    );
    let report = json!({
        "schema": 1,
        "kind": "acceptance-report",
        "profile": profile,
        "input": input.to_string_lossy(),
        "cases": entries,
        "summary": {
            "required": required,
            "passed": passed,
            "blocked": blocked,
            "incomplete": incomplete,
            "state_conflict": conflicts,
            "budget_exhausted": budget,
            "deferred": deferred,
            "all_required_passed": required > 0 && passed == required,
        },
        "notes": [
            "DEFERRED cases are excluded from the required denominator",
            "a case is PASS only when every run and every stage is PASS",
            "budget exhaustion and state conflicts are never reported as PASS",
        ],
    });
    let path = input.join(REPORT_FILE);
    let text = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    fs::write(&path, text).map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(path)
}

struct CaseRuns {
    id: String,
    runs: Vec<Value>,
}

fn is_deferred(id: &str) -> bool {
    let id = id.to_ascii_lowercase();
    DEFERRED_MARKERS.iter().any(|marker| id.contains(marker))
}

fn original_family(id: &str) -> bool {
    ["gpt2-", "gemma3-", "timesfm-"]
        .iter()
        .any(|prefix| id.starts_with(prefix))
}

fn collect_cases(input: &Path) -> Result<Vec<CaseRuns>, String> {
    let profile_matrix = input.join("profile-matrix.json");
    if profile_matrix.exists() {
        return collect_cases_from_profile(&profile_matrix);
    }
    let matrix = input.join("matrix.json");
    if matrix.exists() {
        let text = fs::read_to_string(&matrix)
            .map_err(|error| format!("cannot read {}: {error}", matrix.display()))?;
        let value: Value =
            serde_json::from_str(&text).map_err(|error| format!("invalid matrix.json: {error}"))?;
        let cases = value
            .get("cases")
            .and_then(Value::as_array)
            .ok_or("matrix.json has no cases array")?;
        return Ok(cases
            .iter()
            .map(|case| CaseRuns {
                id: case
                    .get("case_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                runs: case
                    .get("runs")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect());
    }
    collect_cases_from_dirs(input)
}

fn collect_cases_from_profile(path: &Path) -> Result<Vec<CaseRuns>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| format!("invalid profile-matrix.json: {error}"))?;
    let groups = value
        .get("cases")
        .and_then(Value::as_array)
        .ok_or("profile-matrix.json has no cases array")?;
    let mut by_case: std::collections::BTreeMap<String, Vec<Value>> =
        std::collections::BTreeMap::new();
    for group in groups {
        let id = group
            .get("case_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let threads = group.get("threads").and_then(Value::as_u64).unwrap_or(0);
        let runs = group
            .get("runs")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for run in runs {
            let mut lifted = Map::new();
            if let Some(stages) = run.get("stages").and_then(Value::as_object) {
                for (stage, stage_value) in stages {
                    lifted.insert(stage.clone(), stage_value.clone());
                }
            }
            lifted.insert("threads".into(), json!(threads));
            if let Some(exit_code) = run.get("exit_code") {
                lifted.insert("exit_code".into(), exit_code.clone());
            }
            by_case.entry(id.clone()).or_default().push(Value::Object(lifted));
        }
    }
    Ok(by_case
        .into_iter()
        .map(|(id, runs)| CaseRuns { id, runs })
        .collect())
}

fn collect_cases_from_dirs(input: &Path) -> Result<Vec<CaseRuns>, String> {
    let mut cases = Vec::new();
    let mut directories: Vec<PathBuf> = fs::read_dir(input)
        .map_err(|error| format!("cannot read {}: {error}", input.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect();
    directories.sort();
    for directory in directories {
        let id = directory
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut run_dirs: Vec<PathBuf> = fs::read_dir(&directory)
            .map_err(|error| format!("cannot read {}: {error}", directory.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .map(|name| {
                            let name = name.to_string_lossy();
                            name.starts_with("cold-") || name.starts_with("warm-")
                        })
                        .unwrap_or(false)
            })
            .collect();
        run_dirs.sort();
        let runs = run_dirs
            .iter()
            .map(|run_dir| {
                let mut run = Map::new();
                let mut exit_code = 0i64;
                for (stage, subdir) in [
                    ("commit", "commit-run"),
                    ("prove", "prove"),
                    ("verify", "verify"),
                ] {
                    let path = run_dir.join(subdir).join("result.json");
                    match fs::read_to_string(&path).ok().and_then(|text| serde_json::from_str::<Value>(&text).ok()) {
                        Some(value) => {
                            if value.get("outcome").and_then(Value::as_str) != Some("PASS") {
                                exit_code = 1;
                            }
                            run.insert(stage.to_owned(), value);
                        }
                        None => {
                            exit_code = 1;
                        }
                    }
                }
                run.insert("exit_code".into(), json!(exit_code));
                Value::Object(run)
            })
            .collect();
        cases.push(CaseRuns { id, runs });
    }
    Ok(cases)
}

struct RunAnalysis {
    pass: bool,
    complete: bool,
    blocked: bool,
    budget: bool,
    reasons: Vec<String>,
}

fn analyze_run(run: &Value) -> RunAnalysis {
    let mut pass = true;
    let mut complete = true;
    let mut blocked = false;
    let mut budget = false;
    let mut reasons = Vec::new();
    let has_prove = run.get("prove").is_some();
    let has_verify = run.get("verify").is_some();
    let has_stage = run.get("commit").is_some() || has_prove || has_verify;
    if !has_stage {
        complete = false;
        pass = false;
        reasons.push("no stage results".into());
    }
    for stage in ["commit", "prove", "verify"] {
        match run.get(stage).and_then(Value::as_object) {
            Some(stage_value) => {
                let outcome = stage_value
                    .get("outcome")
                    .and_then(Value::as_str)
                    .unwrap_or("MISSING");
                let error = stage_value
                    .get("error_code")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if outcome != "PASS" {
                    pass = false;
                    reasons.push(format!("{stage}:{outcome}"));
                }
                if outcome == "BLOCKED" || !error.is_empty() {
                    blocked = true;
                }
                if BUDGET_ERRORS.contains(&error) {
                    budget = true;
                }
            }
            None => {
                if stage == "commit" && has_prove && has_verify {
                    continue;
                }
                complete = false;
                pass = false;
                reasons.push(format!("{stage}:missing"));
            }
        }
    }
    if let Some(code) = run.get("exit_code").and_then(Value::as_i64) {
        if code != 0 {
            pass = false;
            reasons.push(format!("exit_code:{code}"));
        }
    }
    RunAnalysis {
        pass,
        complete,
        blocked,
        budget,
        reasons,
    }
}

fn analyze_case(case: &CaseRuns) -> Value {
    let analyses: Vec<RunAnalysis> = case.runs.iter().map(analyze_run).collect();
    let passed_runs = analyses.iter().filter(|analysis| analysis.pass).count();
    let required_note = if case.runs.is_empty() {
        "no runs"
    } else {
        "runs were recorded"
    };
    let status = if is_deferred(&case.id) {
        "DEFERRED"
    } else if case.runs.is_empty() {
        "INCOMPLETE"
    } else if analyses.iter().any(|analysis| analysis.budget) {
        "BUDGET_EXHAUSTED"
    } else if passed_runs == case.runs.len() {
        "PASS"
    } else if passed_runs > 0 {
        "STATE_CONFLICT"
    } else if analyses.iter().any(|analysis| analysis.blocked) {
        "BLOCKED"
    } else if analyses.iter().any(|analysis| !analysis.complete) {
        "INCOMPLETE"
    } else {
        "BLOCKED"
    };
    let reasons: Vec<String> = analyses
        .iter()
        .flat_map(|analysis| analysis.reasons.iter().cloned())
        .collect();
    json!({
        "case_id": case.id,
        "status": status,
        "runs": case.runs.len(),
        "passed_runs": passed_runs,
        "reasons": reasons,
        "note": required_note,
    })
}
