//! Structured run result and process-level measurements.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use zkie_core::error::{Error, Result};
use zkie_core::progress::RunOutcome;

#[derive(Clone, Debug)]
pub struct RunReport {
    pub run_id: String,
    pub command: String,
    pub outcome: RunOutcome,
    pub stage_wall_ms: u64,
    pub protocol_wall_ms: u64,
    pub vmhwm_kib: Option<u64>,
    pub proof_bytes: Option<u64>,
    pub error_code: Option<String>,
    pub progress_path: Option<PathBuf>,
    pub last_event_seq: Option<u64>,
    pub last_completed_stage: Option<String>,
    pub output_binding: Option<String>,
    pub authenticated_output_digest: Option<String>,
    pub authenticated_output_elements: Option<u64>,
}

impl RunReport {
    pub fn new(command: impl Into<String>, outcome: RunOutcome) -> Self {
        let command = command.into();
        Self {
            run_id: command.clone(),
            command,
            outcome,
            stage_wall_ms: 0,
            protocol_wall_ms: 0,
            vmhwm_kib: None,
            proof_bytes: None,
            error_code: None,
            progress_path: None,
            last_event_seq: None,
            last_completed_stage: None,
            output_binding: None,
            authenticated_output_digest: None,
            authenticated_output_elements: None,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::json!({
            "run_id": self.run_id,
            "command": self.command,
            "outcome": outcome_code(self.outcome),
            "stage_wall_ms": self.stage_wall_ms,
            "protocol_wall_ms": self.protocol_wall_ms,
            "vmhwm_kib": self.vmhwm_kib,
            "proof_bytes": self.proof_bytes,
            "error_code": self.error_code,
            "progress_path": self.progress_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "last_event_seq": self.last_event_seq,
            "last_completed_stage": self.last_completed_stage,
            "output_binding": self.output_binding,
            "authenticated_output_digest": self.authenticated_output_digest,
            "authenticated_output_elements": self.authenticated_output_elements,
        })
        .to_string()
    }

    pub fn write(&self, run_dir: &Path) -> Result<PathBuf> {
        fs::create_dir_all(run_dir).map_err(|_| Error::Io)?;
        let path = run_dir.join("result.json");
        fs::write(&path, self.to_json()).map_err(|_| Error::Io)?;
        Ok(path)
    }
}

pub fn outcome_code(outcome: RunOutcome) -> &'static str {
    match outcome {
        RunOutcome::Passed => "PASS",
        RunOutcome::Rejected => "FAIL",
        RunOutcome::Failed => "FAIL",
        RunOutcome::Blocked => "BLOCKED",
        RunOutcome::Skipped => "SKIP",
        RunOutcome::Cancelled => "CANCELLED",
        RunOutcome::Incomplete => "INCOMPLETE",
    }
}

pub fn outcome_for_error(error: &Error) -> RunOutcome {
    match error {
        Error::UnsupportedPrimitive
        | Error::UnsupportedKernel
        | Error::UnsupportedBackend
        | Error::MissingCommitArtifact
        | Error::IncompleteArtifact
        | Error::ResourceLimit
        | Error::DiskLimit
        | Error::AssetMismatch => RunOutcome::Blocked,
        Error::Cancelled => RunOutcome::Cancelled,
        _ => RunOutcome::Failed,
    }
}

pub fn dir_size_bytes(dir: &Path) -> Option<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let metadata = entry.metadata().ok()?;
        if metadata.is_file() {
            total = total.checked_add(metadata.len())?;
        } else if metadata.is_dir() {
            total = total.checked_add(dir_size_bytes(&entry.path())?)?;
        }
    }
    Some(total)
}

pub fn vm_hwm_kib() -> Option<u64> {
    zkie_engine::metrics::vm_hwm_kib()
}

pub fn parse_flags(args: &[String], allowed: &[&str]) -> Result<BTreeMap<String, String>> {
    let mut flags = BTreeMap::new();
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        let name = flag
            .strip_prefix("--")
            .filter(|name| !name.is_empty())
            .ok_or(Error::InvalidConfig)?;
        if !allowed.contains(&name) {
            return Err(Error::InvalidConfig);
        }
        let value = args.get(index + 1).ok_or(Error::InvalidConfig)?;
        if flags.insert(name.to_owned(), value.clone()).is_some() {
            return Err(Error::InvalidConfig);
        }
        index += 2;
    }
    Ok(flags)
}

pub fn require_flag<'a>(flags: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str> {
    flags.get(name).map(String::as_str).ok_or(Error::InvalidConfig)
}

pub fn path_flag(flags: &BTreeMap<String, String>, name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(require_flag(flags, name)?))
}

/// One measured stage sample from a profile matrix run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileSample {
    pub case_id: String,
    pub threads: usize,
    pub run_index: u32,
    pub stage: String,
    pub outcome: String,
    pub stage_wall_ms: u64,
    pub protocol_wall_ms: u64,
    pub vmhwm_kib: Option<u64>,
    pub proof_bytes: Option<u64>,
}

fn median_u64(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    if values.len() % 2 == 1 {
        values[values.len() / 2]
    } else {
        let left = values[values.len() / 2 - 1];
        let right = values[values.len() / 2];
        left / 2 + right / 2 + (left % 2 + right % 2) / 2
    }
}

/// Summarize every sample by (case, threads, stage). All raw samples remain in
/// the report; the summary only adds min/median/max and never replaces them.
pub fn summarize_profile(samples: &[ProfileSample]) -> Result<serde_json::Value> {
    #[derive(Default)]
    struct Group {
        walls: Vec<u64>,
        vmhwm: Vec<u64>,
        proof_bytes: Option<u64>,
        outcomes: Vec<String>,
    }
    let mut groups: BTreeMap<(String, usize, String), Group> = BTreeMap::new();
    for sample in samples {
        let group = groups
            .entry((sample.case_id.clone(), sample.threads, sample.stage.clone()))
            .or_default();
        group.walls.push(sample.stage_wall_ms);
        if let Some(vmhwm) = sample.vmhwm_kib {
            group.vmhwm.push(vmhwm);
        }
        if group.proof_bytes.is_none() {
            group.proof_bytes = sample.proof_bytes;
        }
        group.outcomes.push(sample.outcome.clone());
    }
    let mut rows = Vec::with_capacity(groups.len());
    for ((case_id, threads, stage), mut group) in groups {
        group.walls.sort_unstable();
        group.vmhwm.sort_unstable();
        let min_wall = group.walls.first().copied().unwrap_or(0);
        let max_wall = group.walls.last().copied().unwrap_or(0);
        let median_wall = median_u64(&group.walls);
        let max_vmhwm = group.vmhwm.last().copied();
        rows.push(serde_json::json!({
            "case_id": case_id,
            "threads": threads,
            "stage": stage,
            "sample_count": group.walls.len(),
            "stage_wall_ms_all": group.walls,
            "stage_wall_ms_min": min_wall,
            "stage_wall_ms_median": median_wall,
            "stage_wall_ms_max": max_wall,
            "vmhwm_kib_all": group.vmhwm,
            "vmhwm_kib_max": max_vmhwm,
            "proof_bytes": group.proof_bytes,
            "outcomes": group.outcomes,
        }));
    }
    Ok(serde_json::json!({"groups": rows}))
}

/// Writes operator/kernel counters for this stage when profiling is enabled.
pub fn write_profile(run_dir: &Path) -> Result<Option<PathBuf>> {
    if !zkie_core::profile::Scope::is_enabled() {
        return Ok(None);
    }
    let snapshot = zkie_core::profile::snapshot();
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let operators: Vec<serde_json::Value> = snapshot
        .operators
        .iter()
        .map(|entry| {
            let name = match zkie_ops::op::opcode_from_u16(entry.slot) {
                Ok(opcode) => format!("{opcode:?}"),
                Err(_) => format!("opcode_{}", entry.slot),
            };
            serde_json::json!({
                "opcode": entry.slot,
                "name": name,
                "wall_ns": entry.wall_nanos,
                "calls": entry.calls,
            })
        })
        .collect();
    let kernels: Vec<serde_json::Value> = snapshot
        .kernels
        .iter()
        .map(|entry| {
            serde_json::json!({
                "name": entry.name,
                "wall_ns": entry.wall_nanos,
                "calls": entry.calls,
            })
        })
        .collect();
    fs::create_dir_all(run_dir).map_err(|_| Error::Io)?;
    let path = run_dir.join("profile.json");
    let body = serde_json::to_string_pretty(&serde_json::json!({
        "schema": 1,
        "threads": threads,
        "operators": operators,
        "kernels": kernels,
    }))
    .map_err(|_| Error::InvalidEncoding)?;
    fs::write(&path, body).map_err(|_| Error::Io)?;
    Ok(Some(path))
}
