//! Progress events.
//!
//! Progress is observation only. None of these types may appear inside a
//! mathematical statement, a transcript, or proof bytes.

use crate::error::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressKind {
    RunStarted,
    StageStarted,
    WorkAdvanced,
    Heartbeat,
    StageFinished,
    StageFailed,
    StageSkipped,
    RunFinished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    Passed,
    Rejected,
    Failed,
    Blocked,
    Skipped,
    Cancelled,
    Incomplete,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProgressDetail {
    #[default]
    Stage,
    Op,
    Kernel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgressEvent {
    pub schema_version: u16,
    pub run_id: String,
    pub event_seq: u64,
    pub elapsed_ms: u64,
    pub stage_id: String,
    pub stage_instance: u64,
    pub parent_instance: Option<u64>,
    pub kind: ProgressKind,
    pub completed_units: u64,
    pub total_units: Option<u64>,
    pub unit: Option<String>,
    pub last_completed_stage: Option<String>,
    pub outcome: Option<RunOutcome>,
    pub error_code: Option<String>,
}

impl ProgressEvent {
    pub fn new(
        run_id: impl Into<String>,
        stage_id: impl Into<String>,
        kind: ProgressKind,
    ) -> Self {
        Self {
            schema_version: 1,
            run_id: run_id.into(),
            event_seq: 0,
            elapsed_ms: 0,
            stage_id: stage_id.into(),
            stage_instance: 0,
            parent_instance: None,
            kind,
            completed_units: 0,
            total_units: None,
            unit: None,
            last_completed_stage: None,
            outcome: None,
            error_code: None,
        }
    }
}

pub trait ProgressSink: Send + Sync {
    fn emit(&self, event: ProgressEvent) -> Result<()>;
    fn flush(&self) -> Result<()>;
}
