//! JSONL progress writer.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink, RunOutcome};
use zkie_engine::progress::{SequenceAssigner, HEARTBEAT_INTERVAL_SECS};

pub struct JsonlProgressSink {
    path: PathBuf,
    run_id: String,
    file: Mutex<File>,
    assigner: Mutex<SequenceAssigner>,
    started: Instant,
    units: Mutex<(u64, Option<u64>, Option<String>)>,
}

impl JsonlProgressSink {
    pub fn create(run_dir: &Path, run_id: &str) -> Result<Self> {
        fs::create_dir_all(run_dir).map_err(|_| Error::Io)?;
        let path = run_dir.join("progress.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|_| Error::Io)?;
        Ok(Self {
            path,
            run_id: run_id.to_owned(),
            file: Mutex::new(file),
            assigner: Mutex::new(SequenceAssigner::new()),
            started: Instant::now(),
            units: Mutex::new((0, None, None)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn last_event_seq(&self) -> Option<u64> {
        self.assigner
            .lock()
            .ok()
            .and_then(|assigner| assigner.next_seq().checked_sub(1))
    }

    pub fn heartbeat(&self) -> Result<()> {
        let mut event = ProgressEvent::new(
            self.run_id.clone(),
            "heartbeat",
            ProgressKind::Heartbeat,
        );
        if let Ok(units) = self.units.lock() {
            event.completed_units = units.0;
            event.total_units = units.1;
            event.unit = units.2.clone();
        }
        self.emit(event)
    }

    pub fn run_finished(
        &self,
        stage: &str,
        outcome: RunOutcome,
        error_code: Option<&str>,
    ) -> Result<()> {
        let mut event = ProgressEvent::new(self.run_id.clone(), stage, ProgressKind::RunFinished);
        event.outcome = Some(outcome);
        event.last_completed_stage = Some(stage.to_owned());
        event.error_code = error_code.map(str::to_owned);
        self.emit(event)
    }
}

impl ProgressSink for JsonlProgressSink {
    fn emit(&self, mut event: ProgressEvent) -> Result<()> {
        event.elapsed_ms =
            u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        if event.kind == ProgressKind::WorkAdvanced {
            if let Ok(mut units) = self.units.lock() {
                *units = (event.completed_units, event.total_units, event.unit.clone());
            }
        } else if matches!(
            event.kind,
            ProgressKind::StageFinished | ProgressKind::RunFinished
        ) && event.completed_units == 0
        {
            if let Ok(units) = self.units.lock() {
                event.completed_units = units.0;
                event.total_units = units.1;
                event.unit = units.2.clone();
            }
        }
        {
            let mut assigner = self
                .assigner
                .lock()
                .map_err(|_| Error::SynchronizationFailed)?;
            assigner.assign(&mut event)?;
        }
        let line = event_to_json(&event);
        let mut file = self.file.lock().map_err(|_| Error::SynchronizationFailed)?;
        writeln!(file, "{line}").map_err(|_| Error::ObservabilityIo)?;
        file.flush().map_err(|_| Error::ObservabilityIo)?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        self.file
            .lock()
            .map_err(|_| Error::SynchronizationFailed)?
            .flush()
            .map_err(|_| Error::ObservabilityIo)
    }
}

/// Runs a heartbeat every five seconds until dropped. Dropping sends a stop
/// message so the worker wakes immediately instead of sleeping out its interval.
pub struct HeartbeatGuard {
    stop: mpsc::Sender<()>,
    handle: Option<JoinHandle<()>>,
}

impl HeartbeatGuard {
    pub fn start(sink: Arc<JsonlProgressSink>) -> Self {
        Self::start_with_interval(sink, Duration::from_secs(HEARTBEAT_INTERVAL_SECS))
    }

    pub fn start_with_interval(sink: Arc<JsonlProgressSink>, interval: Duration) -> Self {
        let (stop, receiver) = mpsc::channel();
        let handle = thread::spawn(move || loop {
            match receiver.recv_timeout(interval) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    let _ = sink.heartbeat();
                }
            }
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for HeartbeatGuard {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub fn event_to_json(event: &ProgressEvent) -> String {
    serde_json::json!({
        "schema_version": event.schema_version,
        "run_id": event.run_id,
        "event_seq": event.event_seq,
        "elapsed_ms": event.elapsed_ms,
        "stage_id": event.stage_id,
        "stage_instance": event.stage_instance,
        "parent_instance": event.parent_instance,
        "kind": kind_code(event.kind),
        "completed_units": event.completed_units,
        "total_units": event.total_units,
        "unit": event.unit,
        "last_completed_stage": event.last_completed_stage,
        "outcome": event.outcome.map(outcome_code),
        "error_code": event.error_code,
    })
    .to_string()
}

pub fn kind_code(kind: ProgressKind) -> &'static str {
    match kind {
        ProgressKind::RunStarted => "run_started",
        ProgressKind::StageStarted => "stage_started",
        ProgressKind::WorkAdvanced => "work_advanced",
        ProgressKind::Heartbeat => "heartbeat",
        ProgressKind::StageFinished => "stage_finished",
        ProgressKind::StageFailed => "stage_failed",
        ProgressKind::StageSkipped => "stage_skipped",
        ProgressKind::RunFinished => "run_finished",
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
