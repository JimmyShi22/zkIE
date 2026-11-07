//! Progress sequencing, bounded buffering and terminal-state detection.
//!
//! A single writer assigns strictly increasing event sequence numbers. The
//! bounded queue may drop work/heartbeat events under pressure, but lifecycle
//! events are never dropped. A missing terminal event is reported as
//! `INCOMPLETE`.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink, RunOutcome};

pub const DEFAULT_PROGRESS_QUEUE: usize = 4096;
pub const HEARTBEAT_INTERVAL_SECS: u64 = 5;

#[derive(Clone, Copy, Debug, Default)]
pub struct SequenceAssigner {
    next_seq: u64,
}

impl SequenceAssigner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assign(&mut self, event: &mut ProgressEvent) -> Result<()> {
        event.event_seq = self.next_seq;
        self.next_seq = self
            .next_seq
            .checked_add(1)
            .ok_or(Error::InvalidSessionState)?;
        Ok(())
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StageTimer {
    start: Instant,
}

impl StageTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventClass {
    Lifecycle,
    Work,
    Heartbeat,
}

impl EventClass {
    pub const fn of(kind: ProgressKind) -> Self {
        match kind {
            ProgressKind::WorkAdvanced => EventClass::Work,
            ProgressKind::Heartbeat => EventClass::Heartbeat,
            ProgressKind::RunStarted
            | ProgressKind::StageStarted
            | ProgressKind::StageFinished
            | ProgressKind::StageFailed
            | ProgressKind::StageSkipped
            | ProgressKind::RunFinished => EventClass::Lifecycle,
        }
    }
}

#[derive(Debug)]
pub struct ProgressBuffer {
    capacity: usize,
    queue: VecDeque<ProgressEvent>,
    assigner: SequenceAssigner,
    dropped_work: u64,
}

impl Default for ProgressBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_PROGRESS_QUEUE)
    }
}

impl ProgressBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            queue: VecDeque::new(),
            assigner: SequenceAssigner::new(),
            dropped_work: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn emit(&mut self, mut event: ProgressEvent) -> Result<()> {
        self.assigner.assign(&mut event)?;
        if self.queue.len() >= self.capacity {
            if let Some(index) = self
                .queue
                .iter()
                .position(|queued| EventClass::of(queued.kind) != EventClass::Lifecycle)
            {
                self.queue.remove(index);
                self.dropped_work = self.dropped_work.saturating_add(1);
            }
            // If every queued event is a lifecycle event, allow a temporary
            // overshoot rather than dropping lifecycle state.
        }
        self.queue.push_back(event);
        Ok(())
    }

    pub fn drain(&mut self) -> Vec<ProgressEvent> {
        self.queue.drain(..).collect()
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    pub fn dropped_work(&self) -> u64 {
        self.dropped_work
    }

    pub fn last_event_seq(&self) -> Option<u64> {
        self.assigner.next_seq().checked_sub(1)
    }

    pub fn outcome(&self) -> RunOutcome {
        outcome_from_events(self.queue.iter())
    }
}

pub struct BufferedProgressSink {
    inner: Mutex<ProgressBuffer>,
}

impl BufferedProgressSink {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(ProgressBuffer::new(capacity)),
        }
    }

    pub fn drain(&self) -> Vec<ProgressEvent> {
        self.inner
            .lock()
            .map(|mut buffer| buffer.drain())
            .unwrap_or_default()
    }

    pub fn pending(&self) -> usize {
        self.inner
            .lock()
            .map(|buffer| buffer.pending())
            .unwrap_or(0)
    }

    pub fn dropped_work(&self) -> u64 {
        self.inner
            .lock()
            .map(|buffer| buffer.dropped_work())
            .unwrap_or(0)
    }

    pub fn last_event_seq(&self) -> Option<u64> {
        self.inner
            .lock()
            .ok()
            .and_then(|buffer| buffer.last_event_seq())
    }

    pub fn outcome(&self) -> RunOutcome {
        self.inner
            .lock()
            .map(|buffer| buffer.outcome())
            .unwrap_or(RunOutcome::Incomplete)
    }
}

impl ProgressSink for BufferedProgressSink {
    fn emit(&self, event: ProgressEvent) -> Result<()> {
        self.inner
            .lock()
            .map_err(|_| Error::SynchronizationFailed)?
            .emit(event)
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

pub fn is_terminal_event(event: &ProgressEvent) -> bool {
    event.kind == ProgressKind::RunFinished
}

pub fn outcome_from_events<'a>(
    events: impl IntoIterator<Item = &'a ProgressEvent>,
) -> RunOutcome {
    let mut terminal: Option<Option<RunOutcome>> = None;
    for event in events {
        if is_terminal_event(event) {
            terminal = Some(event.outcome);
        }
    }
    match terminal {
        Some(Some(outcome)) => outcome,
        _ => RunOutcome::Incomplete,
    }
}
