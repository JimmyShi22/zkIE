//! Progress buffering, heartbeat and terminal-state contracts.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink, RunOutcome};
use zkie_engine::metrics::{vm_hwm_bytes, vm_hwm_kib, MonotonicTimer};
use zkie_engine::progress::{
    is_terminal_event, outcome_from_events, BufferedProgressSink, EventClass, ProgressBuffer,
    SequenceAssigner, DEFAULT_PROGRESS_QUEUE, HEARTBEAT_INTERVAL_SECS,
};

fn event(stage: &str, kind: ProgressKind) -> ProgressEvent {
    ProgressEvent::new("run-1", stage, kind)
}

fn finished(stage: &str, outcome: RunOutcome) -> ProgressEvent {
    let mut event = event(stage, ProgressKind::RunFinished);
    event.outcome = Some(outcome);
    event
}

#[test]
fn bounded_queue_drops_work_but_never_lifecycle() {
    let mut buffer = ProgressBuffer::new(4);
    buffer.emit(event("run", ProgressKind::RunStarted)).unwrap();
    for _ in 0..8 {
        buffer.emit(event("run", ProgressKind::WorkAdvanced)).unwrap();
    }
    buffer.emit(event("run", ProgressKind::Heartbeat)).unwrap();
    buffer
        .emit(finished("run", RunOutcome::Passed))
        .unwrap();
    assert_eq!(buffer.outcome(), RunOutcome::Passed);
    let events = buffer.drain();
    assert!(events
        .iter()
        .any(|event| event.kind == ProgressKind::RunStarted));
    assert!(events
        .iter()
        .any(|event| event.kind == ProgressKind::RunFinished));
    assert_eq!(buffer.dropped_work(), 7);

    let mut lifecycle_only = ProgressBuffer::new(1);
    for kind in [
        ProgressKind::RunStarted,
        ProgressKind::StageStarted,
        ProgressKind::RunFinished,
    ] {
        lifecycle_only.emit(event("run", kind)).unwrap();
    }
    assert_eq!(lifecycle_only.pending(), 3);
    assert_eq!(lifecycle_only.dropped_work(), 0);
    let seqs: Vec<u64> = lifecycle_only
        .drain()
        .iter()
        .map(|event| event.event_seq)
        .collect();
    assert_eq!(seqs, vec![0, 1, 2]);
}

#[test]
fn missing_terminal_event_is_incomplete() {
    let mut buffer = ProgressBuffer::new(16);
    buffer.emit(event("run", ProgressKind::RunStarted)).unwrap();
    buffer.emit(event("run", ProgressKind::StageStarted)).unwrap();
    buffer.emit(event("run", ProgressKind::StageFinished)).unwrap();
    assert_eq!(buffer.outcome(), RunOutcome::Incomplete);

    let sink = BufferedProgressSink::new(16);
    sink.emit(event("run", ProgressKind::RunStarted)).unwrap();
    assert_eq!(sink.outcome(), RunOutcome::Incomplete);
    sink.emit(finished("run", RunOutcome::Failed)).unwrap();
    assert_eq!(sink.outcome(), RunOutcome::Failed);

    let events = vec![event("run", ProgressKind::StageFinished)];
    assert_eq!(outcome_from_events(&events), RunOutcome::Incomplete);
    assert!(!is_terminal_event(&events[0]));
    assert_eq!(EventClass::of(ProgressKind::RunFinished), EventClass::Lifecycle);
    assert_eq!(EventClass::of(ProgressKind::WorkAdvanced), EventClass::Work);
    assert_eq!(EventClass::of(ProgressKind::Heartbeat), EventClass::Heartbeat);
}

#[test]
fn buffered_sink_assigns_gapless_sequences_across_threads() {
    let sink = Arc::new(BufferedProgressSink::new(DEFAULT_PROGRESS_QUEUE));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let sink = Arc::clone(&sink);
            scope.spawn(move || {
                for _ in 0..100 {
                    sink.emit(event("run", ProgressKind::WorkAdvanced)).unwrap();
                }
            });
        }
    });
    let events = sink.drain();
    assert_eq!(events.len(), 400);
    let seqs: BTreeSet<u64> = events.iter().map(|event| event.event_seq).collect();
    assert_eq!(seqs.len(), 400);
    assert_eq!(*seqs.iter().next().unwrap(), 0);
    assert_eq!(*seqs.iter().next_back().unwrap(), 399);
    assert_eq!(sink.last_event_seq(), Some(399));

    let mut assigner = SequenceAssigner::new();
    let mut first = event("run", ProgressKind::RunStarted);
    let mut second = event("run", ProgressKind::StageStarted);
    assigner.assign(&mut first).unwrap();
    assigner.assign(&mut second).unwrap();
    assert_eq!((first.event_seq, second.event_seq), (0, 1));
    assert_eq!(assigner.next_seq(), 2);
}

#[test]
fn heartbeat_interval_and_process_metrics_are_available() {
    assert_eq!(HEARTBEAT_INTERVAL_SECS, 5);
    assert_eq!(DEFAULT_PROGRESS_QUEUE, 4096);

    let timer = MonotonicTimer::start();
    std::thread::sleep(Duration::from_millis(2));
    let first = timer.elapsed_us();
    let second = timer.elapsed_us();
    assert!(first > 0);
    assert!(second >= first);

    let kib = vm_hwm_kib().expect("VmHWM is readable on Linux");
    let bytes = vm_hwm_bytes().expect("VmHWM converts to bytes");
    assert!(bytes % 1024 == 0);
    assert!(bytes >= kib * 1024);
}
