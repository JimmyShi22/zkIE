//! Interface contract tests for buffers, memory, deterministic tasks and
//! progress sequencing.

use std::sync::{Arc, Mutex};

use zkie_core::compute::{MemoryGovernor, MemoryRequest};
use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink};
use zkie_engine::context::RuntimeConfig;
use zkie_engine::memory::{BudgetedMemoryGovernor, MemoryBudgetConfig};
use zkie_engine::parallel::{
    run_deterministic, LogicalTaskKey, SerialTaskExecutor, TaskExecutor, TaskPhase, TaskResult,
};
use zkie_engine::progress::{SequenceAssigner, StageTimer};

#[derive(Clone, Default)]
struct RecordingSink {
    events: Arc<Mutex<Vec<ProgressEvent>>>,
}

impl ProgressSink for RecordingSink {
    fn emit(&self, event: ProgressEvent) -> Result<()> {
        self.events
            .lock()
            .map_err(|_| Error::SynchronizationFailed)?
            .push(event);
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn memory_budget_is_reserved_before_use_and_released_on_drop() {
    let governor = BudgetedMemoryGovernor::new(1000);
    let request = MemoryRequest {
        resident_bytes: 400,
        ..MemoryRequest::default()
    };

    let first = governor.acquire(&request).unwrap();
    assert_eq!(governor.snapshot().reserved_bytes, 400);
    let second = governor.acquire(&request).unwrap();
    assert_eq!(governor.snapshot().reserved_bytes, 800);
    assert!(matches!(
        governor.acquire(&request),
        Err(Error::ResourceLimit)
    ));

    drop(first);
    assert_eq!(governor.snapshot().reserved_bytes, 400);
    let third = governor.acquire(&request).unwrap();
    assert_eq!(governor.snapshot().reserved_bytes, 800);
    drop(second);
    drop(third);
    assert_eq!(governor.snapshot().reserved_bytes, 0);
    assert!(governor.snapshot().peak_bytes >= 800);
}

type Task = Box<dyn FnOnce() -> Result<u64> + Send>;

fn key(op_id: u32) -> LogicalTaskKey {
    LogicalTaskKey::new(1, TaskPhase::Forward, 0, op_id, 0, op_id)
}

fn tasks() -> Vec<(LogicalTaskKey, Task)> {
    vec![
        (key(2), Box::new(|| Ok(20u64))),
        (key(0), Box::new(|| Ok(0u64))),
        (key(1), Box::new(|| Ok(10u64))),
    ]
}

struct ReverseExecutor;

impl TaskExecutor for ReverseExecutor {
    fn worker_count(&self) -> usize {
        4
    }

    fn execute<T, F>(&self, mut tasks: Vec<(LogicalTaskKey, F)>) -> Result<Vec<TaskResult<T>>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let mut results = Vec::with_capacity(tasks.len());
        while let Some((key, task)) = tasks.pop() {
            results.push(TaskResult {
                key,
                value: task()?,
            });
        }
        Ok(results)
    }
}

#[test]
fn logical_task_order_is_independent_of_worker_count() {
    let serial = SerialTaskExecutor;
    let serial_out = run_deterministic(&serial, tasks()).unwrap();
    let reverse_out = run_deterministic(&ReverseExecutor, tasks()).unwrap();

    let keys = |out: &[TaskResult<u64>]| {
        out.iter()
            .map(|result| result.key.op_id)
            .collect::<Vec<_>>()
    };
    let values = |out: &[TaskResult<u64>]| {
        out.iter().map(|result| result.value).collect::<Vec<_>>()
    };
    assert_eq!(keys(&serial_out), vec![0, 1, 2]);
    assert_eq!(keys(&reverse_out), vec![0, 1, 2]);
    assert_eq!(values(&serial_out), vec![0, 10, 20]);
    assert_eq!(values(&reverse_out), vec![0, 10, 20]);
    assert_eq!(serial.worker_count(), 1);
    assert_eq!(ReverseExecutor.worker_count(), 4);
}

#[test]
fn duplicate_task_keys_are_rejected() {
    let tasks: Vec<(LogicalTaskKey, Task)> = vec![
        (key(1), Box::new(|| Ok(1u64))),
        (key(1), Box::new(|| Ok(2u64))),
    ];
    assert!(matches!(
        run_deterministic(&SerialTaskExecutor, tasks),
        Err(Error::StatementMismatch)
    ));
}

#[test]
fn progress_sequence_is_strictly_increasing() {
    let sink = RecordingSink::default();
    let mut assigner = SequenceAssigner::new();
    for kind in [
        ProgressKind::RunStarted,
        ProgressKind::StageStarted,
        ProgressKind::StageFinished,
    ] {
        let mut event = ProgressEvent::new("run-1", "stage", kind);
        assigner.assign(&mut event).unwrap();
        sink.emit(event).unwrap();
    }
    sink.flush().unwrap();

    let events = sink.events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .map(|event| event.event_seq)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(events[0].kind, ProgressKind::RunStarted);
    drop(events);

    let timer = StageTimer::start();
    assert!(timer.elapsed_ms() < 60_000);
}

#[test]
fn runtime_config_is_validated_before_execution() {
    let config = RuntimeConfig::default();
    config.validate().unwrap();

    let bad_threads = RuntimeConfig {
        compute_threads: 0,
        ..RuntimeConfig::default()
    };
    assert!(matches!(bad_threads.validate(), Err(Error::InvalidConfig)));

    let bad_budget = RuntimeConfig {
        memory_budget_bytes: 0,
        ..RuntimeConfig::default()
    };
    assert!(matches!(bad_budget.validate(), Err(Error::InvalidConfig)));

    let budget = MemoryBudgetConfig {
        budget_bytes: 800,
        system_reserve_bytes: 100,
    };
    assert_eq!(budget.effective_budget(None), 800);
    assert_eq!(budget.effective_budget(Some(500)), 400);
    assert_eq!(budget.effective_budget(Some(1000)), 800);

    let cpu_caps = zkie_core::compute::ComputeCapabilities::cpu();
    let cpu = RuntimeConfig {
        compute: zkie_engine::context::ComputeSelection::Cpu,
        ..RuntimeConfig::default()
    };
    assert_eq!(
        cpu.require_backend(cpu_caps).unwrap(),
        zkie_engine::context::ComputeSelection::Cpu
    );

    let auto = RuntimeConfig {
        compute: zkie_engine::context::ComputeSelection::Auto,
        ..RuntimeConfig::default()
    };
    assert_eq!(
        auto.require_backend(cpu_caps).unwrap(),
        zkie_engine::context::ComputeSelection::Cpu
    );

    let gpu = RuntimeConfig {
        compute: zkie_engine::context::ComputeSelection::RequireGpu,
        ..RuntimeConfig::default()
    };
    assert!(matches!(
        gpu.require_backend(cpu_caps),
        Err(Error::UnsupportedBackend)
    ));
}
