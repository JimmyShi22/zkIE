//! Deterministic task identity, a bounded worker pool and overlap gating.
//!
//! Workers return results carrying a fixed logical key. The coordinator sorts
//! by that key before publishing anything, so completion order and worker count
//! cannot change the normalized result.

use std::collections::BTreeSet;
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

pub const DEFAULT_WORKERS: usize = 64;
pub const DEFAULT_QUEUE_DEPTH: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPhase {
    Commit,
    Forward,
    Witness,
    Prove,
    Verify,
    Serialize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalTaskKey {
    pub request_id: u64,
    pub phase: TaskPhase,
    pub layer_id: u32,
    pub op_id: u32,
    pub round_id: u32,
    pub block_id: u32,
}

impl LogicalTaskKey {
    pub const fn new(
        request_id: u64,
        phase: TaskPhase,
        layer_id: u32,
        op_id: u32,
        round_id: u32,
        block_id: u32,
    ) -> Self {
        Self {
            request_id,
            phase,
            layer_id,
            op_id,
            round_id,
            block_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskResult<T> {
    pub key: LogicalTaskKey,
    pub value: T,
}

pub trait TaskExecutor: Send + Sync {
    fn worker_count(&self) -> usize;

    fn execute<T, F>(&self, tasks: Vec<(LogicalTaskKey, F)>) -> Result<Vec<TaskResult<T>>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SerialTaskExecutor;

impl TaskExecutor for SerialTaskExecutor {
    fn worker_count(&self) -> usize {
        1
    }

    fn execute<T, F>(&self, tasks: Vec<(LogicalTaskKey, F)>) -> Result<Vec<TaskResult<T>>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let mut results = Vec::with_capacity(tasks.len());
        for (key, task) in tasks {
            results.push(TaskResult {
                key,
                value: task()?,
            });
        }
        Ok(results)
    }
}

enum PoolJob {
    Run(Box<dyn FnOnce() + Send>),
    Shutdown,
}

pub struct ThreadPool {
    sender: SyncSender<PoolJob>,
    workers: usize,
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(workers: usize) -> Result<Self> {
        Self::with_queue_depth(workers, DEFAULT_QUEUE_DEPTH)
    }

    pub fn with_queue_depth(workers: usize, queue_depth: usize) -> Result<Self> {
        if workers == 0 || queue_depth == 0 {
            return Err(Error::InvalidConfig);
        }
        let (sender, receiver) = mpsc::sync_channel::<PoolJob>(queue_depth);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let receiver = Arc::clone(&receiver);
            let handle = thread::Builder::new()
                .name(format!("zkie-worker-{index}"))
                .spawn(move || loop {
                    let job = {
                        let guard = match receiver.lock() {
                            Ok(guard) => guard,
                            Err(_) => break,
                        };
                        guard.recv()
                    };
                    match job {
                        Ok(PoolJob::Run(job)) => job(),
                        Ok(PoolJob::Shutdown) | Err(_) => break,
                    }
                })
                .map_err(|_| Error::Io)?;
            handles.push(handle);
        }
        Ok(Self {
            sender,
            workers,
            handles,
        })
    }

    pub fn worker_count(&self) -> usize {
        self.workers
    }

    pub fn execute_tasks<T, F>(
        &self,
        tasks: Vec<(LogicalTaskKey, F)>,
    ) -> Result<Vec<TaskResult<T>>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let count = tasks.len();
        let (result_sender, result_receiver) = mpsc::channel();
        for (key, task) in tasks {
            let sender = result_sender.clone();
            self.sender
                .send(PoolJob::Run(Box::new(move || {
                    let value = task();
                    let _ = sender.send((key, value));
                })))
                .map_err(|_| Error::SynchronizationFailed)?;
        }
        drop(result_sender);

        let mut results = Vec::with_capacity(count);
        for _ in 0..count {
            let (key, value) = result_receiver
                .recv()
                .map_err(|_| Error::SynchronizationFailed)?;
            results.push(TaskResult { key, value: value? });
        }
        Ok(results)
    }
}

impl TaskExecutor for ThreadPool {
    fn worker_count(&self) -> usize {
        ThreadPool::worker_count(self)
    }

    fn execute<T, F>(&self, tasks: Vec<(LogicalTaskKey, F)>) -> Result<Vec<TaskResult<T>>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        ThreadPool::execute_tasks(self, tasks)
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in 0..self.workers {
            let _ = self.sender.send(PoolJob::Shutdown);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Sort by logical key and reject duplicate keys.
pub fn collect_in_key_order<T>(mut results: Vec<TaskResult<T>>) -> Result<Vec<TaskResult<T>>> {
    results.sort_by_key(|result| result.key);
    for pair in results.windows(2) {
        if pair[0].key == pair[1].key {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(results)
}

pub fn run_deterministic<E, T, F>(
    executor: &E,
    tasks: Vec<(LogicalTaskKey, F)>,
) -> Result<Vec<TaskResult<T>>>
where
    E: TaskExecutor,
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    collect_in_key_order(executor.execute(tasks)?)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OverlapKind {
    LayerKernelWithPreviousWitness,
    SecondPassWeightRead,
    CommitWarmupWithProveStart,
    OpeningWithSerialization,
    UnsafeChallengelessRead,
}

pub const ALLOWED_OVERLAP: [OverlapKind; 4] = [
    OverlapKind::LayerKernelWithPreviousWitness,
    OverlapKind::SecondPassWeightRead,
    OverlapKind::CommitWarmupWithProveStart,
    OverlapKind::OpeningWithSerialization,
];

pub fn overlap_allowed(kind: OverlapKind) -> bool {
    ALLOWED_OVERLAP.contains(&kind)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OverlapSwitch {
    enabled: bool,
    kinds: BTreeSet<OverlapKind>,
}

impl OverlapSwitch {
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn is_enabled_for(&self, kind: OverlapKind) -> bool {
        self.enabled && self.kinds.contains(&kind)
    }

    pub fn enable(&mut self, kind: OverlapKind) -> Result<()> {
        if !overlap_allowed(kind) {
            return Err(Error::InvalidConfig);
        }
        self.kinds.insert(kind);
        self.enabled = true;
        Ok(())
    }

    pub fn disable(&mut self) {
        self.enabled = false;
        self.kinds.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerialWitness {
    pub random_stream_index: u64,
    pub transcript_digest: Digest,
    pub ledger_digest: Digest,
}

pub fn overlap_matches(serial: &SerialWitness, overlap: &SerialWitness) -> bool {
    serial == overlap
}

/// Returns true only when the overlap witness matches the serial witness;
/// otherwise the caller must fall back to the serial path.
pub fn choose_overlap_or_serial(serial: &SerialWitness, overlap: &SerialWitness) -> bool {
    overlap_matches(serial, overlap)
}
