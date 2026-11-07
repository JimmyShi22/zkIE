//! Deterministic worker-pool and overlap-gate contracts.

use std::time::Duration;

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::parallel::{
    choose_overlap_or_serial, overlap_allowed, run_deterministic, LogicalTaskKey, OverlapKind,
    OverlapSwitch, SerialTaskExecutor, SerialWitness, TaskExecutor, TaskPhase, ThreadPool,
    DEFAULT_WORKERS,
};

type Task = Box<dyn FnOnce() -> Result<u64> + Send>;

fn key(op_id: u32) -> LogicalTaskKey {
    LogicalTaskKey::new(1, TaskPhase::Forward, 0, op_id, 0, op_id)
}

fn tasks() -> Vec<(LogicalTaskKey, Task)> {
    vec![
        (
            key(2),
            Box::new(|| {
                std::thread::sleep(Duration::from_millis(20));
                Ok(200u64)
            }),
        ),
        (key(0), Box::new(|| Ok(0u64))),
        (
            key(1),
            Box::new(|| {
                std::thread::sleep(Duration::from_millis(5));
                Ok(100u64)
            }),
        ),
    ]
}

fn assert_normalized(results: &[zkie_engine::parallel::TaskResult<u64>]) {
    let keys: Vec<u32> = results.iter().map(|result| result.key.op_id).collect();
    let values: Vec<u64> = results.iter().map(|result| result.value).collect();
    assert_eq!(keys, vec![0, 1, 2]);
    assert_eq!(values, vec![0, 100, 200]);
}

#[test]
fn worker_count_does_not_change_normalized_results() {
    let serial = SerialTaskExecutor;
    assert_normalized(&run_deterministic(&serial, tasks()).unwrap());
    assert_eq!(serial.worker_count(), 1);

    for workers in [1usize, 4, 8] {
        let pool = ThreadPool::new(workers).unwrap();
        assert_eq!(pool.worker_count(), workers);
        assert_normalized(&run_deterministic(&pool, tasks()).unwrap());
    }
    assert_eq!(DEFAULT_WORKERS, 64);
    assert!(ThreadPool::new(0).is_err());
}

#[test]
fn duplicate_keys_are_rejected() {
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
fn overlap_is_off_by_default_and_allowlisted() {
    let mut switch = OverlapSwitch::default();
    assert!(!switch.is_enabled());
    assert!(!switch.is_enabled_for(OverlapKind::LayerKernelWithPreviousWitness));
    assert!(overlap_allowed(
        OverlapKind::LayerKernelWithPreviousWitness
    ));
    assert!(!overlap_allowed(OverlapKind::UnsafeChallengelessRead));
    assert!(matches!(
        switch.enable(OverlapKind::UnsafeChallengelessRead),
        Err(Error::InvalidConfig)
    ));

    switch
        .enable(OverlapKind::LayerKernelWithPreviousWitness)
        .unwrap();
    assert!(switch.is_enabled_for(OverlapKind::LayerKernelWithPreviousWitness));
    switch.disable();
    assert!(!switch.is_enabled());
}

#[test]
fn overlap_must_match_serial_witness_or_fall_back() {
    let serial = SerialWitness {
        random_stream_index: 9,
        transcript_digest: Digest([1u8; 32]),
        ledger_digest: Digest([2u8; 32]),
    };
    assert!(choose_overlap_or_serial(&serial, &serial));

    let mismatch = SerialWitness {
        random_stream_index: 10,
        ..serial
    };
    assert!(!choose_overlap_or_serial(&serial, &mismatch));
}
