//! Memory permit, scratch and fallback contracts.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::common::field::Goldilocks;
use zkie_core::compute::{
    ComputeBackend, ComputeCapabilities, ExecutionPort, KernelId, KernelJob, MemoryGovernor,
    MemoryPermit, MemoryRequest, MemoryUsage, RuntimeBackendId,
};
use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressSink};
use zkie_core::storage::TensorStorageService;
use zkie_core::types::{
    AllocationId, BufferGeneration, BufferHandle, BufferOwner, BufferSpec, DType, ElementRange,
    Fence, TensorSpec, TensorStorageRef, WorkspaceRequirement,
};
use zkie_engine::context::{ExecContext, FallbackPolicy, Metrics, RuntimeConfig};
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::storage::EngineStorage;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zkie-memory-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[derive(Default)]
struct Sink;

impl ProgressSink for Sink {
    fn emit(&self, _event: ProgressEvent) -> Result<()> {
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct MockBackend {
    next: u64,
    released: Vec<AllocationId>,
}

impl ComputeBackend<Goldilocks> for MockBackend {
    fn identity(&self) -> RuntimeBackendId {
        RuntimeBackendId::CPU
    }

    fn capabilities(&self) -> ComputeCapabilities {
        ComputeCapabilities::cpu()
    }

    fn allocate(&mut self, spec: &BufferSpec, _permit: &MemoryPermit) -> Result<BufferHandle> {
        self.next = self.next.checked_add(1).ok_or(Error::ResourceLimit)?;
        Ok(BufferHandle::new(
            BufferOwner(1),
            BufferGeneration(0),
            AllocationId(self.next),
            spec.dtype,
            spec.element_count,
        ))
    }

    fn upload(&mut self, _dst: BufferHandle, _canonical_bytes: &[u8]) -> Result<Fence> {
        Ok(Fence::new(BufferOwner(1), self.next))
    }

    fn download_range(
        &mut self,
        _src: BufferHandle,
        _range: ElementRange,
        _wait_for: &[Fence],
        _dst: &mut [u8],
    ) -> Result<()> {
        Ok(())
    }

    fn submit(
        &mut self,
        _job: &KernelJob<Goldilocks>,
        _inputs: &[BufferHandle],
        _outputs: &[BufferHandle],
        _wait_for: &[Fence],
    ) -> Result<Fence> {
        Ok(Fence::new(BufferOwner(1), self.next))
    }

    fn wait(&mut self, _fence: Fence) -> Result<()> {
        Ok(())
    }

    fn synchronize(&mut self) -> Result<()> {
        Ok(())
    }

    fn release(&mut self, buffer: BufferHandle) -> Result<()> {
        self.released.push(buffer.allocation_id);
        Ok(())
    }

    fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage::default()
    }
}

#[test]
fn snapshot_tracks_categories_and_releases_on_drop() {
    let governor = BudgetedMemoryGovernor::new(2000);
    let request = MemoryRequest {
        resident_bytes: 100,
        mapped_window_bytes: 200,
        pinned_bytes: 50,
        metadata_bytes: 10,
        scratch_disk_bytes: 300,
    };
    let permit = governor.acquire(&request).unwrap();
    let snapshot = governor.snapshot();
    assert_eq!(snapshot.reserved_bytes, 660);
    assert_eq!(snapshot.resident_bytes, 100);
    assert_eq!(snapshot.mapped_window_bytes, 200);
    assert_eq!(snapshot.pinned_bytes, 50);
    assert_eq!(snapshot.metadata_bytes, 10);
    assert_eq!(snapshot.scratch_disk_bytes, 300);
    assert_eq!(snapshot.remaining_bytes(), 1340);
    assert_eq!(snapshot.peak_bytes, 660);

    drop(permit);
    let snapshot = governor.snapshot();
    assert_eq!(snapshot.reserved_bytes, 0);
    assert_eq!(snapshot.resident_bytes, 0);
    assert_eq!(snapshot.scratch_disk_bytes, 0);
}

#[test]
fn insufficient_budget_fails_before_reservation() {
    let governor = BudgetedMemoryGovernor::new(100);
    let request = MemoryRequest {
        resident_bytes: 101,
        ..MemoryRequest::default()
    };
    assert!(matches!(
        governor.acquire(&request),
        Err(Error::ResourceLimit)
    ));
    assert_eq!(governor.snapshot().reserved_bytes, 0);

    let workspace = WorkspaceRequirement {
        minimum_resident_bytes: 17,
        selected_resident_bytes: 16,
        ..WorkspaceRequirement::default()
    };
    assert!(matches!(workspace.validate(), Err(Error::ResourceLimit)));
}

#[test]
fn scratch_is_writable_until_sealed_then_immutable() {
    let temp = TempDir::new();
    let storage = EngineStorage::new(temp.path().join("scratch")).unwrap();
    let spec = TensorSpec::new(DType::I32, vec![4]);
    let governor = BudgetedMemoryGovernor::new(64);
    let permit = governor
        .acquire(&MemoryRequest {
            scratch_disk_bytes: 16,
            ..MemoryRequest::default()
        })
        .unwrap();

    let id = storage.create_scratch(&spec, &permit).unwrap();
    let bytes: Vec<u8> = (0..4i32).flat_map(|value| value.to_le_bytes()).collect();
    storage
        .write_scratch(id, ElementRange::new(0, 4), &bytes)
        .unwrap();
    let view = storage.seal_scratch(id).unwrap();
    let reader = storage.reader(&view).unwrap();
    let mut dst = vec![0u8; 16];
    reader.read_range(0, 4, &mut dst).unwrap();
    assert_eq!(dst, bytes);

    assert!(matches!(
        storage.write_scratch(id, ElementRange::new(0, 4), &bytes),
        Err(Error::InvalidSessionState)
    ));
    storage.release_scratch(id).unwrap();
    assert!(storage.reader(&view).is_err());
}

#[test]
fn exec_context_implements_execution_port() {
    let temp = TempDir::new();
    let storage = EngineStorage::new(temp.path().join("scratch")).unwrap();
    let spec = TensorSpec::new(DType::I32, vec![4]);
    let governor = BudgetedMemoryGovernor::new(1024);
    let permit = governor
        .acquire(&MemoryRequest {
            scratch_disk_bytes: 16,
            ..MemoryRequest::default()
        })
        .unwrap();
    let id = storage.create_scratch(&spec, &permit).unwrap();
    let bytes: Vec<u8> = (0..4i32).flat_map(|value| value.to_le_bytes()).collect();
    storage
        .write_scratch(id, ElementRange::new(0, 4), &bytes)
        .unwrap();
    let input = storage.seal_scratch(id).unwrap();
    drop(permit);

    let sink = Sink;
    let mut backend = MockBackend::default();
    let mut metrics = Metrics::new();
    let runtime = RuntimeConfig::default();
    let mut ctx = ExecContext::new(
        &mut backend,
        &storage,
        &governor,
        &sink,
        &mut metrics,
        &runtime,
    );
    let job = KernelJob::new(KernelId(1), Vec::<Goldilocks>::new());
    let workspace = WorkspaceRequirement {
        minimum_resident_bytes: 0,
        selected_resident_bytes: 0,
        mapped_window_bytes: 0,
        scratch_disk_bytes: 16,
    };
    let outputs = ctx
        .execute(&job, &[input], &[spec], &workspace)
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(matches!(
        outputs[0].storage,
        TensorStorageRef::Resident(_)
    ));
    outputs[0].validate().unwrap();
    assert_eq!(metrics.get("kernel_submits"), 1);
    assert_eq!(backend.released.len(), 1);
}
