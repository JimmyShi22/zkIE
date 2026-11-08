//! mmap range-read contracts.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::compute::{MemoryGovernor, MemoryRequest};
use zkie_core::error::{Error, Result};
use zkie_core::storage::{TensorReader, TensorStorageService};
use zkie_core::types::{
    Digest, DType, ElementRange, MappedTensorId, TensorId, TensorSpec, TensorStorageRef,
    TensorView,
};
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::storage::mmap::MmapTensorReader;
use zkie_engine::storage::EngineStorage;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zkie-mmap-{}-{}",
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

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn try_open_reader(
    path: &Path,
    spec: TensorSpec,
    identity: Digest,
) -> Result<MmapTensorReader> {
    let bytes = spec.byte_len()?;
    let governor = BudgetedMemoryGovernor::new(1024);
    let permit = governor.acquire(&MemoryRequest {
        mapped_window_bytes: bytes,
        ..MemoryRequest::default()
    })?;
    MmapTensorReader::open(path, spec, identity, permit)
}

fn open_reader(path: &Path, spec: TensorSpec, identity: Digest) -> MmapTensorReader {
    try_open_reader(path, spec, identity).unwrap()
}

#[test]
fn mmap_range_matches_independent_slice() {
    let temp = TempDir::new();
    let path = temp.path().join("weights.bin");
    let values = [1i32, -2, 3, -4, 5, -6];
    let bytes = i32_bytes(&values);
    fs::write(&path, &bytes).unwrap();

    let spec = TensorSpec::new(DType::I32, vec![values.len() as u64]);
    let reader = open_reader(&path, spec, Digest([9u8; 32]));
    assert_eq!(reader.element_count(), 6);
    assert_eq!(reader.mapped_len(), bytes.len());

    let mut dst = [0u8; 12];
    reader.read_range(1, 3, &mut dst).unwrap();
    assert_eq!(&dst[..], &bytes[4..16]);
}

#[test]
fn mmap_rejects_bounds_overflow_and_truncation() {
    let temp = TempDir::new();
    let path = temp.path().join("weights.bin");
    let values = [1i32, 2, 3, 4, 5, 6];
    fs::write(&path, i32_bytes(&values)).unwrap();
    let spec = TensorSpec::new(DType::I32, vec![6]);
    let reader = open_reader(&path, spec.clone(), Digest([1u8; 32]));

    let mut dst = [0u8; 8];
    assert!(matches!(
        reader.read_range(5, 2, &mut dst),
        Err(Error::InvalidIndex)
    ));
    assert!(matches!(
        reader.read_range(u64::MAX, 1, &mut dst),
        Err(Error::InvalidIndex)
    ));
    assert!(matches!(
        reader.read_range(0, 3, &mut dst),
        Err(Error::ShapeMismatch)
    ));

    let truncated = temp.path().join("truncated.bin");
    fs::write(&truncated, [0u8; 8]).unwrap();
    assert!(matches!(
        try_open_reader(&truncated, spec.clone(), Digest([2u8; 32])),
        Err(Error::IncompleteArtifact)
    ));

    let governor = BudgetedMemoryGovernor::new(8);
    let permit = governor
        .acquire(&MemoryRequest {
            mapped_window_bytes: 8,
            ..MemoryRequest::default()
        })
        .unwrap();
    assert!(matches!(
        MmapTensorReader::open(&path, spec, Digest([3u8; 32]), permit),
        Err(Error::ResourceLimit)
    ));
}

#[test]
fn engine_storage_serves_mapped_view() {
    let temp = TempDir::new();
    let path = temp.path().join("weights.bin");
    let values = [11i32, 22, 33];
    let bytes = i32_bytes(&values);
    fs::write(&path, &bytes).unwrap();
    let spec = TensorSpec::new(DType::I32, vec![3]);

    let reader = open_reader(&path, spec.clone(), Digest([3u8; 32]));
    let mut storage = EngineStorage::new(temp.path().join("scratch")).unwrap();
    storage
        .register_mapped(MappedTensorId(1), reader)
        .unwrap();
    let view = TensorView {
        tensor_id: TensorId(1),
        spec,
        storage: TensorStorageRef::Mapped(MappedTensorId(1)),
        element_range: ElementRange::new(0, 3),
    };
    let reader = storage.reader(&view).unwrap();
    let mut dst = vec![0u8; 12];
    reader.read_range(0, 3, &mut dst).unwrap();
    assert_eq!(dst, bytes);
}
