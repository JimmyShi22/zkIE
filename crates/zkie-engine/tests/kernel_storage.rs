//! The same kernels over resident, mmap and scratch sources, with ranged reads
//! only and explicit workspace accounting.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use zkie_core::common::field::{GOLDILOCKS, GOLDILOCKS_EXT3};
use zkie_core::compute::{MemoryGovernor, MemoryRequest};
use zkie_core::error::Error;
use zkie_core::storage::{TensorReader, TensorStorageService};
use zkie_core::types::{
    DType, Digest, ElementRange, TensorSpec,
};

use zkie_engine::compute::cpu::fixed_matmul;
use zkie_engine::compute::{encode_i64, read_fp3_tiles, read_fp_tiles, read_i64_tiles};
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::pcs::request::field::{fp_from_u64, fp_to_u64, Fp3, MODULUS};
use zkie_engine::storage::mmap::MmapTensorReader;
use zkie_engine::storage::EngineStorage;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zkie-kernel-{}-{}",
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

fn request(mapped: u64, scratch: u64) -> MemoryRequest {
    MemoryRequest {
        resident_bytes: 0,
        mapped_window_bytes: mapped,
        pinned_bytes: 0,
        metadata_bytes: 0,
        scratch_disk_bytes: scratch,
    }
}

struct CountingReader {
    inner: Arc<dyn TensorReader>,
    calls: AtomicU64,
    max_bytes: AtomicU64,
}

impl CountingReader {
    fn new(inner: Arc<dyn TensorReader>) -> Self {
        Self {
            inner,
            calls: AtomicU64::new(0),
            max_bytes: AtomicU64::new(0),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    fn max_bytes(&self) -> u64 {
        self.max_bytes.load(Ordering::Relaxed)
    }
}

impl TensorReader for CountingReader {
    fn spec(&self) -> &TensorSpec {
        self.inner.spec()
    }

    fn element_count(&self) -> u64 {
        self.inner.element_count()
    }

    fn identity(&self) -> Digest {
        self.inner.identity()
    }

    fn read_range(
        &self,
        start_element: u64,
        element_count: usize,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.max_bytes
            .fetch_max(dst.len() as u64, Ordering::Relaxed);
        self.inner.read_range(start_element, element_count, dst)
    }
}

fn open_mmap(
    path: &Path,
    spec: &TensorSpec,
    governor: &BudgetedMemoryGovernor,
    bytes: u64,
) -> Arc<dyn TensorReader> {
    let permit = governor.acquire(&request(bytes, 0)).unwrap();
    let reader = MmapTensorReader::open(path, spec.clone(), Digest([7u8; 32]), permit).unwrap();
    Arc::new(reader)
}

#[test]
fn resident_mmap_and_scratch_sources_agree_with_tiled_reads() {
    let temp = TempDir::new();
    let a = vec![1i64, -2, 3, 4, 5, -6, 7, 8, -9, 10, 11, -12, 13, -14, 15, 16];
    let spec = TensorSpec::new(DType::I64, vec![4, 4]);
    let byte_len = spec.byte_len().unwrap();
    let governor = BudgetedMemoryGovernor::new(1 << 20);

    let path = temp.path().join("a.bin");
    fs::write(&path, encode_i64(&a)).unwrap();
    let mapped_reader = CountingReader::new(open_mmap(&path, &spec, &governor, byte_len));
    assert_eq!(governor.snapshot().reserved_bytes, byte_len);
    let mapped_values = read_i64_tiles(&mapped_reader, 3).unwrap();
    assert_eq!(mapped_values, a);
    assert_eq!(mapped_reader.max_bytes(), 3 * 8);
    assert!(mapped_reader.calls() >= 6);
    assert_eq!(governor.snapshot().reserved_bytes, byte_len);

    let storage = EngineStorage::new(temp.path().join("scratch")).unwrap();
    let scratch_permit = governor.acquire(&request(0, byte_len)).unwrap();
    let id = storage.create_scratch(&spec, &scratch_permit).unwrap();
    storage
        .write_scratch(id, ElementRange::new(0, a.len() as u64), &encode_i64(&a))
        .unwrap();
    let view = storage.seal_scratch(id).unwrap();
    let scratch_reader = storage.reader(&view).unwrap();
    let scratch_values = read_i64_tiles(scratch_reader.as_ref(), 4).unwrap();
    assert_eq!(scratch_values, a);

    let expected = fixed_matmul(&a, &a, 4, 4, 4).unwrap();
    let from_mapped = fixed_matmul(&mapped_values, &mapped_values, 4, 4, 4).unwrap();
    let from_scratch = fixed_matmul(&scratch_values, &scratch_values, 4, 4, 4).unwrap();
    assert_eq!(from_mapped, expected);
    assert_eq!(from_scratch, expected);

    drop(mapped_reader);
    assert_eq!(governor.snapshot().reserved_bytes, byte_len);
    drop(scratch_permit);
    assert_eq!(governor.snapshot().reserved_bytes, 0);
    storage.release_scratch(id).unwrap();
}

#[test]
fn field_sources_reject_non_canonical_words_and_mismatched_dtypes() {
    let temp = TempDir::new();
    let governor = BudgetedMemoryGovernor::new(1 << 20);
    let spec = TensorSpec::new(DType::BaseField(&GOLDILOCKS), vec![3]);

    let good = temp.path().join("good.bin");
    let mut bytes = Vec::new();
    for value in [1u64, 2, 3] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&good, &bytes).unwrap();
    let reader = open_mmap(&good, &spec, &governor, spec.byte_len().unwrap());
    let values = read_fp_tiles(reader.as_ref(), 2).unwrap();
    assert_eq!(
        values.iter().map(|value| fp_to_u64(*value)).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert!(matches!(
        read_i64_tiles(reader.as_ref(), 2),
        Err(Error::UnsupportedPrimitive)
    ));

    let bad = temp.path().join("bad.bin");
    let mut bad_bytes = Vec::new();
    for value in [1u64, MODULUS, 3] {
        bad_bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&bad, &bad_bytes).unwrap();
    let reader = open_mmap(&bad, &spec, &governor, spec.byte_len().unwrap());
    assert!(matches!(
        read_fp_tiles(reader.as_ref(), 3),
        Err(Error::InvalidEncoding)
    ));

    let extension_spec = TensorSpec::new(DType::ExtensionField(&GOLDILOCKS_EXT3), vec![2]);
    let storage = EngineStorage::new(temp.path().join("scratch")).unwrap();
    let permit = governor
        .acquire(&request(0, extension_spec.byte_len().unwrap()))
        .unwrap();
    let id = storage.create_scratch(&extension_spec, &permit).unwrap();
    let values = [
        Fp3::new(fp_from_u64(1), fp_from_u64(2), fp_from_u64(3)),
        Fp3::new(fp_from_u64(4), fp_from_u64(5), fp_from_u64(6)),
    ];
    let mut encoded = Vec::new();
    for value in &values {
        encoded.extend_from_slice(&value.encode());
    }
    storage
        .write_scratch(id, ElementRange::new(0, 2), &encoded)
        .unwrap();
    let view = storage.seal_scratch(id).unwrap();
    let reader = storage.reader(&view).unwrap();
    assert_eq!(read_fp3_tiles(reader.as_ref(), 1).unwrap(), values);
}

#[test]
fn workspace_requests_are_enforced_and_released() {
    let governor = BudgetedMemoryGovernor::new(1000);
    let mapped = governor.acquire(&request(128, 0)).unwrap();
    assert_eq!(governor.snapshot().reserved_bytes, 128);
    assert!(matches!(
        governor.acquire(&request(900, 0)),
        Err(Error::ResourceLimit)
    ));
    let scratch = governor.acquire(&request(0, 800)).unwrap();
    assert_eq!(governor.snapshot().reserved_bytes, 928);
    drop(scratch);
    drop(mapped);
    assert_eq!(governor.snapshot().reserved_bytes, 0);

    let temp = TempDir::new();
    let storage = EngineStorage::new(temp.path().join("scratch")).unwrap();
    let spec = TensorSpec::new(DType::I64, vec![16]);
    let permit = governor.acquire(&request(0, 64)).unwrap();
    assert!(matches!(
        storage.create_scratch(&spec, &permit),
        Err(Error::ResourceLimit)
    ));
}

#[test]
fn tile_bounds_do_not_change_results_across_sources() {
    let temp = TempDir::new();
    let a = vec![2i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    let b = vec![1i64, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6];
    let spec = TensorSpec::new(DType::I64, vec![3, 4]);
    let governor = BudgetedMemoryGovernor::new(1 << 20);

    let path = temp.path().join("a.bin");
    fs::write(&path, encode_i64(&a)).unwrap();
    let reader = open_mmap(&path, &spec, &governor, spec.byte_len().unwrap());
    let mut expected = None;
    for tile in [1usize, 2, 5, 12] {
        let values = read_i64_tiles(reader.as_ref(), tile).unwrap();
        let product = fixed_matmul(&values, &b, 3, 3, 4).unwrap();
        match &expected {
            None => expected = Some(product),
            Some(first) => assert_eq!(&product, first),
        }
    }
}
