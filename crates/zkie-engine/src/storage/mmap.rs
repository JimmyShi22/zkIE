//! Read-only memory-mapped tensor files.
//!
//! A range read copies only the requested bytes. The file is mapped, not read
//! into a whole-file buffer; pages touched by a copy are charged to RSS by the
//! OS.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use zkie_core::compute::MemoryPermit;
use zkie_core::error::{Error, Result};
use zkie_core::storage::TensorReader;
use zkie_core::types::{Digest, ElementRange, MappedTensorId, TensorSpec};

const PAGE_BYTES: u64 = 4096;

pub struct MmapTensorReader {
    mmap: Mmap,
    spec: TensorSpec,
    identity: Digest,
    // The permit stays with the mapping so the mapped window is accounted for
    // until the reader is dropped.
    _permit: MemoryPermit,
}

impl MmapTensorReader {
    pub fn open(
        path: impl AsRef<Path>,
        spec: TensorSpec,
        identity: Digest,
        permit: MemoryPermit,
    ) -> Result<Self> {
        spec.validate()?;
        let expected = spec.byte_len()?;
        if permit.request().mapped_window_bytes < expected {
            return Err(Error::ResourceLimit);
        }
        let file = File::open(path).map_err(|_| Error::Io)?;
        let mmap = unsafe { Mmap::map(&file).map_err(|_| Error::Io)? };
        if (mmap.len() as u64) < expected {
            return Err(Error::IncompleteArtifact);
        }
        Ok(Self {
            mmap,
            spec,
            identity,
            _permit: permit,
        })
    }

    pub fn mapped_len(&self) -> usize {
        self.mmap.len()
    }
}

impl TensorReader for MmapTensorReader {
    fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    fn element_count(&self) -> u64 {
        self.spec.element_count().unwrap_or(0)
    }

    fn identity(&self) -> Digest {
        self.identity
    }

    fn read_range(
        &self,
        start_element: u64,
        element_count: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        let count = u64::try_from(element_count).map_err(|_| Error::InvalidIndex)?;
        let range = ElementRange::new(start_element, count);
        range.validate(&self.spec)?;
        let (start_byte, byte_len) = range.byte_range(&self.spec)?;
        let end_byte = start_byte
            .checked_add(byte_len)
            .ok_or(Error::InvalidIndex)?;
        if end_byte > self.mmap.len() as u64 {
            return Err(Error::IncompleteArtifact);
        }
        if dst.len() as u64 != byte_len {
            return Err(Error::ShapeMismatch);
        }

        let aligned_start = (start_byte / PAGE_BYTES) * PAGE_BYTES;
        let aligned_end = end_byte
            .checked_add(PAGE_BYTES - 1)
            .ok_or(Error::InvalidIndex)?
            / PAGE_BYTES
            * PAGE_BYTES;
        let aligned_end = aligned_end.min(self.mmap.len() as u64);
        let slice = &self.mmap[aligned_start as usize..aligned_end as usize];
        let offset = (start_byte - aligned_start) as usize;
        dst.copy_from_slice(&slice[offset..offset + byte_len as usize]);
        Ok(())
    }
}

/// A read window budget for one asset reader. Every range read is charged to
/// the window limit and to a total limit; a request above either is rejected
/// before the mapping is touched. The budget counts bytes copied out of the
/// mapping, not the mapping itself.
pub struct WindowBudget {
    max_window_bytes: u64,
    max_total_bytes: u64,
    used_bytes: u64,
    peak_window_bytes: u64,
}

impl WindowBudget {
    pub fn new(max_window_bytes: u64, max_total_bytes: u64) -> Result<Self> {
        if max_window_bytes == 0 || max_total_bytes == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            max_window_bytes,
            max_total_bytes,
            used_bytes: 0,
            peak_window_bytes: 0,
        })
    }

    pub fn max_window_bytes(&self) -> u64 {
        self.max_window_bytes
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn peak_window_bytes(&self) -> u64 {
        self.peak_window_bytes
    }

    fn charge(&mut self, bytes: u64) -> Result<()> {
        if bytes == 0 || bytes > self.max_window_bytes {
            return Err(Error::ResourceLimit);
        }
        let total = self
            .used_bytes
            .checked_add(bytes)
            .ok_or(Error::ResourceLimit)?;
        if total > self.max_total_bytes {
            return Err(Error::ResourceLimit);
        }
        self.used_bytes = total;
        self.peak_window_bytes = self.peak_window_bytes.max(bytes);
        Ok(())
    }
}

/// A read-only asset reader whose copies out of the mapping are bounded by a
/// window budget. It exposes no whole-file copy: callers read element ranges,
/// and each read is charged before it touches the mapping. The mapping itself
/// stays read-only and is accounted for by the permit that came with it.
pub struct BoundedAssetReader {
    reader: MmapTensorReader,
    budget: WindowBudget,
    mapped_window_bytes: u64,
}

impl BoundedAssetReader {
    pub fn open(
        path: impl AsRef<Path>,
        spec: TensorSpec,
        identity: Digest,
        permit: MemoryPermit,
        max_window_bytes: u64,
        max_total_bytes: u64,
    ) -> Result<Self> {
        let mapped_window_bytes = permit.request().mapped_window_bytes;
        let budget = WindowBudget::new(max_window_bytes, max_total_bytes)?;
        let reader = MmapTensorReader::open(path, spec, identity, permit)?;
        Ok(Self {
            reader,
            budget,
            mapped_window_bytes,
        })
    }

    pub fn spec(&self) -> &TensorSpec {
        self.reader.spec()
    }

    pub fn identity(&self) -> Digest {
        self.reader.identity()
    }

    pub fn mapped_window_bytes(&self) -> u64 {
        self.mapped_window_bytes
    }

    pub fn budget(&self) -> &WindowBudget {
        &self.budget
    }

    pub fn read_range(
        &mut self,
        start_element: u64,
        element_count: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        let bytes = u64::try_from(dst.len()).map_err(|_| Error::InvalidIndex)?;
        self.budget.charge(bytes)?;
        self.reader.read_range(start_element, element_count, dst)
    }
}

#[derive(Clone, Default)]
pub struct MmapRegistry {
    readers: BTreeMap<MappedTensorId, Arc<MmapTensorReader>>,
}

impl MmapRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, id: MappedTensorId, reader: MmapTensorReader) -> Result<()> {
        if self.readers.insert(id, Arc::new(reader)).is_some() {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    pub fn reader(&self, id: MappedTensorId) -> Result<Arc<dyn TensorReader>> {
        let reader = self.readers.get(&id).ok_or(Error::InvalidIndex)?;
        Ok(Arc::clone(reader) as Arc<dyn TensorReader>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use zkie_core::compute::{MemoryGovernor, MemoryRequest};
    use crate::memory::BudgetedMemoryGovernor;
    use zkie_core::types::DType;

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_file(bytes: &[u8]) -> std::path::PathBuf {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zkie-mmap-window-{}-{}-{}",
            std::process::id(),
            id,
            bytes.len()
        ));
        std::fs::write(&path, bytes).expect("write");
        path
    }

    fn spec(elements: u64) -> TensorSpec {
        TensorSpec::new(DType::I32, vec![elements])
    }

    fn permit(bytes: u64) -> MemoryPermit {
        let governor = BudgetedMemoryGovernor::new(bytes);
        governor
            .acquire(&MemoryRequest {
                mapped_window_bytes: bytes,
                ..MemoryRequest::default()
            })
            .expect("permit")
    }

    #[test]
    fn bounded_reads_charge_the_window_and_the_total() {
        let bytes: Vec<u8> = (0..4096u32).flat_map(|v| v.to_le_bytes()).collect();
        let path = temp_file(&bytes);
        let file_bytes = bytes.len() as u64;
        let mut reader = BoundedAssetReader::open(
            &path,
            spec(1024),
            Digest::ZERO,
            permit(file_bytes),
            64,
            128,
        )
        .expect("reader");
        assert_eq!(reader.mapped_window_bytes(), file_bytes);
        let mut first = [0u8; 32];
        reader.read_range(0, 8, &mut first).expect("first");
        assert_eq!(first[..4], 0u32.to_le_bytes());
        assert_eq!(reader.budget().used_bytes(), 32);
        assert_eq!(reader.budget().peak_window_bytes(), 32);
        let mut second = [0u8; 64];
        reader.read_range(8, 16, &mut second).expect("second");
        assert_eq!(reader.budget().used_bytes(), 96);
        assert_eq!(reader.budget().peak_window_bytes(), 64);
        let mut too_large = [0u8; 72];
        assert!(matches!(
            reader.read_range(0, 18, &mut too_large),
            Err(Error::ResourceLimit)
        ));
        let mut beyond_total = [0u8; 40];
        assert!(matches!(
            reader.read_range(0, 10, &mut beyond_total),
            Err(Error::ResourceLimit)
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_truncated_file_is_rejected_before_any_read() {
        let bytes = vec![0u8; 64];
        let path = temp_file(&bytes);
        let result = BoundedAssetReader::open(
            &path,
            spec(1024),
            Digest::ZERO,
            permit(4096),
            64,
            128,
        );
        assert!(matches!(result, Err(Error::IncompleteArtifact)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn out_of_range_reads_are_rejected() {
        let bytes = vec![0u8; 64];
        let path = temp_file(&bytes);
        let mut reader = BoundedAssetReader::open(
            &path,
            spec(16),
            Digest::ZERO,
            permit(64),
            32,
            64,
        )
        .expect("reader");
        let mut dst = [0u8; 8];
        assert!(reader.read_range(15, 2, &mut dst).is_err());
        assert_eq!(reader.budget().used_bytes(), 8);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_zero_budget_is_rejected() {
        assert!(WindowBudget::new(0, 1).is_err());
        assert!(WindowBudget::new(1, 0).is_err());
    }

    #[test]
    fn a_tampered_file_reads_the_tampered_bytes() {
        // The reader authenticates the file through the committed table, not
        // here: a changed byte is returned as it is and the table tree or
        // record check rejects the change later.
        let bytes: Vec<u8> = (0..16u32).flat_map(|v| v.to_le_bytes()).collect();
        let path = temp_file(&bytes);
        let mut tampered = bytes.clone();
        tampered[0] ^= 1;
        std::fs::write(&path, &tampered).expect("tamper");
        let mut reader = BoundedAssetReader::open(
            &path,
            spec(16),
            Digest::ZERO,
            permit(64),
            32,
            64,
        )
        .expect("reader");
        let mut dst = [0u8; 4];
        reader.read_range(0, 1, &mut dst).expect("read");
        assert_eq!(dst, [1u8, 0, 0, 0]);
        let _ = std::fs::remove_file(path);
    }
}
