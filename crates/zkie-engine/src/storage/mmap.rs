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
