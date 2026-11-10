//! Engine storage services.

pub mod encoded_store;
pub mod mmap;
pub mod scratch;

use std::path::PathBuf;
use std::sync::Arc;

use zkie_core::compute::MemoryPermit;
use zkie_core::error::{Error, Result};
use zkie_core::storage::{TensorReader, TensorStorageService};
use zkie_core::types::{
    ElementRange, MappedTensorId, ScratchTensorId, TensorSpec, TensorStorageRef, TensorView,
};

use self::mmap::{MmapRegistry, MmapTensorReader};
use self::scratch::ScratchStorage;

pub struct EngineStorage {
    mmap: MmapRegistry,
    scratch: ScratchStorage,
}

impl EngineStorage {
    pub fn new(scratch_dir: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            mmap: MmapRegistry::new(),
            scratch: ScratchStorage::new(scratch_dir)?,
        })
    }

    pub fn register_mapped(
        &mut self,
        id: MappedTensorId,
        reader: MmapTensorReader,
    ) -> Result<()> {
        self.mmap.register(id, reader)
    }
}

impl TensorStorageService for EngineStorage {
    fn reader(&self, view: &TensorView) -> Result<Arc<dyn TensorReader>> {
        match view.storage {
            TensorStorageRef::Mapped(id) => self.mmap.reader(id),
            TensorStorageRef::Scratch(_) => self.scratch.reader(view),
            TensorStorageRef::Resident(_) => Err(Error::UnsupportedPrimitive),
        }
    }

    fn create_scratch(
        &self,
        spec: &TensorSpec,
        permit: &MemoryPermit,
    ) -> Result<ScratchTensorId> {
        self.scratch.create_scratch(spec, permit)
    }

    fn write_scratch(
        &self,
        id: ScratchTensorId,
        range: ElementRange,
        canonical_bytes: &[u8],
    ) -> Result<()> {
        self.scratch.write_scratch(id, range, canonical_bytes)
    }

    fn seal_scratch(&self, id: ScratchTensorId) -> Result<TensorView> {
        self.scratch.seal_scratch(id)
    }

    fn release_scratch(&self, id: ScratchTensorId) -> Result<()> {
        self.scratch.release_scratch(id)
    }
}
