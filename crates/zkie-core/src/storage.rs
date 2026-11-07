//! Controlled tensor storage.
//!
//! Reads are always ranged. There is no `read_all`/`materialize_all` method on
//! the port: a kernel that needs a resident tile must ask for that tile
//! explicitly and account for it.

use std::sync::Arc;

use crate::compute::MemoryPermit;
use crate::error::Result;
use crate::types::{Digest, ElementRange, ScratchTensorId, TensorSpec, TensorView};

pub trait TensorReader: Send + Sync {
    fn spec(&self) -> &TensorSpec;
    fn element_count(&self) -> u64;
    fn identity(&self) -> Digest;

    /// Read exactly `element_count` elements starting at `start_element` into
    /// `dst`. Implementations must not read the whole backing file first.
    fn read_range(
        &self,
        start_element: u64,
        element_count: usize,
        dst: &mut [u8],
    ) -> Result<()>;
}

pub trait TensorStorageService: Send + Sync {
    fn reader(&self, view: &TensorView) -> Result<Arc<dyn TensorReader>>;

    fn create_scratch(&self, spec: &TensorSpec, permit: &MemoryPermit)
        -> Result<ScratchTensorId>;

    fn write_scratch(
        &self,
        id: ScratchTensorId,
        range: ElementRange,
        canonical_bytes: &[u8],
    ) -> Result<()>;

    fn seal_scratch(&self, id: ScratchTensorId) -> Result<TensorView>;

    fn release_scratch(&self, id: ScratchTensorId) -> Result<()>;
}
