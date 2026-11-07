//! Buffer identity and validity checks.
//!
//! A handle is valid only for the owner and generation that produced it. Stale
//! handles, cross-owner handles and dtype mismatches fail before any read or
//! kernel submission.

use std::collections::BTreeMap;

use zkie_core::error::{Error, Result};
use zkie_core::types::{
    AllocationId, BufferGeneration, BufferHandle, BufferOwner, DType, Fence,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRecord {
    pub owner: BufferOwner,
    pub generation: BufferGeneration,
    pub dtype: DType,
    pub element_count: u64,
    pub live: bool,
}

#[derive(Clone, Debug, Default)]
pub struct BufferTable {
    records: BTreeMap<AllocationId, BufferRecord>,
}

impl BufferTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a handle. A retired allocation may be reused with a newer
    /// generation; a live allocation cannot be overwritten.
    pub fn register(&mut self, handle: BufferHandle) -> Result<()> {
        if let Some(record) = self.records.get(&handle.allocation_id) {
            if record.live {
                return Err(Error::InvalidHandle);
            }
            if handle.generation <= record.generation {
                return Err(Error::StateRollbackSuspected);
            }
        }
        self.records.insert(
            handle.allocation_id,
            BufferRecord {
                owner: handle.owner,
                generation: handle.generation,
                dtype: handle.dtype,
                element_count: handle.element_count,
                live: true,
            },
        );
        Ok(())
    }

    pub fn validate(
        &self,
        handle: BufferHandle,
        expected_owner: BufferOwner,
        expected_dtype: DType,
    ) -> Result<()> {
        let record = self
            .records
            .get(&handle.allocation_id)
            .ok_or(Error::InvalidHandle)?;
        if !record.live {
            return Err(Error::InvalidHandle);
        }
        if handle.owner != expected_owner || record.owner != expected_owner {
            return Err(Error::StatementMismatch);
        }
        if handle.generation != record.generation {
            return Err(Error::DeviceLost);
        }
        if handle.dtype != expected_dtype || record.dtype != expected_dtype {
            return Err(Error::ShapeMismatch);
        }
        if handle.element_count != record.element_count {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }

    pub fn retire(&mut self, handle: BufferHandle) -> Result<()> {
        let record = self
            .records
            .get_mut(&handle.allocation_id)
            .ok_or(Error::InvalidHandle)?;
        if !record.live {
            return Err(Error::InvalidHandle);
        }
        if record.owner != handle.owner {
            return Err(Error::StatementMismatch);
        }
        if record.generation != handle.generation {
            return Err(Error::DeviceLost);
        }
        record.live = false;
        Ok(())
    }

    pub fn is_live(&self, allocation_id: AllocationId) -> bool {
        self.records
            .get(&allocation_id)
            .map(|record| record.live)
            .unwrap_or(false)
    }
}

pub fn validate_fence(fence: Fence, expected_owner: BufferOwner) -> Result<()> {
    fence.check_owner(expected_owner)
}
