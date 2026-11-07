//! KV cache append at an explicit position.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCacheOp {
    pub capacity: usize,
    pub width: usize,
    pub position: usize,
}

impl PureForward for KvCacheOp {
    fn opcode(&self) -> OpCode {
        OpCode::KvCache
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 4 || self.position >= self.capacity || self.width == 0 {
            return Err(Error::InvalidConfig);
        }
        let cache = self.capacity * self.width;
        if io_count(&inputs[0])? != cache
            || io_count(&inputs[1])? != cache
            || io_count(&inputs[2])? != self.width
            || io_count(&inputs[3])? != self.width
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![
            crate::registry::spec_1d(cache),
            crate::registry::spec_1d(cache),
        ])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 4 {
            return Err(Error::ShapeMismatch);
        }
        let mut keys = inputs[0].clone();
        let mut values = inputs[1].clone();
        if self.position >= self.capacity || self.width == 0 {
            return Err(Error::InvalidConfig);
        }
        let start = self.position * self.width;
        keys[start..start + self.width].copy_from_slice(&inputs[2]);
        values[start..start + self.width].copy_from_slice(&inputs[3]);
        Ok(KernelRecord {
            outputs: vec![keys, values],
            ..KernelRecord::default()
        })
    }
}
