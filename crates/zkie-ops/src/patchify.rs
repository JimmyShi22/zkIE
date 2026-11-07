//! Patchify: split a sequence into fixed-size patches.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchifyOp {
    pub patch: usize,
}

impl PureForward for PatchifyOp {
    fn opcode(&self) -> OpCode {
        OpCode::Patchify
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if self.patch == 0 {
            return Err(Error::InvalidConfig);
        }
        let count = io_count(inputs.first().ok_or(Error::ShapeMismatch)?)?;
        if count % self.patch != 0 {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        if self.patch == 0 || values.len() % self.patch != 0 {
            return Err(Error::ShapeMismatch);
        }
        let patches = values.len() / self.patch;
        let mut output = Vec::with_capacity(values.len());
        for patch in 0..patches {
            output.extend_from_slice(&values[patch * self.patch..(patch + 1) * self.patch]);
        }
        Ok(KernelRecord {
            outputs: vec![output],
            auxiliaries: vec![vec![patches as i64, self.patch as i64]],
            ..KernelRecord::default()
        })
    }
}
