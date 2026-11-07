//! Fixed-point requantization with mathematical floor.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequantizeOp {
    pub shift: u32,
}

impl PureForward for RequantizeOp {
    fn opcode(&self) -> OpCode {
        OpCode::Requantize
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let spec = inputs.first().ok_or(Error::ShapeMismatch)?;
        let count = io_count(spec)?;
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        let mut output = Vec::with_capacity(values.len());
        for value in values {
            output.push(to_i64_checked(requantize(*value as i128, self.shift)?)?);
        }
        Ok(KernelRecord::with_output(output))
    }
}
